#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

static void ggml_log_callback_default(ggml_log_level level, const char *text,
                                      void *user_data) {
  (void)level;
  (void)user_data;
  fputs(text, stderr);
  fflush(stderr);
}

// This is a simple model with two tensors a and b
struct simple_model {
  struct ggml_tensor *a{};
  struct ggml_tensor *b{};
  ggml_backend_buffer_t backend_buffer{};
  ggml_gallocr_t allocr{};

  // CPU backend only
  ggml_backend_t backend_cpu{};

  // storage for the graph + context
  std::vector<uint8_t> buf;
  ggml_context *ctx{};
};

// initialize data of matrices to perform matrix multiplication
const int rows_A = 4, cols_A = 2;
float matrix_A[rows_A * cols_A] = {2, 8, 5, 1, 4, 2, 8, 6};

const int rows_B = 3, cols_B = 2;
/* Transpose([
    10, 9, 5,
    5, 9, 4
]) 2 rows, 3 cols */
float matrix_B[rows_B * cols_B] = {10, 5, 9, 9, 5, 4};

void init_model(simple_model &model) {
  ggml_log_set(ggml_log_callback_default, nullptr);

  // load backends, but we will only use CPU
  ggml_backend_load_all();
  model.backend_cpu =
      ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

  assert(model.backend_cpu != nullptr);
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph *build_graph(simple_model &model) {
  // bigger than minimal overhead to be safe
  size_t buf_size =
      ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
  model.buf.resize(buf_size);

  // no_alloc = false: ggml will allocate tensor data in CPU memory
  struct ggml_init_params params0 = {
      /*.mem_size   =*/buf_size,
      /*.mem_buffer =*/model.buf.data(),
      /*.no_alloc   =*/true,
  };

  // create a context to build the graph AND hold tensor data
  model.ctx = ggml_init(params0);
  assert(model.ctx != nullptr);

  // create tensors (CPU data will be allocated because no_alloc=false)
  model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
  model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_B, rows_B);

  // 1. allocate tensors in CPU backend
  model.backend_buffer =
      ggml_backend_alloc_ctx_tensors(model.ctx, model.backend_cpu);
  ggml_backend_tensor_set(model.a, matrix_A, 0, ggml_nbytes(model.a));
  ggml_backend_tensor_set(model.b, matrix_B, 0, ggml_nbytes(model.b));

  // 2. build graph
  struct ggml_cgraph *gf = ggml_new_graph(model.ctx);
  // result = a*b^T (ggml_mul_mat does (b x a) in ggml layout; keep your
  // original usage)
  struct ggml_tensor *result = ggml_mul_mat(model.ctx, model.a, model.b);
  // build operations nodes
  ggml_build_forward_expand(gf, result);

  // 3. allocate graph tensors in CPU backend
  model.allocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend_cpu));
  ggml_gallocr_alloc_graph(model.allocr, gf);

  return gf;
}

// compute with CPU backend (no scheduler)
struct ggml_tensor *compute(simple_model &model, struct ggml_cgraph *gf) {
  // copy inputs into CPU tensor buffers
  assert(model.a->data != nullptr);
  assert(model.b->data != nullptr);

  // run on CPU backend
  int n_threads = 1;
  ggml_backend_cpu_set_n_threads(model.backend_cpu, n_threads);
  ggml_backend_graph_compute(model.backend_cpu, gf);

  // output tensor is last node
  return ggml_graph_node(gf, -1);
}

int main(void) {
  ggml_time_init();

  simple_model model;
  init_model(model);

  struct ggml_cgraph *gf = build_graph(model);

  // perform computation
  struct ggml_tensor *result = compute(model, gf);

  // read result directly from CPU memory
  std::vector<float> out_data(ggml_nelements(result));
  memcpy(out_data.data(), result->data, ggml_nbytes(result));

  printf("mul mat (%d x %d) (transposed result):\n[", (int)result->ne[0],
         (int)result->ne[1]);
  for (int j = 0; j < result->ne[1]; j++) {
    if (j > 0)
      printf("\n");
    for (int i = 0; i < result->ne[0]; i++) {
      printf(" %.2f", out_data[j * result->ne[0] + i]);
    }
  }
  printf(" ]\n");

  // cleanup
  ggml_backend_buffer_free(model.backend_buffer);
  ggml_backend_free(model.backend_cpu);
  ggml_free(model.ctx); // frees tensors + graph objects allocated in ctx
  return 0;
}
