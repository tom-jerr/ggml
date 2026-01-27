#include "model.h"
#include "ggml-backend.h"
#include <fstream>
#include <ggml-opt.h>
#include <spdlog/spdlog.h>

// ======== helper function to load gguf model from file ========
bool load_from_gguf(const char *fname, struct ggml_context *ctx_ggml,
                    struct gguf_context *ctx_gguf) {
  FILE *f = ggml_fopen(fname, "rb");
  if (!f) {
    spdlog::error("failed to open model file: {}", fname);
    return false;
  }

  const size_t buf_size = 4 * 1024 * 1024;
  void *buf = malloc(buf_size);

  const int n_tensors = gguf_get_n_tensors(ctx_gguf);
  for (int i = 0; i < n_tensors; i++) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);

    struct ggml_tensor *tensor = ggml_get_tensor(ctx_ggml, name);
    if (!tensor) {
      continue;
    }

    const size_t offs =
        gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);

    if (fseek(f, offs, SEEK_SET) != 0) {
      fclose(f);
      free(buf);
      return false;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    for (size_t pos = 0; pos < nbytes; pos += buf_size) {
      const size_t nbytes_cpy =
          buf_size < nbytes - pos ? buf_size : nbytes - pos;

      if (fread(buf, 1, nbytes_cpy, f) != nbytes_cpy) {
        fclose(f);
        free(buf);
        return false;
      }

      ggml_backend_tensor_set(tensor, buf, pos, nbytes_cpy);
    }
  }

  fclose(f);
  free(buf);
  return true;
}

// ======== MnistCNN ========

MnistCNN::MnistCNN(const std::string &model_file, const int nbatch_logical,
                   const int nbatch_physical)
    : nbatch_logical(nbatch_logical), nbatch_physical(nbatch_physical) {

  const int ncores_logical = std::thread::hardware_concurrency();
  const int nthreads = std::min(ncores_logical, (ncores_logical + 4) / 2);

  // just use CPU backend for this demo
  backend_cpu =
      ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
  ggml_backend_cpu_set_n_threads(backend_cpu, nthreads);

  {
    const size_t size_meta = 1024 * ggml_tensor_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/size_meta,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ctx_static = ggml_init(params);
  }

  {
    // The compute context needs a total of 3 compute graphs: forward pass +
    // backwards pass (with/without optimizer step).
    const size_t size_meta = GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() +
                             3 * ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/size_meta,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ctx_compute = ggml_init(params);
  }
  if (!model_file.empty()) {
    init_from_file(model_file);
  } else {
    init_random();
  }
  init_input();
}

MnistCNN::~MnistCNN() {
  ggml_free(ctx_gguf);
  ggml_free(ctx_static);
  ggml_free(ctx_compute);

  ggml_backend_buffer_free(buf_gguf);
  ggml_backend_buffer_free(buf_static);
  ggml_backend_free(backend_cpu);
}

bool MnistCNN::init_input() {
  images = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, MNIST_NINPUT,
                              nbatch_physical);

  ggml_set_name(images, "images");
  ggml_set_input(images);

  buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, backend_cpu);
  spdlog::info("Successfully initialized input tensor");
  return true;
}

bool MnistCNN::init_from_file(const std::string &fname) {
  struct gguf_context *ctx;
  {
    struct gguf_init_params params = {
        /*.no_alloc   =*/true,
        /*.ctx        =*/&this->ctx_gguf,
    };
    ctx = gguf_init_from_file(fname.c_str(), params);
    if (!ctx) {
      spdlog::error("failed to load model from file: {}", fname);
      return false;
    }
  }
  // Allocate tensor metadata in ctx_gguf
  this->conv1_kernel = ggml_get_tensor(this->ctx_gguf, "conv1.kernel");
  GGML_ASSERT(this->conv1_kernel->type == GGML_TYPE_F32);
  GGML_ASSERT(this->conv1_kernel->ne[0] == 3);
  GGML_ASSERT(this->conv1_kernel->ne[1] == 3);
  GGML_ASSERT(this->conv1_kernel->ne[2] == 1);
  GGML_ASSERT(this->conv1_kernel->ne[3] == MNIST_CNN_NCB);
  this->conv1_bias = ggml_get_tensor(this->ctx_gguf, "conv1.bias");
  GGML_ASSERT(this->conv1_bias->type == GGML_TYPE_F32);
  GGML_ASSERT(this->conv1_bias->ne[0] == 1);
  GGML_ASSERT(this->conv1_bias->ne[1] == 1);
  GGML_ASSERT(this->conv1_bias->ne[2] == MNIST_CNN_NCB);
  GGML_ASSERT(this->conv1_bias->ne[3] == 1);

  this->conv2_kernel = ggml_get_tensor(this->ctx_gguf, "conv2.kernel");
  GGML_ASSERT(this->conv2_kernel->type == GGML_TYPE_F32);
  GGML_ASSERT(this->conv2_kernel->ne[0] == 3);
  GGML_ASSERT(this->conv2_kernel->ne[1] == 3);
  GGML_ASSERT(this->conv2_kernel->ne[2] == MNIST_CNN_NCB);
  GGML_ASSERT(this->conv2_kernel->ne[3] == MNIST_CNN_NCB * 2);

  this->conv2_bias = ggml_get_tensor(this->ctx_gguf, "conv2.bias");
  GGML_ASSERT(this->conv2_bias->type == GGML_TYPE_F32);
  GGML_ASSERT(this->conv2_bias->ne[0] == 1);
  GGML_ASSERT(this->conv2_bias->ne[1] == 1);
  GGML_ASSERT(this->conv2_bias->ne[2] == MNIST_CNN_NCB * 2);
  GGML_ASSERT(this->conv2_bias->ne[3] == 1);

  this->dense_weight = ggml_get_tensor(this->ctx_gguf, "dense.weight");
  GGML_ASSERT(this->dense_weight->type == GGML_TYPE_F32);
  GGML_ASSERT(this->dense_weight->ne[0] ==
              (MNIST_HW / 4) * (MNIST_HW / 4) * (MNIST_CNN_NCB * 2));
  GGML_ASSERT(this->dense_weight->ne[1] == MNIST_NCLASSES);
  GGML_ASSERT(this->dense_weight->ne[2] == 1);
  GGML_ASSERT(this->dense_weight->ne[3] == 1);

  this->dense_bias = ggml_get_tensor(this->ctx_gguf, "dense.bias");
  GGML_ASSERT(this->dense_bias->type == GGML_TYPE_F32);
  GGML_ASSERT(this->dense_bias->ne[0] == MNIST_NCLASSES);
  GGML_ASSERT(this->dense_bias->ne[1] == 1);
  GGML_ASSERT(this->dense_bias->ne[2] == 1);
  GGML_ASSERT(this->dense_bias->ne[3] == 1);
  // Load weights
  this->buf_gguf =
      ggml_backend_alloc_ctx_tensors(this->ctx_gguf, this->backend_cpu);
  if (!load_from_gguf(fname.c_str(), this->ctx_gguf, ctx)) {
    spdlog::error("loading weights from {} failed", fname);
    return false;
  }
  spdlog::info("Successfully loaded weights from {}", fname);
  return true;
}

bool MnistCNN::init_random() {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> nd{0.0f, 1e-2f};
  std::vector<ggml_tensor *> init_tensors;
  this->conv1_kernel = ggml_new_tensor_4d(this->ctx_static, GGML_TYPE_F32, 3, 3,
                                          1, MNIST_CNN_NCB);
  this->conv1_bias =
      ggml_new_tensor_3d(this->ctx_static, GGML_TYPE_F32, 1, 1, MNIST_CNN_NCB);
  this->conv2_kernel = ggml_new_tensor_4d(this->ctx_static, GGML_TYPE_F32, 3, 3,
                                          MNIST_CNN_NCB, MNIST_CNN_NCB * 2);
  this->conv2_bias = ggml_new_tensor_3d(this->ctx_static, GGML_TYPE_F32, 1, 1,
                                        MNIST_CNN_NCB * 2);
  this->dense_weight = ggml_new_tensor_2d(
      this->ctx_static, GGML_TYPE_F32,
      (MNIST_HW / 4) * (MNIST_HW / 4) * (MNIST_CNN_NCB * 2), MNIST_NCLASSES);
  this->dense_bias =
      ggml_new_tensor_1d(this->ctx_static, GGML_TYPE_F32, MNIST_NCLASSES);

  ggml_set_name(this->conv1_kernel, "conv1.kernel");
  ggml_set_name(this->conv1_bias, "conv1.bias");
  ggml_set_name(this->conv2_kernel, "conv2.kernel");
  ggml_set_name(this->conv2_bias, "conv2.bias");
  ggml_set_name(this->dense_weight, "dense.weight");
  ggml_set_name(this->dense_bias, "dense.bias");

  init_tensors.push_back(this->conv1_kernel);
  init_tensors.push_back(this->conv1_bias);
  init_tensors.push_back(this->conv2_kernel);
  init_tensors.push_back(this->conv2_bias);
  init_tensors.push_back(this->dense_weight);
  init_tensors.push_back(this->dense_bias);

  // Set allocator
  this->buf_static =
      ggml_backend_alloc_ctx_tensors(this->ctx_static, this->backend_cpu);

  for (ggml_tensor *t : init_tensors) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    const int64_t ne = ggml_nelements(t);
    std::vector<float> tmp(ne);

    for (int64_t i = 0; i < ne; ++i) {
      tmp[i] = nd(gen);
    }
    ggml_backend_tensor_set(t, tmp.data(), 0, ggml_nbytes(t));
  }
  spdlog::info("Successfully initialized random weights");
  return true;
}

bool MnistCNN::load_dataset(const std::string &image_fname,
                            const std::string &label_fname,
                            ggml_opt_dataset_t dataset) {
  auto image_in = std::ifstream(image_fname, std::ios::binary);
  if (!image_in) {
    spdlog::error("failed to open image file: {}", image_fname);
    return false;
  }
  auto label_in = std::ifstream(label_fname, std::ios::binary);
  if (!label_in) {
    spdlog::error("failed to open label file: {}", label_fname);
    return false;
  }

  // image must skip 16 bytes header
  image_in.seekg(16, std::ios::beg);
  // label must skip 8 bytes header
  label_in.seekg(8, std::ios::beg);

  uint8_t image_buf[MNIST_NINPUT];
  uint8_t label_buf;
  struct ggml_tensor *data = ggml_opt_dataset_data(dataset);
  struct ggml_tensor *labels = ggml_opt_dataset_labels(dataset);
  float *data_ptr = ggml_get_data_f32(data);
  float *labels_ptr = ggml_get_data_f32(labels);
  int ndata = data->ne[1]; // number of dataset

  for (int64_t iex = 0; iex < ndata; ++iex) {
    image_in.read((char *)image_buf, MNIST_NINPUT);
    label_in.read((char *)&label_buf, sizeof(label_buf));

    for (int i = 0; i < MNIST_NINPUT; ++i) {
      data_ptr[iex * MNIST_NINPUT + i] = image_buf[i] / 255.0f;
    }
    for (int i = 0; i < MNIST_NCLASSES; ++i) {
      labels_ptr[iex * MNIST_NCLASSES + i] = i == label_buf ? 1.0f : 0.0f;
    }
  }
  return true;
}

void MnistCNN::build(ggml_opt_dataset_t dataset) {
  // build model computation graph here
  ggml_set_param(this->conv1_kernel);
  ggml_set_param(this->conv1_bias);
  ggml_set_param(this->conv2_kernel);
  ggml_set_param(this->conv2_bias);
  ggml_set_param(this->dense_weight);
  ggml_set_param(this->dense_bias);

  struct ggml_tensor *images_2D =
      ggml_reshape_4d(this->ctx_compute, this->images, MNIST_HW, MNIST_HW, 1,
                      this->images->ne[1]);

  // conv2d params: stride_h, stride_w, pad_h, pad_w, dilation_w, dilation_h
  struct ggml_tensor *conv1_out =
      ggml_relu(this->ctx_compute,
                ggml_add(this->ctx_compute,
                         ggml_conv_2d(this->ctx_compute, this->conv1_kernel,
                                      images_2D, 1, 1, 1, 1, 1, 1),
                         this->conv1_bias));

  // shape [H, W, C, B] -> (conv) -> shape [H, W, NCB, B]
  GGML_ASSERT(conv1_out->ne[0] == MNIST_HW);
  GGML_ASSERT(conv1_out->ne[1] == MNIST_HW);
  GGML_ASSERT(conv1_out->ne[2] == MNIST_CNN_NCB);
  GGML_ASSERT(conv1_out->ne[3] == this->nbatch_physical);

  // shape [H, W, NCB, B] -> (maxpool) -> shape [H/2, W/2, NCB, B]
  struct ggml_tensor *conv2_in = ggml_pool_2d(
      this->ctx_compute, conv1_out, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  GGML_ASSERT(conv2_in->ne[0] == MNIST_HW / 2);
  GGML_ASSERT(conv2_in->ne[1] == MNIST_HW / 2);
  GGML_ASSERT(conv2_in->ne[2] == MNIST_CNN_NCB);
  GGML_ASSERT(conv2_in->ne[3] == this->nbatch_physical);

  // shape [H/2, W/2, NCB, B] -> (conv) -> shape [H/2, W/2, NCB*2, B]
  struct ggml_tensor *conv2_out =
      ggml_relu(this->ctx_compute,
                ggml_add(this->ctx_compute,
                         ggml_conv_2d(this->ctx_compute, this->conv2_kernel,
                                      conv2_in, 1, 1, 1, 1, 1, 1),
                         this->conv2_bias));
  GGML_ASSERT(conv2_out->ne[0] == MNIST_HW / 2);
  GGML_ASSERT(conv2_out->ne[1] == MNIST_HW / 2);
  GGML_ASSERT(conv2_out->ne[2] == MNIST_CNN_NCB * 2);
  GGML_ASSERT(conv2_out->ne[3] == this->nbatch_physical);

  // shape [H/2, W/2, NCB*2, B] -> (maxpool) -> shape [H/4, W/4, NCB*2, B]
  struct ggml_tensor *dense_in = ggml_pool_2d(
      this->ctx_compute, conv2_out, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  GGML_ASSERT(dense_in->ne[0] == MNIST_HW / 4);
  GGML_ASSERT(dense_in->ne[1] == MNIST_HW / 4);
  GGML_ASSERT(dense_in->ne[2] == MNIST_CNN_NCB * 2);
  GGML_ASSERT(dense_in->ne[3] == this->nbatch_physical);

  // shape [H/4, W/4, NCB*2, B] -> (reshape) -> shape [features, B]
  dense_in = ggml_reshape_2d(
      this->ctx_compute,
      ggml_cont(this->ctx_compute,
                ggml_permute(this->ctx_compute, dense_in, 1, 2, 0, 3)),
      (MNIST_HW / 4) * (MNIST_HW / 4) * (MNIST_CNN_NCB * 2),
      this->nbatch_physical);
  GGML_ASSERT(dense_in->ne[0] ==
              (MNIST_HW / 4) * (MNIST_HW / 4) * (MNIST_CNN_NCB * 2));
  GGML_ASSERT(dense_in->ne[1] == this->nbatch_physical);
  GGML_ASSERT(dense_in->ne[2] == 1);
  GGML_ASSERT(dense_in->ne[3] == 1);

  // shape [features, B] -> (fc) -> shape [10, B]
  this->logits =
      ggml_add(this->ctx_compute,
               ggml_mul_mat(this->ctx_compute, this->dense_weight, dense_in),
               this->dense_bias);

  ggml_set_name(this->logits, "logits");
  ggml_set_output(this->logits);
  GGML_ASSERT(this->logits->type == GGML_TYPE_F32);
  GGML_ASSERT(this->logits->ne[0] == MNIST_NCLASSES);
  GGML_ASSERT(this->logits->ne[1] == this->nbatch_physical);
  GGML_ASSERT(this->logits->ne[2] == 1);
  GGML_ASSERT(this->logits->ne[3] == 1);
}

void MnistCNN::train(ggml_opt_dataset_t dataset, const int nepoch,
                     const float val_split) {
  ggml_opt_fit_backend(backend_cpu, ctx_compute, images, logits, dataset,
                       GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
                       GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                       ggml_opt_get_default_optimizer_params, nepoch,
                       nbatch_logical, val_split, false);
}

void MnistCNN::save_model(const std::string &fname) {
  spdlog::info("saving model to '{}'", fname);

  struct ggml_context *ggml_ctx;
  {
    struct ggml_init_params params = {
        /*.mem_size   =*/100 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };
    ggml_ctx = ggml_init(params);
  }

  gguf_context *gguf_ctx = gguf_init_empty();
  gguf_set_val_str(gguf_ctx, "general.architecture", "mnist-cnn");

  std::vector<struct ggml_tensor *> weights = {
      this->conv1_kernel, this->conv1_bias,   this->conv2_kernel,
      this->conv2_bias,   this->dense_weight, this->dense_bias};
  for (struct ggml_tensor *t : weights) {
    struct ggml_tensor *copy = ggml_dup_tensor(ggml_ctx, t);
    ggml_set_name(copy, t->name);
    ggml_backend_tensor_get(t, copy->data, 0, ggml_nbytes(t));
    gguf_add_tensor(gguf_ctx, copy);
  }
  gguf_write_to_file(gguf_ctx, fname.c_str(), false);

  ggml_free(ggml_ctx);
  gguf_free(gguf_ctx);
}

ggml_opt_result_t MnistCNN::eval(ggml_opt_dataset_t dataset) {
  ggml_opt_result_t result = ggml_opt_result_init();

  ggml_backend_t backends[1] = {backend_cpu};
  ggml_backend_sched_t backend_sched = ggml_backend_sched_new(
      backends, /*bufts =*/nullptr, /*n_backends =*/1, GGML_DEFAULT_GRAPH_SIZE,
      /*parallel =*/false, /*op_offload =*/true);
  ggml_opt_params params =
      ggml_opt_default_params(backend_sched, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
  params.ctx_compute = ctx_compute;
  params.inputs = images;
  params.outputs = logits;
  params.build_type = GGML_OPT_BUILD_TYPE_FORWARD;
  ggml_opt_context_t opt_ctx = ggml_opt_init(params);

  {
    const int64_t t_start_us = ggml_time_us();

    ggml_opt_epoch(opt_ctx, dataset, nullptr, result, /*idata_split =*/0,
                   nullptr, nullptr);

    const int64_t t_total_us = ggml_time_us() - t_start_us;
    const double t_total_ms = 1e-3 * t_total_us;
    const int nex = ggml_opt_dataset_data(dataset)->ne[1];
    spdlog::info(
        "{}: model evaluation on {} images took {:.2f} ms, {:.2f} us/image",
        __func__, nex, t_total_ms, (double)t_total_us / nex);
  }

  ggml_opt_free(opt_ctx);

  return result;
}

void MnistCNN::print_image(FILE *stream, ggml_opt_dataset_t dataset,
                           const int iex) {
  ggml_tensor *images = ggml_opt_dataset_data(dataset);
  GGML_ASSERT(images && images->ne[0] == MNIST_NINPUT);
  GGML_ASSERT(iex >= 0 && iex < (int)images->ne[1]);

  const float *base = ggml_get_data_f32(images);
  const float *image = base + (size_t)iex * (size_t)MNIST_NINPUT;

  for (int row = 0; row < MNIST_HW; ++row) {
    for (int col = 0; col < MNIST_HW; ++col) {
      float v = std::clamp(image[row * MNIST_HW + col], 0.0f, 1.0f);
      int gray = (int)std::lround(v * 23.0f); // 0..23
      int color = 232 + gray;                 // ANSI 256灰度区 232..255

      // 背景色块 + 两个空格
      std::fprintf(stream, "\033[48;5;%dm  \033[0m", color);
    }
    std::fprintf(stream, "\n");
  }
}