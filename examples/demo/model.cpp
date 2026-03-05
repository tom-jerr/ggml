#include "model.h"
#include "ggml-backend.h"
#include <cstdint>
#include <cstdlib>
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
    : model_file(model_file), nbatch_logical(nbatch_logical),
      nbatch_physical(nbatch_physical) {

  const int ncores_logical = std::thread::hardware_concurrency();
  const int nthreads = std::min(ncores_logical, (ncores_logical + 4) / 2);

  init_backends();
  ggml_backend_cpu_set_n_threads(backend_cpu, nthreads);

  const size_t size_meta = 1024 * ggml_tensor_overhead() +
                           GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() +
                           3 * ggml_graph_overhead();
  struct ggml_init_params params = {
      /*.mem_size   =*/size_meta,
      /*.mem_buffer =*/nullptr,
      /*.no_alloc   =*/true,
  };
  ctx_compute = ggml_init(params);

  init_input();   // load images as input
  init_weights(); // load or initialize weights
}

MnistCNN::~MnistCNN() {
  ggml_free(ctx_gguf);
  ggml_free(ctx_compute);

  ggml_backend_buffer_free(buf_weights_gpu);
  ggml_backend_buffer_free(buf_weights_cpu);

  ggml_backend_free(backend_gpu);
  ggml_backend_free(backend_cpu);
}

void MnistCNN::init_backends() {
  ggml_backend_load_all();

  backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
  if (!backend_cpu) {
    GGML_ABORT("failed to init CPU backend");
  }

  backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
  if (backend_gpu) {
    spdlog::info("backend: GPU available: {}", ggml_backend_name(backend_gpu));
  } else {
    spdlog::warn("backend: GPU backend not available, falling back to CPU only");
  }
}

void MnistCNN::ensure_sched_debug_env() const {
  // Scheduler prints splits/assignments via GGML_LOG_DEBUG when this env var is set.
  if (std::getenv("GGML_SCHED_DEBUG") == nullptr) {
    setenv("GGML_SCHED_DEBUG", "2", /*overwrite =*/0);
  }
}

void MnistCNN::alloc_weights_split(bool enable_gpu_conv) {
  GGML_ASSERT(conv1_kernel && conv1_bias && conv2_kernel && conv2_bias);
  GGML_ASSERT(dense_weight && dense_bias);

  // Free previous buffers if re-initializing.
  ggml_backend_buffer_free(buf_weights_gpu);
  ggml_backend_buffer_free(buf_weights_cpu);
  buf_weights_gpu = nullptr;
  buf_weights_cpu = nullptr;

  const bool use_gpu_for_conv = enable_gpu_conv && backend_gpu != nullptr;

  std::vector<ggml_tensor *> w_conv = {conv1_kernel, conv1_bias, conv2_kernel, conv2_bias};
  std::vector<ggml_tensor *> w_dense = {dense_weight, dense_bias};

  size_t size_gpu = 0;
  size_t size_cpu = 0;

  auto add_size = [](size_t &acc, ggml_tensor *t) { acc += ggml_nbytes(t) + 512; };

  if (use_gpu_for_conv) {
    for (ggml_tensor *t : w_conv) {
      add_size(size_gpu, t);
    }
    for (ggml_tensor *t : w_dense) {
      add_size(size_cpu, t);
    }
  } else {
    for (ggml_tensor *t : w_conv) {
      add_size(size_cpu, t);
    }
    for (ggml_tensor *t : w_dense) {
      add_size(size_cpu, t);
    }
  }

  if (size_gpu > 0) {
    buf_weights_gpu = ggml_backend_alloc_buffer(backend_gpu, size_gpu);
    GGML_ASSERT(buf_weights_gpu);
    ggml_backend_buffer_set_usage(buf_weights_gpu,
                                 GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  }
  if (size_cpu > 0) {
    buf_weights_cpu = ggml_backend_alloc_buffer(backend_cpu, size_cpu);
    GGML_ASSERT(buf_weights_cpu);
    ggml_backend_buffer_set_usage(buf_weights_cpu,
                                 GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  }

  if (use_gpu_for_conv) {
    ggml_tallocr alloc_gpu = ggml_tallocr_new(buf_weights_gpu);
    for (ggml_tensor *t : w_conv) {
      const enum ggml_status st = ggml_tallocr_alloc(&alloc_gpu, t);
      GGML_ASSERT(st == GGML_STATUS_SUCCESS);
    }

    ggml_tallocr alloc_cpu = ggml_tallocr_new(buf_weights_cpu);
    for (ggml_tensor *t : w_dense) {
      const enum ggml_status st = ggml_tallocr_alloc(&alloc_cpu, t);
      GGML_ASSERT(st == GGML_STATUS_SUCCESS);
    }
  } else {
    ggml_tallocr alloc_cpu = ggml_tallocr_new(buf_weights_cpu);
    for (ggml_tensor *t : w_conv) {
      const enum ggml_status st = ggml_tallocr_alloc(&alloc_cpu, t);
      GGML_ASSERT(st == GGML_STATUS_SUCCESS);
    }
    for (ggml_tensor *t : w_dense) {
      const enum ggml_status st = ggml_tallocr_alloc(&alloc_cpu, t);
      GGML_ASSERT(st == GGML_STATUS_SUCCESS);
    }
  }

  log_weight_placement();
}

void MnistCNN::log_weight_placement() const {
  auto log_w = [&](ggml_tensor *t) {
    const char *buf_name = t->buffer ? ggml_backend_buffer_name(t->buffer) : "NULL";
    const char *where = "UNALLOC";
    if (t->buffer == buf_weights_gpu) {
      where = backend_gpu ? ggml_backend_name(backend_gpu) : "GPU(NULL)";
    } else if (t->buffer == buf_weights_cpu) {
      where = backend_cpu ? ggml_backend_name(backend_cpu) : "CPU(NULL)";
    }
    spdlog::info("weight placement: {:<14} -> {:<8} buffer={} bytes={}",
                 t->name, where, buf_name, ggml_nbytes(t));
  };

  log_w(conv1_kernel);
  log_w(conv1_bias);
  log_w(conv2_kernel);
  log_w(conv2_bias);
  log_w(dense_weight);
  log_w(dense_bias);
}

bool MnistCNN::init_input() {
  images = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, MNIST_NINPUT,
                              nbatch_physical);
  spdlog::info("Successfully initialized input tensor");
  return true;
}

bool MnistCNN::init_weights() {
  if (!model_file.empty()) {
    return init_from_file();
  } else {
    return init_random();
  }
}

bool MnistCNN::init_from_file() {
  struct gguf_context *ctx;

  struct gguf_init_params params = {
      /*.no_alloc   =*/true,
      /*.ctx        =*/&this->ctx_gguf,
  };
  ctx = gguf_init_from_file(this->model_file.c_str(), params);
  if (!ctx) {
    spdlog::error("failed to load model from file: {}", this->model_file);
    return false;
  }

  // Allocate tensor metadata in ctx_gguf
  this->conv1_kernel = ggml_get_tensor(this->ctx_gguf, "conv1.kernel");
  this->conv1_bias = ggml_get_tensor(this->ctx_gguf, "conv1.bias");
  this->conv2_kernel = ggml_get_tensor(this->ctx_gguf, "conv2.kernel");
  this->conv2_bias = ggml_get_tensor(this->ctx_gguf, "conv2.bias");
  this->dense_weight = ggml_get_tensor(this->ctx_gguf, "dense.weight");
  this->dense_bias = ggml_get_tensor(this->ctx_gguf, "dense.bias");

  // Allocate persistent weight storage (conv on GPU, dense on CPU when possible).
  alloc_weights_split(/*enable_gpu_conv =*/true);

  // Load weights
  if (!load_from_gguf(this->model_file.c_str(), this->ctx_gguf, ctx)) {
    spdlog::error("loading weights from {} failed", this->model_file);
    return false;
  }
  spdlog::info("Successfully loaded weights from {}", this->model_file);
  return true;
}

bool MnistCNN::init_random() {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> nd{0.0f, 1e-2f};
  std::vector<ggml_tensor *> init_tensors;
  this->conv1_kernel = ggml_new_tensor_4d(this->ctx_compute, GGML_TYPE_F32, 3,
                                          3, 1, MNIST_CNN_NCB);
  this->conv1_bias =
      ggml_new_tensor_3d(this->ctx_compute, GGML_TYPE_F32, 1, 1, MNIST_CNN_NCB);
  this->conv2_kernel = ggml_new_tensor_4d(this->ctx_compute, GGML_TYPE_F32, 3,
                                          3, MNIST_CNN_NCB, MNIST_CNN_NCB * 2);
  this->conv2_bias = ggml_new_tensor_3d(this->ctx_compute, GGML_TYPE_F32, 1, 1,
                                        MNIST_CNN_NCB * 2);
  this->dense_weight = ggml_new_tensor_2d(
      this->ctx_compute, GGML_TYPE_F32,
      (MNIST_HW / 4) * (MNIST_HW / 4) * (MNIST_CNN_NCB * 2), MNIST_NCLASSES);
  this->dense_bias =
      ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_F32, MNIST_NCLASSES);

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

  // Allocate persistent weight storage (conv on GPU, dense on CPU when possible).
  alloc_weights_split(/*enable_gpu_conv =*/true);

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

void MnistCNN::build_compute_graph() {
  // build model computation graph here
  struct ggml_tensor *images_2D =
      ggml_reshape_4d(this->ctx_compute, this->images, MNIST_HW, MNIST_HW, 1,
                      this->images->ne[1]);
  ggml_set_name(images_2D, "images_2d");

  // conv2d params: stride_h, stride_w, pad_h, pad_w, dilation_w, dilation_h
  // shape [H, W, C, B] -> (conv) -> shape [H, W, NCB, B]
  struct ggml_tensor *conv1_out =
      ggml_relu(this->ctx_compute,
                ggml_add(this->ctx_compute,
                         ggml_conv_2d(this->ctx_compute, this->conv1_kernel,
                                      images_2D, 1, 1, 1, 1, 1, 1),
                         this->conv1_bias));
  ggml_set_name(conv1_out, "conv1_out");

  // shape [H, W, NCB, B] -> (maxpool) -> shape [H/2, W/2, NCB, B]
  struct ggml_tensor *conv2_in = ggml_pool_2d(
      this->ctx_compute, conv1_out, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  ggml_set_name(conv2_in, "conv2_in");

  // shape [H/2, W/2, NCB, B] -> (conv) -> shape [H/2, W/2, NCB*2, B]
  struct ggml_tensor *conv2_out =
      ggml_relu(this->ctx_compute,
                ggml_add(this->ctx_compute,
                         ggml_conv_2d(this->ctx_compute, this->conv2_kernel,
                                      conv2_in, 1, 1, 1, 1, 1, 1),
                         this->conv2_bias));
  ggml_set_name(conv2_out, "conv2_out");

  // shape [H/2, W/2, NCB*2, B] -> (maxpool) -> shape [H/4, W/4, NCB*2, B]
  struct ggml_tensor *dense_in = ggml_pool_2d(
      this->ctx_compute, conv2_out, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  ggml_set_name(dense_in, "dense_in_pool");

  // shape [H/4, W/4, NCB*2, B] -> (reshape) -> shape [features, B]
  dense_in = ggml_reshape_2d(
      this->ctx_compute,
      ggml_cont(this->ctx_compute,
                ggml_permute(this->ctx_compute, dense_in, 1, 2, 0, 3)),
      (MNIST_HW / 4) * (MNIST_HW / 4) * (MNIST_CNN_NCB * 2),
      this->nbatch_physical);
  ggml_set_name(dense_in, "dense_in");

  // shape [features, B] -> (fc) -> shape [10, B]
  this->logits =
      ggml_add(this->ctx_compute,
               ggml_mul_mat(this->ctx_compute, this->dense_weight, dense_in),
               this->dense_bias);

  // set input
  ggml_set_name(this->images, "images");
  ggml_set_input(this->images);
  // param means weights that will be optimized during training
  ggml_set_param(this->conv1_kernel);
  ggml_set_param(this->conv1_bias);
  ggml_set_param(this->conv2_kernel);
  ggml_set_param(this->conv2_bias);
  ggml_set_param(this->dense_weight);
  ggml_set_param(this->dense_bias);

  ggml_set_name(this->logits, "logits");
  ggml_set_output(this->logits);
}

void MnistCNN::train(ggml_opt_dataset_t dataset, const int nepoch,
                     const float val_split) {
  // 计算后面的 pred，需要 dataset 中的 labels 作为 Input

  ensure_sched_debug_env();

  ggml_backend_t backends[2] = {nullptr, nullptr};
  int n_backends = 0;
  if (backend_gpu) {
    backends[n_backends++] = backend_gpu;
  }
  backends[n_backends++] = backend_cpu;

  std::string sched_desc;
  for (int i = 0; i < n_backends; ++i) {
    if (i) {
      sched_desc += " -> ";
    }
    sched_desc += ggml_backend_name(backends[i]);
  }
  spdlog::info("scheduler backends (prio high->low): {}", sched_desc);

  ggml_backend_sched_t backend_sched = ggml_backend_sched_new(
      backends, /*bufts =*/nullptr, /*n_backends =*/n_backends,
      GGML_DEFAULT_GRAPH_SIZE,
      /*parallel =*/false, /*op_offload =*/false);

  ggml_time_init();
  const int64_t t_start_us = ggml_time_us();
  const int64_t ndata = ggml_opt_dataset_data(dataset)->ne[1];
  const int64_t opt_period = nbatch_logical / nbatch_physical;
  const int64_t nbatches_logical = ndata / nbatch_logical;
  const int64_t ibatch_split =
      int64_t(((1.0f - val_split) * nbatches_logical)) *
      opt_period; // train <-> val split index (physical)
  const int64_t idata_split = ibatch_split * nbatch_physical;

  int64_t epoch = 1;

  ggml_opt_params params =
      ggml_opt_default_params(backend_sched, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
  params.ctx_compute = ctx_compute;
  params.inputs = this->images;
  params.outputs = this->logits;
  params.opt_period = opt_period;
  params.get_opt_pars = ggml_opt_get_default_optimizer_params;
  params.get_opt_pars_ud = &epoch;
  params.optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;

  // here we build both forward and backward graphs
  ggml_opt_context_t opt_ctx = ggml_opt_init(params);

  // Shuffling the data is generally useful but there is only a point if not all
  // data is used in a single batch.
  if (nbatch_logical < ndata) {
    ggml_opt_dataset_shuffle(opt_ctx, dataset,
                             -1); // Shuffle all data (train + validation).
  }

  ggml_opt_result_t result_train = ggml_opt_result_init();
  struct ggml_tensor *inputs = ggml_opt_inputs(opt_ctx);
  struct ggml_tensor *labels = ggml_opt_labels(opt_ctx);
  for (; epoch <= nepoch; ++epoch) {
    if (nbatch_logical < idata_split) {
      ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
    }

    ggml_opt_result_reset(result_train);
    spdlog::info("epoch {}/{}:", epoch, nepoch);

    int64_t t_loop_start = ggml_time_us();
    int64_t ibatch = 0;

    for (; ibatch < ibatch_split; ++ibatch) {
      // copy compute graph metadata to a new context, then allocate memory for
      // tensor data at backend memory
      ggml_opt_alloc(opt_ctx, /*backward =*/true);
      // get a batch of data from dataset(images and labels)
      ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
      // compute graph and update optimizer
      ggml_opt_eval(opt_ctx, result_train);

      ggml_opt_epoch_callback_progress_bar(true, opt_ctx, dataset, result_train,
                                           ibatch + 1, ibatch_split,
                                           t_loop_start);
    }
  }

  int64_t t_total_s = (ggml_time_us() - t_start_us) / 1000000;
  const int64_t t_total_h = t_total_s / 3600;
  t_total_s -= t_total_h * 3600;
  const int64_t t_total_m = t_total_s / 60;
  t_total_s -= t_total_m * 60;
  spdlog::info("training completed in {:02}h:{:02}m:{:02}s", t_total_h,
               t_total_m, t_total_s);

  ggml_opt_free(opt_ctx);
  ggml_opt_result_free(result_train);
  ggml_backend_sched_free(backend_sched);
}

void MnistCNN::save_model(const std::string &fname) {
  spdlog::info("saving model to '{}'", fname);

  struct ggml_context *ggml_ctx;
  struct ggml_init_params params = {
      /*.mem_size   =*/100 * 1024 * 1024,
      /*.mem_buffer =*/NULL,
      /*.no_alloc   =*/false,
  };
  ggml_ctx = ggml_init(params);

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

ggml_opt_result_t MnistCNN::eval(const float *image_data, int label) {

  ensure_sched_debug_env();

  ggml_backend_t backends[2] = {nullptr, nullptr};
  int n_backends = 0;
  if (backend_gpu) {
    backends[n_backends++] = backend_gpu;
  }
  backends[n_backends++] = backend_cpu;

  ggml_backend_sched_t backend_sched = ggml_backend_sched_new(
      backends, /*bufts =*/nullptr, /*n_backends =*/n_backends,
      GGML_DEFAULT_GRAPH_SIZE,
      /*parallel =*/false, /*op_offload =*/false);
  ggml_opt_params params =
      ggml_opt_default_params(backend_sched, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
  params.ctx_compute = ctx_compute;
  params.inputs = images;
  params.outputs = logits;
  params.build_type = GGML_OPT_BUILD_TYPE_FORWARD;
  ggml_opt_context_t opt_ctx = ggml_opt_init(params);

  int64_t t_start_us = ggml_time_us();
  struct ggml_tensor *inputs = ggml_opt_inputs(opt_ctx);
  struct ggml_tensor *labels_tensor = ggml_opt_labels(opt_ctx);
  ggml_opt_result_t result = ggml_opt_result_init();

  // 先分配内存，再设置数据
  ggml_opt_alloc(opt_ctx, /*backward =*/false);

  // 设置图像数据
  ggml_backend_tensor_set(inputs, image_data, 0, MNIST_NINPUT * sizeof(float));

  // 设置 one-hot 标签
  std::vector<float> label_onehot(MNIST_NCLASSES, 0.0f);
  label_onehot[label] = 1.0f;
  ggml_backend_tensor_set(labels_tensor, label_onehot.data(), 0,
                          MNIST_NCLASSES * sizeof(float));

  ggml_opt_eval(opt_ctx, result);

  const int64_t t_total_us = ggml_time_us() - t_start_us;
  const double t_total_ms = 1e-3 * t_total_us;

  spdlog::info("eval: sample evaluated in {:.2f} ms", t_total_ms);

  ggml_opt_free(opt_ctx);
  ggml_backend_sched_free(backend_sched);

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
