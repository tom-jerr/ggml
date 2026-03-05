#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"
#include "ggml.h"
#include "gguf.h"

#define MNIST_NTRAIN 60000
#define MNIST_NTEST 10000

// Gradient accumulation can be achieved by setting the logical batch size to a
// multiple of the physical one. The logical batch size determines how many
// datapoints are used for a gradient update. The physical batch size determines
// how many datapoints are processed in parallel, larger values utilize compute
// better but need more memory.
#define MNIST_NBATCH_LOGICAL 500
#define MNIST_NBATCH_PHYSICAL 500

static_assert(MNIST_NBATCH_LOGICAL % MNIST_NBATCH_PHYSICAL == 0,
              "MNIST_NBATCH_LOGICAL % MNIST_NBATCH_PHYSICAL != 0");
static_assert(MNIST_NTRAIN % MNIST_NBATCH_LOGICAL == 0,
              "MNIST_NTRAIN % MNIST_NBATCH_LOGICAL != 0");
static_assert(MNIST_NTEST % MNIST_NBATCH_LOGICAL == 0,
              "MNIST_NTRAIN % MNIST_NBATCH_LOGICAL != 0");

#define MNIST_HW 28
#define MNIST_NINPUT (MNIST_HW * MNIST_HW)
#define MNIST_NCLASSES 10

#define MNIST_NHIDDEN 500

// NCB = number of channels base
#define MNIST_CNN_NCB 8

struct mnist_model {};

class MnistCNN {
  /**
   * @brief A simple CNN model for MNIST digit classification
   * @details The model consists of:
   *   - Conv2D(3x3, 1->NCB) + ReLU
   *   - MaxPool2D(2x2)
   *   - Conv2D(3x3, NCB->NCB*2) + ReLU
   *   - MaxPool2D(2x2)
   *   - Fully connected layer to 10 classes
   * shape conventions:
   *   - images: [H, W, C, B]
   *   - first conv: [H, W, NCB, B]
   *   - first maxpool: [H/2, W/2, NCB, B]
   *   - second conv: [H/2, W/2, NCB*2, B]
   *   - second maxpool: [H/4, W/4, NCB*2, B]
   *   - dense input: [features, B] ( after reshape )
   *   - fc: [10, B]
   */
private:
  std::string model_file;
  ggml_backend_t backend_cpu = nullptr;
  ggml_backend_t backend_gpu = nullptr;

  const int nbatch_logical;
  const int nbatch_physical;

  struct ggml_tensor *images = nullptr; // input
  struct ggml_tensor *logits = nullptr; // output

  struct ggml_tensor *fc1_weight = nullptr;
  struct ggml_tensor *fc1_bias = nullptr;
  struct ggml_tensor *fc2_weight = nullptr;
  struct ggml_tensor *fc2_bias = nullptr;

  struct ggml_tensor *conv1_kernel = nullptr;
  struct ggml_tensor *conv1_bias = nullptr;
  struct ggml_tensor *conv2_kernel = nullptr;
  struct ggml_tensor *conv2_bias = nullptr;
  struct ggml_tensor *dense_weight = nullptr;
  struct ggml_tensor *dense_bias = nullptr;

  struct ggml_context *ctx_gguf = nullptr; // for weights tensor metadata
  // struct ggml_context *ctx_static = nullptr;
  struct ggml_context *ctx_compute = nullptr;

  // Persistent weight buffers (split across backends).
  ggml_backend_buffer_t buf_weights_gpu = nullptr;
  ggml_backend_buffer_t buf_weights_cpu = nullptr;

public:
  MnistCNN(const std::string &model_file, const int nbatch_logical,
           const int nbatch_physical);

  ~MnistCNN();

  /**
   * @brief 加载对应数据集到 dataset 中
   * @param  image_fname      const string &
   * @param  label_fname      const string &
   * @param  dataset          ggml_opt_dataset_t
   */
  bool load_dataset(const std::string &image_fname,
                    const std::string &label_fname, ggml_opt_dataset_t dataset);
  void build_compute_graph();
  void train(ggml_opt_dataset_t dataset, const int nepoch,
             const float val_split);
  /**
   * @brief 使用索引从数据中评估单个样本
   * @param image_data 图像数据指针 (MNIST_NINPUT 个 float)
   * @param label 真实标签 (0-9)
   */
  ggml_opt_result_t eval(const float *image_data, int label);
  void save_model(const std::string &fname);

  // print image for debugging
  void print_image(FILE *stream, ggml_opt_dataset_t dataset, const int iex);

private:
  /**
   * @brief init model helper functions
   *
   */
  void init_backends();
  void ensure_sched_debug_env() const;
  void alloc_weights_split(bool enable_gpu_conv);
  void log_weight_placement() const;

  bool init_input();
  bool init_weights();
  bool init_from_file();
  bool init_random();
};

/**
 * @brief helper function to load gguf model from file
 * @param  fname            My Param doc
 * @param  ctx_ggml         My Param doc
 * @param  ctx_gguf         My Param doc
 * @return true
 * @return false
 */
bool load_from_gguf(const char *fname, struct ggml_context *ctx_ggml,
                    struct gguf_context *ctx_gguf);
