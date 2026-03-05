#include "ggml-opt.h"
#include "model.h"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

static bool ends_with(const std::string &s, const std::string &suffix) {
  if (suffix.size() > s.size()) {
    return false;
  }
  return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

static void print_usage(const char *argv0) {
  spdlog::info("Usage: {} <config.json>", argv0);
  spdlog::info("Config keys (all optional; defaults shown):");
  spdlog::info("  model_name    = \"mnist-cnn\"");
  spdlog::info("  model_file    = \"\"  (if empty: {model_name}-f32.gguf)");
  spdlog::info("  eval_images   = \"data/MNIST/raw/t10k-images-idx3-ubyte\"");
  spdlog::info("  eval_labels   = \"data/MNIST/raw/t10k-labels-idx1-ubyte\"");
}

int main(int argc, char **argv) {
  if (argc != 2) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string cfg_path = argv[1];

  nlohmann::json cfg;
  try {
    std::ifstream fin(cfg_path);
    if (!fin) {
      spdlog::error("failed to open config: {}", cfg_path);
      return 1;
    }
    const std::string text((std::istreambuf_iterator<char>(fin)),
                           (std::istreambuf_iterator<char>()));
    cfg = nlohmann::json::parse(text);
  } catch (const std::exception &e) {
    spdlog::error("failed to parse json '{}': {}", cfg_path, e.what());
    return 1;
  }

  const std::string model_name =
      cfg.value<std::string>("model_name", "mnist-cnn");
  std::string model_file =
      cfg.value<std::string>("model_file", "models/mnist-cnn-f32.gguf");

  const std::string eval_images = cfg.value<std::string>(
      "eval_images", "data/MNIST/raw/t10k-images-idx3-ubyte");
  const std::string eval_labels = cfg.value<std::string>(
      "eval_labels", "data/MNIST/raw/t10k-labels-idx1-ubyte");

  spdlog::info("config: {}", cfg_path);
  spdlog::info("model_name:   {}", model_name);
  spdlog::info("eval_images:  {}", eval_images);
  spdlog::info("eval_labels:  {}", eval_labels);

  ggml_time_init();

  // Eval dataset
  ggml_opt_dataset_t eval_ds = ggml_opt_dataset_init(
      GGML_TYPE_F32, GGML_TYPE_F32, MNIST_NINPUT, MNIST_NCLASSES, 1, 1);

  MnistCNN model_eval(model_file, 1, 1);

  if (!model_eval.load_dataset(eval_images, eval_labels, eval_ds)) {
    ggml_opt_dataset_free(eval_ds);
    return 1;
  }

  model_eval.build_compute_graph();

  spdlog::info("Evaluating model on eval dataset...");

  // 从 dataset 中获取第 0 个样本的数据
  struct ggml_tensor *data_tensor = ggml_opt_dataset_data(eval_ds);
  struct ggml_tensor *labels_tensor = ggml_opt_dataset_labels(eval_ds);
  const float *all_images = ggml_get_data_f32(data_tensor);
  const float *all_labels = ggml_get_data_f32(labels_tensor);

  // 获取第 0 个样本
  const int sample_idx = 0;
  const float *image_data = all_images + sample_idx * MNIST_NINPUT;
  // 从 one-hot 标签中找到真实标签
  const float *label_onehot = all_labels + sample_idx * MNIST_NCLASSES;
  int true_label = 0;
  for (int i = 0; i < MNIST_NCLASSES; ++i) {
    if (label_onehot[i] > 0.5f) {
      true_label = i;
      break;
    }
  }

  ggml_opt_result_t result = model_eval.eval(image_data, true_label);
  double loss = 0.0, loss_unc = 0.0;
  double acc = 0.0, acc_unc = 0.0;
  int32_t pred;
  ggml_opt_result_loss(result, &loss, &loss_unc);
  ggml_opt_result_accuracy(result, &acc, &acc_unc);
  ggml_opt_result_pred(result, &pred);
  spdlog::info("eval: loss={:.6f} (±{:.6f}), accuracy={:.4f} (±{:.4f})", loss,
               loss_unc, acc, acc_unc);
  spdlog::info("True label: {}, Predicted label: {}", true_label, pred);

  // ============ Print first eval image ============
  model_eval.print_image(stdout, eval_ds, 0);

  // =========== Cleanup ============
  ggml_opt_result_free(result);
  ggml_opt_dataset_free(eval_ds);
  return 0;
}