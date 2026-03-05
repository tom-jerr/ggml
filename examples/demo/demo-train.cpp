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
  spdlog::info("  model_dir     = \"models\"");
  spdlog::info("  model_file    = \"\"  (if empty: {model_name}-f32.gguf)");
  spdlog::info("  init_model    = \"\"  (optional: start from existing .gguf)");
  spdlog::info("  train_images  = \"data/MNIST/raw/train-images-idx3-ubyte\"");
  spdlog::info("  train_labels  = \"data/MNIST/raw/train-labels-idx1-ubyte\"");
  spdlog::info("  eval_images   = \"data/MNIST/raw/t10k-images-idx3-ubyte\"");
  spdlog::info("  eval_labels   = \"data/MNIST/raw/t10k-labels-idx1-ubyte\"");
  spdlog::info("  nepoch        = 5");
  spdlog::info("  val_split     = 0.05");
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
  const std::string model_dir = cfg.value<std::string>("model_dir", "models");
  std::string model_file = cfg.value<std::string>("model_file", "");

  const std::string train_images = cfg.value<std::string>(
      "train_images", "data/MNIST/raw/train-images-idx3-ubyte");
  const std::string train_labels = cfg.value<std::string>(
      "train_labels", "data/MNIST/raw/train-labels-idx1-ubyte");

  const int nepoch = cfg.value<int>("nepoch", 5);
  const float val_split = cfg.value<float>("val_split", 0.05f);

  if (model_file.empty()) {
    spdlog::info("This is Training !!!");
  }

  const std::filesystem::path out_dir = std::filesystem::path(model_dir);
  std::filesystem::path out_path = out_dir / (model_name + "-f32.gguf");

  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    spdlog::error("failed to create model_dir '{}': {}", out_dir.string(),
                  ec.message());
    return 1;
  }

  spdlog::info("config: {}", cfg_path);
  spdlog::info("model_name:   {}", model_name);
  spdlog::info("output_model: {}", out_path.string());
  spdlog::info("train_images: {}", train_images);
  spdlog::info("train_labels: {}", train_labels);
  spdlog::info("nepoch: {}, val_split: {:.3f}", nepoch, (double)val_split);

  ggml_time_init();

  // Train dataset
  ggml_opt_dataset_t train_ds = ggml_opt_dataset_init(
      GGML_TYPE_F32, GGML_TYPE_F32, MNIST_NINPUT, MNIST_NCLASSES, MNIST_NTRAIN,
      /*ndata_shard=*/MNIST_NBATCH_PHYSICAL);

  MnistCNN model(model_file, MNIST_NBATCH_LOGICAL, MNIST_NBATCH_PHYSICAL);

  if (!model.load_dataset(train_images, train_labels, train_ds)) {
    ggml_opt_dataset_free(train_ds);
    return 1;
  }

  model.build_compute_graph();
  model.train(train_ds, nepoch, val_split);
  model.save_model(out_path.string());
  spdlog::info("Training completed and model saved.");

  // =========== Cleanup ============
  ggml_opt_dataset_free(train_ds);
  return 0;
}
