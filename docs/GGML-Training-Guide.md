# GGML 深度技术指南：训练与推理全解析

> 本文档以 MNIST CNN 训练/推理为主线，从高层 API 入手，逐步揭示 GGML 的核心设计与内存管理机制。

---

## 目录

- [第 1 章：GGML 简介与核心设计](#第-1-章ggml-简介与核心设计)
- [第 2 章：完整示例 — MNIST CNN 训练与推理](#第-2-章完整示例--mnist-cnn-训练与推理)
- [第 3 章：高层训练 API — ggml_opt](#第-3-章高层训练-api--ggml_opt)
- [第 4 章：计算图内存管理 — 两阶段分配](#第-4-章计算图内存管理--两阶段分配)
- [第 5 章：基础组件详解](#第-5-章基础组件详解)
- [第 6 章：优化器与自动微分](#第-6-章优化器与自动微分)
- [第 7 章：权重管理 — GGUF 格式](#第-7-章权重管理--gguf-格式)
- [附录：算子系统与卷积实现](#附录算子系统与卷积实现)

---

## 第 1 章：GGML 简介与核心设计

### 1.1 定位与目标

GGML（Georgi Gerganov's Machine Learning library）是一个面向边缘设备的机器学习张量计算库，具有以下特点：

- **极简依赖**：纯 C/C++ 实现，无第三方依赖
- **零运行时分配**：内存在初始化阶段预分配，计算阶段无 malloc
- **广泛量化支持**：40+ 种量化类型（Q4_0 ~ K-quants），定义于 `include/ggml.h`
- **多后端支持**：CPU / CUDA / Metal / Vulkan / SYCL / OpenCL / CANN / Hexagon 等
- **完整训练能力**：自动微分 + AdamW/SGD 优化器

### 1.2 设计原则

#### 静态图执行模型

GGML 采用**先定义、后计算**的静态图模型，而非 PyTorch 的 eager execution：

1. **定义阶段**：创建 tensor、构建计算图（纯元数据操作）
2. **分配阶段**：为计算图中的 tensor 分配后端内存
3. **执行阶段**：提交计算图到后端执行

#### 双 Context 模式

GGML 推荐使用两个分离的 context：

| Context 类型 | `no_alloc` | 用途 | 生命周期 |
|--------------|------------|------|----------|
| **静态 Context** | `true` | 存储模型权重和输入的元数据 | 整个程序生命周期 |
| **计算 Context** | `true` | 存储计算图中间 tensor | 每次图执行时重分配 |

> **设计动机**：权重 tensor 需要持久存储，而中间激活值可以复用内存空间。

### 1.3 架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            User Code                                     │
│   (定义模型、构建计算图、调用 ggml_opt_fit / ggml_opt_eval)              │
└─────────────────────────────────────────────────────────────┬───────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ggml_context                                     │
│   (tensor 元数据容器：type, shape, stride, op, src[], flags)            │
└─────────────────────────────────────────────────────────────┬───────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ggml_backend_buffer                                  │
│   (真实内存：CPU malloc / CUDA cudaMalloc / Metal MTLBuffer)            │
│   └── 由 ggml_backend_alloc_ctx_tensors() 为 context 中所有 tensor 分配 │
└─────────────────────────────────────────────────────────────┬───────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ggml_gallocr                                     │
│   (计算图分配器 — 两阶段分配)                                            │
│   ├── Reserve：模拟分配，计算峰值内存，记录 offset                       │
│   └── Alloc：设置 tensor->data = buffer_base + offset                   │
└─────────────────────────────────────────────────────────────┬───────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ggml_backend_sched                                   │
│   (后端调度器 — 多后端图分割)                                            │
│   ├── 5 遍分图算法：根据 tensor 放置决策分割图                           │
│   └── 执行：逐 split 拷贝输入 → ggml_backend_graph_compute_async()      │
└─────────────────────────────────────────────────────────────┬───────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         thread_pool                                      │
│   (多线程并行执行计算图中的每个 node)                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 核心头文件导航

| 头文件 | 用途 |
|--------|------|
| `include/ggml.h` | 核心 API：tensor、计算图、算子 |
| `include/ggml-backend.h` | 后端抽象：buffer、device、scheduler |
| `include/ggml-alloc.h` | 内存分配器：gallocr |
| `include/ggml-opt.h` | 高层训练 API：dataset、optimizer、fit |
| `include/gguf.h` | 模型文件格式 |

---

*下一章将通过完整的 MNIST CNN 示例展示这些组件如何协同工作。*

---

## 第 2 章：完整示例 — MNIST CNN 训练与推理

> 本章以 `examples/demo/` 中的 MNIST CNN 为例，展示从数据加载到模型保存的完整工作流。

### 2.1 模型架构

MNIST CNN 是一个简单的卷积神经网络，用于手写数字分类：

```
输入 [28, 28, 1, B]
    │
    ▼
Conv2D(3×3, 1→8) + ReLU    → [28, 28, 8, B]
    │
    ▼
MaxPool2D(2×2)              → [14, 14, 8, B]
    │
    ▼
Conv2D(3×3, 8→16) + ReLU   → [14, 14, 16, B]
    │
    ▼
MaxPool2D(2×2)              → [7, 7, 16, B]
    │
    ▼
Reshape + Dense(784→10)    → [10, B]
    │
    ▼
Softmax + CrossEntropy Loss
```

对应的 GGUF 模型结构：

```
Tensor count: 6
[0] conv1.kernel  [3, 3, 1, 8]     288 bytes
[1] conv1.bias    [1, 1, 8]         32 bytes
[2] conv2.kernel  [3, 3, 8, 16]   4608 bytes
[3] conv2.bias    [1, 1, 16]        64 bytes
[4] dense.weight  [784, 10]      31360 bytes
[5] dense.bias    [10]              40 bytes
```

### 2.2 训练流程总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 数据集初始化                                                          │
│    ggml_opt_dataset_init(F32, F32, 784, 10, 60000, ndata_shard=10)     │
│    └── 分配 CPU buffer 存储 data[784, 60000] + labels[10, 60000]       │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. 加载数据                                                              │
│    load_dataset(images_file, labels_file, dataset)                      │
│    └── 读取 MNIST 文件 → 归一化 [0,1] → one-hot 编码                    │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. 模型初始化                                                            │
│    MnistCNN(model_file="", nbatch_logical=500, nbatch_physical=500)    │
│    ├── 创建 ctx_compute (no_alloc=true)                                 │
│    ├── init_input(): 创建 images tensor                                 │
│    └── init_random(): 随机初始化权重 → ggml_backend_alloc_ctx_tensors() │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. 构建计算图                                                            │
│    build_compute_graph()                                                 │
│    ├── 定义前向计算：Conv → ReLU → Pool → Conv → ReLU → Pool → Dense   │
│    ├── ggml_set_input(images)    // 标记输入                            │
│    ├── ggml_set_param(weights)   // 标记可训练参数                       │
│    └── ggml_set_output(logits)   // 标记输出                            │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. 创建优化上下文                                                        │
│    ggml_opt_init(params)                                                 │
│    ├── 构建 gf (前向图)                                                  │
│    ├── 添加 loss/pred/ncorrect 节点                                     │
│    ├── ggml_build_backward_expand() → gb_grad (反向图)                  │
│    ├── 添加 ggml_opt_step_adamw 节点 → gb_opt (优化图)                  │
│    └── ggml_backend_alloc_ctx_tensors(ctx_static) → 分配梯度/momenta   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. 训练循环 (per epoch)                                                  │
│    for epoch in 1..nepoch:                                              │
│        ggml_opt_dataset_shuffle()   // 打乱数据 (按 shard)              │
│        for ibatch in 0..nbatches:                                       │
│            ┌─────────────────────────────────────────────────────────┐  │
│            │ a. ggml_opt_alloc(backward=true)                        │  │
│            │    ├── 复制计算图元数据 (dup_graph)                     │  │
│            │    ├── ggml_backend_sched_reset()                       │  │
│            │    └── ggml_backend_sched_alloc_graph() → 两阶段分配    │  │
│            ├─────────────────────────────────────────────────────────┤  │
│            │ b. ggml_opt_dataset_get_batch()                         │  │
│            │    └── ggml_backend_tensor_set() → 拷贝数据到 backend   │  │
│            ├─────────────────────────────────────────────────────────┤  │
│            │ c. ggml_opt_eval(opt_ctx, result)                       │  │
│            │    ├── 写入优化器参数 (alpha, beta1, beta2, ...)        │  │
│            │    ├── ggml_backend_sched_graph_compute() → 执行计算    │  │
│            │    └── 提取 loss/pred/ncorrect 到 result                │  │
│            └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. 保存模型                                                              │
│    save_model("mnist-cnn-f32.gguf")                                     │
│    ├── gguf_init_empty() → gguf_set_val_str("general.architecture")    │
│    ├── for weight in weights: gguf_add_tensor(gguf_ctx, weight)        │
│    └── gguf_write_to_file()                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 核心代码解读

#### 2.3.1 模型初始化 (`model.cpp`)

```cpp
MnistCNN::MnistCNN(const std::string &model_file, 
                   const int nbatch_logical,
                   const int nbatch_physical)
    : model_file(model_file), 
      nbatch_logical(nbatch_logical),
      nbatch_physical(nbatch_physical) {
  
  // 1. 初始化 CPU 后端
  backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
  ggml_backend_cpu_set_n_threads(backend_cpu, nthreads);

  // 2. 创建 compute context (no_alloc=true)
  //    只存储 tensor 元数据，不分配数据
  const size_t size_meta = 1024 * ggml_tensor_overhead() + 
                           GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() +
                           3 * ggml_graph_overhead();
  struct ggml_init_params params = {
      .mem_size   = size_meta,
      .mem_buffer = nullptr,
      .no_alloc   = true,  // 关键：只分配元数据
  };
  ctx_compute = ggml_init(params);

  init_input();   // 创建输入 tensor
  init_weights(); // 加载或随机初始化权重
}
```

**要点**：
- `no_alloc = true` 表示 context 只存储 tensor 的元数据（类型、形状、算子等），不分配实际数据
- 数据分配由后续的 `ggml_backend_alloc_ctx_tensors()` 完成

#### 2.3.2 构建计算图 (`build_compute_graph`)

```cpp
void MnistCNN::build_compute_graph() {
  // 将 1D 输入 reshape 为 4D
  struct ggml_tensor *images_2D = ggml_reshape_4d(
      ctx_compute, images, MNIST_HW, MNIST_HW, 1, images->ne[1]);

  // Conv1: [28,28,1,B] → [28,28,8,B]
  struct ggml_tensor *conv1_out = ggml_relu(ctx_compute,
      ggml_add(ctx_compute,
          ggml_conv_2d(ctx_compute, conv1_kernel, images_2D, 
                       /*s0=*/1, /*s1=*/1, /*p0=*/1, /*p1=*/1, /*d0=*/1, /*d1=*/1),
          conv1_bias));

  // MaxPool1: [28,28,8,B] → [14,14,8,B]
  struct ggml_tensor *pool1_out = ggml_pool_2d(
      ctx_compute, conv1_out, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);

  // Conv2: [14,14,8,B] → [14,14,16,B]
  struct ggml_tensor *conv2_out = ggml_relu(ctx_compute,
      ggml_add(ctx_compute,
          ggml_conv_2d(ctx_compute, conv2_kernel, pool1_out,
                       1, 1, 1, 1, 1, 1),
          conv2_bias));

  // MaxPool2: [14,14,16,B] → [7,7,16,B]
  struct ggml_tensor *pool2_out = ggml_pool_2d(
      ctx_compute, conv2_out, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);

  // Flatten + Dense: [7*7*16,B] → [10,B]
  struct ggml_tensor *dense_in = ggml_reshape_2d(ctx_compute,
      ggml_cont(ctx_compute, ggml_permute(ctx_compute, pool2_out, 1, 2, 0, 3)),
      (MNIST_HW/4) * (MNIST_HW/4) * (MNIST_CNN_NCB*2), nbatch_physical);

  logits = ggml_add(ctx_compute,
      ggml_mul_mat(ctx_compute, dense_weight, dense_in),
      dense_bias);

  // 标记 tensor 角色
  ggml_set_input(images);         // 输入
  ggml_set_param(conv1_kernel);   // 可训练参数
  ggml_set_param(conv1_bias);
  ggml_set_param(conv2_kernel);
  ggml_set_param(conv2_bias);
  ggml_set_param(dense_weight);
  ggml_set_param(dense_bias);
  ggml_set_output(logits);        // 输出
}
```

**要点**：
- `ggml_set_param()` 标记的 tensor 会被自动微分系统追踪
- `ggml_set_input()` / `ggml_set_output()` 影响计算图的边界

#### 2.3.3 训练循环 (`train`)

```cpp
void MnistCNN::train(ggml_opt_dataset_t dataset, 
                     const int nepoch, 
                     const float val_split) {
  // 创建后端调度器
  ggml_backend_t backends[1] = {backend_cpu};
  ggml_backend_sched_t backend_sched = ggml_backend_sched_new(
      backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

  // 配置优化参数
  ggml_opt_params params = ggml_opt_default_params(
      backend_sched, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
  params.ctx_compute = ctx_compute;
  params.inputs = images;
  params.outputs = logits;
  params.opt_period = nbatch_logical / nbatch_physical;  // 梯度累积
  params.optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;

  // 初始化优化上下文 (自动构建反向图和优化器节点)
  ggml_opt_context_t opt_ctx = ggml_opt_init(params);

  // 打乱数据
  ggml_opt_dataset_shuffle(opt_ctx, dataset, -1);

  ggml_opt_result_t result_train = ggml_opt_result_init();
  struct ggml_tensor *inputs = ggml_opt_inputs(opt_ctx);
  struct ggml_tensor *labels = ggml_opt_labels(opt_ctx);

  for (int64_t epoch = 1; epoch <= nepoch; ++epoch) {
    ggml_opt_result_reset(result_train);
    
    for (int64_t ibatch = 0; ibatch < ibatch_split; ++ibatch) {
      // a. 分配计算图内存 (两阶段分配)
      ggml_opt_alloc(opt_ctx, /*backward=*/true);
      
      // b. 获取批次数据 (拷贝到 backend)
      ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
      
      // c. 执行前向+反向+优化器更新
      ggml_opt_eval(opt_ctx, result_train);
    }
  }

  ggml_opt_free(opt_ctx);
  ggml_opt_result_free(result_train);
}
```

### 2.4 推理流程

推理与训练的主要区别：

| 方面 | 训练 | 推理 |
|------|------|------|
| `build_type` | `GGML_OPT_BUILD_TYPE_OPT` | `GGML_OPT_BUILD_TYPE_FORWARD` |
| 计算图 | gf + gb_grad + gb_opt | 仅 gf |
| `ggml_opt_alloc()` 参数 | `backward=true` | `backward=false` |
| 权重来源 | 随机初始化 | 从 GGUF 文件加载 |

```cpp
ggml_opt_result_t MnistCNN::eval(const float *image_data, int label) {
  // 配置为仅前向计算
  ggml_opt_params params = ggml_opt_default_params(
      backend_sched, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
  params.ctx_compute = ctx_compute;
  params.inputs = images;
  params.outputs = logits;
  params.build_type = GGML_OPT_BUILD_TYPE_FORWARD;  // 关键区别
  
  ggml_opt_context_t opt_ctx = ggml_opt_init(params);
  ggml_opt_result_t result = ggml_opt_result_init();

  // 分配前向图内存
  ggml_opt_alloc(opt_ctx, /*backward=*/false);

  // 设置输入数据
  ggml_backend_tensor_set(inputs, image_data, 0, MNIST_NINPUT * sizeof(float));
  
  // 设置标签 (用于计算 loss 和 accuracy)
  std::vector<float> label_onehot(MNIST_NCLASSES, 0.0f);
  label_onehot[label] = 1.0f;
  ggml_backend_tensor_set(labels_tensor, label_onehot.data(), 0, 
                          MNIST_NCLASSES * sizeof(float));

  // 执行前向计算
  ggml_opt_eval(opt_ctx, result);

  ggml_opt_free(opt_ctx);
  return result;
}
```

### 2.5 内存分配时序图

```
时间轴 ──────────────────────────────────────────────────────────────────▶

┌──────────────────┐
│ 程序启动         │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 1. ggml_init(no_alloc=true)                                          │
│    └── 仅分配 ctx 内部的元数据池 (几 KB)                              │
└────────┬─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. ggml_new_tensor_*()                                               │
│    └── 在 ctx 内分配 tensor 元数据 (~200 bytes/tensor)               │
│        此时 tensor->data = NULL                                       │
└────────┬─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 3. ggml_backend_alloc_ctx_tensors(ctx, backend)                      │
│    ├── 遍历 ctx 中所有 tensor                                        │
│    ├── 计算总共需要的内存 (含对齐)                                    │
│    ├── ggml_backend_buft_alloc_buffer() 分配后端 buffer              │
│    └── 设置每个 tensor->data, tensor->buffer                         │
│                                                                       │
│    *** 权重 tensor 的数据在此时分配！***                              │
└────────┬─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 4. ggml_opt_init(params)                                             │
│    ├── ggml_new_graph_custom() 创建计算图                            │
│    ├── ggml_build_forward_expand() 添加节点                          │
│    ├── ggml_opt_build()                                              │
│    │   ├── 创建 ctx_static (梯度累加器 + momenta)                    │
│    │   ├── ggml_build_backward_expand() 构建反向图                   │
│    │   └── 添加优化器节点                                             │
│    └── ggml_backend_alloc_ctx_tensors(ctx_static, backend)           │
│                                                                       │
│    *** 梯度/momenta 的内存在此时分配！***                             │
└────────┬─────────────────────────────────────────────────────────────┘
         │
         │ for each batch:
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 5. ggml_opt_alloc(backward=true)                                     │
│    ├── dup_graph() 复制计算图元数据                                   │
│    ├── ggml_backend_sched_reset() 清除上次分配状态                   │
│    └── ggml_backend_sched_alloc_graph()                              │
│        ├── ggml_gallocr_reserve_n() [Reserve 阶段]                   │
│        │   ├── 模拟遍历图                                             │
│        │   ├── ggml_dyn_tallocr_alloc() 记录 offset + max_size       │
│        │   └── ggml_vbuffer_alloc() 分配真实后端 buffer              │
│        └── 设置 tensor->data = buffer_base + offset [Alloc 阶段]     │
│                                                                       │
│    *** 计算图中间激活值的内存在此时分配！***                          │
│    *** 由于生命周期分析，中间 tensor 会复用内存 ***                   │
└────────┬─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 6. ggml_opt_dataset_get_batch()                                      │
│    └── ggml_backend_tensor_set() 将数据从 host 拷贝到 device         │
└────────┬─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 7. ggml_opt_eval()                                                   │
│    └── ggml_backend_sched_graph_compute()                            │
│        └── 多线程执行计算图                                           │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.6 关键常量

```cpp
// examples/demo/model.h

#define MNIST_NTRAIN   60000   // 训练集大小
#define MNIST_NTEST    10000   // 测试集大小
#define MNIST_HW       28      // 图像宽高
#define MNIST_NINPUT   784     // 28 * 28
#define MNIST_NCLASSES 10      // 数字 0-9

#define MNIST_NBATCH_LOGICAL   500   // 逻辑批次大小 (梯度更新单位)
#define MNIST_NBATCH_PHYSICAL  500   // 物理批次大小 (并行处理单位)

#define MNIST_CNN_NCB  8       // 卷积通道基数
```

**梯度累积**：当 `NBATCH_LOGICAL > NBATCH_PHYSICAL` 时：
- `opt_period = NBATCH_LOGICAL / NBATCH_PHYSICAL`
- 前 `opt_period-1` 次迭代只累积梯度
- 第 `opt_period` 次迭代执行权重更新

---

*下一章将深入讲解 `ggml_opt` 高层 API 的内部实现。*

---

## 第 3 章：高层训练 API — ggml_opt

> `ggml_opt` 是 GGML 提供的高层训练接口，封装了数据集管理、计算图构建、自动微分、优化器更新等复杂逻辑。

### 3.1 核心数据结构

#### 3.1.1 数据集 (`ggml_opt_dataset_t`)

```cpp
// src/ggml-opt.cpp
struct ggml_opt_dataset {
  struct ggml_context *ctx = nullptr;     // tensor 元数据上下文
  ggml_backend_buffer_t buf = nullptr;    // CPU buffer 存储实际数据
  struct ggml_tensor *data = nullptr;     // [ne_datapoint, ndata]
  struct ggml_tensor *labels = nullptr;   // [ne_label, ndata]

  int64_t ndata = -1;                     // 数据总量
  int64_t ndata_shard = -1;               // shard 大小 (shuffle 粒度)
  size_t nbs_data = -1;                   // 每个 shard 的 data 字节数
  size_t nbs_labels = -1;                 // 每个 shard 的 labels 字节数

  std::vector<int64_t> permutation;       // shuffle 用的索引排列
};
```

**Shard 机制**：数据按 `ndata_shard` 分组打乱，减少大数据集的 shuffle 开销。

#### 3.1.2 优化上下文 (`ggml_opt_context_t`)

```cpp
struct ggml_opt_context {
  // === 后端调度 ===
  ggml_backend_sched_t backend_sched = nullptr;
  
  // === 计算图 ===
  ggml_cgraph *gf = nullptr;          // 前向图
  ggml_cgraph *gb_grad = nullptr;     // 前向 + 反向图
  ggml_cgraph *gb_opt = nullptr;      // 前向 + 反向 + 优化器更新图
  ggml_cgraph *allocated_graph = nullptr;       // 当前分配的图
  ggml_cgraph *allocated_graph_copy = nullptr;  // 图的拷贝（用于执行）
  
  // === 上下文 ===
  struct ggml_context *ctx_static = nullptr;   // 静态 tensor (梯度累加器, momenta)
  struct ggml_context *ctx_cpu = nullptr;      // CPU tensor (优化器参数)
  struct ggml_context *ctx_compute = nullptr;  // 计算图临时 tensor
  struct ggml_context *ctx_copy = nullptr;     // 图拷贝用的 context
  
  // === 缓冲区 ===
  ggml_backend_buffer_t buf_static = nullptr;
  ggml_backend_buffer_t buf_cpu = nullptr;
  
  // === 输入/输出 tensor ===
  struct ggml_tensor *inputs = nullptr;
  struct ggml_tensor *outputs = nullptr;
  struct ggml_tensor *labels = nullptr;
  struct ggml_tensor *loss = nullptr;
  struct ggml_tensor *pred = nullptr;
  struct ggml_tensor *ncorrect = nullptr;
  
  // === 梯度和 Momenta ===
  std::vector<struct ggml_tensor *> grad_accs;  // 梯度累加器 (每个 PARAM 一个)
  std::vector<struct ggml_tensor *> grad_m;     // AdamW 一阶动量
  std::vector<struct ggml_tensor *> grad_v;     // AdamW 二阶动量
  
  // === 优化器配置 ===
  struct ggml_tensor *opt_step_params = nullptr;  // 存储 alpha, beta1, beta2, ...
  enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
  ggml_opt_get_optimizer_params get_opt_pars = nullptr;
  void *get_opt_pars_ud = nullptr;
  
  // === 状态 ===
  int64_t iter = 1;           // 迭代计数 (用于 AdamW bias correction)
  int32_t opt_period = 1;     // 梯度累积周期
  int32_t opt_i = 0;          // 当前累积索引
  bool static_graphs = false; // 是否使用静态图
  bool eval_ready = false;    // 是否已调用 alloc
  
  std::mt19937 rng;           // 随机数生成器 (用于 shuffle)
};
```

#### 3.1.3 结果容器 (`ggml_opt_result_t`)

```cpp
struct ggml_opt_result {
  int64_t ndata = 0;              // 已处理的数据点数
  std::vector<float> loss;        // 每个 batch 的 loss
  std::vector<int32_t> pred;      // 预测结果
  int64_t ncorrect = 0;           // 正确预测数

  int64_t opt_period = -1;        // 梯度累积周期
  bool loss_per_datapoint = false; // loss 是否按数据点平均
};
```

### 3.2 Dataset API

#### 创建数据集

```cpp
ggml_opt_dataset_t ggml_opt_dataset_init(
    enum ggml_type type_data,   // 数据类型 (通常 F32)
    enum ggml_type type_label,  // 标签类型 (通常 F32 for one-hot)
    int64_t ne_datapoint,       // 每个数据点的元素数 (如 784)
    int64_t ne_label,           // 每个标签的元素数 (如 10)
    int64_t ndata,              // 数据总量 (如 60000)
    int64_t ndata_shard);       // shard 大小 (如 10)
```

**内部实现**：
1. 创建 `ggml_context`（`no_alloc=true`）
2. 创建 `data` 和 `labels` tensor
3. 调用 `ggml_backend_alloc_ctx_tensors_from_buft(ctx, cpu_buffer_type)` 分配 CPU buffer
4. 初始化 `permutation` 数组

#### 获取批次数据

```cpp
void ggml_opt_dataset_get_batch(
    ggml_opt_dataset_t dataset,
    struct ggml_tensor *data_batch,   // [ne_datapoint, batch_size]
    struct ggml_tensor *labels_batch, // [ne_label, batch_size]
    int64_t ibatch);
```

**内部实现**：
1. 根据 `permutation` 确定要访问的 shard 索引
2. 调用 `ggml_backend_tensor_set()` 将数据从 CPU buffer 拷贝到目标 tensor

### 3.3 训练三级图

`ggml_opt` 根据 `build_type` 构建三种不同的计算图：

```
┌─────────────────────────────────────────────────────────────────────┐
│                           gf (Forward)                               │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐              │
│  │ inputs  │──▶│  Conv   │──▶│  Pool   │──▶│ outputs │              │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘              │
│                                               │                      │
│                                               ▼                      │
│                                          ┌─────────┐                 │
│                                          │  loss   │                 │
│                                          └─────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ ggml_build_backward_expand()
┌─────────────────────────────────────────────────────────────────────┐
│                        gb_grad (Gradient)                            │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐              │
│  │ inputs  │──▶│  Conv   │──▶│  Pool   │──▶│ outputs │              │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘              │
│                    │              │              │                   │
│                    ▼              ▼              ▼                   │
│               ┌─────────┐   ┌─────────┐   ┌─────────┐               │
│               │ ∂L/∂W   │◀──│ ∂L/∂x   │◀──│ ∂L/∂out │◀── loss      │
│               └────┬────┘   └─────────┘   └─────────┘               │
│                    │                                                 │
│                    ▼                                                 │
│               ┌──────────┐                                           │
│               │ grad_acc │  (梯度累加器)                              │
│               └──────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ 添加优化器节点
┌─────────────────────────────────────────────────────────────────────┐
│                        gb_opt (Optimize)                             │
│                                                                      │
│  [ ... gb_grad 的所有节点 ... ]                                      │
│                    │                                                 │
│                    ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │            ggml_opt_step_adamw / ggml_opt_step_sgd             │ │
│  │                                                                 │ │
│  │  inputs: param, grad, m (momentum), v (velocity), params_tensor│ │
│  │  output: 原地更新 param                                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**图选择逻辑** (`ggml_opt_alloc`)：

```cpp
if (backward) {
  const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
  opt_ctx->build_type = (opt_i_next == 0) 
      ? GGML_OPT_BUILD_TYPE_OPT    // 累积完成，执行更新
      : GGML_OPT_BUILD_TYPE_GRAD;  // 继续累积
} else {
  opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;  // 仅推理
}
```

### 3.4 梯度累积实现

```
opt_period = 2 的情况：

Step 0 (opt_i=0):
  ├── 重置 grad_accs 为 0
  ├── 使用 gb_grad (累积梯度)
  └── opt_i → 1

Step 1 (opt_i=1):
  ├── 使用 gb_opt (累积梯度 + 权重更新)
  ├── opt_i → 0
  └── iter++

Step 2 (opt_i=0):
  ├── 重置 grad_accs 为 0
  ├── 使用 gb_grad (累积梯度)
  └── opt_i → 1

... 重复 ...
```

### 3.5 `ggml_opt_build()` 内部流程

```cpp
static void ggml_opt_build(ggml_opt_context_t opt_ctx) {
  // 1. 标记输入/输出
  ggml_set_input(opt_ctx->inputs);
  ggml_set_output(opt_ctx->outputs);

  // 2. 构建损失计算节点
  switch (opt_ctx->loss_type) {
    case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY:
      opt_ctx->labels = ggml_dup_tensor(ctx, opt_ctx->outputs);
      ggml_set_input(opt_ctx->labels);
      opt_ctx->loss = ggml_cross_entropy_loss(ctx, opt_ctx->outputs, opt_ctx->labels);
      
      // 添加预测和准确率统计
      opt_ctx->pred = ggml_argmax(ctx, opt_ctx->outputs);
      opt_ctx->ncorrect = ggml_count_equal(ctx, opt_ctx->pred, 
                                           ggml_argmax(ctx, opt_ctx->labels));
      break;
    // ... 其他损失类型 ...
  }
  
  ggml_set_output(opt_ctx->loss);
  ggml_set_loss(opt_ctx->loss);
  ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

  // 3. 创建梯度累加器 (如果需要)
  if (accumulate) {
    for (每个 PARAM 节点) {
      opt_ctx->grad_accs[i] = ggml_new_tensor(...);  // 在 ctx_static 中分配
    }
  }

  // 4. 构建反向图
  opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, true);
  ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, 
                             opt_ctx->grad_accs.data());

  // 5. 添加优化器节点
  opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, true);
  
  for (每个 PARAM 节点) {
    struct ggml_tensor *grad = ggml_graph_get_grad(opt_ctx->gb_opt, node);
    struct ggml_tensor *opt_step = ggml_opt_step_adamw(
        ctx, node, grad, m, v, adamw_params);
    ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
  }

  // 6. 分配静态 buffer
  opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
      opt_ctx->ctx_static, backend);
  opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(
      opt_ctx->ctx_cpu, ggml_backend_cpu_buffer_type());
}
```

### 3.6 `ggml_opt_eval()` 执行流程

```cpp
void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result) {
  // 1. 如果是优化图，写入优化器参数
  if (opt_ctx->allocated_graph == opt_ctx->gb_opt) {
    float *params = ggml_get_data_f32(opt_ctx->opt_step_params);
    params[0] = opt_pars.adamw.alpha;   // 学习率
    params[1] = opt_pars.adamw.beta1;
    params[2] = opt_pars.adamw.beta2;
    params[3] = opt_pars.adamw.eps;
    params[4] = opt_pars.adamw.wd;
    params[5] = 1.0f / (1.0f - powf(beta1, iter));  // bias correction 1
    params[6] = 1.0f / (1.0f - powf(beta2, iter));  // bias correction 2
  }

  // 2. 执行计算图
  ggml_backend_sched_graph_compute(opt_ctx->backend_sched, 
                                   opt_ctx->allocated_graph_copy);

  // 3. 更新迭代计数
  opt_ctx->iter += (opt_ctx->allocated_graph == opt_ctx->gb_opt);
  opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

  // 4. 收集结果
  if (result) {
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, sizeof(float));
    result->loss.push_back(loss);
    
    if (opt_ctx->pred) {
      std::vector<int32_t> pred(ndata);
      ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ...);
      result->pred.insert(result->pred.end(), pred.begin(), pred.end());
    }
    
    if (opt_ctx->ncorrect) {
      int64_t ncorrect;
      ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ...);
      result->ncorrect += ncorrect;
    }
  }
}
```

### 3.7 高层便捷函数

#### `ggml_opt_fit()` — 完整训练流程

```cpp
void ggml_opt_fit(
    ggml_backend_sched_t backend_sched,
    struct ggml_context *ctx_compute,
    struct ggml_tensor *inputs,
    struct ggml_tensor *outputs,
    ggml_opt_dataset_t dataset,
    enum ggml_opt_loss_type loss_type,
    enum ggml_opt_optimizer_type optimizer,
    ggml_opt_get_optimizer_params get_opt_pars,
    int64_t nepoch,
    int64_t nbatch_logical,
    float val_split,
    bool silent);
```

该函数封装了完整的训练循环：
1. 初始化 `ggml_opt_context`
2. 数据 shuffle
3. 训练/验证分割
4. epoch 循环 → batch 循环 → alloc → get_batch → eval
5. 打印进度和结果

#### `ggml_opt_epoch()` — 单个 epoch

```cpp
void ggml_opt_epoch(
    ggml_opt_context_t opt_ctx,
    ggml_opt_dataset_t dataset,
    ggml_opt_result_t result_train,
    int64_t idata_split,
    ggml_opt_epoch_callback callback_train);
```

适用于需要更多控制的场景（如自定义回调、学习率调度）。

### 3.8 API 速查表

| 函数 | 用途 |
|------|------|
| **Dataset** | |
| `ggml_opt_dataset_init()` | 创建数据集 |
| `ggml_opt_dataset_free()` | 释放数据集 |
| `ggml_opt_dataset_data()` | 获取 data tensor |
| `ggml_opt_dataset_labels()` | 获取 labels tensor |
| `ggml_opt_dataset_shuffle()` | 打乱数据 |
| `ggml_opt_dataset_get_batch()` | 获取批次 |
| **Context** | |
| `ggml_opt_default_params()` | 获取默认参数 |
| `ggml_opt_init()` | 初始化上下文 |
| `ggml_opt_free()` | 释放上下文 |
| `ggml_opt_reset()` | 重置梯度/优化器 |
| `ggml_opt_inputs()` | 获取输入 tensor |
| `ggml_opt_labels()` | 获取标签 tensor |
| `ggml_opt_loss()` | 获取 loss tensor |
| **Computation** | |
| `ggml_opt_alloc()` | 分配计算图内存 |
| `ggml_opt_eval()` | 执行计算 |
| **High-Level** | |
| `ggml_opt_fit()` | 完整训练流程 |
| `ggml_opt_epoch()` | 单个 epoch |
| **Result** | |
| `ggml_opt_result_init()` | 创建结果容器 |
| `ggml_opt_result_free()` | 释放结果 |
| `ggml_opt_result_loss()` | 获取 loss |
| `ggml_opt_result_accuracy()` | 获取准确率 |
| `ggml_opt_result_pred()` | 获取预测 |

---

*下一章将深入讲解计算图内存管理的两阶段分配机制。*

---

## 第 4 章：计算图内存管理 — 两阶段分配

> GGML 的内存管理是其高性能的关键。本章深入讲解 `ggml_gallocr` 的两阶段分配算法，揭示如何实现零运行时分配和内存复用。

### 4.1 核心组件关系

```
ggml_opt_alloc(backward=true)
         │
         ▼
ggml_backend_sched_alloc_graph()
         │
         ├─────────────────────────────────────────────────────────────┐
         ▼                                                             │
ggml_gallocr_alloc_graph()                                            │
         │                                                             │
         ├── ggml_gallocr_needs_realloc() 检查是否需要重新分配          │
         │                                                             │
         ├── [Reserve 阶段] ggml_gallocr_reserve_n()                   │
         │       │                                                     │
         │       ├── ggml_gallocr_alloc_graph_impl() [模拟分配]        │
         │       │       │                                             │
         │       │       ├── 遍历图，按拓扑序处理每个 tensor           │
         │       │       ├── ggml_dyn_tallocr_alloc() 记录 offset      │
         │       │       ├── 引用计数跟踪生命周期                       │
         │       │       └── ggml_dyn_tallocr_free_bytes() 释放空间    │
         │       │                                                     │
         │       ├── 保存 node_allocs[] / leaf_allocs[]                │
         │       │                                                     │
         │       └── ggml_vbuffer_alloc() 分配真实后端 buffer          │
         │                                                             │
         └── [Alloc 阶段]                                              │
                 │                                                     │
                 └── 设置 tensor->data = buffer_base + offset          │
                                                                       │
                                                                       │
ggml_backend_sched_graph_compute() ◀───────────────────────────────────┘
```

### 4.2 核心数据结构

#### 4.2.1 图分配器 (`ggml_gallocr`)

```cpp
// src/ggml-alloc.c
struct ggml_gallocr {
    ggml_backend_buffer_type_t *bufts;     // 各后端的 buffer 类型
    struct vbuffer **buffers;               // 真实分配的 buffer
    struct ggml_dyn_tallocr **buf_tallocs;  // 动态分配器 (每个后端一个)
    int n_buffers;
    
    struct ggml_hash_set hash_set;          // tensor 哈希表
    struct hash_node *hash_values;          // 哈希值 (n_children, n_views, buffer_id)
    
    struct node_alloc *node_allocs;         // 每个 node 的分配方案
    int n_nodes;
    
    struct leaf_alloc *leaf_allocs;         // 每个 leaf 的分配方案
    int n_leafs;
};

// 分配方案
struct node_alloc {
    int buffer_id;                  // 使用哪个 buffer
    struct tensor_alloc dst;        // 输出 tensor 的分配
    struct tensor_alloc src[GGML_MAX_SRC]; // 输入 tensor 的分配
};

struct tensor_alloc {
    int chunk_id;                   // 在 vbuffer 中的 chunk 索引
    size_t offset;                  // 在 chunk 中的偏移量
};
```

#### 4.2.2 动态张量分配器 (`ggml_dyn_tallocr`)

```cpp
struct ggml_dyn_tallocr {
    size_t alignment;               // 内存对齐 (通常 32 或 64)
    size_t max_chunk_size;          // 单个 chunk 的最大大小
    struct tallocr_chunk *chunks[GGML_VBUFFER_MAX_CHUNKS];
    int n_chunks;
};

struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS]; // 空闲块列表
    int n_free_blocks;
    size_t max_size;                // 该 chunk 需要的峰值内存
};

struct free_block {
    size_t offset;                  // 起始偏移
    size_t size;                    // 块大小
};
```

**核心思想**：`ggml_dyn_tallocr` **不真正分配内存**，只是模拟分配过程，记录每个 tensor 需要的 offset 和整体峰值内存。

#### 4.2.3 虚拟缓冲区 (`vbuffer`)

```cpp
struct vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};
```

将多个后端 buffer 组合为逻辑连续的缓冲区，支持超大模型（超过单次分配限制）。

### 4.3 Reserve 阶段详解

Reserve 阶段的目标是**模拟分配，计算峰值内存，记录每个 tensor 的分配方案**。

#### 4.3.1 模拟分配算法

```cpp
static bool ggml_gallocr_alloc_graph_impl(
    ggml_gallocr_t galloc, 
    ggml_cgraph *graph) 
{
    // 1. 重置动态分配器
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // 2. 初始化引用计数
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor *node = graph->nodes[i];
        
        // 计算 n_children: 有多少节点依赖此 tensor
        hn->n_children = 0;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                get_hash_node(node->src[j])->n_children++;
            }
        }
        
        // 计算 n_views: 有多少 view 依赖此 tensor
        if (node->view_src) {
            get_hash_node(node->view_src)->n_views++;
        }
    }

    // 3. 按拓扑序遍历图
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor *node = graph->nodes[i];
        
        // 3.1 分配输出 tensor
        if (需要分配(node)) {
            // 尝试 inplace 优化
            if (ggml_op_can_inplace(node->op)) {
                // 检查是否可以复用 src[0] 的内存
                if (src[0] 是最后一个消费者) {
                    // 直接使用 src[0] 的 offset
                    node_alloc->dst = src_alloc;
                    goto allocated;
                }
            }
            
            // 正常分配
            ggml_dyn_tallocr_alloc(talloc, tensor_size, node);
        }
        
        // 3.2 处理 src 的引用计数
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                struct hash_node *hn = get_hash_node(node->src[j]);
                hn->n_children--;
                
                // 如果是最后一个消费者，释放空间
                if (hn->n_children == 0 && hn->n_views == 0) {
                    ggml_dyn_tallocr_free_bytes(talloc, offset, size);
                }
            }
        }
    }
}
```

#### 4.3.2 最佳适配分配算法

```cpp
static size_t ggml_dyn_tallocr_alloc(
    struct ggml_dyn_tallocr *alloc,
    size_t size,
    ggml_tensor *tensor) 
{
    size = GGML_PAD(size, alloc->alignment);  // 对齐
    
    // 搜索空闲块，找到最小的足够大的块
    int best_fit_block = -1;
    size_t best_fit_size = SIZE_MAX;
    
    for (int i = 0; i < chunk->n_free_blocks; i++) {
        if (chunk->free_blocks[i].size >= size &&
            chunk->free_blocks[i].size < best_fit_size) {
            best_fit_block = i;
            best_fit_size = chunk->free_blocks[i].size;
        }
    }
    
    if (best_fit_block != -1) {
        // 使用找到的块
        size_t offset = chunk->free_blocks[best_fit_block].offset;
        
        // 如果块更大，分割它
        if (best_fit_size > size) {
            chunk->free_blocks[best_fit_block].offset += size;
            chunk->free_blocks[best_fit_block].size -= size;
        } else {
            // 移除该块
            remove_free_block(chunk, best_fit_block);
        }
        
        return offset;
    }
    
    // 没有合适的块，扩展 chunk
    size_t offset = chunk->max_size;
    chunk->max_size += size;
    return offset;
}
```

#### 4.3.3 释放与合并

```cpp
static void ggml_dyn_tallocr_free_bytes(
    struct ggml_dyn_tallocr *alloc,
    size_t offset,
    size_t size)
{
    // 添加为空闲块
    add_free_block(chunk, offset, size);
    
    // 尝试与相邻块合并
    for (int i = 0; i < chunk->n_free_blocks - 1; i++) {
        struct free_block *cur = &chunk->free_blocks[i];
        struct free_block *next = &chunk->free_blocks[i + 1];
        
        // 如果两个块相邻，合并
        if (cur->offset + cur->size == next->offset) {
            cur->size += next->size;
            remove_free_block(chunk, i + 1);
            i--;  // 重新检查当前块
        }
    }
}
```

#### 4.3.4 保存分配方案

```cpp
// 为每个 node 保存分配结果
galloc->node_allocs[i] = {
    .buffer_id = buffer_id,
    .dst = {
        .chunk_id = chunk_id,
        .offset = offset,
    },
    .src = { ... }
};
```

#### 4.3.5 分配真实后端 buffer

```cpp
static struct vbuffer *ggml_vbuffer_alloc(
    ggml_backend_buffer_type_t buft,
    const struct ggml_dyn_tallocr *talloc,
    enum ggml_backend_buffer_usage usage)
{
    struct vbuffer *buf = calloc(1, sizeof(struct vbuffer));
    
    for (int n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        
        // 调用后端分配真实内存
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    
    return buf;
}
```

### 4.4 Alloc 阶段详解

Alloc 阶段非常简单：根据 Reserve 阶段的方案，设置每个 tensor 的 data 指针。

```cpp
static void ggml_gallocr_init_tensor(
    ggml_gallocr_t galloc,
    struct ggml_tensor *tensor,
    struct tensor_alloc *tensor_alloc)
{
    int buffer_id = tensor_alloc->buffer_id;
    int chunk_id = tensor_alloc->chunk_id;
    size_t offset = tensor_alloc->offset;
    
    struct vbuffer *vbuf = galloc->buffers[buffer_id];
    ggml_backend_buffer_t chunk = vbuf->chunks[chunk_id];
    
    void *base = ggml_backend_buffer_get_base(chunk);
    
    tensor->data = (char *)base + offset;
    tensor->buffer = chunk;
}
```

### 4.5 内存复用示例

考虑以下计算图：

```
A ─┬─▶ B ─┬─▶ D
   │      │
   └─▶ C ─┘
```

其中 `D = f(B, C)`，`B = g(A)`，`C = h(A)`。

**内存分配过程**：

```
Step 1: 分配 A
  Memory: [  A  |  free  |  free  ]
  
Step 2: 分配 B (需要 A，A 还被 C 引用)
  Memory: [  A  |   B    |  free  ]
  
Step 3: 分配 C (需要 A，A 的最后一个消费者)
  → 释放 A 的空间
  Memory: [ free |   B    |   C   ]
  
Step 4: 分配 D (需要 B 和 C，都是最后一个消费者)
  → 释放 B 和 C 的空间
  → D 可以复用 A 的空间 (inplace 优化)
  Memory: [  D  | free   | free  ]
```

**峰值内存** = max(A + B, A + B + C, D) = A + B + C

### 4.6 缓存与重分配策略

```cpp
static bool ggml_gallocr_needs_realloc(
    ggml_gallocr_t galloc,
    ggml_cgraph *graph)
{
    // 检查图结构是否变化
    if (galloc->n_nodes != graph->n_nodes) return true;
    if (galloc->n_leafs != graph->n_leafs) return true;
    
    // 检查每个 node 是否相同
    for (int i = 0; i < graph->n_nodes; i++) {
        if (galloc->node_allocs[i].node != graph->nodes[i]) {
            return true;
        }
        // 检查 src 是否变化
        // ...
    }
    
    return false;
}
```

**关键特性**：
- 结构不变时，直接使用缓存的分配方案（只设置指针）
- **只扩容不缩容**：即使需求变小，也不会 free + realloc
- 训练/推理切换时会触发重分配（因为图结构不同）

### 4.7 训练时的内存分配特点

| 图类型 | 节点数 | 中间 tensor 数 | 峰值内存 |
|--------|--------|----------------|----------|
| gf (forward) | N | ~N | M |
| gb_grad (forward + backward) | ~2N | ~2N | ~2M |
| gb_opt (forward + backward + optimizer) | ~2N + P | ~2N | ~2M |

其中 N = 前向图节点数，P = 参数数量，M = 前向图峰值内存。

**注意**：由于训练和推理使用不同的图（gf vs gb_opt），每次切换时都会触发 `ggml_gallocr_needs_realloc() = true`，导致重新 reserve。但由于只扩容不缩容，第一次分配后就会稳定。

### 4.8 调试工具

设置环境变量可以导出计算图：

```bash
# 导出 DOT 格式的计算图
export GGML_OPT_DUMP_DOT=1

# 打印计算图信息
export GGML_OPT_PRINT_GRAPH=1
```

生成的 `.dot` 文件可以用 Graphviz 可视化：

```bash
dot -Tpng ggml_opt_graph_000000_train_opt.dot -o graph.png
```

---

*下一章将详细介绍 GGML 的基础组件：tensor、context、backend、buffer。*

---

## 第 5 章：基础组件详解

> 本章深入讲解 GGML 的核心抽象：张量、上下文、后端、缓冲区。理解这些组件是掌握 GGML 的基础。

### 5.1 张量 (`ggml_tensor`)

#### 5.1.1 结构定义

```cpp
// include/ggml.h
struct ggml_tensor {
    enum ggml_type type;        // 数据类型 (F32, F16, Q4_0, ...)
    
    int64_t ne[GGML_MAX_DIMS];  // shape: ne[0]=cols, ne[1]=rows, ne[2]=channels, ne[3]=batch
    size_t  nb[GGML_MAX_DIMS];  // strides: nb[0]=element_size, nb[1]=row_stride, ...
    
    enum ggml_op op;            // 操作类型 (GGML_OP_ADD, GGML_OP_MUL_MAT, ...)
    
    struct ggml_tensor *src[GGML_MAX_SRC]; // 输入 tensor (最多 10 个)
    
    void *data;                 // 数据指针 (指向 backend buffer)
    struct ggml_backend_buffer *buffer; // 所属的后端缓冲区
    
    struct ggml_tensor *view_src;  // 如果是 view，指向源 tensor
    size_t view_offs;               // view 的偏移量
    
    int32_t flags;              // 标志位 (INPUT, OUTPUT, PARAM, LOSS)
    
    char name[GGML_MAX_NAME];   // tensor 名称 (调试用)
    
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]; // 算子参数
    
    void *extra;                // 后端特定数据 (CUDA stream, Metal buffer, ...)
};
```

#### 5.1.2 数据类型

GGML 支持 40+ 种数据类型，定义于 `include/ggml.h`：

| 类型 | 说明 | 用途 |
|------|------|------|
| `GGML_TYPE_F32` | 32 位浮点 | 训练、精确推理 |
| `GGML_TYPE_F16` | 16 位浮点 | 推理加速 |
| `GGML_TYPE_BF16` | bfloat16 | 训练加速 |
| `GGML_TYPE_Q4_0` | 4 位量化 (对称) | 推理压缩 |
| `GGML_TYPE_Q4_K` | 4 位 K-quant | 高质量量化 |
| `GGML_TYPE_I8` | 8 位整数 | 量化中间值 |
| `GGML_TYPE_I32` | 32 位整数 | 索引、argmax |
| `GGML_TYPE_I64` | 64 位整数 | 计数器 |

#### 5.1.3 创建 API

```cpp
// 创建 tensor (在 context 中分配元数据)
struct ggml_tensor *ggml_new_tensor_1d(ctx, type, ne0);
struct ggml_tensor *ggml_new_tensor_2d(ctx, type, ne0, ne1);
struct ggml_tensor *ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
struct ggml_tensor *ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

// 复制 tensor 元数据 (共享数据)
struct ggml_tensor *ggml_dup_tensor(ctx, src);

// 创建 view (指向同一数据)
struct ggml_tensor *ggml_view_tensor(ctx, src);

// 创建带偏移的 view
struct ggml_tensor *ggml_view_1d(ctx, src, ne0, offset);
struct ggml_tensor *ggml_view_2d(ctx, src, ne0, ne1, nb1, offset);
// ...
```

#### 5.1.4 标记 API

```cpp
// 标记为计算图输入
void ggml_set_input(struct ggml_tensor *tensor);

// 标记为计算图输出
void ggml_set_output(struct ggml_tensor *tensor);

// 标记为可训练参数 (自动微分会追踪)
void ggml_set_param(struct ggml_tensor *tensor);

// 标记为损失函数输出
void ggml_set_loss(struct ggml_tensor *tensor);
```

**Flags 位定义**：

```cpp
enum ggml_tensor_flag {
    GGML_TENSOR_FLAG_INPUT  = 1 << 0,  // 图输入
    GGML_TENSOR_FLAG_OUTPUT = 1 << 1,  // 图输出
    GGML_TENSOR_FLAG_PARAM  = 1 << 2,  // 可训练参数
    GGML_TENSOR_FLAG_LOSS   = 1 << 3,  // 损失函数
};
```

#### 5.1.5 数据访问

```cpp
// Host → Device: 将数据从 CPU 拷贝到 tensor 的后端 buffer
void ggml_backend_tensor_set(
    struct ggml_tensor *tensor,
    const void *data,
    size_t offset,
    size_t size);

// Device → Host: 将数据从 tensor 的后端 buffer 拷贝到 CPU
void ggml_backend_tensor_get(
    const struct ggml_tensor *tensor,
    void *data,
    size_t offset,
    size_t size);

// 直接获取 CPU tensor 的数据指针 (仅限 CPU buffer)
float *ggml_get_data_f32(const struct ggml_tensor *tensor);
```

### 5.2 上下文 (`ggml_context`)

#### 5.2.1 概念

`ggml_context` 是一个**内存池**，用于存储 tensor 的**元数据**（类型、形状、算子、依赖关系等）。

```cpp
// 初始化参数
struct ggml_init_params {
    size_t mem_size;     // 内存池大小 (只存元数据，几 KB~几 MB 足够)
    void  *mem_buffer;   // 外部提供的内存 (NULL 则自动分配)
    bool   no_alloc;     // true=只分配元数据，不为 tensor data 分配内存
};
```

#### 5.2.2 两种使用模式

**模式 1：静态 Context（权重和输入）**

```cpp
// 创建 context，不分配 tensor data
struct ggml_init_params params = {
    .mem_size = 1024 * ggml_tensor_overhead(),  // 假设最多 1024 个 tensor
    .mem_buffer = nullptr,
    .no_alloc = true,  // 关键：只分配元数据
};
struct ggml_context *ctx_static = ggml_init(params);

// 创建 tensor (此时 data=NULL)
struct ggml_tensor *weight = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, 784, 10);

// 为 context 中所有 tensor 分配后端 buffer
ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend);

// 现在 weight->data 指向 backend buffer 中的内存
ggml_backend_tensor_set(weight, data, 0, ggml_nbytes(weight));
```

**模式 2：计算 Context（计算图中间 tensor）**

```cpp
// 创建 context，不分配 tensor data
struct ggml_context *ctx_compute = ggml_init({
    .mem_size = 1024 * ggml_tensor_overhead() + 3 * ggml_graph_overhead(),
    .no_alloc = true,
});

// 定义计算图 (tensor data 不分配)
struct ggml_tensor *a = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, 100, 100);
struct ggml_tensor *b = ggml_mul_mat(ctx_compute, weight, a);
struct ggml_tensor *c = ggml_relu(ctx_compute, b);

// 创建计算图
struct ggml_cgraph *graph = ggml_new_graph(ctx_compute);
ggml_build_forward_expand(graph, c);

// 由 ggml_gallocr 分配 tensor data (两阶段分配)
ggml_gallocr_t allocr = ggml_gallocr_new(...);
ggml_gallocr_alloc_graph(allocr, graph);
```

#### 5.2.3 API

```cpp
// 初始化 context
struct ggml_context *ggml_init(struct ggml_init_params params);

// 释放 context (不释放 backend buffer)
void ggml_free(struct ggml_context *ctx);

// 重置 context (保留元数据池，但标记所有内存为未使用)
void ggml_reset(struct ggml_context *ctx);

// 获取已使用的内存量
size_t ggml_used_mem(const struct ggml_context *ctx);

// 获取 tensor 元数据大小
size_t ggml_tensor_overhead(void);  // ~200 bytes

// 获取计算图元数据大小
size_t ggml_graph_overhead(void);
```

### 5.3 后端 (`ggml_backend`)

#### 5.3.1 抽象层级

```
ggml_backend_reg_t     后端注册表 (CPU, CUDA, Metal, ...)
       │
       ▼
ggml_backend_dev_t     后端设备 (GPU 0, GPU 1, ...)
       │
       ▼
ggml_backend_t         后端实例 (包含线程池、CUDA stream 等)
       │
       ▼
ggml_backend_buffer_type_t   Buffer 类型 (决定内存分配方式)
       │
       ▼
ggml_backend_buffer_t  Buffer 实例 (真实内存块)
```

#### 5.3.2 设备类型

```cpp
enum ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_CPU,     // CPU
    GGML_BACKEND_DEVICE_TYPE_GPU,     // 独立 GPU (CUDA, Vulkan, ...)
    GGML_BACKEND_DEVICE_TYPE_IGPU,    // 集成 GPU (Intel UHD, ...)
    GGML_BACKEND_DEVICE_TYPE_ACCEL,   // 加速器 (NPU, ...)
};
```

#### 5.3.3 初始化后端

```cpp
// 按类型初始化 (简单方式)
ggml_backend_t backend = ggml_backend_init_by_type(
    GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

// 设置 CPU 线程数
ggml_backend_cpu_set_n_threads(backend, 8);

// 按名称初始化
ggml_backend_t cuda_backend = ggml_backend_init_by_name("CUDA0", nullptr);

// 释放后端
ggml_backend_free(backend);
```

#### 5.3.4 后端调度器 (`ggml_backend_sched`)

当有多个后端时，调度器负责决定每个 tensor 在哪个后端执行：

```cpp
// 创建调度器
ggml_backend_t backends[] = {cuda_backend, cpu_backend};
ggml_backend_sched_t sched = ggml_backend_sched_new(
    backends,           // 后端数组 (优先级递减)
    nullptr,            // buffer 类型 (NULL 则自动选择)
    2,                  // 后端数量
    GGML_DEFAULT_GRAPH_SIZE,
    false,              // 是否并行
    true                // 是否启用算子卸载
);

// 分配计算图
ggml_backend_sched_alloc_graph(sched, graph);

// 执行计算
ggml_backend_sched_graph_compute(sched, graph);

// 重置 (清除分配状态)
ggml_backend_sched_reset(sched);

// 释放
ggml_backend_sched_free(sched);
```

**调度算法**（5 遍分图）：

1. **Pass 1**：根据权重 buffer 的后端预分配 tensor
2. **Pass 2**：扩展 GPU 后端覆盖范围
3. **Pass 3**：升级到更高优先级的兼容后端
4. **Pass 4**：根据输出和 view 源分配输入
5. **Pass 5**：实际分割图，插入跨后端拷贝节点

### 5.4 缓冲区 (`ggml_backend_buffer`)

#### 5.4.1 概念

`ggml_backend_buffer` 是实际的**设备内存块**：
- CPU：`malloc` / `mmap`
- CUDA：`cudaMalloc` / `cudaMallocHost`
- Metal：`MTLBuffer`

#### 5.4.2 Buffer 类型

```cpp
// 获取后端的默认 buffer 类型
ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

// CPU 后端的 buffer 类型
ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();

// 分配 buffer
ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, size);

// 获取 buffer 基地址
void *base = ggml_backend_buffer_get_base(buf);

// 获取 buffer 大小
size_t size = ggml_backend_buffer_get_size(buf);

// 释放 buffer
ggml_backend_buffer_free(buf);
```

#### 5.4.3 Buffer 用途标记

```cpp
enum ggml_backend_buffer_usage {
    GGML_BACKEND_BUFFER_USAGE_ANY = 0,      // 通用
    GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,  // 权重 (优先放 GPU)
    GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,  // 计算 (中间激活)
};

ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
```

#### 5.4.4 便捷分配函数

```cpp
// 为 context 中所有 tensor 分配内存 (使用后端的默认 buffer 类型)
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(
    struct ggml_context *ctx,
    ggml_backend_t backend);

// 使用指定的 buffer 类型
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(
    struct ggml_context *ctx,
    ggml_backend_buffer_type_t buft);
```

**内部流程**：
1. 遍历 context 中所有 tensor
2. 累加内存需求（含对齐）
3. 调用 `ggml_backend_buft_alloc_buffer()` 分配一块大 buffer
4. 为每个 tensor 设置 `data` 和 `buffer` 字段

### 5.5 计算图 (`ggml_cgraph`)

#### 5.5.1 结构

```cpp
struct ggml_cgraph {
    int size;                       // 最大节点数
    int n_nodes;                    // 当前节点数
    int n_leafs;                    // 叶子节点数
    
    struct ggml_tensor **nodes;     // 计算节点 (有操作的 tensor)
    struct ggml_tensor **leafs;     // 叶子节点 (输入 tensor)
    
    struct ggml_hash_set visited_hash_set;  // 已访问 tensor 集合
    
    struct ggml_tensor **grads;     // 梯度 tensor (反向图专用)
    struct ggml_tensor **grad_accs; // 梯度累加器
};
```

#### 5.5.2 创建和构建

```cpp
// 创建计算图 (默认 2048 节点)
struct ggml_cgraph *ggml_new_graph(struct ggml_context *ctx);

// 创建自定义大小的计算图
struct ggml_cgraph *ggml_new_graph_custom(
    struct ggml_context *ctx,
    size_t size,
    bool grads);  // 是否分配梯度存储

// 添加 tensor 及其所有依赖到计算图
void ggml_build_forward_expand(struct ggml_cgraph *cgraph, struct ggml_tensor *tensor);

// 构建反向图 (自动微分)
void ggml_build_backward_expand(
    struct ggml_context *ctx,
    struct ggml_cgraph *cgraph,
    struct ggml_tensor **grad_accs);  // 梯度累加器数组

// 复制计算图拓扑 (tensor 共享)
struct ggml_cgraph *ggml_graph_dup(
    struct ggml_context *ctx,
    struct ggml_cgraph *src,
    bool force_grads);
```

#### 5.5.3 获取梯度

```cpp
// 获取 tensor 的梯度
struct ggml_tensor *ggml_graph_get_grad(
    const struct ggml_cgraph *cgraph,
    const struct ggml_tensor *node);

// 获取 tensor 的梯度累加器
struct ggml_tensor *ggml_graph_get_grad_acc(
    const struct ggml_cgraph *cgraph,
    const struct ggml_tensor *node);
```

#### 5.5.4 重置图

```cpp
// 重置梯度为 0
void ggml_graph_reset(struct ggml_cgraph *cgraph);

// 打印图信息
void ggml_graph_print(const struct ggml_cgraph *cgraph);

// 导出 DOT 格式
void ggml_graph_dump_dot(
    const struct ggml_cgraph *gb,
    const struct ggml_cgraph *gf,
    const char *filename);
```

### 5.6 组件关系总结

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ggml_context                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Tensor 元数据池 (type, shape, op, src[], flags, name, ...)      │   │
│  │                                                                   │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐      │   │
│  │  │ tensor A  │  │ tensor B  │  │ tensor C  │  │   ...     │      │   │
│  │  │ data=NULL │  │ data=NULL │  │ data=NULL │  │           │      │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                │                                         │
│                                │ ggml_backend_alloc_ctx_tensors()        │
│                                ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    ggml_backend_buffer                            │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │              真实设备内存 (CPU/GPU/...)                    │  │   │
│  │  │  ┌─────┐  ┌─────┐  ┌─────┐                                │  │   │
│  │  │  │ A   │  │ B   │  │ C   │  ...                           │  │   │
│  │  │  └─────┘  └─────┘  └─────┘                                │  │   │
│  │  │  ▲                                                        │  │   │
│  │  │  │ tensor->data = buffer_base + offset                    │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  与 ggml_backend 关联                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  ggml_backend (CPU/CUDA/Metal/...)                               │   │
│  │  - 线程池                                                        │   │
│  │  - CUDA stream / Metal command queue                             │   │
│  │  - 算子实现 (compute kernel)                                     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*下一章将介绍优化器与自动微分的实现细节。*

---

## 第 6 章：优化器与自动微分

> GGML 内置了完整的自动微分系统和常用优化器，支持端到端训练。

### 6.1 自动微分

#### 6.1.1 基本原理

GGML 使用**反向模式自动微分**（Reverse-mode AD），也称为反向传播（Backpropagation）：

1. **前向传播**：计算输出和中间值
2. **反向传播**：从 loss 开始，逐层计算梯度

```
前向: x → [f] → y → [g] → z → [h] → loss
               
反向: ∂L/∂x ← [∂f/∂x] ← ∂L/∂y ← [∂g/∂y] ← ∂L/∂z ← [∂h/∂z] ← 1
```

#### 6.1.2 构建反向图

```cpp
// 前向图
struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, size, /*grads=*/true);
ggml_build_forward_expand(gf, loss);

// 设置 loss 的梯度为 1
ggml_set_loss(loss);

// 构建反向图
// grad_accs: 每个 PARAM 节点的梯度累加器数组
struct ggml_tensor **grad_accs = ...;  // 需要预先分配
ggml_build_backward_expand(ctx, gf, grad_accs);
```

**`ggml_build_backward_expand` 内部流程**：

1. 找到所有标记为 `GGML_TENSOR_FLAG_PARAM` 的节点
2. 从 `GGML_TENSOR_FLAG_LOSS` 节点开始反向遍历
3. 为每个算子生成对应的梯度计算节点
4. 将梯度累加到 `grad_accs` 数组

#### 6.1.3 支持的可微分算子

GGML 为以下算子实现了梯度计算：

| 算子类别 | 算子 |
|----------|------|
| **算术** | ADD, SUB, MUL, DIV, SQR, SQRT, LOG, EXP |
| **矩阵** | MUL_MAT, OUT_PROD |
| **归约** | SUM, SUM_ROWS, MEAN |
| **归一化** | NORM, RMS_NORM, SOFT_MAX |
| **激活** | RELU, GELU, SILU, TANH, SIGMOID |
| **数据操作** | RESHAPE, VIEW, PERMUTE, TRANSPOSE, CONT |
| **卷积** | CONV_2D (通过 IM2COL + MUL_MAT) |
| **池化** | POOL_2D |
| **损失** | CROSS_ENTROPY_LOSS |
| **注意力** | FLASH_ATTN_EXT |

#### 6.1.4 梯度累加器

梯度累加器用于实现**梯度累积**（Gradient Accumulation）：

```cpp
// 在 ctx_static 中分配梯度累加器
for (int i = 0; i < gf->n_nodes; ++i) {
    ggml_tensor *node = gf->nodes[i];
    if (node->flags & GGML_TENSOR_FLAG_PARAM) {
        grad_accs[i] = ggml_new_tensor(ctx_static, GGML_TYPE_F32, 
                                       GGML_MAX_DIMS, node->ne);
    }
}
```

**工作流程**：

```
Step 1: grad_acc = 0 (重置)
Step 2: grad_acc += ∂L/∂W (第 1 个 mini-batch 的梯度)
Step 3: grad_acc += ∂L/∂W (第 2 个 mini-batch 的梯度)
...
Step N: W = W - α * grad_acc (优化器更新)
        grad_acc = 0 (重置)
```

### 6.2 优化器

#### 6.2.1 支持的优化器

| 优化器 | 类型常量 | 参数 |
|--------|----------|------|
| **AdamW** | `GGML_OPT_OPTIMIZER_TYPE_ADAMW` | alpha, beta1, beta2, eps, wd |
| **SGD** | `GGML_OPT_OPTIMIZER_TYPE_SGD` | alpha, wd |

#### 6.2.2 AdamW 优化器

**数学公式**：

$$
\begin{align}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= m_t / (1 - \beta_1^t) \\
\hat{v}_t &= v_t / (1 - \beta_2^t) \\
\theta_t &= \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
\end{align}
$$

其中：
- $g_t$：当前梯度
- $m_t$：一阶动量（梯度的指数移动平均）
- $v_t$：二阶动量（梯度平方的指数移动平均）
- $\hat{m}_t, \hat{v}_t$：Bias correction（初期修正）
- $\alpha$：学习率
- $\lambda$：权重衰减（Weight Decay）

**参数配置**：

```cpp
struct ggml_opt_optimizer_params {
    struct {
        float alpha;  // 学习率，默认 0.001
        float beta1;  // 一阶动量系数，默认 0.9
        float beta2;  // 二阶动量系数，默认 0.999
        float eps;    // 数值稳定性，默认 1e-8
        float wd;     // 权重衰减，默认 0.0
    } adamw;
};
```

**GGML 实现**：

```cpp
// 创建优化器步骤节点
struct ggml_tensor *ggml_opt_step_adamw(
    struct ggml_context *ctx,
    struct ggml_tensor *a,       // 参数 tensor
    struct ggml_tensor *grad,    // 梯度
    struct ggml_tensor *m,       // 一阶动量
    struct ggml_tensor *v,       // 二阶动量
    struct ggml_tensor *adamw_params);  // [alpha, beta1, beta2, eps, wd, beta1h, beta2h]
```

该节点是 **in-place** 操作，直接修改参数 `a`、动量 `m` 和 `v`。

**参数张量布局**：

```cpp
float *params = ggml_get_data_f32(opt_step_params);
params[0] = alpha;   // 学习率
params[1] = beta1;   // β₁
params[2] = beta2;   // β₂
params[3] = eps;     // ε
params[4] = wd;      // 权重衰减
params[5] = 1.0f / (1.0f - powf(beta1, iter));  // bias correction 1
params[6] = 1.0f / (1.0f - powf(beta2, iter));  // bias correction 2
```

#### 6.2.3 SGD 优化器

**数学公式**：

$$
\theta_t = \theta_{t-1} - \alpha (g_t + \lambda \theta_{t-1})
$$

**GGML 实现**：

```cpp
struct ggml_tensor *ggml_opt_step_sgd(
    struct ggml_context *ctx,
    struct ggml_tensor *a,       // 参数 tensor
    struct ggml_tensor *grad,    // 梯度
    struct ggml_tensor *sgd_params);  // [alpha, wd]
```

### 6.3 损失函数

#### 6.3.1 Cross-Entropy Loss

用于分类任务：

```cpp
// API
struct ggml_tensor *ggml_cross_entropy_loss(
    struct ggml_context *ctx,
    struct ggml_tensor *a,       // 预测 logits [n_classes, batch]
    struct ggml_tensor *b);      // 真实标签 (one-hot) [n_classes, batch]

// 数学公式
// L = -sum(y * log(softmax(x))) / batch_size
```

**构建方式**：

```cpp
opt_ctx->labels = ggml_dup_tensor(ctx, opt_ctx->outputs);
ggml_set_input(opt_ctx->labels);
opt_ctx->loss = ggml_cross_entropy_loss(ctx, opt_ctx->outputs, opt_ctx->labels);
ggml_set_loss(opt_ctx->loss);
```

#### 6.3.2 Mean Squared Error

用于回归任务：

```cpp
// GGML 中的实现 (组合基本算子)
struct ggml_tensor *error = ggml_sub(ctx, outputs, labels);
struct ggml_tensor *squared = ggml_sqr(ctx, error);
struct ggml_tensor *loss = ggml_sum(ctx, squared);
loss = ggml_scale(ctx, loss, 1.0f / n_elements);
```

#### 6.3.3 损失类型枚举

```cpp
enum ggml_opt_loss_type {
    GGML_OPT_LOSS_TYPE_MEAN,            // sum(outputs) / n
    GGML_OPT_LOSS_TYPE_SUM,             // sum(outputs)
    GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,   // cross_entropy(outputs, labels)
    GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,  // mse(outputs, labels)
};
```

### 6.4 训练辅助指标

#### 6.4.1 预测

```cpp
// 获取每个样本的预测类别
struct ggml_tensor *pred = ggml_argmax(ctx, outputs);  // [batch]
ggml_set_output(pred);
```

#### 6.4.2 准确率统计

```cpp
// 计算预测正确的数量
struct ggml_tensor *true_labels = ggml_argmax(ctx, labels);
struct ggml_tensor *ncorrect = ggml_count_equal(ctx, pred, true_labels);  // scalar
ggml_set_output(ncorrect);
```

### 6.5 学习率调度

GGML 通过回调函数支持自定义学习率调度：

```cpp
// 回调函数类型
typedef struct ggml_opt_optimizer_params (*ggml_opt_get_optimizer_params)(
    void *userdata);

// 示例：线性衰减学习率
struct ggml_opt_optimizer_params get_opt_pars_with_decay(void *userdata) {
    int64_t *epoch = (int64_t *)userdata;
    
    ggml_opt_optimizer_params params = ggml_opt_get_default_optimizer_params(nullptr);
    
    // 线性衰减：从 0.001 衰减到 0.0001
    float initial_lr = 0.001f;
    float final_lr = 0.0001f;
    float total_epochs = 10.0f;
    
    params.adamw.alpha = initial_lr - (*epoch / total_epochs) * (initial_lr - final_lr);
    
    return params;
}

// 使用
int64_t epoch = 1;
ggml_opt_params params = ggml_opt_default_params(backend_sched, loss_type);
params.get_opt_pars = get_opt_pars_with_decay;
params.get_opt_pars_ud = &epoch;
```

### 6.6 完整的训练图结构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Forward Pass (gf)                                 │
│                                                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐              │
│  │  input  │───▶│  Conv   │───▶│  Pool   │───▶│  Dense  │───▶ logits   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘              │
│       │              │              │              │                    │
│       │              │              │              │                    │
│       ▼              ▼              ▼              ▼                    │
│  labels ─────────────────────────────────────────────────▶ cross_entropy│
│                                                                │        │
│                                                                ▼        │
│                                                             loss        │
│                                                                │        │
│                                            pred ◀── argmax ◀───┤        │
│                                                                │        │
│                                       ncorrect ◀── count_equal─┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                         │
                                         │ ggml_build_backward_expand()
                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Backward Pass (gb_grad)                             │
│                                                                          │
│  [ Forward Pass 的所有节点 ]                                             │
│                                                                          │
│  ∂L/∂logits ◀── ∂cross_entropy/∂logits ◀── ∂L/∂loss = 1                 │
│       │                                                                  │
│       ▼                                                                  │
│  ∂L/∂W_dense ◀── ∂Dense/∂W ◀── ∂L/∂logits                               │
│       │                                                                  │
│       ▼                                                                  │
│  grad_acc_dense += ∂L/∂W_dense                                          │
│       │                                                                  │
│       ▼ (继续反向传播到 Pool, Conv, ...)                                 │
│                                                                          │
│  grad_acc_conv2 += ∂L/∂W_conv2                                          │
│  grad_acc_conv1 += ∂L/∂W_conv1                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                         │
                                         │ 添加优化器节点
                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Optimizer Step (gb_opt)                             │
│                                                                          │
│  [ gb_grad 的所有节点 ]                                                  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    ggml_opt_step_adamw                              │ │
│  │                                                                     │ │
│  │  for each param W ∈ {conv1, conv2, dense}:                         │ │
│  │      m[W] = β₁ * m[W] + (1 - β₁) * grad_acc[W]                     │ │
│  │      v[W] = β₂ * v[W] + (1 - β₂) * grad_acc[W]²                    │ │
│  │      W = W - α * (m̂[W] / (√v̂[W] + ε) + λ * W)                       │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*下一章将介绍 GGUF 模型文件格式和权重管理。*

---

## 第 7 章：权重管理 — GGUF 格式

> GGUF（GGML Universal Format）是 GGML 的标准模型文件格式，支持元数据、多种量化类型和高效的内存映射。

### 7.1 GGUF 文件结构

```
┌──────────────────────────────────────────────────────────────────────┐
│                           GGUF File                                   │
├──────────────────────────────────────────────────────────────────────┤
│  Header (固定大小)                                                    │
│  ├── magic: "GGUF" (4 bytes)                                         │
│  ├── version: uint32 (当前为 3)                                      │
│  ├── n_tensors: uint64 (tensor 数量)                                 │
│  └── n_kv: uint64 (KV 对数量)                                        │
├──────────────────────────────────────────────────────────────────────┤
│  KV Pairs (元数据)                                                    │
│  ├── [key_length, key_string, value_type, value] × n_kv              │
│  │                                                                    │
│  │  例如:                                                             │
│  │  - "general.architecture" = "mnist-cnn"                           │
│  │  - "general.name" = "MNIST CNN Model"                             │
│  │  - "mnist.conv1.kernel_size" = 3                                  │
├──────────────────────────────────────────────────────────────────────┤
│  Tensor Info (每个 tensor 的元数据)                                   │
│  ├── [name_length, name_string, n_dims, dims[], type, offset] × n    │
│  │                                                                    │
│  │  例如:                                                             │
│  │  - "conv1.kernel": dims=[3,3,1,8], type=F32, offset=0             │
│  │  - "conv1.bias": dims=[1,1,8], type=F32, offset=288               │
├──────────────────────────────────────────────────────────────────────┤
│  Alignment Padding                                                    │
│  └── 填充到 alignment 边界 (默认 32 bytes)                           │
├──────────────────────────────────────────────────────────────────────┤
│  Tensor Data (二进制 blob)                                           │
│  ├── [tensor_0_data][padding][tensor_1_data][padding]...             │
│  │                                                                    │
│  │  每个 tensor 按 alignment 对齐                                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 KV 值类型

```cpp
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};
```

### 7.3 加载模型

#### 7.3.1 两阶段加载

```cpp
// 阶段 1：解析元数据，创建 tensor 结构
struct ggml_context *ctx_gguf = nullptr;
struct gguf_init_params params = {
    .no_alloc = true,     // 不分配 tensor 数据
    .ctx = &ctx_gguf,     // 输出 ggml_context
};
struct gguf_context *gguf_ctx = gguf_init_from_file("model.gguf", params);

// 此时 ctx_gguf 包含所有 tensor 的元数据，但 data=NULL

// 阶段 2：分配后端 buffer
ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_gguf, backend);

// 阶段 3：加载 tensor 数据
load_from_gguf("model.gguf", ctx_gguf, gguf_ctx);
```

#### 7.3.2 数据加载实现

```cpp
bool load_from_gguf(const char *fname, 
                    struct ggml_context *ctx_ggml, 
                    struct gguf_context *ctx_gguf) 
{
    FILE *f = ggml_fopen(fname, "rb");
    const size_t buf_size = 4 * 1024 * 1024;  // 4MB 缓冲区
    void *buf = malloc(buf_size);

    const int n_tensors = gguf_get_n_tensors(ctx_gguf);
    
    for (int i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor *tensor = ggml_get_tensor(ctx_ggml, name);
        
        // 计算文件中的偏移量
        const size_t offs = gguf_get_data_offset(ctx_gguf) + 
                           gguf_get_tensor_offset(ctx_gguf, i);
        
        fseek(f, offs, SEEK_SET);
        
        // 分块读取并设置到 backend buffer
        const size_t nbytes = ggml_nbytes(tensor);
        for (size_t pos = 0; pos < nbytes; pos += buf_size) {
            const size_t nbytes_cpy = min(buf_size, nbytes - pos);
            fread(buf, 1, nbytes_cpy, f);
            ggml_backend_tensor_set(tensor, buf, pos, nbytes_cpy);
        }
    }
    
    fclose(f);
    free(buf);
    return true;
}
```

#### 7.3.3 获取 tensor

```cpp
// 按名称获取 tensor
struct ggml_tensor *weight = ggml_get_tensor(ctx_gguf, "conv1.kernel");

// 或者通过索引遍历
const int n_tensors = gguf_get_n_tensors(gguf_ctx);
for (int i = 0; i < n_tensors; i++) {
    const char *name = gguf_get_tensor_name(gguf_ctx, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_gguf, name);
    printf("Tensor %s: [%lld, %lld, %lld, %lld]\n", 
           name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
}
```

### 7.4 保存模型

#### 7.4.1 基本流程

```cpp
void save_model(const std::string &fname, 
                const std::vector<ggml_tensor *> &weights) 
{
    // 1. 创建临时 context (用于复制 tensor)
    struct ggml_context *ggml_ctx;
    struct ggml_init_params params = {
        .mem_size = 100 * 1024 * 1024,  // 100MB
        .mem_buffer = NULL,
        .no_alloc = false,  // 需要分配数据
    };
    ggml_ctx = ggml_init(params);

    // 2. 创建 GGUF context
    gguf_context *gguf_ctx = gguf_init_empty();

    // 3. 设置元数据
    gguf_set_val_str(gguf_ctx, "general.architecture", "mnist-cnn");
    gguf_set_val_str(gguf_ctx, "general.name", "MNIST CNN Model");
    gguf_set_val_u32(gguf_ctx, "mnist.version", 1);

    // 4. 添加 tensor
    for (struct ggml_tensor *t : weights) {
        // 复制 tensor (元数据 + 数据)
        struct ggml_tensor *copy = ggml_dup_tensor(ggml_ctx, t);
        ggml_set_name(copy, t->name);
        
        // 从 backend buffer 复制数据到 CPU
        ggml_backend_tensor_get(t, copy->data, 0, ggml_nbytes(t));
        
        // 添加到 GGUF
        gguf_add_tensor(gguf_ctx, copy);
    }

    // 5. 写入文件
    gguf_write_to_file(gguf_ctx, fname.c_str(), /*only_meta=*/false);

    // 6. 清理
    ggml_free(ggml_ctx);
    gguf_free(gguf_ctx);
}
```

#### 7.4.2 设置 KV 元数据

```cpp
// 设置各种类型的值
gguf_set_val_str(ctx, "general.architecture", "mnist-cnn");
gguf_set_val_u32(ctx, "mnist.hidden_size", 512);
gguf_set_val_f32(ctx, "mnist.dropout", 0.1f);
gguf_set_val_bool(ctx, "mnist.use_bias", true);

// 设置数组
const char *authors[] = {"Alice", "Bob"};
gguf_set_arr_str(ctx, "general.authors", authors, 2);

uint32_t dims[] = {28, 28};
gguf_set_arr_data(ctx, "mnist.input_dims", GGUF_TYPE_UINT32, dims, 2);
```

#### 7.4.3 读取 KV 元数据

```cpp
// 查找 key
int key_id = gguf_find_key(ctx, "general.architecture");
if (key_id != -1) {
    const char *arch = gguf_get_val_str(ctx, key_id);
    printf("Architecture: %s\n", arch);
}

// 按类型读取
uint32_t hidden = gguf_get_val_u32(ctx, gguf_find_key(ctx, "mnist.hidden_size"));
float dropout = gguf_get_val_f32(ctx, gguf_find_key(ctx, "mnist.dropout"));
```

### 7.5 量化

#### 7.5.1 支持的量化类型

| 类型 | 位宽 | 精度 | 压缩比 | 用途 |
|------|------|------|--------|------|
| F32 | 32 | 最高 | 1x | 训练 |
| F16 | 16 | 高 | 2x | 推理 |
| Q8_0 | 8 | 高 | 4x | 高质量量化 |
| Q4_0 | 4 | 中 | 8x | 平衡压缩 |
| Q4_K | 4 | 较高 | ~8x | K-quant（推荐） |
| Q2_K | 2 | 低 | 16x | 极限压缩 |

#### 7.5.2 量化原理

**Q4_0 量化**（对称，无零点）：

$$
Q(x) = \text{round}\left(\frac{x}{s}\right) \cdot s
$$

其中 $s = \max(|x|) / 7$ （4 位有符号整数范围 -8 到 7）。

**K-quant**（分块超参，更高精度）：

```
每 256 个元素为一组
├── d: float16 (scale)
├── dmin: float16 (minimum scale)
├── scales: uint8[12] (子块 scale)
└── qs: uint8[128] (量化值)
```

#### 7.5.3 量化 API

```cpp
// 计算量化后的大小
size_t ggml_row_size(enum ggml_type type, int64_t ne0);

// 量化函数
void ggml_quantize_chunk(
    enum ggml_type type,
    const float *src,
    void *dst,
    int64_t start,
    int64_t nrows,
    int64_t n_per_row,
    const float *imatrix);  // 重要性矩阵（可选）
```

### 7.6 MNIST 模型示例

训练后保存的 MNIST CNN 模型结构：

```
$ gguf-parser models/mnist-cnn-f32.gguf

GGUF File: models/mnist-cnn-f32.gguf
Size: 36864 bytes
GGUF version: 3
KV count: 1
Tensor count: 6
Alignment: 32
Tensor data blob offset (file): 448

=== Metadata (KV) ===
general.architecture : string = mnist-cnn

=== Tensors ===
[   0] conv1.kernel
       dims=[3, 3, 1, 8]  n_elems=72  type=F32
       rel_off=0  abs_off=448  size=288 bytes

[   1] conv1.bias
       dims=[1, 1, 8]  n_elems=8  type=F32
       rel_off=288  abs_off=736  size=32 bytes

[   2] conv2.kernel
       dims=[3, 3, 8, 16]  n_elems=1152  type=F32
       rel_off=320  abs_off=768  size=4608 bytes

[   3] conv2.bias
       dims=[1, 1, 16]  n_elems=16  type=F32
       rel_off=4928  abs_off=5376  size=64 bytes

[   4] dense.weight
       dims=[784, 10]  n_elems=7840  type=F32
       rel_off=4992  abs_off=5440  size=31360 bytes

[   5] dense.bias
       dims=[10]  n_elems=10  type=F32
       rel_off=36352  abs_off=36800  size=40 bytes
```

### 7.7 API 速查表

| 函数 | 用途 |
|------|------|
| **初始化** | |
| `gguf_init_empty()` | 创建空 GGUF context |
| `gguf_init_from_file()` | 从文件加载 GGUF |
| `gguf_free()` | 释放 GGUF context |
| **KV 操作** | |
| `gguf_get_n_kv()` | 获取 KV 对数量 |
| `gguf_find_key()` | 查找 key 的索引 |
| `gguf_get_val_*()` | 获取 KV 值 |
| `gguf_set_val_*()` | 设置 KV 值 |
| **Tensor 操作** | |
| `gguf_get_n_tensors()` | 获取 tensor 数量 |
| `gguf_get_tensor_name()` | 获取 tensor 名称 |
| `gguf_get_tensor_offset()` | 获取 tensor 数据偏移 |
| `gguf_add_tensor()` | 添加 tensor |
| **文件操作** | |
| `gguf_get_data_offset()` | 获取数据区偏移 |
| `gguf_write_to_file()` | 写入文件 |

---

*下一节是附录，介绍算子系统和卷积实现。*

---

## 附录：算子系统与卷积实现

### A.1 算子总览

GGML 内置 80+ 种算子，定义于 `include/ggml.h`：

| 类别 | 算子 |
|------|------|
| **算术** | ADD, SUB, MUL, DIV, SQR, SQRT, LOG, EXP, SIN, COS |
| **归约** | SUM, SUM_ROWS, MEAN, ARGMAX, COUNT_EQUAL |
| **矩阵** | MUL_MAT, MUL_MAT_ID, OUT_PROD |
| **归一化** | NORM, RMS_NORM, GROUP_NORM, LAYER_NORM |
| **激活** | RELU, GELU, GELU_QUICK, SILU, TANH, SIGMOID, LEAKY_RELU |
| **数据操作** | RESHAPE, VIEW, PERMUTE, TRANSPOSE, CONT, GET_ROWS, CPY |
| **卷积** | CONV_1D, CONV_2D, CONV_TRANSPOSE_1D, CONV_TRANSPOSE_2D |
| **池化** | POOL_1D, POOL_2D |
| **注意力** | FLASH_ATTN_EXT, FLASH_ATTN_BACK |
| **RoPE** | ROPE, ROPE_BACK |
| **损失** | CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_BACK |
| **优化** | OPT_STEP_ADAMW, OPT_STEP_SGD |
| **自定义** | MAP_UNARY, MAP_BINARY, MAP_CUSTOM1~3 |

### A.2 CONV2D 算子

#### A.2.1 API 签名

```cpp
struct ggml_tensor *ggml_conv_2d(
    struct ggml_context *ctx,
    struct ggml_tensor  *a,    // kernel: [OC, IC, KH, KW]
    struct ggml_tensor  *b,    // input:  [N, IC, IH, IW]
    int s0, int s1,            // stride (水平, 垂直)
    int p0, int p1,            // padding (水平, 垂直)
    int d0, int d1);           // dilation (水平, 垂直)
// 返回: [N, OC, OH, OW]
```

#### A.2.2 参数说明

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `s0, s1` | **步长 (Stride)**：卷积核每次滑动的像素距离 | 1 (逐像素) 或 2 (降采样) |
| `p0, p1` | **填充 (Padding)**：图像边缘补零的圈数 | 0 (valid) 或 1 (same for 3×3) |
| `d0, d1` | **空洞 (Dilation)**：卷积核元素间距 | 1 (标准) 或 2 (扩大感受野) |

#### A.2.3 输出尺寸计算

$$
OH = \frac{IH + 2 \cdot p1 - d1 \cdot (KH - 1) - 1}{s1} + 1
$$

$$
OW = \frac{IW + 2 \cdot p0 - d0 \cdot (KW - 1) - 1}{s0} + 1
$$

### A.3 Img2col + GEMM 实现

GGML 的 CONV2D 通过 **Img2col + GEMM** 算法实现，这是高效卷积的经典方法。

#### A.3.1 算法原理

将卷积运算转换为矩阵乘法：

1. **Im2col**：将输入的每个感受野拉成一列
2. **Reshape**：将权重拉成矩阵
3. **GEMM**：矩阵乘法
4. **Reshape**：还原为 4D 张量

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Standard Convolution                            │
│                                                                         │
│  Input [C,H,W]           Kernel [OC,C,KH,KW]          Output [OC,OH,OW] │
│  ┌─────────┐             ┌─────────┐                  ┌─────────┐       │
│  │ ░░░░░░░ │             │ K₀      │                  │ Y₀      │       │
│  │ ░░▓▓▓░░ │      ★      │ K₁      │         =        │ Y₁      │       │
│  │ ░░▓▓▓░░ │             │ ...     │                  │ ...     │       │
│  │ ░░▓▓▓░░ │             │ K_OC    │                  │ Y_OC    │       │
│  └─────────┘             └─────────┘                  └─────────┘       │
│                                                                         │
│  滑动窗口，局部加权求和                                                  │
└────────────────────────────────────────────────────────────────────────┘

                                ↓ Im2col 变换

┌────────────────────────────────────────────────────────────────────────┐
│                         Matrix Multiplication                           │
│                                                                         │
│  Kernel Matrix             Im2col Matrix              Output Matrix     │
│  [OC, C×KH×KW]             [C×KH×KW, OH×OW]          [OC, OH×OW]       │
│  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐   │
│  │ K₀ ───────▶ │           │ ↓ ↓ ↓ ↓ ↓  │           │ Y₀ Y₀ Y₀... │   │
│  │ K₁ ───────▶ │     ×     │ x x x x x  │     =     │ Y₁ Y₁ Y₁... │   │
│  │ ...         │           │ x x x x x  │           │ ...          │   │
│  │ K_OC ─────▶ │           │ x x x x x  │           │ Y_OC ...     │   │
│  └─────────────┘           └─────────────┘           └─────────────┘   │
│                                                                         │
│  GEMM: Y = W × X_col                                                   │
└────────────────────────────────────────────────────────────────────────┘
```

#### A.3.2 数学推导

标准卷积公式（单样本，忽略 batch）：

$$
y_{k,p,q} = \sum_{c=0}^{C-1} \sum_{r=0}^{R-1} \sum_{s=0}^{S-1} W_{k,c,r,s} \times X_{c, \ p \cdot s_1 + r, \ q \cdot s_0 + s}
$$

其中：
- $X$：输入 $[C \times H \times W]$
- $W$：权重 $[K \times C \times R \times S]$（K=输出通道，R/S=核高宽）
- $Y$：输出 $[K \times P \times Q]$

**Im2col 变换**：

1. 将权重矩阵 $W$ reshape 为 $[K, C \cdot R \cdot S]$
2. 将输入的每个感受野拉成一列，形成 $X_{col}$：$[C \cdot R \cdot S, P \cdot Q]$

**矩阵乘法等价性**：

$$
\mathbf{Y}_{mat}[k, l] = \sum_{j=0}^{D-1} \mathbf{W}_{mat}[k, j] \times \mathbf{X}_{col}[j, l]
$$

其中 $D = C \cdot R \cdot S$，$l$ 编码空间位置 $(p, q)$。

将 $j$ 展开为 $(c, r, s)$：

$$
\sum_{j} \rightarrow \sum_{c=0}^{C-1} \sum_{r=0}^{R-1} \sum_{s=0}^{S-1}
$$

代入：
- $\mathbf{W}_{mat}[k, j] \rightarrow W_{k, c, r, s}$
- $\mathbf{X}_{col}[j, l] \rightarrow X_{c, p+r, q+s}$

即得标准卷积公式，证明等价性。

#### A.3.3 GGML 实现

```cpp
// include/ggml.h
struct ggml_tensor *ggml_conv_2d(
    struct ggml_context *ctx,
    struct ggml_tensor  *a,    // kernel
    struct ggml_tensor  *b,    // input
    int s0, int s1, int p0, int p1, int d0, int d1) 
{
    // 1. Im2col: 将输入的感受野展开为列
    struct ggml_tensor *im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, GGML_TYPE_F16);
    
    // 2. Reshape kernel: [OC, IC, KH, KW] -> [OC, IC*KH*KW]
    struct ggml_tensor *a_reshape = ggml_reshape_2d(ctx, a, 
        ggml_element_size(a) * a->ne[0] * a->ne[1] * a->ne[2],  // IC*KH*KW
        a->ne[3]);  // OC
    
    // 3. GEMM: kernel × im2col
    struct ggml_tensor *result = ggml_mul_mat(ctx, a_reshape, im2col);
    
    // 4. Reshape 回 4D: [OC, OH*OW] -> [OC, OH, OW, N]
    result = ggml_reshape_4d(ctx, result, OH, OW, OC, N);
    
    return result;
}
```

**Im2col 函数**：

```cpp
struct ggml_tensor *ggml_im2col(
    struct ggml_context *ctx,
    struct ggml_tensor  *a,    // kernel (用于获取 KH, KW)
    struct ggml_tensor  *b,    // input
    int s0, int s1,            // stride
    int p0, int p1,            // padding
    int d0, int d1,            // dilation
    bool is_2D,
    enum ggml_type dst_type);  // 输出类型 (通常 F16 节省带宽)
```

### A.4 MaxPool2D 算子

#### A.4.1 API 签名

```cpp
struct ggml_tensor *ggml_pool_2d(
    struct ggml_context *ctx,
    struct ggml_tensor  *a,    // input: [N, C, H, W]
    enum ggml_op_pool op,      // GGML_OP_POOL_MAX 或 GGML_OP_POOL_AVG
    int k0, int k1,            // kernel size (宽, 高)
    int s0, int s1,            // stride (宽, 高)
    float p0, float p1);       // padding (宽, 高)
// 返回: [N, C, OH, OW]
```

#### A.4.2 池化类型

```cpp
enum ggml_op_pool {
    GGML_OP_POOL_MAX,    // 最大池化
    GGML_OP_POOL_AVG,    // 平均池化
    GGML_OP_POOL_COUNT,
};
```

#### A.4.3 使用示例

```cpp
// 2×2 最大池化，步长 2（尺寸减半）
struct ggml_tensor *pool_out = ggml_pool_2d(
    ctx, conv_out, 
    GGML_OP_POOL_MAX,
    /*k0=*/2, /*k1=*/2,  // 池化窗口
    /*s0=*/2, /*s1=*/2,  // 步长
    /*p0=*/0, /*p1=*/0); // 无填充

// 输入 [B, C, 28, 28] → 输出 [B, C, 14, 14]
```

### A.5 完整 CNN 前向传播示例

```cpp
void build_cnn_forward(ggml_context *ctx,
                       ggml_tensor *input,    // [B, 1, 28, 28]
                       ggml_tensor *conv1_k,  // [8, 1, 3, 3]
                       ggml_tensor *conv1_b,  // [8]
                       ggml_tensor *conv2_k,  // [16, 8, 3, 3]
                       ggml_tensor *conv2_b,  // [16]
                       ggml_tensor *fc_w,     // [10, 784]
                       ggml_tensor *fc_b,     // [10]
                       ggml_tensor **output)
{
    // Conv1: [B,1,28,28] → [B,8,28,28]
    ggml_tensor *x = ggml_conv_2d(ctx, conv1_k, input, 
                                  1, 1, 1, 1, 1, 1);
    x = ggml_add(ctx, x, conv1_b);
    x = ggml_relu(ctx, x);
    
    // Pool1: [B,8,28,28] → [B,8,14,14]
    x = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    
    // Conv2: [B,8,14,14] → [B,16,14,14]
    x = ggml_conv_2d(ctx, conv2_k, x, 1, 1, 1, 1, 1, 1);
    x = ggml_add(ctx, x, conv2_b);
    x = ggml_relu(ctx, x);
    
    // Pool2: [B,16,14,14] → [B,16,7,7]
    x = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    
    // Flatten: [B,16,7,7] → [B,784]
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
    x = ggml_reshape_2d(ctx, x, 7*7*16, ggml_ne(input, 3));
    
    // FC: [B,784] → [B,10]
    x = ggml_mul_mat(ctx, fc_w, x);
    x = ggml_add(ctx, x, fc_b);
    
    *output = x;
}
```

---

## 总结

本文档详细介绍了 GGML 的训练与推理机制：

1. **第 1 章**：GGML 的设计理念和架构总览
2. **第 2 章**：MNIST CNN 完整示例（训练 + 推理流程）
3. **第 3 章**：`ggml_opt` 高层训练 API（Dataset、Context、训练三级图）
4. **第 4 章**：两阶段内存分配（Reserve + Alloc）
5. **第 5 章**：基础组件（tensor、context、backend、buffer、cgraph）
6. **第 6 章**：优化器与自动微分（AdamW、SGD、反向传播）
7. **第 7 章**：GGUF 模型格式（加载、保存、量化）
8. **附录**：卷积算子和 Img2col+GEMM 实现

### 快速参考

| 任务 | 关键 API |
|------|----------|
| 创建 tensor | `ggml_new_tensor_*d()` |
| 分配后端内存 | `ggml_backend_alloc_ctx_tensors()` |
| 构建计算图 | `ggml_build_forward_expand()` |
| 创建数据集 | `ggml_opt_dataset_init()` |
| 训练模型 | `ggml_opt_fit()` 或 `ggml_opt_epoch()` |
| 推理 | `ggml_opt_alloc() + ggml_opt_eval()` |
| 保存模型 | `gguf_add_tensor() + gguf_write_to_file()` |
| 加载模型 | `gguf_init_from_file() + load_from_gguf()` |

### 相关文件

- `include/ggml.h` - 核心 API
- `include/ggml-opt.h` - 训练 API
- `include/ggml-backend.h` - 后端 API
- `include/gguf.h` - 文件格式 API
- `examples/demo/` - MNIST 示例
- `examples/mnist/` - 官方 MNIST 示例

---

*文档最后更新：2026-03-04*
