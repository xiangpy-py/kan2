Linear Time B-splines Kolmogorov-Arnold Network (LTBs-KAN) Rust reproduction.

## 项目状态

- 当前已完成：`basis / coeffs / grid / layer` 核心模块、训练闭环、真实数据实验入口。
- 当前目标：先对齐线性层版本的论文主干，再扩展到论文中的卷积架构实验。

## 目录说明

- `src/spline/basis.rs`：Bernstein 基函数。
- `src/spline/coeffs.rs`：Algorithm 1 系数张量 `C` 计算。
- `src/spline/grid.rs`：Algorithm 2 自适应网格更新。
- `src/spline/layer.rs`：LTBs 线性层（含参数缩减与前向模式）。
- `src/convnet.rs`：卷积骨架（Conv 特征提取 + LTBs 分类头）。
- `src/training.rs`：训练与评估脚手架（合成数据 + 真实数据入口 + Conv 骨架入口）。
- `src/main.rs`：最小运行入口。

## 运行方式

1. 编译检查

```bash
cargo check
```

2. 运行单测

```bash
cargo test
```

3. 运行示例（默认先跑合成数据，再尝试真实数据）

```bash
cargo run
```

## 真实数据目录约定

默认根目录为 `datasets`，可通过环境变量覆盖。

- `datasets/mnist`
- `datasets/fashion-mnist`
- `datasets/cifar-10-batches-bin`

## 可配置环境变量

- `KAN_DATASETS_ROOT`：数据根目录，默认 `datasets`。
- `KAN_EPOCHS`：训练轮数，默认 `1`。
- `KAN_BATCH_SIZE`：batch 大小，默认 `64`。
- `KAN_LR`：学习率，默认 `1e-2`。
- `KAN_USE_FACTOR_LINEAR`：是否启用参数缩减线性项，默认 `true`。
- `KAN_RUN_CONVNET`：是否运行卷积骨架实验，默认 `false`。

示例：

```bash
$env:KAN_DATASETS_ROOT="datasets"
$env:KAN_EPOCHS="2"
$env:KAN_BATCH_SIZE="128"
$env:KAN_LR="0.005"
$env:KAN_USE_FACTOR_LINEAR="true"
$env:KAN_RUN_CONVNET="true"
cargo run
```

## 注意事项

- 目前实现优先保证论文主路径可运行与可验证，性能优化仍可继续迭代。
- 如果真实数据目录不存在，程序会自动跳过对应实验并打印提示信息。
