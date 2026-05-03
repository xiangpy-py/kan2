use std::io::Result;
use std::env;
use std::path::Path;

use tch::vision::{cifar, dataset::Dataset, mnist};
use tch::{Device, Kind, Tensor};

use crate::convnet::KanConvNet;
use crate::spline::layer::LtbSplineLayer;

/// 训练配置：先给轻量默认值，保证本地可快速跑通。
pub struct TrainingConfig {
    pub epochs: i64,
    pub batch_size: i64,
    pub learning_rate: f64,
    pub use_factorized_linear: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 64,
            learning_rate: 1.0e-2,
            use_factorized_linear: true,
        }
    }
}

impl TrainingConfig {
    /// 从环境变量读取训练配置，未设置时回退默认值。
    pub fn from_env() -> Self {
        Self {
            epochs: read_env_or_default_i64("KAN_EPOCHS", 1),
            batch_size: read_env_or_default_i64("KAN_BATCH_SIZE", 64),
            learning_rate: read_env_or_default_f64("KAN_LR", 1.0e-2),
            use_factorized_linear: read_env_or_default_bool("KAN_USE_FACTOR_LINEAR", true),
        }
    }
}

/// 合成数据训练演示：用于验证前向、反向和参数更新链路是连通的。
pub fn run_synthetic_training_demo(
    steps: i64,
    batch_size: i64,
    input_dim: i64,
    output_dim: i64,
    device: Device,
) {
    let mut layer = LtbSplineLayer::new(input_dim, output_dim, input_dim, 3, device);
    let learning_rate = 1.0e-2;

    for step in 0..steps {
        let input = Tensor::randn([batch_size, input_dim], (Kind::Float, device));
        let target = Tensor::randn([batch_size, output_dim], (Kind::Float, device));

        let prediction = layer.forward_with_mode(&input, true);
        let loss = (prediction - target).pow_tensor_scalar(2.0).mean(Kind::Float);
        loss.backward();
        layer.sgd_step(learning_rate);

        if step % 10 == 0 || step == steps - 1 {
            println!("step={step:03}, loss={:.6}", loss.double_value(&[]));
        }
    }
}

/// 自动探测并运行真实数据实验（MNIST / Fashion-MNIST / CIFAR-10）。
pub fn run_real_dataset_experiments_if_available(
    datasets_root: &str,
    device: Device,
    config: &TrainingConfig,
) -> Result<()> {
    let root = Path::new(datasets_root);
    if !root.exists() {
        println!("datasets root not found: {datasets_root}, skip real-dataset experiments");
        return Ok(());
    }

    let mnist_dir = root.join("mnist");
    if mnist_dir.exists() {
        let dataset = mnist::load_dir(&mnist_dir)?;
        run_dataset_experiment("MNIST", &dataset, device, config);
    } else {
        println!("MNIST directory not found: {}", mnist_dir.display());
    }

    // Fashion-MNIST 与 MNIST 文件格式一致，可复用 mnist::load_dir。
    let fashion_mnist_dir = root.join("fashion-mnist");
    if fashion_mnist_dir.exists() {
        let dataset = mnist::load_dir(&fashion_mnist_dir)?;
        run_dataset_experiment("Fashion-MNIST", &dataset, device, config);
    } else {
        println!("Fashion-MNIST directory not found: {}", fashion_mnist_dir.display());
    }

    let cifar10_dir = root.join("cifar-10-batches-bin");
    if cifar10_dir.exists() {
        let dataset = cifar::load_dir(&cifar10_dir)?;
        run_dataset_experiment("CIFAR-10", &dataset, device, config);
    } else {
        println!("CIFAR-10 directory not found: {}", cifar10_dir.display());
    }

    Ok(())
}

/// 自动探测并运行卷积骨架实验（Conv + LTBs Head）。
pub fn run_real_dataset_convnet_experiments_if_available(
    datasets_root: &str,
    device: Device,
    config: &TrainingConfig,
) -> Result<()> {
    let root = Path::new(datasets_root);
    if !root.exists() {
        println!("datasets root not found: {datasets_root}, skip convnet experiments");
        return Ok(());
    }

    let mnist_dir = root.join("mnist");
    if mnist_dir.exists() {
        let dataset = mnist::load_dir(&mnist_dir)?;
        run_dataset_convnet_experiment("MNIST-Conv", &dataset, device, config, 1, 28, 28);
    } else {
        println!("MNIST directory not found: {}", mnist_dir.display());
    }

    let fashion_mnist_dir = root.join("fashion-mnist");
    if fashion_mnist_dir.exists() {
        let dataset = mnist::load_dir(&fashion_mnist_dir)?;
        run_dataset_convnet_experiment(
            "FashionMNIST-Conv",
            &dataset,
            device,
            config,
            1,
            28,
            28,
        );
    } else {
        println!("Fashion-MNIST directory not found: {}", fashion_mnist_dir.display());
    }

    let cifar10_dir = root.join("cifar-10-batches-bin");
    if cifar10_dir.exists() {
        let dataset = cifar::load_dir(&cifar10_dir)?;
        run_dataset_convnet_experiment("CIFAR10-Conv", &dataset, device, config, 3, 32, 32);
    } else {
        println!("CIFAR-10 directory not found: {}", cifar10_dir.display());
    }

    Ok(())
}

/// 在给定数据集上运行最小分类训练与评估流程。
fn run_dataset_experiment(name: &str, dataset: &Dataset, device: Device, config: &TrainingConfig) {
    let flattened_train = flatten_features(&dataset.train_images);
    let input_dim = flattened_train.size()[1];
    let output_dim = dataset.labels;

    let mut layer = LtbSplineLayer::new(input_dim, output_dim, input_dim, 3, device);

    println!(
        "[{name}] start training: train_samples={}, test_samples={}, input_dim={}, labels={}",
        dataset.train_images.size()[0],
        dataset.test_images.size()[0],
        input_dim,
        output_dim
    );

    for epoch in 0..config.epochs {
        let train_loss = train_one_epoch(&mut layer, dataset, device, config);
        let test_accuracy = evaluate_accuracy(&mut layer, dataset, device, config);
        println!(
            "[{name}] epoch={epoch:02}, train_loss={train_loss:.6}, test_acc={:.2}%",
            test_accuracy * 100.0
        );
    }
}

/// 在给定数据集上运行卷积骨架实验。
fn run_dataset_convnet_experiment(
    name: &str,
    dataset: &Dataset,
    device: Device,
    config: &TrainingConfig,
    input_channels: i64,
    input_height: i64,
    input_width: i64,
) {
    let mut model = KanConvNet::new(input_channels, input_height, input_width, dataset.labels, device);
    println!(
        "[{name}] start training: train_samples={}, test_samples={}, channels={}, h={}, w={}, labels={}",
        dataset.train_images.size()[0],
        dataset.test_images.size()[0],
        input_channels,
        input_height,
        input_width,
        dataset.labels
    );

    for epoch in 0..config.epochs {
        let train_loss = train_convnet_one_epoch(
            &mut model,
            dataset,
            device,
            config,
            input_channels,
            input_height,
            input_width,
        );
        let test_accuracy = evaluate_convnet_accuracy(
            &mut model,
            dataset,
            device,
            config,
            input_channels,
            input_height,
            input_width,
        );
        println!(
            "[{name}] epoch={epoch:02}, train_loss={train_loss:.6}, test_acc={:.2}%",
            test_accuracy * 100.0
        );
    }
}

/// 单个 epoch 的训练循环（交叉熵分类）。
fn train_one_epoch(
    layer: &mut LtbSplineLayer,
    dataset: &Dataset,
    device: Device,
    config: &TrainingConfig,
) -> f64 {
    let mut iterator = dataset.train_iter(config.batch_size);
    iterator.shuffle().to_device(device).return_smaller_last_batch();

    let mut loss_sum = 0.0_f64;
    let mut batch_count = 0_i64;

    for (features, labels) in iterator {
        let inputs = flatten_features(&features);
        let logits = layer.forward_with_mode(&inputs, config.use_factorized_linear);
        let loss = logits.cross_entropy_for_logits(&labels);
        loss.backward();
        layer.sgd_step(config.learning_rate);
        loss_sum += loss.double_value(&[]);
        batch_count += 1;
    }

    if batch_count == 0 {
        0.0
    } else {
        loss_sum / batch_count as f64
    }
}

/// 卷积骨架单个 epoch 的训练循环（交叉熵分类）。
fn train_convnet_one_epoch(
    model: &mut KanConvNet,
    dataset: &Dataset,
    device: Device,
    config: &TrainingConfig,
    input_channels: i64,
    input_height: i64,
    input_width: i64,
) -> f64 {
    let mut iterator = dataset.train_iter(config.batch_size);
    iterator.shuffle().to_device(device).return_smaller_last_batch();

    let mut loss_sum = 0.0_f64;
    let mut batch_count = 0_i64;

    for (features, labels) in iterator {
        let inputs = ensure_image_features(&features, input_channels, input_height, input_width);
        let logits = model.forward_with_adaptive_grid(&inputs, 0.1, config.use_factorized_linear);
        let loss = logits.cross_entropy_for_logits(&labels);
        loss.backward();
        model.sgd_step(config.learning_rate);
        loss_sum += loss.double_value(&[]);
        batch_count += 1;
    }

    if batch_count == 0 {
        0.0
    } else {
        loss_sum / batch_count as f64
    }
}

/// 评估分类准确率。
fn evaluate_accuracy(
    layer: &mut LtbSplineLayer,
    dataset: &Dataset,
    device: Device,
    config: &TrainingConfig,
) -> f64 {
    let mut iterator = dataset.test_iter(config.batch_size);
    iterator.to_device(device).return_smaller_last_batch();

    let mut correct = 0.0_f64;
    let mut total = 0.0_f64;

    for (features, labels) in iterator {
        let inputs = flatten_features(&features);
        let logits = layer.forward_with_mode(&inputs, config.use_factorized_linear);
        let predictions = logits.argmax(-1, false);
        let batch_correct = predictions
            .eq_tensor(&labels)
            .to_kind(Kind::Float)
            .sum(Kind::Float)
            .double_value(&[]);
        correct += batch_correct;
        total += labels.size()[0] as f64;
    }

    if total == 0.0 { 0.0 } else { correct / total }
}

/// 评估卷积骨架分类准确率。
fn evaluate_convnet_accuracy(
    model: &mut KanConvNet,
    dataset: &Dataset,
    device: Device,
    config: &TrainingConfig,
    input_channels: i64,
    input_height: i64,
    input_width: i64,
) -> f64 {
    let mut iterator = dataset.test_iter(config.batch_size);
    iterator.to_device(device).return_smaller_last_batch();

    let mut correct = 0.0_f64;
    let mut total = 0.0_f64;

    for (features, labels) in iterator {
        let inputs = ensure_image_features(&features, input_channels, input_height, input_width);
        let logits = model.forward(&inputs, config.use_factorized_linear);
        let predictions = logits.argmax(-1, false);
        let batch_correct = predictions
            .eq_tensor(&labels)
            .to_kind(Kind::Float)
            .sum(Kind::Float)
            .double_value(&[]);
        correct += batch_correct;
        total += labels.size()[0] as f64;
    }

    if total == 0.0 { 0.0 } else { correct / total }
}

/// 将图像或已展平特征统一为二维 `[batch, feature_dim]`。
fn flatten_features(features: &Tensor) -> Tensor {
    let size = features.size();
    if size.len() <= 2 {
        features.shallow_clone()
    } else {
        features.view([size[0], -1])
    }
}

/// 将输入转换为卷积骨架所需图像形状 `[batch, channels, height, width]`。
fn ensure_image_features(
    features: &Tensor,
    input_channels: i64,
    input_height: i64,
    input_width: i64,
) -> Tensor {
    let size = features.size();
    if size.len() == 4 {
        features.shallow_clone()
    } else {
        features.view([size[0], input_channels, input_height, input_width])
    }
}

/// 读取 i64 环境变量，失败则返回默认值。
fn read_env_or_default_i64(key: &str, default_value: i64) -> i64 {
    match env::var(key) {
        Ok(value) => value.parse::<i64>().unwrap_or(default_value),
        Err(_) => default_value,
    }
}

/// 读取 f64 环境变量，失败则返回默认值。
fn read_env_or_default_f64(key: &str, default_value: f64) -> f64 {
    match env::var(key) {
        Ok(value) => value.parse::<f64>().unwrap_or(default_value),
        Err(_) => default_value,
    }
}

/// 读取 bool 环境变量，支持 1/0、true/false（不区分大小写）。
fn read_env_or_default_bool(key: &str, default_value: bool) -> bool {
    match env::var(key) {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => default_value,
        },
        Err(_) => default_value,
    }
}
