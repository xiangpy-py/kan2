use std::env;

use tch::{Device, Kind, Tensor};

use kan2::spline::layer::LtbSplineLayer;
use kan2::training::{
    run_real_dataset_convnet_experiments_if_available, run_real_dataset_experiments_if_available,
    run_synthetic_training_demo, TrainingConfig,
};

fn main() {
    let device = Device::Cpu;
    let batch_size = 4;
    let input_dim = 8;
    let output_dim = 4;
    let spline_order = 3;
    let node_count = input_dim;

    // 构建最小可运行层，并执行一次前向用于连通性验证。
    let layer = LtbSplineLayer::new(input_dim, output_dim, node_count, spline_order, device);
    let input = Tensor::randn([batch_size, input_dim], (Kind::Float, device));
    let output = layer.forward(&input);

    println!("input shape:  {:?}", input.size());
    println!("output shape: {:?}", output.size());

    // 训练链路连通性演示（合成数据）。
    run_synthetic_training_demo(5, 8, input_dim, output_dim, device);

    // 若存在真实数据目录，则自动执行最小实验脚手架。
    let training_config = TrainingConfig::from_env();
    let datasets_root = env::var("KAN_DATASETS_ROOT").unwrap_or_else(|_| "assets/datasets".to_string());
    if let Err(error) =
        run_real_dataset_experiments_if_available(&datasets_root, device, &training_config)
    {
        eprintln!("real-dataset experiments failed: {error}");
    }

    // 可选：卷积骨架实验（默认关闭，防止每次运行过慢）。
    let run_convnet = parse_bool_env(
        &env::var("KAN_RUN_CONVNET").unwrap_or_else(|_| "false".to_string()),
        false,
    );
    if run_convnet {
        if let Err(error) =
            run_real_dataset_convnet_experiments_if_available(&datasets_root, device, &training_config)
        {
            eprintln!("convnet experiments failed: {error}");
        }
    }
}

/// 解析 bool 文本，支持 1/0、true/false（不区分大小写）。
fn parse_bool_env(value: &str, default_value: bool) -> bool {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" => true,
        "0" | "false" => false,
        _ => default_value,
    }
}
