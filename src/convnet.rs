use tch::{no_grad, Device, Kind, Tensor};

use crate::spline::layer::LtbSplineLayer;

/// 论文卷积化路径的最小骨架：Conv 特征提取 + LTBs 分类头。
pub struct KanConvNet {
    /// 第一层卷积核，形状 `[16, in_channels, 3, 3]`。
    pub conv1_weight: Tensor,
    /// 第一层卷积偏置，形状 `[16]`。
    pub conv1_bias: Tensor,
    /// 第二层卷积核，形状 `[32, 16, 3, 3]`。
    pub conv2_weight: Tensor,
    /// 第二层卷积偏置，形状 `[32]`。
    pub conv2_bias: Tensor,
    /// LTBs 分类头。
    pub classifier: LtbSplineLayer,
}

impl KanConvNet {
    /// 构建卷积骨架网络。
    pub fn new(
        input_channels: i64,
        input_height: i64,
        input_width: i64,
        num_classes: i64,
        device: Device,
    ) -> Self {
        let conv1_weight =
            (Tensor::randn([16, input_channels, 3, 3], (Kind::Float, device)) * 0.01)
                .set_requires_grad(true);
        let conv1_bias = Tensor::zeros([16], (Kind::Float, device)).set_requires_grad(true);
        let conv2_weight =
            (Tensor::randn([32, 16, 3, 3], (Kind::Float, device)) * 0.01).set_requires_grad(true);
        let conv2_bias = Tensor::zeros([32], (Kind::Float, device)).set_requires_grad(true);

        let flattened_dim =
            infer_flattened_dim(input_channels, input_height, input_width, &conv1_weight, &conv1_bias, &conv2_weight, &conv2_bias, device);
        let classifier = LtbSplineLayer::new(flattened_dim, num_classes, flattened_dim, 3, device);

        Self {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            classifier,
        }
    }

    /// 执行 `Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> LTBs` 前向。
    pub fn forward(&self, input: &Tensor, use_factorized_linear: bool) -> Tensor {
        let conv_features = self.extract_conv_features(input);
        self.classifier
            .forward_with_mode(&conv_features, use_factorized_linear)
    }

    /// 使用当前 batch 做一次网格更新再前向。
    pub fn forward_with_adaptive_grid(
        &mut self,
        input: &Tensor,
        margin: f64,
        use_factorized_linear: bool,
    ) -> Tensor {
        let conv_features = self.extract_conv_features(input);
        self.classifier
            .forward_with_adaptive_grid(&conv_features, margin, use_factorized_linear)
    }

    /// 执行一次简单 SGD 参数更新。
    pub fn sgd_step(&mut self, learning_rate: f64) {
        no_grad(|| {
            update_parameter_with_sgd(&mut self.conv1_weight, learning_rate);
            update_parameter_with_sgd(&mut self.conv1_bias, learning_rate);
            update_parameter_with_sgd(&mut self.conv2_weight, learning_rate);
            update_parameter_with_sgd(&mut self.conv2_bias, learning_rate);
        });
        self.classifier.sgd_step(learning_rate);
    }

    /// 计算卷积特征并展平到 `[batch, feature_dim]`。
    fn extract_conv_features(&self, input: &Tensor) -> Tensor {
        let x = input
            .conv2d(
                &self.conv1_weight,
                Some(&self.conv1_bias),
                [1, 1],
                [1, 1],
                [1, 1],
                1,
            )
            .relu()
            .max_pool2d_default(2)
            .conv2d(
                &self.conv2_weight,
                Some(&self.conv2_bias),
                [1, 1],
                [1, 1],
                [1, 1],
                1,
            )
            .relu()
            .max_pool2d_default(2)
            // 压缩卷积特征维度，避免当前 LTBs 头在超高输入维上计算过重。
            .adaptive_avg_pool2d([2, 2]);

        x.view([x.size()[0], -1])
    }
}

/// 推断卷积特征展平后的维度。
fn infer_flattened_dim(
    input_channels: i64,
    input_height: i64,
    input_width: i64,
    conv1_weight: &Tensor,
    conv1_bias: &Tensor,
    conv2_weight: &Tensor,
    conv2_bias: &Tensor,
    device: Device,
) -> i64 {
    let dummy = Tensor::zeros([1, input_channels, input_height, input_width], (Kind::Float, device));
    let output = dummy
        .conv2d(conv1_weight, Some(conv1_bias), [1, 1], [1, 1], [1, 1], 1)
        .relu()
        .max_pool2d_default(2)
        .conv2d(conv2_weight, Some(conv2_bias), [1, 1], [1, 1], [1, 1], 1)
        .relu()
        .max_pool2d_default(2)
        .adaptive_avg_pool2d([2, 2]);
    output.size()[1] * output.size()[2] * output.size()[3]
}

/// 对单个参数执行 SGD 更新。
fn update_parameter_with_sgd(parameter: &mut Tensor, learning_rate: f64) {
    let gradient = parameter.grad();
    if gradient.defined() {
        let _ = parameter.f_sub_(&(gradient * learning_rate));
        parameter.zero_grad();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convnet_forward_shape() {
        let device = Device::Cpu;
        let net = KanConvNet::new(1, 28, 28, 10, device);
        let input = Tensor::randn([4, 1, 28, 28], (Kind::Float, device));
        let output = net.forward(&input, true);
        assert_eq!(output.size(), vec![4, 10]);
    }
}
