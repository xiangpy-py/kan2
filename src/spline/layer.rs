use tch::{no_grad, Device, Kind, Tensor};

use crate::spline::basis::bernstein_basis;
use crate::spline::coeffs::compute_ltb_coeffs;
use crate::spline::grid::{build_uniform_knot_grid, update_knot_grid_adaptive};

/// LTBs-KAN 的最小线性层：`y = linear(x) + spline(x)`。
pub struct LtbSplineLayer {
    /// 输入特征维度。
    pub input_dim: i64,
    /// 输出特征维度。
    pub output_dim: i64,
    /// 样条阶数（论文中常用 `m=3`）。
    pub spline_order: i64,
    /// 网格节点数 `n`。
    pub node_count: i64,
    /// 线性分支权重，形状 `[input_dim, output_dim]`。
    pub linear_weight: Tensor,
    /// 样条分支权重，形状 `[output_dim, input_dim, D]`，其中 `D = n + m`。
    pub spline_weight: Tensor,
    /// LTBs 系数张量，来自论文 Algorithm 1。
    pub ltb_coeffs: Tensor,
    /// 当前 knot grid（用于后续接入自适应更新）。
    pub knot_grid: Tensor,
    /// 参数缩减中的 p（论文 3.3）。
    pub factorization_p: i64,
    /// 参数缩减中的 s（论文 3.3）。
    pub factorization_s: i64,
    /// 参数缩减可学习系数 a_{jk}，形状 `[p, s]`。
    pub factorization_coefficients: Tensor,
    /// 固定基矩阵 M_{jk}，形状 `[p, s, input_dim, output_dim]`。
    pub factorization_basis_matrices: Tensor,
    /// LayerNorm 数值稳定参数。
    pub layer_norm_eps: f64,
}

impl LtbSplineLayer {
    /// 构造最小可运行的 LTBs spline layer。
    pub fn new(
        input_dim: i64,
        output_dim: i64,
        node_count: i64,
        spline_order: i64,
        device: Device,
    ) -> Self {
        // 论文中 p,s 取小常数（例如 p=2/3, s=4/5），此处给默认可运行配置。
        Self::new_with_factorization_params(input_dim, output_dim, node_count, spline_order, 2, 4, device)
    }

    /// 构造可配置 `p,s` 的 LTBs spline layer。
    pub fn new_with_factorization_params(
        input_dim: i64,
        output_dim: i64,
        node_count: i64,
        spline_order: i64,
        factorization_p: i64,
        factorization_s: i64,
        device: Device,
    ) -> Self {
        assert!(input_dim > 0, "input_dim must be positive");
        assert!(output_dim > 0, "output_dim must be positive");
        assert!(node_count > 0, "node_count must be positive");
        assert!(spline_order >= 0, "spline_order must be non-negative");
        assert!(factorization_p > 0, "factorization_p must be positive");
        assert!(factorization_s > 0, "factorization_s must be positive");
        assert!(
            node_count == input_dim,
            "current minimal implementation requires node_count == input_dim"
        );

        let linear_weight =
            (Tensor::randn([input_dim, output_dim], (Kind::Float, device)) * 0.01).set_requires_grad(true);
        let basis_count = node_count + spline_order;
        let spline_weight = (Tensor::randn([output_dim, input_dim, basis_count], (Kind::Float, device))
            * 0.01)
            .set_requires_grad(true);

        let ltb_coeffs = compute_ltb_coeffs(node_count, spline_order, device);
        let knot_grid = build_uniform_knot_grid(-1.0, 1.0, node_count, spline_order, device);
        let factorization_coefficients =
            (Tensor::randn([factorization_p, factorization_s], (Kind::Float, device)) * 0.01)
                .set_requires_grad(true);
        let factorization_basis_matrices = build_factorization_basis_matrices(
            input_dim,
            output_dim,
            factorization_p,
            factorization_s,
            device,
        );

        Self {
            input_dim,
            output_dim,
            spline_order,
            node_count,
            linear_weight,
            spline_weight,
            ltb_coeffs,
            knot_grid,
            factorization_p,
            factorization_s,
            factorization_coefficients,
            factorization_basis_matrices,
            layer_norm_eps: 1.0e-5,
        }
    }

    /// 执行总前向：线性分支 + spline 分支。
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_with_mode(input, false)
    }

    /// 按模式执行前向：普通线性项或参数缩减线性项。
    pub fn forward_with_mode(&self, input: &Tensor, use_factorized_linear: bool) -> Tensor {
        let linear_output = if use_factorized_linear {
            self.compute_factorized_linear_output(input)
        } else {
            // 使用 SiLU 基激活以贴合论文 Eq.(23) 后的 baseOutput 定义。
            input.silu().matmul(&self.linear_weight)
        };
        let spline_output = self.forward_spline_branch(input);
        let merged_output = linear_output + spline_output;
        // 对齐论文 Eq.(26)：最终输出执行 LayerNorm。
        merged_output.layer_norm(
            [self.output_dim],
            Option::<&Tensor>::None,
            Option::<&Tensor>::None,
            self.layer_norm_eps,
            false,
        )
    }

    /// 按当前 batch 执行一次 gridUpdate，并刷新系数张量后再前向。
    pub fn forward_with_adaptive_grid(
        &mut self,
        input: &Tensor,
        margin: f64,
        use_factorized_linear: bool,
    ) -> Tensor {
        let device = input.device();
        self.knot_grid = update_knot_grid_adaptive(
            input,
            margin,
            self.node_count,
            self.spline_order,
            device,
        );
        // 当前版本先按论文 Algorithm 1 重算系数；后续可进一步接入“由 knot_grid 驱动”的系数重算。
        self.ltb_coeffs = compute_ltb_coeffs(self.node_count, self.spline_order, device);
        self.forward_with_mode(input, use_factorized_linear)
    }

    /// 执行 spline 分支前向，便于与论文公式逐步对齐。
    pub fn forward_spline_branch(&self, input: &Tensor) -> Tensor {
        let basis_values = self.compute_bspline_basis_values(input);
        // 对齐论文 Eq.(24)/(25)：按 (input_dim, D) 双重收缩得到输出。
        Tensor::einsum("bid,oid->bo", &[&basis_values, &self.spline_weight], None::<&[i64]>)
    }

    /// 按论文 Eq.(22) 计算 factorized baseOutput：`sum_{j,k} a_jk * (phi(X) M_jk)`。
    fn compute_factorized_linear_output(&self, input: &Tensor) -> Tensor {
        let activated_input = input.silu();
        // 先组装有效权重 W = sum_{j,k} a_jk M_jk，再进行一次矩阵乘。
        let effective_weight = (self.factorization_coefficients.unsqueeze(-1).unsqueeze(-1)
            * &self.factorization_basis_matrices)
            .sum_dim_intlist(&[0_i64, 1_i64][..], false, Kind::Float);
        activated_input.matmul(&effective_weight)
    }

    /// 计算每个样本、每个输入维度上的 B-spline 基值 `B_j(x_i)`，输出形状 `[B, d_in, D]`。
    fn compute_bspline_basis_values(&self, input: &Tensor) -> Tensor {
        let batch_size = input.size()[0];
        let basis_count = self.node_count + self.spline_order;
        let mut all_values = Vec::with_capacity((batch_size * self.input_dim) as usize);

        for batch_index in 0..batch_size {
            for input_index in 0..self.input_dim {
                let x = input.double_value(&[batch_index, input_index]);
                // 当前实现使用 [-1,1] 到 [0,1] 的线性映射，便于与默认 grid 对齐。
                let normalized_x = ((x + 1.0) * 0.5).clamp(0.0, 1.0 - 1.0e-8);
                let span_index = (normalized_x * self.node_count as f64).floor() as i64;
                let bounded_span_index = span_index.clamp(0, self.node_count - 1);

                // 每个 span 上使用 Bernstein 多项式计算局部基值。
                let local_parameter = normalized_x * self.node_count as f64 - bounded_span_index as f64;
                let bernstein_values = evaluate_bernstein_scalar(local_parameter, self.spline_order);

                let mut basis_values = Vec::with_capacity(basis_count as usize);
                for basis_index in 0..basis_count {
                    let mut value = 0.0_f64;
                    for degree_index in 0..=self.spline_order {
                        let coefficient = self
                            .ltb_coeffs
                            .double_value(&[bounded_span_index, basis_index, degree_index]);
                        value += coefficient * bernstein_values[degree_index as usize];
                    }
                    basis_values.push(value as f32);
                }

                all_values.push(Tensor::from_slice(&basis_values).unsqueeze(0));
            }
        }

        Tensor::cat(&all_values, 0).reshape([batch_size, self.input_dim, basis_count])
    }

    /// 执行一次简单 SGD 参数更新（仅更新可学习参数）。
    pub fn sgd_step(&mut self, learning_rate: f64) {
        no_grad(|| {
            update_parameter_with_sgd(&mut self.linear_weight, learning_rate);
            update_parameter_with_sgd(&mut self.spline_weight, learning_rate);
            update_parameter_with_sgd(&mut self.factorization_coefficients, learning_rate);
        });
    }
}

/// 构造固定基矩阵 `M_{jk}`，作为参数缩减的子空间基。
fn build_factorization_basis_matrices(
    input_dim: i64,
    output_dim: i64,
    factorization_p: i64,
    factorization_s: i64,
    device: Device,
) -> Tensor {
    let basis = Tensor::randn(
        [factorization_p, factorization_s, input_dim, output_dim],
        (Kind::Float, device),
    );
    basis.set_requires_grad(false)
}

/// 计算标量输入的 Bernstein 基值，用于 span 内局部多项式求值。
fn evaluate_bernstein_scalar(local_parameter: f64, degree: i64) -> Vec<f64> {
    let parameter_tensor = Tensor::from_slice(&[local_parameter as f32]).to_kind(Kind::Float);
    let basis_tensor = bernstein_basis(&parameter_tensor, degree).squeeze_dim(0);
    let mut values = Vec::with_capacity((degree + 1) as usize);
    for index in 0..=degree {
        values.push(basis_tensor.double_value(&[index]));
    }
    values
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
    fn test_layer_forward_shape_default_mode() {
        let device = Device::Cpu;
        let layer = LtbSplineLayer::new(8, 4, 8, 3, device);
        let input = Tensor::randn([2, 8], (Kind::Float, device));
        let output = layer.forward(&input);
        assert_eq!(output.size(), vec![2, 4]);
    }

    #[test]
    fn test_layer_forward_shape_factorized_mode() {
        let device = Device::Cpu;
        let layer = LtbSplineLayer::new_with_factorization_params(8, 4, 8, 3, 2, 4, device);
        let input = Tensor::randn([3, 8], (Kind::Float, device));
        let output = layer.forward_with_mode(&input, true);
        assert_eq!(output.size(), vec![3, 4]);
    }

    #[test]
    fn test_layer_forward_with_adaptive_grid_shape() {
        let device = Device::Cpu;
        let mut layer = LtbSplineLayer::new(8, 4, 8, 3, device);
        let input = Tensor::randn([2, 8], (Kind::Float, device));
        let output = layer.forward_with_adaptive_grid(&input, 0.1, true);
        assert_eq!(output.size(), vec![2, 4]);
    }
}
