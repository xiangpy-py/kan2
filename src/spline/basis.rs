use tch::Tensor;

/// 计算 Bernstein 基函数值，返回末维为 `degree + 1` 的张量。
pub fn bernstein_basis(x: &Tensor, degree: i64) -> Tensor {
    assert!(degree >= 0, "degree must be non-negative");

    // 使用显式公式避免 x/(1-x) 形式在端点处产生数值不稳定。
    let coefficients = binomial_row(degree);
    let one_minus_x = Tensor::ones_like(x) - x;
    let mut basis_values = Vec::with_capacity((degree + 1) as usize);

    for k in 0..=degree {
        let coefficient = coefficients[k as usize];
        let x_power = x.pow_tensor_scalar(k as f64);
        let one_minus_x_power = one_minus_x.pow_tensor_scalar((degree - k) as f64);
        let basis_k = x_power * one_minus_x_power * coefficient;
        basis_values.push(basis_k.unsqueeze(-1));
    }

    Tensor::cat(&basis_values, -1)
}

/// 计算二项式系数行：`C(n,0)..C(n,n)`。
fn binomial_row(n: i64) -> Vec<f64> {
    let mut coefficients = Vec::with_capacity((n + 1) as usize);
    let mut current = 1.0;

    for k in 0..=n {
        coefficients.push(current);
        current = current * (n - k) as f64 / (k + 1) as f64;
    }

    coefficients
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_bernstein_basis_shape() {
        let input = Tensor::rand([5, 3], (Kind::Float, Device::Cpu));
        let basis = bernstein_basis(&input, 3);
        assert_eq!(basis.size(), vec![5, 3, 4]);
    }

    #[test]
    fn test_bernstein_partition_of_unity() {
        let input = Tensor::rand([4, 2], (Kind::Float, Device::Cpu));
        let basis = bernstein_basis(&input, 3);
        let sums = basis.sum_dim_intlist(&[2_i64][..], false, Kind::Float);
        let ones = Tensor::ones([4, 2], (Kind::Float, Device::Cpu));
        let max_error = (sums - ones).abs().max().double_value(&[]);
        assert!(max_error < 1.0e-5, "max error too large: {max_error}");
    }
}
