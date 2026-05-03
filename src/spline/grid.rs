use tch::{Device, Kind, Tensor};

const MIN_SPAN: f64 = 1.0e-3;

/// 基于给定区间构建等距 knot grid，步数为 `n + 2m + 1`（对应论文 Algorithm 2）。
pub fn build_uniform_knot_grid(
    low: f64,
    high: f64,
    node_count: i64,
    spline_order: i64,
    device: Device,
) -> Tensor {
    assert!(node_count > 0, "node_count must be positive");
    assert!(spline_order >= 0, "spline_order must be non-negative");

    let steps = node_count + 2 * spline_order + 1;
    Tensor::linspace(low, high, steps, (Kind::Float, device))
}

/// 根据输入样本自适应更新 knot grid（论文 Algorithm 2: gridUpdate）。
pub fn update_knot_grid_adaptive(
    samples: &Tensor,
    margin: f64,
    node_count: i64,
    spline_order: i64,
    device: Device,
) -> Tensor {
    assert!(samples.numel() > 0, "samples must be non-empty");

    let sample_min = samples.min().double_value(&[]);
    let sample_max = samples.max().double_value(&[]);
    let span = (sample_max - sample_min).max(MIN_SPAN);

    // 论文中的 low/high 扩展策略：low = min - delta*span, high = max + delta*span
    let low = sample_min - margin * span;
    let high = sample_max + margin * span;
    build_uniform_knot_grid(low, high, node_count, spline_order, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_uniform_knot_grid_size() {
        let grid = build_uniform_knot_grid(-1.0, 1.0, 8, 3, Device::Cpu);
        assert_eq!(grid.size(), vec![15]);
    }

    #[test]
    fn test_adaptive_grid_monotonic() {
        let samples = Tensor::randn([16, 8], (Kind::Float, Device::Cpu));
        let grid = update_knot_grid_adaptive(&samples, 0.1, 8, 3, Device::Cpu);
        let left = grid.narrow(0, 0, grid.size()[0] - 1);
        let right = grid.narrow(0, 1, grid.size()[0] - 1);
        let is_monotonic = right.ge_tensor(&left).all().int64_value(&[]) != 0;
        assert!(is_monotonic, "adaptive grid must be non-decreasing");
    }
}
