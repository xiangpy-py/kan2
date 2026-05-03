use tch::{Device, IndexOp, Kind, Tensor};

const EPSILON: f64 = 1.0e-8;

/// 根据论文 Algorithm 1 计算 LTBs 系数张量 C。
/// 维度约定：`C[span_index, basis_index, degree_index]`。
pub fn compute_ltb_coeffs(n: i64, m: i64, device: Device) -> Tensor {
    assert!(n > 0, "n must be positive");
    assert!(m >= 0, "m must be non-negative");

    let coeffs = Tensor::zeros([n, n + m, m + 1], (Kind::Float, device));
    let knots = build_uniform_knots(n, m);

    initialize_boundary_block(&coeffs, &knots, n, m);
    apply_backward_recursion_block(&coeffs, &knots, n, m);
    coeffs
}

/// Block 1：初始化上下对角边界系数。
fn initialize_boundary_block(coeffs: &Tensor, knots: &[f64], n: i64, m: i64) {
    if m == 0 {
        for j in 0..n {
            let _ = coeffs.i((j, j, 0)).fill_(1.0);
        }
        return;
    }

    let knots_len = knots.len() as i64;
    for j in 0..n {
        if j + m + 1 >= knots_len {
            continue;
        }

        let delta = (knots[(j + m + 1) as usize] - knots[(j + m) as usize]).max(EPSILON);
        let num = delta.powf((m - 1) as f64);

        let mut val_m = num;
        for k in 2..=m {
            let right_index = j + k + m;
            if right_index < knots_len {
                let denominator = (knots[right_index as usize] - knots[(j + m) as usize]).max(EPSILON);
                val_m /= denominator;
            }
        }
        let _ = coeffs.i((j, j + m, m)).fill_(val_m);

        let mut val_0 = num;
        for k in 2..=m {
            let knot_index = j + 1 - k + m;
            if knot_index >= 0 && knot_index < knots_len {
                let denominator =
                    (knots[(j + m + 1) as usize] - knots[knot_index as usize]).max(EPSILON);
                val_0 /= denominator;
            }
        }
        let _ = coeffs.i((j, j, 0)).fill_(val_0);
    }
}

/// Block 2：按论文式 (17) 反向递推填充其余系数。
fn apply_backward_recursion_block(coeffs: &Tensor, knots: &[f64], n: i64, m: i64) {
    if n < 2 || m == 0 {
        return;
    }

    let knots_len = knots.len() as i64;
    let start = n.saturating_sub(m + 1);
    let end = n - 2;

    if start > end {
        return;
    }

    for i in (start..=end).rev() {
        if i + m >= knots_len || n + m >= knots_len {
            continue;
        }

        let numerator_1 = knots[(n - 1 + m) as usize] - knots[(i + m) as usize];
        let denominator_1 = (knots[(n + m) as usize] - knots[(i + m) as usize]) + EPSILON;
        let coefficient_1 = numerator_1 / denominator_1;

        let numerator_2 = knots[(n + m) as usize] - knots[(n - 1 + m) as usize];
        let denominator_2 = (knots[(n + m) as usize] - knots[(i + 1 + m) as usize]) + EPSILON;
        let coefficient_2 = numerator_2 / denominator_2;

        for k in (0..m).rev() {
            // C[n-1, i+m, k] = c1 * C[n-1, i+m, k+1] + c2 * C[n-1, i+1+m, k+1]
            let left = coeffs.i((n - 1, i + m, k + 1)) * coefficient_1;
            let right = coeffs.i((n - 1, i + 1 + m, k + 1)) * coefficient_2;
            let updated = left + right;
            coeffs.i((n - 1, i + m, k)).copy_(&updated);
        }
    }
}

/// 构建默认等距 knot 向量；后续可替换为自适应 grid 更新版本。
fn build_uniform_knots(n: i64, m: i64) -> Vec<f64> {
    let knot_count = (n + m + 1) as usize;
    let mut knots = Vec::with_capacity(knot_count);

    for index in 0..knot_count {
        knots.push(index as f64);
    }

    knots
}
