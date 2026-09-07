//! Generic multi-variable differential correction (damped Newton / Gauss-Newton).
//!
//! Generalizes the multi-start Newton-targeting pattern proven in the Artemis
//! free-return targeting (`AstroProbs/Artemis/src/bin/target.rs`): central-difference
//! Jacobian, damped step, per-parameter step clamp, multi-start seeding to guard
//! against a bad initial guess. The residual function is supplied by the caller
//! via closure — this crate stays ephemeris-free; callers (e.g. the
//! `MissionPlanner` design stage) provide the orbital-mechanics evaluation.

use nalgebra::{DMatrix, DVector};

/// Damped Newton / Gauss-Newton corrector for `eval(params) -> residuals`.
///
/// Square systems (`residuals.len() == params.len()`) use a direct Newton
/// step. Non-square systems use the Gauss-Newton normal-equations step
/// `Δ = -(JᵗJ)⁻¹Jᵗr`, which reduces to the direct step in the square case.
pub struct DiffCorrectionSolver {
    /// Maximum Newton iterations per start.
    pub max_iter: usize,
    /// Central-difference stencil half-width, one per free parameter.
    pub fd_step: Vec<f64>,
    /// Convergence tolerance, one per residual component (all must be satisfied).
    pub tol: Vec<f64>,
    /// Fraction of the Newton step applied each iteration (0, 1].
    pub damping: f64,
    /// Maximum |step| per parameter per iteration — bounds divergence from a bad seed.
    pub max_step: Vec<f64>,
}

/// Outcome of a single Newton solve.
#[derive(Clone, Debug)]
pub struct DiffCorrectionResult {
    pub params: Vec<f64>,
    pub residuals: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

impl DiffCorrectionSolver {
    /// Run a single Newton solve from `seed`. `eval` returns the residual
    /// vector for a given parameter vector (same length as `tol`).
    pub fn solve<F>(&self, seed: &[f64], mut eval: F) -> DiffCorrectionResult
    where
        F: FnMut(&[f64]) -> Vec<f64>,
    {
        let n = seed.len();
        let mut params = seed.to_vec();
        let mut residuals = eval(&params);

        for iter in 0..self.max_iter {
            if Self::converged(&residuals, &self.tol) {
                return DiffCorrectionResult { params, residuals, iterations: iter, converged: true };
            }

            let m = residuals.len();
            let mut jac = DMatrix::<f64>::zeros(m, n);
            for j in 0..n {
                let h = self.fd_step[j];
                let mut p_plus = params.clone();
                let mut p_minus = params.clone();
                p_plus[j] += h;
                p_minus[j] -= h;
                let r_plus = eval(&p_plus);
                let r_minus = eval(&p_minus);
                for i in 0..m {
                    jac[(i, j)] = (r_plus[i] - r_minus[i]) / (2.0 * h);
                }
            }

            let r_vec = DVector::from_vec(residuals.clone());
            let Some(delta) = newton_step(&jac, &r_vec) else {
                break; // singular Jacobian — report non-convergence at current point
            };

            for j in 0..n {
                params[j] += (self.damping * delta[j]).clamp(-self.max_step[j], self.max_step[j]);
            }

            residuals = eval(&params);
        }

        let converged = Self::converged(&residuals, &self.tol);
        DiffCorrectionResult { params, residuals, iterations: self.max_iter, converged }
    }

    /// Run [`solve`](Self::solve) from each seed; return the converged result
    /// with the smallest residual norm, or `None` if no seed converged.
    pub fn solve_multi_start<F>(&self, seeds: &[Vec<f64>], mut eval: F) -> Option<DiffCorrectionResult>
    where
        F: FnMut(&[f64]) -> Vec<f64>,
    {
        seeds
            .iter()
            .map(|seed| self.solve(seed, &mut eval))
            .filter(|r| r.converged)
            .min_by(|a, b| residual_norm(&a.residuals).partial_cmp(&residual_norm(&b.residuals)).unwrap())
    }

    fn converged(residuals: &[f64], tol: &[f64]) -> bool {
        residuals.iter().zip(tol).all(|(r, t)| r.abs() < *t)
    }
}

/// `Δ = -J⁻¹r` (square) or `Δ = -(JᵗJ)⁻¹Jᵗr` (Gauss-Newton, rectangular).
fn newton_step(jac: &DMatrix<f64>, r: &DVector<f64>) -> Option<DVector<f64>> {
    if jac.nrows() == jac.ncols() {
        let inv = jac.clone().try_inverse()?;
        Some(-(inv * r))
    } else {
        let jt = jac.transpose();
        let jtj = &jt * jac;
        let inv = jtj.try_inverse()?;
        Some(-(inv * &jt * r))
    }
}

fn residual_norm(residuals: &[f64]) -> f64 {
    residuals.iter().map(|r| r * r).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic 2x2 system: circle ∩ line, exact root at (√2, √2).
    /// Exercises the square-system Newton path with no orbital mechanics involved.
    #[test]
    fn solves_circle_line_intersection() {
        let solver = DiffCorrectionSolver {
            max_iter: 50,
            fd_step: vec![1e-4, 1e-4],
            tol: vec![1e-8, 1e-8],
            damping: 1.0,
            max_step: vec![1.0, 1.0],
        };

        let result = solver.solve(&[1.0, 1.0], |p| {
            let (x, y) = (p[0], p[1]);
            vec![x * x + y * y - 4.0, x - y]
        });

        assert!(result.converged, "expected convergence, got {result:?}");
        let expected = 2.0_f64.sqrt();
        assert!((result.params[0] - expected).abs() < 1e-6);
        assert!((result.params[1] - expected).abs() < 1e-6);
    }

    /// A seed on the wrong side of a singular point should still be rescued
    /// by multi-start: one of several seeds must land in a well-conditioned basin.
    #[test]
    fn multi_start_recovers_from_bad_seed() {
        let solver = DiffCorrectionSolver {
            max_iter: 50,
            fd_step: vec![1e-4, 1e-4],
            tol: vec![1e-8, 1e-8],
            damping: 1.0,
            max_step: vec![1.0, 1.0],
        };

        let eval = |p: &[f64]| {
            let (x, y) = (p[0], p[1]);
            vec![x * x + y * y - 4.0, x - y]
        };

        // Seed at the origin gives a singular Jacobian for this system; a second
        // seed away from the singularity should converge.
        let seeds = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let result = solver.solve_multi_start(&seeds, eval).expect("at least one seed should converge");
        assert!(result.converged);
    }
}
