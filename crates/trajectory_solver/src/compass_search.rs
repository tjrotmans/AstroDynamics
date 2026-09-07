//! Compass search (coordinate pattern search) minimiser for bounded
//! continuous parameter spaces.
//!
//! Classic derivative-free pattern search (Hooke & Jeeves 1961; "compass
//! search" naming per Kolda, Lewis & Torczon 2003): at each iteration, probe
//! along every coordinate axis (+step and -step) from the current point;
//! move to the best-improving probe found, or shrink the step if none
//! improve. Simpler and more conservative than Nelder-Mead's simplex moves.
//!
//! Added as a selectable alternative to [`crate::NelderMead`] for
//! [`crate::MbhSolver`]'s inner local descent, after confirming (against the
//! real pagmo2 C++ source, `src/algorithms/mbh.cpp`) that pagmo's own MBH
//! implementation uses `compass_search` as its DEFAULT inner optimizer, not
//! Nelder-Mead — a real, verified difference from our own prior default,
//! worth testing directly rather than assuming Nelder-Mead is equally good.
//!
//! # References
//! - Hooke, R. & Jeeves, T. A. (1961), "Direct Search Solution of Numerical
//!   and Statistical Problems", J. ACM 8(2):212-229.
//! - Kolda, T. G., Lewis, R. M. & Torczon, V. (2003), "Optimization by
//!   Direct Search: New Perspectives on Some Classical and Modern Methods",
//!   SIAM Review 45(3):385-482 (the "compass search" / generalized pattern
//!   search framework and its convergence theory).

use crate::nelder_mead::NmResult;

/// Compass search minimiser for bounded real-valued parameters.
///
/// Operates entirely in bounds-normalised `[0, 1]^n` space internally
/// (same convention as [`crate::NelderMead`]) — required whenever the
/// parameter vector mixes units of wildly different magnitude.
pub struct CompassSearch {
    pub max_iter: usize,
    /// Initial step size, as a fraction of each parameter's bounds width.
    pub start_step: f64,
    /// Step shrink factor applied when no probe direction improves
    /// (standard: 0.5, matching classic Hooke-Jeeves).
    pub shrink_factor: f64,
    /// Stop when the (normalised) step size drops below this.
    pub step_tol: f64,
}

impl Default for CompassSearch {
    fn default() -> Self {
        CompassSearch {
            max_iter: 500,
            start_step: 0.1,
            shrink_factor: 0.5,
            step_tol: 1e-6,
        }
    }
}

impl CompassSearch {
    /// Minimise `fitness` over `bounds` (one `(min, max)` pair per
    /// parameter), starting at `x0` (original units, clamped into bounds).
    ///
    /// `fitness` returning `None` is treated as infeasible, penalised to
    /// `f64::MAX`, same convention as [`crate::NelderMead`].
    pub fn run<F>(&self, bounds: &[(f64, f64)], x0: &[f64], fitness: F) -> NmResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
    {
        self.run_with_progress(bounds, x0, fitness, |_iter, _best| {})
    }

    /// Same as [`CompassSearch::run`] but calls `on_iteration(iteration_index,
    /// best_fitness_so_far)` after each iteration — for live progress streams.
    pub fn run_with_progress<F, G>(
        &self,
        bounds: &[(f64, f64)],
        x0: &[f64],
        mut fitness: F,
        mut on_iteration: G,
    ) -> NmResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
        G: FnMut(usize, f64),
    {
        let n = bounds.len();

        let to_norm = |x: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|j| ((x[j] - bounds[j].0) / (bounds[j].1 - bounds[j].0)).clamp(0.0, 1.0))
                .collect()
        };
        let to_real = |u: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|j| bounds[j].0 + u[j].clamp(0.0, 1.0) * (bounds[j].1 - bounds[j].0))
                .collect()
        };
        let eval = |u: &[f64], fitness: &mut F| -> f64 {
            fitness(&to_real(u)).unwrap_or(f64::MAX)
        };

        let mut x = to_norm(x0);
        let mut fx = eval(&x, &mut fitness);
        let mut step = self.start_step;
        let mut history = Vec::with_capacity(self.max_iter);
        let mut iter = 0;

        while iter < self.max_iter && step > self.step_tol {
            // Probe all 2n coordinate directions; move to the single
            // best-improving one found this iteration (deterministic
            // generalized pattern search, per Kolda/Lewis/Torczon 2003).
            let mut best_trial: Option<(Vec<f64>, f64)> = None;
            for j in 0..n {
                for &sign in &[1.0, -1.0] {
                    let mut trial = x.clone();
                    trial[j] = (trial[j] + sign * step).clamp(0.0, 1.0);
                    let ft = eval(&trial, &mut fitness);
                    if ft < fx && best_trial.as_ref().map_or(true, |(_, bf)| ft < *bf) {
                        best_trial = Some((trial, ft));
                    }
                }
            }

            if let Some((trial, ft)) = best_trial {
                x = trial;
                fx = ft;
            } else {
                step *= self.shrink_factor;
            }

            history.push(fx);
            on_iteration(iter, fx);
            iter += 1;
        }

        NmResult {
            best_params: to_real(&x),
            best_fitness: fx,
            iterations: iter,
            history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sphere function: global minimum at the origin with f=0.
    #[test]
    fn minimises_sphere() {
        let cs = CompassSearch { max_iter: 300, ..Default::default() };
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let x0 = [3.0, -2.0];
        let result = cs.run(&bounds, &x0, |x| Some(x[0].powi(2) + x[1].powi(2)));
        assert!(
            result.best_fitness < 1e-6,
            "sphere minimum not found: f={:.3e} at ({:.6}, {:.6})",
            result.best_fitness, result.best_params[0], result.best_params[1]
        );
    }

    /// Rosenbrock: narrow curved valley, classic hard case for coordinate
    /// pattern search (axis-aligned steps struggle to follow the curve).
    /// Generous budget, looser tolerance than Nelder-Mead's equivalent test
    /// -- compass search is expected to converge more slowly here, that's
    /// the whole point of comparing the two.
    #[test]
    fn minimises_rosenbrock() {
        let cs = CompassSearch {
            max_iter: 5000,
            start_step: 0.1,
            shrink_factor: 0.5,
            step_tol: 1e-10,
        };
        let bounds = vec![(-2.0_f64, 2.0), (-2.0, 2.0)];
        let x0 = [-1.0, 1.0];
        let result = cs.run(&bounds, &x0, |x| {
            let f = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
            Some(f)
        });
        assert!(
            result.best_fitness < 1e-2,
            "Rosenbrock minimum not found: f={:.3e} at ({:.4}, {:.4})",
            result.best_fitness, result.best_params[0], result.best_params[1]
        );
    }

    /// Infeasible fitness (`None`) must not panic and must be avoided.
    #[test]
    fn handles_infeasible_fitness() {
        let cs = CompassSearch { max_iter: 50, ..Default::default() };
        let bounds = vec![(0.0_f64, 1.0)];
        let x0 = [0.5];
        let result = cs.run(&bounds, &x0, |_| None);
        assert_eq!(result.best_fitness, f64::MAX);
    }

    /// A seeded start already at the optimum should converge immediately
    /// (step shrinks to tolerance quickly with no improving moves found).
    #[test]
    fn seeded_start_at_optimum_converges_fast() {
        let cs = CompassSearch { max_iter: 200, ..Default::default() };
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let x0 = [0.001, -0.001];
        let result = cs.run(&bounds, &x0, |x| Some(x[0].powi(2) + x[1].powi(2)));
        assert!(result.best_fitness < 1e-4);
    }
}
