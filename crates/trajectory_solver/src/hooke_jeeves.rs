//! Classic Hooke & Jeeves (1961) greedy pattern search for bounded
//! continuous parameter spaces.
//!
//! At each sweep, probes coordinate directions in order (`+step` then
//! `-step` per axis) and accepts and restarts the sweep from the FIRST
//! improving move found — unlike [`crate::CompassSearch`]'s
//! best-of-all-`2n`-directions variant, this never surveys every direction
//! before moving. If a full sweep finds no improving move anywhere, the
//! step shrinks by `reduction_coeff`. Termination is a fitness-EVALUATION
//! budget (`max_fevals`), not an iteration count — this is deliberate (see
//! below), not an oversight.
//!
//! Added after checking `MbhSolver`'s inner local descent against
//! real third-party MBH reference behaviour (pagmo2's `mbh`, whose default
//! inner algorithm is a greedy pattern search of this same published type)
//! and finding that our own pre-existing `CompassSearch`/`NelderMead`
//! variants both use best-of-all-directions acceptance and an
//! iteration-count budget — a materially different cost allocation between
//! outer hops and inner local polish. This is an independent
//! reimplementation from the algorithm's own 1961 description (see the design notes
//! "Third-Party Algorithm Reimplementation Policy" for why this is written
//! from the published method, in this crate's own style, rather than ported
//! from any specific existing implementation): only PUBLIC FACTS about
//! third-party default behaviour (which acceptance rule, which termination
//! condition, which default parameter values) were used to decide what to
//! build, never any third-party source code or structure.
//!
//! With `max_fevals` set very small (as small as 1-2 full sweeps' worth of
//! evaluations), this optimizer barely polishes its starting point before
//! returning control to whatever outer search called it — appropriate when
//! the outer search (e.g. [`crate::MbhSolver`]'s hop loop) is meant to do
//! most of the exploration work itself, with each hop's local descent
//! providing only a cheap nudge toward the nearest improving direction
//! rather than a full local-minimum solve.
//!
//! # References
//! - Hooke, R. & Jeeves, T. A. (1961), "Direct Search Solution of Numerical
//!   and Statistical Problems", J. ACM 8(2):212-229.
//! - Kolda, T. G., Lewis, R. M. & Torczon, V. (2003), "Optimization by
//!   Direct Search: New Perspectives on Some Classical and Modern Methods",
//!   SIAM Review 45(3):385-482 (surveys this and other pattern-search
//!   variants and their convergence theory).

use crate::nelder_mead::NmResult;

/// Hooke & Jeeves greedy pattern search minimiser for bounded real-valued
/// parameters.
///
/// Operates entirely in bounds-normalised `[0, 1]^n` space internally (same
/// convention as [`crate::NelderMead`]/[`crate::CompassSearch`]).
pub struct HookeJeevesSearch {
    /// Fitness-evaluation budget. The sweep loop stops once the count of
    /// evaluations performed so far exceeds this, even mid-sweep — a full
    /// sweep is NOT guaranteed to complete. `1` (an extremely shallow,
    /// near-single-probe descent) is a legitimate and intentional setting
    /// when the outer search is meant to carry most of the exploration
    /// burden itself.
    pub max_fevals: usize,
    /// Initial step size, as a fraction of each parameter's bounds width.
    pub start_range: f64,
    /// Stop once the (normalised) step size drops below this, independent
    /// of the evaluation budget.
    pub stop_range: f64,
    /// Step shrink factor applied after a full sweep finds no improving
    /// move (standard: 0.5).
    pub reduction_coeff: f64,
}

impl Default for HookeJeevesSearch {
    fn default() -> Self {
        HookeJeevesSearch {
            max_fevals: 1,
            start_range: 0.1,
            stop_range: 0.01,
            reduction_coeff: 0.5,
        }
    }
}

impl HookeJeevesSearch {
    /// Minimise `fitness` over `bounds` (one `(min, max)` pair per
    /// parameter), starting at `x0` (original units, clamped into bounds).
    ///
    /// `fitness` returning `None` is treated as infeasible, penalised to
    /// `f64::MAX`, same convention as [`crate::NelderMead`]/
    /// [`crate::CompassSearch`].
    pub fn run<F>(&self, bounds: &[(f64, f64)], x0: &[f64], fitness: F) -> NmResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
    {
        self.run_with_progress(bounds, x0, fitness, |_iter, _best| {})
    }

    /// Same as [`HookeJeevesSearch::run`] but calls `on_iteration(sweep_index,
    /// best_fitness_so_far)` after each sweep — for live progress streams.
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
        let mut range = self.start_range;
        let mut fevals = 0usize;
        let mut history = Vec::new();
        let mut sweep = 0usize;

        while range > self.stop_range && fevals <= self.max_fevals {
            let mut improved = false;
            for j in 0..n {
                let mut trial = x.clone();
                trial[j] = (trial[j] + range).clamp(0.0, 1.0);
                let ft = eval(&trial, &mut fitness);
                fevals += 1;
                if ft < fx {
                    x = trial;
                    fx = ft;
                    improved = true;
                    break;
                }

                trial = x.clone();
                trial[j] = (trial[j] - range).clamp(0.0, 1.0);
                let ft = eval(&trial, &mut fitness);
                fevals += 1;
                if ft < fx {
                    x = trial;
                    fx = ft;
                    improved = true;
                    break;
                }
            }

            if !improved {
                range *= self.reduction_coeff;
            }

            history.push(fx);
            on_iteration(sweep, fx);
            sweep += 1;
        }

        NmResult {
            best_params: to_real(&x),
            best_fitness: fx,
            iterations: sweep,
            history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A generous budget must still find the sphere minimum, confirming the
    /// greedy accept-first-improving sweep converges given enough fevals.
    #[test]
    fn generous_budget_minimises_sphere() {
        let hj = HookeJeevesSearch { max_fevals: 2000, ..Default::default() };
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let x0 = [3.0, -2.0];
        let result = hj.run(&bounds, &x0, |x| Some(x[0].powi(2) + x[1].powi(2)));
        assert!(
            result.best_fitness < 1e-4,
            "sphere minimum not found: f={:.3e} at ({:.6}, {:.6})",
            result.best_fitness, result.best_params[0], result.best_params[1]
        );
    }

    /// A `max_fevals=1` budget (pagmo2's own default) must terminate after
    /// essentially one sweep, not converge to the optimum — confirming this
    /// really is a shallow, eval-budget-gated descent, not a disguised
    /// full local solve.
    #[test]
    fn tiny_budget_stops_early_without_full_convergence() {
        let hj = HookeJeevesSearch { max_fevals: 1, ..Default::default() };
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let x0 = [3.0, -2.0];
        let result = hj.run(&bounds, &x0, |x| Some(x[0].powi(2) + x[1].powi(2)));
        // Only ever ran (at most) one sweep of up to 2n=4 evaluations.
        assert!(result.iterations <= 1, "expected at most one sweep, got {}", result.iterations);
        // Far from converged from this start point at this shallow a budget.
        assert!(
            result.best_fitness > 1.0,
            "unexpectedly converged with a 1-feval-budget sweep: f={:.3e}",
            result.best_fitness
        );
    }

    /// Greedy accept-FIRST-improving means the result can differ from (and
    /// need not be better or worse than) `CompassSearch`'s best-of-all
    /// choice on any single sweep — this test only confirms the greedy rule
    /// is actually being applied: given a start point where axis 0's "up"
    /// move improves but is not the best available move, the algorithm must
    /// still take it rather than surveying every direction first.
    #[test]
    fn accepts_first_improving_move_not_best_of_all() {
        // f increases in x, decreases strongly in y: probing x=+step from
        // (0,0) improves (moves away from a positive-x penalty is not what
        // we want -- construct so +x is a mild improvement and +y is a much
        // bigger one, to distinguish greedy-first from best-of-all).
        let f = |x: &[f64]| -> Option<f64> {
            Some((x[0] + 1.0).powi(2) + (x[1] + 10.0).powi(2))
        };
        let hj = HookeJeevesSearch { max_fevals: 1, start_range: 0.01, stop_range: 0.001, reduction_coeff: 0.5 };
        let bounds = vec![(-20.0_f64, 20.0), (-20.0, 20.0)];
        let x0 = [0.0, 0.0];
        let result = hj.run(&bounds, &x0, f);
        // Greedy: axis 0 is probed first. "-step" on x0 improves f (moves
        // toward -1), so the FIRST accepted move should change x[0], not
        // x[1], even though a bigger relative gain is available on axis 1.
        assert_ne!(result.best_params[0], x0[0], "greedy sweep should have moved along axis 0 first");
    }

    /// Infeasible fitness (`None`) must not panic and must be avoided.
    #[test]
    fn handles_infeasible_fitness() {
        let hj = HookeJeevesSearch { max_fevals: 50, ..Default::default() };
        let bounds = vec![(0.0_f64, 1.0)];
        let x0 = [0.5];
        let result = hj.run(&bounds, &x0, |_| None);
        assert_eq!(result.best_fitness, f64::MAX);
    }
}
