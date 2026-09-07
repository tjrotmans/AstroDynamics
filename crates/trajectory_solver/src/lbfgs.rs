//! Bounded quasi-Newton (projected L-BFGS) local minimiser with
//! finite-difference gradients.
//!
//! The joint gradient-based "NLP polish" stage for MGA winners (Phase 9x,
//!): unlike the existing per-coordinate polishes (eta re-polish's
//! coordinate descent, Nelder-Mead's simplex), a quasi-Newton step moves ALL
//! decision variables simultaneously along a curvature-informed direction —
//! the only local method in this crate that can exploit cross-variable
//! tradeoffs ("give leg 1 a slightly worse departure angle so leg 2's DSM
//! gets much cheaper") in a single step. This is the standard final stage in
//! real MGA design tools (EMTG, pykep pipelines run an NLP solver after
//! their global search); the full SQP machinery those tools use is not
//! needed here because this codebase's MGA fitness has no separate equality
//! constraints — each leg's Lambert arc hits its target by construction, so
//! the problem is a plain bound-constrained scalar minimisation.
//!
//! Every parameter is internally rescaled to `[0, 1]` using the supplied
//! bounds (same rationale as `nelder_mead.rs` and the multiple-shooting
//! scaling lesson: an unscaled step sized for a radian-scale coordinate is
//! either negligible or explosive on a days-scale one).
//!
//! Gradients are central finite differences in the normalised space. The
//! MGA fitness has genuine discontinuities (Lambert branch edges, graded
//! penalty clamps), so the line search is a plain backtracking
//! sufficient-decrease (Armijo) search that only ever accepts strict
//! improvement — a cliff in the gradient can waste an iteration but can
//! never make the result worse than the seed, which is the property the
//! polish stage actually needs (same "safe by construction" contract as
//! `repolish_leg_etas` and the bidirectional min() comparison in
//! `MissionPlanner/src/mga.rs`).
//!
//! # References
//! - Nocedal, J. & Wright, S. (2006), *Numerical Optimization*, 2nd ed.,
//!   Springer — §7.2 (L-BFGS two-loop recursion), §3.1 (Armijo backtracking).
//! - Byrd, Lu, Nocedal & Zhu (1995), "A Limited Memory Algorithm for Bound
//!   Constrained Optimization", SIAM J. Sci. Comput. 16(5) — the full
//!   L-BFGS-B algorithm this module deliberately simplifies: instead of the
//!   generalized-Cauchy-point subspace machinery, bounds are handled by
//!   projection (clamp) plus zeroing gradient components that point out of
//!   an active bound, which is adequate for a polish stage seeded at an
//!   interior near-optimum.

/// Result of an [`LbfgsSolver::run`] call.
#[derive(Clone, Debug)]
pub struct LbfgsResult {
    pub best_params: Vec<f64>,
    pub best_fitness: f64,
    /// Number of outer iterations actually run.
    pub iterations: usize,
    /// Best fitness after each iteration, for convergence plots.
    pub history: Vec<f64>,
}

/// Projected L-BFGS minimiser for bounded real-valued parameters.
pub struct LbfgsSolver {
    pub max_iter: usize,
    /// L-BFGS memory: number of past (s, y) correction pairs kept.
    pub memory: usize,
    /// Central finite-difference half-step in NORMALISED (`[0,1]`) space.
    pub fd_step: f64,
    /// Stop when an accepted step improves fitness by less than this.
    pub fitness_tol: f64,
    /// Armijo sufficient-decrease coefficient (standard: 1e-4).
    pub armijo_c: f64,
    /// Maximum backtracking halvings per line search before giving up.
    pub max_backtracks: usize,
}

impl Default for LbfgsSolver {
    fn default() -> Self {
        LbfgsSolver {
            max_iter: 100,
            memory: 8,
            // Large enough to step over fitness-evaluation noise (Lambert /
            // Kepler iteration tolerances), small enough to resolve the
            // local slope of a smooth region.
            fd_step: 1.0e-6,
            fitness_tol: 1.0e-6,
            armijo_c: 1.0e-4,
            max_backtracks: 30,
        }
    }
}

impl LbfgsSolver {
    /// Minimise `fitness` over `bounds` (one `(min, max)` pair per
    /// parameter), starting from `x0` (original units, clamped into bounds).
    ///
    /// `fitness` returning `None` is treated as infeasible: an infeasible
    /// trial point is simply rejected by the line search (never accepted),
    /// and an infeasible point encountered inside the finite-difference
    /// stencil falls back to the centre value (zero contribution to that
    /// gradient component) — the search stays inside the feasible region it
    /// was seeded in rather than panicking or wandering.
    ///
    /// Returns a result whose `best_fitness` is never worse than the seed's
    /// own fitness (strict-improvement acceptance). If the seed itself is
    /// infeasible, returns it unchanged with `best_fitness = f64::MAX`.
    pub fn run<F>(&self, bounds: &[(f64, f64)], x0: &[f64], mut fitness: F) -> LbfgsResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
    {
        let n = bounds.len();
        let to_real = |u: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|j| bounds[j].0 + u[j].clamp(0.0, 1.0) * (bounds[j].1 - bounds[j].0))
                .collect()
        };
        let eval = |u: &[f64], fitness: &mut F| -> Option<f64> { fitness(&to_real(u)) };

        let mut x: Vec<f64> = (0..n)
            .map(|j| ((x0[j] - bounds[j].0) / (bounds[j].1 - bounds[j].0)).clamp(0.0, 1.0))
            .collect();
        let mut f = match eval(&x, &mut fitness) {
            Some(v) => v,
            None => {
                return LbfgsResult {
                    best_params: x0.to_vec(),
                    best_fitness: f64::MAX,
                    iterations: 0,
                    history: Vec::new(),
                }
            }
        };

        // Central-difference gradient in normalised space. Components whose
        // stencil leaves the box or hits infeasibility degrade to one-sided
        // or zero — see `run`'s doc comment.
        let grad = |x: &[f64], f_center: f64, eval: &mut dyn FnMut(&[f64]) -> Option<f64>| -> Vec<f64> {
            let h = self.fd_step;
            (0..n)
                .map(|j| {
                    let mut xp = x.to_vec();
                    let mut xm = x.to_vec();
                    xp[j] = (x[j] + h).min(1.0);
                    xm[j] = (x[j] - h).max(0.0);
                    let dx = xp[j] - xm[j];
                    if dx <= 0.0 {
                        return 0.0;
                    }
                    let fp = eval(&xp).unwrap_or(f_center);
                    let fm = eval(&xm).unwrap_or(f_center);
                    (fp - fm) / dx
                })
                .collect()
        };

        // L-BFGS correction-pair history (s = x_{k+1} − x_k, y = g_{k+1} − g_k).
        let mut s_hist: Vec<Vec<f64>> = Vec::new();
        let mut y_hist: Vec<Vec<f64>> = Vec::new();

        let mut eval_dyn = |u: &[f64]| -> Option<f64> { fitness(&to_real(u)) };
        let mut g = grad(&x, f, &mut eval_dyn);

        let mut history = Vec::with_capacity(self.max_iter);
        let mut iters = 0usize;

        for _ in 0..self.max_iter {
            // Project the gradient: zero any component that points out of an
            // active bound (moving along it would immediately be clamped
            // back, poisoning the curvature pairs with zero-length moves).
            let mut g_proj = g.clone();
            for j in 0..n {
                if (x[j] <= 0.0 && g_proj[j] > 0.0) || (x[j] >= 1.0 && g_proj[j] < 0.0) {
                    g_proj[j] = 0.0;
                }
            }
            let g_norm = g_proj.iter().map(|v| v * v).sum::<f64>().sqrt();
            if g_norm < 1.0e-12 {
                break;
            }

            // Two-loop recursion (Nocedal & Wright §7.2) for d = −H·g_proj.
            let mut q = g_proj.clone();
            let m = s_hist.len();
            let mut alpha = vec![0.0; m];
            let rho: Vec<f64> = (0..m)
                .map(|i| {
                    let sy: f64 = s_hist[i].iter().zip(&y_hist[i]).map(|(a, b)| a * b).sum();
                    if sy.abs() > 1.0e-18 { 1.0 / sy } else { 0.0 }
                })
                .collect();
            for i in (0..m).rev() {
                alpha[i] = rho[i] * s_hist[i].iter().zip(&q).map(|(a, b)| a * b).sum::<f64>();
                for j in 0..n {
                    q[j] -= alpha[i] * y_hist[i][j];
                }
            }
            // Initial Hessian scaling γ = sᵀy / yᵀy from the newest pair.
            let gamma = if m > 0 {
                let sy: f64 = s_hist[m - 1].iter().zip(&y_hist[m - 1]).map(|(a, b)| a * b).sum();
                let yy: f64 = y_hist[m - 1].iter().map(|v| v * v).sum();
                if yy > 1.0e-18 && sy > 0.0 { sy / yy } else { 1.0 }
            } else {
                // First iteration: scale so the initial trial step has a
                // sensible normalised length regardless of fitness units.
                0.01 / g_norm.max(1.0e-12)
            };
            for v in q.iter_mut() {
                *v *= gamma;
            }
            for i in 0..m {
                let beta = rho[i] * y_hist[i].iter().zip(&q).map(|(a, b)| a * b).sum::<f64>();
                for j in 0..n {
                    q[j] += s_hist[i][j] * (alpha[i] - beta);
                }
            }
            let d: Vec<f64> = q.iter().map(|v| -v).collect();

            // Descent check: a stale curvature history near a discontinuity
            // can produce an ascent direction — fall back to steepest descent.
            let dg: f64 = d.iter().zip(&g_proj).map(|(a, b)| a * b).sum();
            let d = if dg < 0.0 {
                d
            } else {
                g_proj.iter().map(|v| -v * gamma.max(1.0e-6)).collect()
            };
            let dg = d.iter().zip(&g_proj).map(|(a, b)| a * b).sum::<f64>();

            // Backtracking Armijo line search on the PROJECTED trial point.
            let mut t = 1.0_f64;
            let mut accepted: Option<(Vec<f64>, f64)> = None;
            for _ in 0..self.max_backtracks {
                let x_trial: Vec<f64> = (0..n).map(|j| (x[j] + t * d[j]).clamp(0.0, 1.0)).collect();
                if let Some(f_trial) = eval_dyn(&x_trial) {
                    if f_trial < f + self.armijo_c * t * dg && f_trial < f {
                        accepted = Some((x_trial, f_trial));
                        break;
                    }
                }
                t *= 0.5;
            }

            let (x_new, f_new) = match accepted {
                Some(v) => v,
                None => break, // no improving step found — converged (or at a cliff)
            };
            iters += 1;
            let improvement = f - f_new;

            let g_new = grad(&x_new, f_new, &mut eval_dyn);
            let s: Vec<f64> = (0..n).map(|j| x_new[j] - x[j]).collect();
            let y: Vec<f64> = (0..n).map(|j| g_new[j] - g[j]).collect();
            // Only keep curvature pairs with positive sᵀy (the L-BFGS
            // positive-definiteness condition); skip pairs polluted by a
            // discontinuity.
            let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
            if sy > 1.0e-14 {
                s_hist.push(s);
                y_hist.push(y);
                if s_hist.len() > self.memory {
                    s_hist.remove(0);
                    y_hist.remove(0);
                }
            }

            x = x_new;
            f = f_new;
            g = g_new;
            history.push(f);

            if improvement < self.fitness_tol {
                break;
            }
        }

        LbfgsResult {
            best_params: to_real(&x),
            best_fitness: f,
            iterations: iters,
            history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convex quadratic with strongly mixed units (mimicking the MGA
    /// chromosome's days / m/s / radians mix) — must converge to the known
    /// interior optimum despite the raw-unit scale spread.
    #[test]
    fn minimises_scaled_quadratic() {
        let solver = LbfgsSolver::default();
        // Optimum at (100 days, 3000 m/s, 0.5 rad) inside generous bounds.
        let bounds = vec![(0.0_f64, 1000.0), (0.0, 10_000.0), (-3.14, 3.14)];
        let x0 = [400.0, 8000.0, -2.0];
        let result = solver.run(&bounds, &x0, |x| {
            Some((x[0] - 100.0).powi(2) / 1.0e4 + (x[1] - 3000.0).powi(2) / 1.0e6 + (x[2] - 0.5).powi(2))
        });
        assert!(result.best_fitness < 1.0e-6, "f={:.3e}", result.best_fitness);
        assert!((result.best_params[0] - 100.0).abs() < 1.0);
        assert!((result.best_params[1] - 3000.0).abs() < 10.0);
        assert!((result.best_params[2] - 0.5).abs() < 1.0e-2);
    }

    /// Rosenbrock: the classic curved-valley test where coordinate descent
    /// stalls but a quasi-Newton method following the valley succeeds —
    /// exactly the cross-variable-coupling capability this solver exists to
    /// add over `repolish_leg_etas`-style coordinate descent.
    #[test]
    fn minimises_rosenbrock() {
        let solver = LbfgsSolver { max_iter: 500, ..Default::default() };
        let bounds = vec![(-2.0_f64, 2.0), (-2.0, 2.0)];
        let x0 = [-1.0, 1.0];
        let result = solver.run(&bounds, &x0, |x| {
            Some((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
        });
        assert!(result.best_fitness < 1.0e-4, "f={:.3e} at ({:.4}, {:.4})",
            result.best_fitness, result.best_params[0], result.best_params[1]);
    }

    /// Optimum outside the box: the solver must converge to the nearest
    /// bound and stop cleanly, not oscillate or escape the box.
    #[test]
    fn respects_active_bounds() {
        let solver = LbfgsSolver::default();
        let bounds = vec![(0.0_f64, 1.0)];
        let x0 = [0.9];
        // Unconstrained optimum at x = 3, so the bounded optimum is x = 1.
        let result = solver.run(&bounds, &x0, |x| Some((x[0] - 3.0).powi(2)));
        assert!((result.best_params[0] - 1.0).abs() < 1.0e-6,
            "bounded optimum should be at the upper bound: x={:.6}", result.best_params[0]);
        assert!(result.best_fitness <= (0.9_f64 - 3.0).powi(2));
    }

    /// Strict-improvement contract: seeded exactly at the optimum, the
    /// result must equal the seed (no wandering), and an infeasible seed
    /// must be returned unchanged with f64::MAX.
    #[test]
    fn never_worse_than_seed() {
        let solver = LbfgsSolver::default();
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let result = solver.run(&bounds, &[0.0, 0.0], |x| Some(x[0].powi(2) + x[1].powi(2)));
        assert!(result.best_fitness <= 1.0e-12);

        let infeasible = solver.run(&bounds, &[1.0, 1.0], |_| None);
        assert_eq!(infeasible.best_fitness, f64::MAX);
        assert_eq!(infeasible.best_params, vec![1.0, 1.0]);
    }

    /// Infeasible pockets inside the box (mimicking Lambert-failure regions
    /// in the MGA fitness): the search must survive them and still improve.
    #[test]
    fn tolerates_infeasible_pockets() {
        let solver = LbfgsSolver::default();
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let x0 = [3.0, 3.0];
        let result = solver.run(&bounds, &x0, |x| {
            // A thin infeasible band the descent path must cross or skirt.
            if (x[0] - 1.5).abs() < 0.05 { return None; }
            Some(x[0].powi(2) + x[1].powi(2))
        });
        assert!(result.best_fitness < 18.0, "must improve on the seed (f0=18): f={:.3}", result.best_fitness);
    }
}
