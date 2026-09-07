//! Nelder-Mead simplex minimiser for bounded continuous parameter spaces.
//!
//! Derivative-free local optimiser — appropriate as the descent step for
//! Monotonic Basin Hopping (Phase 9x-v Stage 3) because the MGA-1DSM fitness
//! pipeline has near-boundary discontinuities (Lambert branch edges, graded-
//! penalty clamps) that make finite-difference gradients noisy.
//!
//! Every parameter is internally rescaled to `[0, 1]` using the supplied
//! bounds before the simplex operations run, and rescaled back for fitness
//! evaluation and the returned result. This is required whenever the
//! parameter vector mixes units of wildly different magnitude (days, m/s,
//! radians, dimensionless norms, as in an MGA-1DSM chromosome) — an
//! unscaled simplex step sized for a radian-scale coordinate is either
//! negligible or explosive on a days- or metres-per-second-scale coordinate.
//!
//! # References
//! - Nelder, J. A. & Mead, R. (1965), "A Simplex Method for Function
//!   Minimization", The Computer Journal 7(4):308–313.

/// Result of a [`NelderMead::run`] call.
#[derive(Clone, Debug)]
pub struct NmResult {
    pub best_params: Vec<f64>,
    pub best_fitness: f64,
    /// Number of iterations actually run (may be less than `max_iter` if
    /// convergence tolerances were met first).
    pub iterations: usize,
    /// Best fitness found so far, one entry per iteration — for convergence
    /// history plots.
    pub history: Vec<f64>,
}

/// Nelder-Mead simplex minimiser for bounded real-valued parameters.
///
/// Standard reflect/expand/contract/shrink update (Nelder & Mead 1965) with
/// the classic default coefficients. Operates entirely in bounds-normalised
/// `[0, 1]^n` space internally; `bounds` and `x0` passed to [`NelderMead::run`]
/// are in the caller's original units.
pub struct NelderMead {
    pub max_iter: usize,
    /// Convergence when the spread of fitness values across the simplex
    /// drops below this.
    pub fitness_tol: f64,
    /// Convergence when the simplex's normalised (`[0,1]^n`) extent drops
    /// below this.
    pub simplex_tol: f64,
    /// Reflection coefficient (standard: 1.0).
    pub alpha: f64,
    /// Expansion coefficient (standard: 2.0).
    pub gamma: f64,
    /// Contraction coefficient (standard: 0.5).
    pub rho: f64,
    /// Shrink coefficient (standard: 0.5).
    pub sigma: f64,
}

impl Default for NelderMead {
    fn default() -> Self {
        NelderMead {
            max_iter: 500,
            fitness_tol: 1e-10,
            simplex_tol: 1e-8,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

impl NelderMead {
    /// Minimise `fitness` over `bounds` (one `(min, max)` pair per parameter),
    /// starting the simplex at `x0` (original units, clamped into bounds).
    ///
    /// `fitness` returning `None` is treated as infeasible, penalised to
    /// `f64::MAX` so the simplex steers away from it without panicking.
    pub fn run<F>(&self, bounds: &[(f64, f64)], x0: &[f64], fitness: F) -> NmResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
    {
        self.run_with_progress(bounds, x0, fitness, |_iter, _best| {})
    }

    /// Same as [`NelderMead::run`] but calls `on_iteration(iteration_index,
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

        // Initial simplex: x0 plus one point per axis, perturbed by a fixed
        // fraction of the normalised range (reflected inward at a bound).
        const INIT_STEP: f64 = 0.05;
        let x0n = to_norm(x0);
        let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        simplex.push(x0n.clone());
        for j in 0..n {
            let mut p = x0n.clone();
            let bumped = p[j] + INIT_STEP;
            p[j] = if bumped <= 1.0 { bumped } else { (p[j] - INIT_STEP).max(0.0) };
            simplex.push(p);
        }
        let mut fvals: Vec<f64> = simplex.iter().map(|u| eval(u, &mut fitness)).collect();

        let mut history = Vec::with_capacity(self.max_iter);
        let mut iter = 0;

        while iter < self.max_iter {
            // Sort simplex vertices by fitness, best first.
            let mut order: Vec<usize> = (0..=n).collect();
            order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());
            simplex = order.iter().map(|&i| simplex[i].clone()).collect();
            fvals = order.iter().map(|&i| fvals[i]).collect();

            history.push(fvals[0]);
            on_iteration(iter, fvals[0]);

            let fitness_spread = fvals[n] - fvals[0];
            let simplex_extent = simplex[1..]
                .iter()
                .map(|p| {
                    p.iter()
                        .zip(&simplex[0])
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(0.0_f64, f64::max);
            if fitness_spread < self.fitness_tol && simplex_extent < self.simplex_tol {
                break;
            }

            // Centroid of all points except the worst.
            let mut centroid = vec![0.0; n];
            for p in &simplex[..n] {
                for j in 0..n {
                    centroid[j] += p[j] / n as f64;
                }
            }

            let worst = simplex[n].clone();
            let reflect = |scale: f64, base: &[f64], from: &[f64]| -> Vec<f64> {
                (0..n)
                    .map(|j| (base[j] + scale * (base[j] - from[j])).clamp(0.0, 1.0))
                    .collect()
            };

            let xr = reflect(self.alpha, &centroid, &worst);
            let fr = eval(&xr, &mut fitness);

            if fr < fvals[0] {
                // Reflection beat the best — try expanding further.
                let xe = reflect(self.alpha * self.gamma, &centroid, &worst);
                let fe = eval(&xe, &mut fitness);
                if fe < fr {
                    simplex[n] = xe;
                    fvals[n] = fe;
                } else {
                    simplex[n] = xr;
                    fvals[n] = fr;
                }
            } else if fr < fvals[n - 1] {
                // Reflection is better than the second-worst — accept it.
                simplex[n] = xr;
                fvals[n] = fr;
            } else {
                // Contraction: outside if reflection beat the worst, else inside.
                let (xc, fc, threshold) = if fr < fvals[n] {
                    let xc: Vec<f64> = (0..n)
                        .map(|j| (centroid[j] + self.rho * (xr[j] - centroid[j])).clamp(0.0, 1.0))
                        .collect();
                    let fc = eval(&xc, &mut fitness);
                    (xc, fc, fr)
                } else {
                    let xc: Vec<f64> = (0..n)
                        .map(|j| (centroid[j] + self.rho * (worst[j] - centroid[j])).clamp(0.0, 1.0))
                        .collect();
                    let fc = eval(&xc, &mut fitness);
                    (xc, fc, fvals[n])
                };
                if fc < threshold {
                    simplex[n] = xc;
                    fvals[n] = fc;
                } else {
                    // Shrink the whole simplex toward the best point.
                    for i in 1..=n {
                        let shrunk: Vec<f64> = (0..n)
                            .map(|j| {
                                (simplex[0][j] + self.sigma * (simplex[i][j] - simplex[0][j]))
                                    .clamp(0.0, 1.0)
                            })
                            .collect();
                        fvals[i] = eval(&shrunk, &mut fitness);
                        simplex[i] = shrunk;
                    }
                }
            }

            iter += 1;
        }

        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());
        let best = order[0];

        NmResult {
            best_params: to_real(&simplex[best]),
            best_fitness: fvals[best],
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
        let nm = NelderMead {
            max_iter: 300,
            ..Default::default()
        };
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let x0 = [3.0, -2.0];
        let result = nm.run(&bounds, &x0, |x| Some(x[0].powi(2) + x[1].powi(2)));
        assert!(
            result.best_fitness < 1e-8,
            "sphere minimum not found: f={:.3e} at ({:.6}, {:.6})",
            result.best_fitness,
            result.best_params[0],
            result.best_params[1]
        );
        assert!(result.best_params[0].abs() < 1e-3);
        assert!(result.best_params[1].abs() < 1e-3);
    }

    /// Rosenbrock function: global minimum at (1, 1) with f=0. Classic
    /// non-convex, narrow-curved-valley test — harder for a simplex method
    /// than sphere, needs a generous iteration budget.
    #[test]
    fn minimises_rosenbrock() {
        let nm = NelderMead {
            max_iter: 3000,
            fitness_tol: 1e-12,
            simplex_tol: 1e-10,
            ..Default::default()
        };
        let bounds = vec![(-2.0_f64, 2.0), (-2.0, 2.0)];
        let x0 = [-1.0, 1.0];
        let result = nm.run(&bounds, &x0, |x| {
            let f = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
            Some(f)
        });
        assert!(
            result.best_fitness < 1e-4,
            "Rosenbrock minimum not found: f={:.3e} at ({:.4}, {:.4})",
            result.best_fitness,
            result.best_params[0],
            result.best_params[1]
        );
        assert!((result.best_params[0] - 1.0).abs() < 1e-2);
        assert!((result.best_params[1] - 1.0).abs() < 1e-2);
    }

    /// Infeasible fitness (`None`) must not panic and must be avoided.
    #[test]
    fn handles_infeasible_fitness() {
        let nm = NelderMead {
            max_iter: 50,
            ..Default::default()
        };
        let bounds = vec![(0.0_f64, 1.0)];
        let x0 = [0.5];
        let result = nm.run(&bounds, &x0, |_| None);
        assert_eq!(result.best_fitness, f64::MAX);
    }

    /// A seeded start already at the optimum should converge immediately
    /// with a tiny iteration budget.
    #[test]
    fn seeded_start_at_optimum_converges_fast() {
        let nm = NelderMead {
            max_iter: 20,
            ..Default::default()
        };
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let x0 = [0.001, -0.001];
        let result = nm.run(&bounds, &x0, |x| Some(x[0].powi(2) + x[1].powi(2)));
        assert!(result.best_fitness < 1e-5);
    }
}
