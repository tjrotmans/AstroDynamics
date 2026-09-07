//! RK45 integrators for the CRTBP (3-D primary, 2-D legacy).
//!
//! # 3-D (primary)
//! - `Crtbp3dOde`     – 6-state  [x, y, z, vx, vy, vz]
//! - `Crtbp3dOdeBack` – same, time-reversed
//! - `Crtbp3dStmOde`  – 42-state [state | Φ cols 0..5]  (6×6 STM, column-major)
//!
//! # 2-D (legacy — kept for reference, not used by the main code path)
//! - `Crtbp2dOde`    – 4-state  [x, y, vx, vy]
//! - `Crtbp2dStmOde` – 20-state [x, y, vx, vy | φ col-0 .. col-3]  (4×4 STM)

use ode_solvers::dopri5::Dopri5;
use ode_solvers::dop_shared::OutputType;
use ode_solvers::{SVector as OdeVec, System};

/// Maximum integrator steps — raised from the library default (100 000) so that
/// large-amplitude orbits and long manifold arcs do not abort prematurely.
const N_MAX: u32 = 2_000_000;

use crate::crtbp::{eom_2d, jacobian_2d, eom_3d, jacobian_3d};
use orbital_models::constants::{MU_EARTH, MU_MOON, MU_SUN, AU, EARTH_MOON_DISTANCE, MOON_SIDEREAL_PERIOD_DAYS, JULIAN_YEAR_DAYS};

// ─── Type aliases ─────────────────────────────────────────────────────────────

pub type State4  = OdeVec<f64, 4>;
pub type State6  = OdeVec<f64, 6>;
pub type State20 = OdeVec<f64, 20>;
pub type State42 = OdeVec<f64, 42>;  // 6-state + 6×6 STM (column-major)

// ─── One logged step ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Step2d {
    pub time: f64,
    pub x:    f64,
    pub y:    f64,
    pub vx:   f64,
    pub vy:   f64,
}

impl Step2d {
    pub fn state(&self) -> [f64; 4] { [self.x, self.y, self.vx, self.vy] }
}

// ─── 3-D step type ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Step3d {
    pub time: f64,
    pub x:    f64,
    pub y:    f64,
    pub z:    f64,
    pub vx:   f64,
    pub vy:   f64,
    pub vz:   f64,
}

impl Step3d {
    pub fn state(&self) -> [f64; 6] { [self.x, self.y, self.z, self.vx, self.vy, self.vz] }
}

// ─── ODE systems ─────────────────────────────────────────────────────────────

// ─── 2-D step type ───────────────────────────────────────────────────────────

/// CRTBP 4-state ODE (planar), forward time.
pub struct Crtbp2dOde { pub mu: f64 }

impl System<f64, State4> for Crtbp2dOde {
    fn system(&self, _t: f64, y: &State4, dy: &mut State4) {
        let s = [y[0], y[1], y[2], y[3]];
        let d = eom_2d(self.mu, &s);
        dy[0] = d[0]; dy[1] = d[1]; dy[2] = d[2]; dy[3] = d[3];
    }
}

/// CRTBP 4-state ODE (planar), backward time (RHS negated).
/// Integrating this forward over [0, T] is equivalent to stepping -T in real time.
pub struct Crtbp2dOdeBack { pub mu: f64 }

impl System<f64, State4> for Crtbp2dOdeBack {
    fn system(&self, _t: f64, y: &State4, dy: &mut State4) {
        let s = [y[0], y[1], y[2], y[3]];
        let d = eom_2d(self.mu, &s);
        dy[0] = -d[0]; dy[1] = -d[1]; dy[2] = -d[2]; dy[3] = -d[3];
    }
}

/// CRTBP 20-state ODE: 4-state + 4×4 STM (column-major).
///
/// Layout of the 20-element vector:
/// ```text
/// [0..4]   main state  [x, y, vx, vy]
/// [4..8]   STM col 0   [φ₀₀, φ₁₀, φ₂₀, φ₃₀]
/// [8..12]  STM col 1   [φ₀₁, φ₁₁, φ₂₁, φ₃₁]
/// [12..16] STM col 2   [φ₀₂, φ₁₂, φ₂₂, φ₃₂]
/// [16..20] STM col 3   [φ₀₃, φ₁₃, φ₂₃, φ₃₃]
/// ```
/// d(STM col j)/dt = A(t) · (STM col j)
pub struct Crtbp2dStmOde { pub mu: f64 }

impl System<f64, State20> for Crtbp2dStmOde {
    fn system(&self, _t: f64, y: &State20, dy: &mut State20) {
        // ── main state ──────────────────────────────────────────────────────
        let s = [y[0], y[1], y[2], y[3]];
        let d = eom_2d(self.mu, &s);
        dy[0] = d[0]; dy[1] = d[1]; dy[2] = d[2]; dy[3] = d[3];

        // ── Jacobian A at current state ─────────────────────────────────────
        let a = jacobian_2d(self.mu, &s);

        // ── dφ/dt = A · φ  (column by column) ──────────────────────────────
        for col in 0..4_usize {
            let base = 4 + col * 4;
            let phi_col = [y[base], y[base+1], y[base+2], y[base+3]];
            for row in 0..4_usize {
                dy[base + row] =
                    a[row][0]*phi_col[0]
                  + a[row][1]*phi_col[1]
                  + a[row][2]*phi_col[2]
                  + a[row][3]*phi_col[3];
            }
        }
    }
}

// ─── Propagation helpers ──────────────────────────────────────────────────────

/// Integrate the planar CRTBP from `state0` for `t_end` normalized time units.
/// Logs a step every `log_dt`.
pub fn propagate_2d(
    mu:      f64,
    state0:  [f64; 4],
    t_end:   f64,
    log_dt:  f64,
    rtol:    f64,
    atol:    f64,
) -> Vec<Step2d> {
    let y0 = State4::from_column_slice(&state0);
    let mut stepper = Dopri5::new(
        Crtbp2dOde { mu },
        0.0, t_end, log_dt, y0, rtol, atol,
    );
    stepper.integrate().expect("CRTBP 2d integration failed");

    stepper.x_out().iter()
        .zip(stepper.y_out().iter())
        .map(|(&t, y)| Step2d { time: t, x: y[0], y: y[1], vx: y[2], vy: y[3] })
        .collect()
}

/// Integrate the planar CRTBP + 4×4 STM from `state0` for `t_end`.
/// Returns the final 20-element state (4-state + STM columns).
pub fn propagate_2d_stm(
    mu:      f64,
    state0:  [f64; 4],
    t_end:   f64,
    rtol:    f64,
    atol:    f64,
) -> (Vec<Step2d>, [[f64; 4]; 4]) {
    // Initial 20-state: main state + identity STM
    let mut y0_arr = [0.0_f64; 20];
    y0_arr[0..4].copy_from_slice(&state0);
    // Identity matrix columns
    y0_arr[4]  = 1.0; // col 0, row 0
    y0_arr[9]  = 1.0; // col 1, row 1
    y0_arr[14] = 1.0; // col 2, row 2
    y0_arr[19] = 1.0; // col 3, row 3

    let y0 = State20::from_column_slice(&y0_arr);
    let step_hint = t_end / 200.0;
    let mut stepper = Dopri5::new(
        Crtbp2dStmOde { mu },
        0.0, t_end, step_hint, y0, rtol, atol,
    );
    stepper.integrate().expect("CRTBP+STM integration failed");

    let traj: Vec<Step2d> = stepper.x_out().iter()
        .zip(stepper.y_out().iter())
        .map(|(&t, y)| Step2d { time: t, x: y[0], y: y[1], vx: y[2], vy: y[3] })
        .collect();

    // Extract monodromy matrix from final state
    let yf = stepper.y_out().last().expect("no output states");
    let mut mono = [[0.0_f64; 4]; 4];
    for col in 0..4_usize {
        let base = 4 + col * 4;
        for row in 0..4_usize {
            mono[row][col] = yf[base + row];
        }
    }

    (traj, mono)
}

/// Extract the STM at every step of a STM propagation.
/// Returns (trajectory, Vec<STM at each step>).
pub fn propagate_2d_stm_full(
    mu:      f64,
    state0:  [f64; 4],
    t_end:   f64,
    log_dt:  f64,
    rtol:    f64,
    atol:    f64,
) -> (Vec<Step2d>, Vec<[[f64; 4]; 4]>) {
    let mut y0_arr = [0.0_f64; 20];
    y0_arr[0..4].copy_from_slice(&state0);
    y0_arr[4]  = 1.0;
    y0_arr[9]  = 1.0;
    y0_arr[14] = 1.0;
    y0_arr[19] = 1.0;

    let y0 = State20::from_column_slice(&y0_arr);
    let mut stepper = Dopri5::new(
        Crtbp2dStmOde { mu },
        0.0, t_end, log_dt, y0, rtol, atol,
    );
    stepper.integrate().expect("CRTBP+STM full integration failed");

    let mut traj  = Vec::new();
    let mut stms: Vec<[[f64; 4]; 4]> = Vec::new();

    for (t, y) in stepper.x_out().iter().zip(stepper.y_out().iter()) {
        traj.push(Step2d { time: *t, x: y[0], y: y[1], vx: y[2], vy: y[3] });
        let mut phi = [[0.0_f64; 4]; 4];
        for col in 0..4_usize {
            let base = 4 + col * 4;
            for row in 0..4_usize { phi[row][col] = y[base + row]; }
        }
        stms.push(phi);
    }

    (traj, stms)
}

/// Integrate the planar CRTBP **backward** in time from `state0` for `t_back`
/// normalized time units (positive value).  Uses the time-reversed ODE so that
/// `ode_solvers` always sees a forward integration.
pub fn propagate_2d_backward(
    mu:      f64,
    state0:  [f64; 4],
    t_back:  f64,
    log_dt:  f64,
    rtol:    f64,
    atol:    f64,
) -> Vec<Step2d> {
    let y0 = State4::from_column_slice(&state0);
    let mut stepper = Dopri5::new(
        Crtbp2dOdeBack { mu },
        0.0, t_back.abs(), log_dt, y0, rtol, atol,
    );
    stepper.integrate().expect("CRTBP 2d backward integration failed");

    stepper.x_out().iter()
        .zip(stepper.y_out().iter())
        .map(|(&t, y)| Step2d { time: -t, x: y[0], y: y[1], vx: y[2], vy: y[3] })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3-D propagators
// ═══════════════════════════════════════════════════════════════════════════════
//
// State layout:  [x, y, z, vx, vy, vz]
//
// STM layout in 42-element vector (column-major, 6×6):
//   [0..6]   main state
//   [6..12]  STM col 0   (∂state/∂x₀)
//   [12..18] STM col 1   (∂state/∂y₀)
//   [18..24] STM col 2   (∂state/∂z₀)
//   [24..30] STM col 3   (∂state/∂vx₀)
//   [30..36] STM col 4   (∂state/∂vy₀)
//   [36..42] STM col 5   (∂state/∂vz₀)

/// 3-D CRTBP forward ODE.
pub struct Crtbp3dOde { pub mu: f64 }

impl System<f64, State6> for Crtbp3dOde {
    fn system(&self, _t: f64, y: &State6, dy: &mut State6) {
        let s = [y[0], y[1], y[2], y[3], y[4], y[5]];
        let d = eom_3d(self.mu, &s);
        for i in 0..6 { dy[i] = d[i]; }
    }
}

/// 3-D CRTBP backward ODE (negated RHS).
pub struct Crtbp3dOdeBack { pub mu: f64 }

impl System<f64, State6> for Crtbp3dOdeBack {
    fn system(&self, _t: f64, y: &State6, dy: &mut State6) {
        let s = [y[0], y[1], y[2], y[3], y[4], y[5]];
        let d = eom_3d(self.mu, &s);
        for i in 0..6 { dy[i] = -d[i]; }
    }
}

/// 3-D CRTBP + 6×6 STM ODE (42-state).
pub struct Crtbp3dStmOde { pub mu: f64 }

impl System<f64, State42> for Crtbp3dStmOde {
    fn system(&self, _t: f64, y: &State42, dy: &mut State42) {
        // ── main state ──────────────────────────────────────────────────────
        let s = [y[0], y[1], y[2], y[3], y[4], y[5]];
        let d = eom_3d(self.mu, &s);
        for i in 0..6 { dy[i] = d[i]; }

        // ── Jacobian A at current state ─────────────────────────────────────
        let a = jacobian_3d(self.mu, &s);

        // ── dΦ/dt = A·Φ  (column by column) ────────────────────────────────
        for col in 0..6_usize {
            let base = 6 + col * 6;
            let phi_col = [y[base], y[base+1], y[base+2], y[base+3], y[base+4], y[base+5]];
            for row in 0..6_usize {
                dy[base + row] = (0..6).map(|k| a[row][k] * phi_col[k]).sum();
            }
        }
    }
}

// ─── 3-D propagation helpers ──────────────────────────────────────────────────

/// Integrate the 3-D CRTBP from `state0` for `t_end` normalized time units.
pub fn propagate_3d(
    mu:      f64,
    state0:  [f64; 6],
    t_end:   f64,
    log_dt:  f64,
    rtol:    f64,
    atol:    f64,
) -> Vec<Step3d> {
    let y0 = State6::from_column_slice(&state0);
    let mut stepper = Dopri5::new(Crtbp3dOde { mu }, 0.0, t_end, log_dt, y0, rtol, atol);
    stepper.integrate().expect("CRTBP 3d integration failed");
    stepper.x_out().iter().zip(stepper.y_out().iter())
        .map(|(&t, y)| Step3d { time: t, x: y[0], y: y[1], z: y[2], vx: y[3], vy: y[4], vz: y[5] })
        .collect()
}

/// Integrate the 3-D CRTBP + 6×6 STM.  Returns trajectory + monodromy matrix.
pub fn propagate_3d_stm(
    mu:      f64,
    state0:  [f64; 6],
    t_end:   f64,
    rtol:    f64,
    atol:    f64,
) -> (Vec<Step3d>, [[f64; 6]; 6]) {
    let mut y0_arr = [0.0_f64; 42];
    y0_arr[0..6].copy_from_slice(&state0);
    for i in 0..6 { y0_arr[6 + i*6 + i] = 1.0; }  // identity STM

    let y0 = State42::from_column_slice(&y0_arr);
    let step_hint = t_end / 200.0;
    let mut stepper = Dopri5::from_param(
        Crtbp3dStmOde { mu },
        0.0, t_end, step_hint, y0, rtol, atol,
        0.9, 0.04, 0.333, 6.0, t_end, step_hint,
        N_MAX, 1000, OutputType::Sparse,
    );
    stepper.integrate().expect("CRTBP 3d+STM integration failed");

    let traj: Vec<Step3d> = stepper.x_out().iter().zip(stepper.y_out().iter())
        .map(|(&t, y)| Step3d { time: t, x: y[0], y: y[1], z: y[2], vx: y[3], vy: y[4], vz: y[5] })
        .collect();

    let yf = stepper.y_out().last().expect("no output states");
    let mut mono = [[0.0_f64; 6]; 6];
    for col in 0..6 {
        let base = 6 + col * 6;
        for row in 0..6 { mono[row][col] = yf[base + row]; }
    }
    (traj, mono)
}

/// Integrate the 3-D CRTBP + STM, logging the full STM at every output step.
pub fn propagate_3d_stm_full(
    mu:      f64,
    state0:  [f64; 6],
    t_end:   f64,
    log_dt:  f64,
    rtol:    f64,
    atol:    f64,
) -> Result<(Vec<Step3d>, Vec<[[f64; 6]; 6]>), String> {
    let mut y0_arr = [0.0_f64; 42];
    y0_arr[0..6].copy_from_slice(&state0);
    for i in 0..6 { y0_arr[6 + i*6 + i] = 1.0; }

    let y0 = State42::from_column_slice(&y0_arr);
    let mut stepper = Dopri5::from_param(
        Crtbp3dStmOde { mu },
        0.0, t_end, log_dt, y0, rtol, atol,
        0.9, 0.04, 0.333, 6.0, t_end, log_dt,
        N_MAX, 1000, OutputType::Sparse,
    );
    stepper.integrate().map_err(|e| format!("CRTBP 3d+STM integration failed: {e:?}"))?;

    let mut traj  = Vec::new();
    let mut stms: Vec<[[f64; 6]; 6]> = Vec::new();

    for (t, y) in stepper.x_out().iter().zip(stepper.y_out().iter()) {
        traj.push(Step3d { time: *t, x: y[0], y: y[1], z: y[2], vx: y[3], vy: y[4], vz: y[5] });
        let mut phi = [[0.0_f64; 6]; 6];
        for col in 0..6 {
            let base = 6 + col * 6;
            for row in 0..6 { phi[row][col] = y[base + row]; }
        }
        stms.push(phi);
    }
    Ok((traj, stms))
}

/// 3-D CRTBP ODE that stops via `solout` once x crosses `x_target`.
/// Used by `propagate_3d_until_x`.
pub struct Crtbp3dOdeStop {
    pub mu:       f64,
    pub x_target: f64,
    prev_x:       f64,
}

impl System<f64, State6> for Crtbp3dOdeStop {
    fn system(&self, _t: f64, y: &State6, dy: &mut State6) {
        let s = [y[0], y[1], y[2], y[3], y[4], y[5]];
        let d = eom_3d(self.mu, &s);
        for i in 0..6 { dy[i] = d[i]; }
    }

    /// Return false (stop) the first time x crosses x_target.
    fn solout(&mut self, _t: f64, y: &State6, _dy: &State6) -> bool {
        let x = y[0];
        let crossed = (self.prev_x - self.x_target) * (x - self.x_target) < 0.0;
        self.prev_x = x;
        !crossed
    }
}

/// Integrate the 3-D CRTBP from `state0`, stopping as soon as x crosses
/// `x_target` (or at `t_end` if never reached).  Returns the crossing time
/// and state interpolated to x = x_target, or `None` if there is no crossing.
///
/// Uses a coarser `log_dt` than the full propagator because only the crossing
/// state is needed — the intermediate trajectory is discarded immediately.
pub fn propagate_3d_until_x(
    mu:       f64,
    state0:   [f64; 6],
    x_target: f64,
    t_end:    f64,
    rtol:     f64,
    atol:     f64,
) -> Option<(f64, [f64; 6])> {
    let y0 = State6::from_column_slice(&state0);
    let ode = Crtbp3dOdeStop { mu, x_target, prev_x: state0[0] };
    // Coarse logging — we only need the two points that bracket the crossing.
    let log_dt = (t_end / 200.0).min(0.1);
    let mut stepper = Dopri5::new(ode, 0.0, t_end, log_dt, y0, rtol, atol);
    let _ = stepper.integrate();   // early stop returns error, which we ignore

    let t_out = stepper.x_out();
    let y_out = stepper.y_out();

    // Find the first pair of logged points that bracket x_target.
    for i in 1..y_out.len() {
        let x0 = y_out[i - 1][0];
        let x1 = y_out[i][0];
        if (x0 - x_target) * (x1 - x_target) <= 0.0 {
            let abs0   = (x0 - x_target).abs();
            let abs1   = (x1 - x_target).abs();
            let frac   = abs0 / (abs0 + abs1);
            let t_c    = t_out[i - 1] + frac * (t_out[i] - t_out[i - 1]);
            let lerp   = |a: f64, b: f64| a + frac * (b - a);
            return Some((t_c, [
                x_target,
                lerp(y_out[i-1][1], y_out[i][1]),
                lerp(y_out[i-1][2], y_out[i][2]),
                lerp(y_out[i-1][3], y_out[i][3]),
                lerp(y_out[i-1][4], y_out[i][4]),
                lerp(y_out[i-1][5], y_out[i][5]),
            ]));
        }
    }
    None
}

/// Integrate the 3-D CRTBP **backward** in time from `state0` for `t_back` (positive).
pub fn propagate_3d_backward(
    mu:      f64,
    state0:  [f64; 6],
    t_back:  f64,
    log_dt:  f64,
    rtol:    f64,
    atol:    f64,
) -> Vec<Step3d> {
    let y0 = State6::from_column_slice(&state0);
    let mut stepper = Dopri5::new(
        Crtbp3dOdeBack { mu }, 0.0, t_back.abs(), log_dt, y0, rtol, atol,
    );
    stepper.integrate().expect("CRTBP 3d backward integration failed");
    stepper.x_out().iter().zip(stepper.y_out().iter())
        .map(|(&t, y)| Step3d { time: -t, x: y[0], y: y[1], z: y[2], vx: y[3], vy: y[4], vz: y[5] })
        .collect()
}

// ─── Bicircular Restricted 4-Body Problem (BCR4BP) ───────────────────────────
//
// Earth-Moon CRTBP + Sun perturbation in the EM rotating frame.
// Sun orbits the EM barycenter at distance a_s with angular velocity omega_s.
//
// Perturbation accelerations (direct + indirect):
//   ax += -mu_s*(x - xs)/rs³  -  mu_s*xs/as³
//   ay += -mu_s*(y - ys)/rs³  -  mu_s*ys/as³
//   az += -mu_s*z/rs³
//
// Physical constants for Earth-Moon-Sun:
//   mu_s   = G·M_Sun / G·(M_E + M_M) ≈ 328 900
//   a_s    = 1 AU / L*              ≈ 389.2  nd
//   omega_s = T_EM/T_year - 1        ≈ -0.9252  (retrograde in EM frame)

/// Parameters for the Bicircular Restricted 4-Body Problem.
#[derive(Clone, Copy, Debug)]
pub struct Bcr4bpParams {
    /// Sun's gravitational parameter in EM-normalized units ≈ 328 900.
    pub mu_s:     f64,
    /// Sun–EM-barycenter distance in EM-normalized units ≈ 389.2.
    pub a_s:      f64,
    /// Sun's angular velocity in the EM rotating frame [nd/nd].
    /// Negative because the EM frame rotates faster than the Sun (≈ −0.9252).
    pub omega_s:  f64,
    /// Initial Sun phase angle in the EM rotating frame [rad].
    pub theta_s0: f64,
}

impl Bcr4bpParams {
    /// Standard Earth-Moon-Sun parameters with a given initial Sun phase.
    pub fn earth_moon_sun(theta_s0: f64) -> Self {
        let mu_em = MU_EARTH + MU_MOON;
        Self {
            mu_s:    MU_SUN / mu_em,
            a_s:     AU / EARTH_MOON_DISTANCE,
            omega_s: MOON_SIDEREAL_PERIOD_DAYS / JULIAN_YEAR_DAYS - 1.0,
            theta_s0,
        }
    }
}

/// BCR4BP equations of motion (Sun perturbation added to CRTBP EOM).
pub struct Bcr4bp3dOde {
    pub mu:     f64,
    pub params: Bcr4bpParams,
}

impl System<f64, State6> for Bcr4bp3dOde {
    fn system(&self, t: f64, y: &State6, dy: &mut State6) {
        let s  = [y[0], y[1], y[2], y[3], y[4], y[5]];
        let d  = eom_3d(self.mu, &s);
        let p  = &self.params;
        let th = p.omega_s * t + p.theta_s0;
        let xs = p.a_s * th.cos();
        let ys = p.a_s * th.sin();
        let dxs = s[0] - xs;
        let dys = s[1] - ys;
        let rs3 = (dxs*dxs + dys*dys + s[2]*s[2]).powf(1.5);
        let as3 = p.a_s * p.a_s * p.a_s;
        dy[0] = d[0];
        dy[1] = d[1];
        dy[2] = d[2];
        dy[3] = d[3] - p.mu_s * (dxs / rs3 + xs / as3);
        dy[4] = d[4] - p.mu_s * (dys / rs3 + ys / as3);
        dy[5] = d[5] - p.mu_s * s[2] / rs3;
    }
}

/// BCR4BP ODE that stops propagation the moment the spacecraft enters the Moon.
/// `solout` returns false (halt) when r_moon < r_moon_nd.
pub struct Bcr4bp3dOdeMoonStop {
    pub mu:        f64,
    pub params:    Bcr4bpParams,
    pub moon_x:    f64,   // 1.0 - mu
    pub r_moon_nd: f64,   // Moon radius in non-dimensional units
}

impl System<f64, State6> for Bcr4bp3dOdeMoonStop {
    fn system(&self, t: f64, y: &State6, dy: &mut State6) {
        let s  = [y[0], y[1], y[2], y[3], y[4], y[5]];
        let d  = eom_3d(self.mu, &s);
        let p  = &self.params;
        let th = p.omega_s * t + p.theta_s0;
        let xs = p.a_s * th.cos();
        let ys = p.a_s * th.sin();
        let dxs = s[0] - xs;
        let dys = s[1] - ys;
        let rs3 = (dxs*dxs + dys*dys + s[2]*s[2]).powf(1.5);
        let as3 = p.a_s * p.a_s * p.a_s;
        dy[0] = d[0];
        dy[1] = d[1];
        dy[2] = d[2];
        dy[3] = d[3] - p.mu_s * (dxs / rs3 + xs / as3);
        dy[4] = d[4] - p.mu_s * (dys / rs3 + ys / as3);
        dy[5] = d[5] - p.mu_s * s[2] / rs3;
    }

    /// Stop (return false) the first time the spacecraft enters the Moon.
    fn solout(&mut self, _t: f64, y: &State6, _dy: &State6) -> bool {
        let dx = y[0] - self.moon_x;
        let r2 = dx*dx + y[1]*y[1] + y[2]*y[2];
        r2 >= self.r_moon_nd * self.r_moon_nd
    }
}

/// Integrate BCR4BP forward, stopping early if the spacecraft hits the Moon.
/// Returns the trajectory up to (and including) the last step before surface entry.
pub fn propagate_bcr4bp_stop_moon(
    mu:        f64,
    params:    Bcr4bpParams,
    state0:    [f64; 6],
    t_end:     f64,
    log_dt:    f64,
    rtol:      f64,
    atol:      f64,
    r_moon_nd: f64,
) -> Vec<Step3d> {
    let moon_x = 1.0 - mu;
    let y0 = State6::from_column_slice(&state0);
    let mut stepper = Dopri5::new(
        Bcr4bp3dOdeMoonStop { mu, params, moon_x, r_moon_nd },
        0.0, t_end, log_dt, y0, rtol, atol,
    );
    let _ = stepper.integrate();
    stepper.x_out().iter().zip(stepper.y_out().iter())
        .map(|(&t, y)| Step3d { time: t, x: y[0], y: y[1], z: y[2],
                                vx: y[3], vy: y[4], vz: y[5] })
        .collect()
}

/// Integrate the BCR4BP logging every accepted internal step (OutputType::Sparse).
///
/// `h_max` caps the adaptive step so cruise segments don't become too coarse for
/// plotting.  A value of ~0.02 nd (≈ 1.7 h) works well for WSB trajectories.
pub fn propagate_bcr4bp_raw(
    mu:     f64,
    params: Bcr4bpParams,
    state0: [f64; 6],
    t_end:  f64,
    h_max:  f64,
    rtol:   f64,
    atol:   f64,
) -> Vec<Step3d> {
    use ode_solvers::dop_shared::OutputType;
    let y0 = State6::from_column_slice(&state0);
    let h_init = h_max.min(t_end / 10.0);
    let mut stepper = Dopri5::from_param(
        Bcr4bp3dOde { mu, params },
        0.0, t_end, h_init, y0, rtol, atol,
        0.9, 0.04, 0.333, 6.0, h_max, h_init,
        N_MAX, 1000, OutputType::Sparse,
    );
    let _ = stepper.integrate();
    stepper.x_out().iter().zip(stepper.y_out().iter())
        .map(|(&t, y)| Step3d { time: t, x: y[0], y: y[1], z: y[2], vx: y[3], vy: y[4], vz: y[5] })
        .collect()
}

/// Integrate the BCR4BP (EM CRTBP + Sun) forward from `state0`.
pub fn propagate_bcr4bp(
    mu:      f64,
    params:  Bcr4bpParams,
    state0:  [f64; 6],
    t_end:   f64,
    log_dt:  f64,
    rtol:    f64,
    atol:    f64,
) -> Vec<Step3d> {
    let y0 = State6::from_column_slice(&state0);
    let mut stepper = Dopri5::new(
        Bcr4bp3dOde { mu, params }, 0.0, t_end, log_dt, y0, rtol, atol,
    );
    let _ = stepper.integrate();
    stepper.x_out().iter().zip(stepper.y_out().iter())
        .map(|(&t, y)| Step3d { time: t, x: y[0], y: y[1], z: y[2], vx: y[3], vy: y[4], vz: y[5] })
        .collect()
}
