//! Deep Space Network (DSN) ground orbit-determination filter and uplink model.
//!
//! Shared by both the cruise and proximity-operations binaries so there is a
//! single implementation of the ground OD pipeline.
//!
//! Architecture (mirrors the real mission):
//!
//!   Raw DSN observables (simulated from truth)
//!     ├── Two-way range     [2 m 1-σ]
//!     ├── Range-rate/Doppler [0.1 mm/s 1-σ]
//!     └── Delta-DOR transverse angles [10 nrad ≈ 1500 m at 1 AU, daily]
//!            │
//!            ▼
//!   `GroundOdEkf::predict` + `update_range` / `update_rrate` / `update_ddor`
//!   (7-state EKF, heliocentric ecliptic J2000, full 7×7 STM)
//!            │
//!      `GroundOdEkf::uplink()` → `DsnUplink`
//!            │
//!   Spacecraft onboard EKF:
//!     ├── Cruise  : `GroundOdEkf::inject_uplink` (hard state reset, same heliocentric EKF)
//!     └── Proximity: `navigation::ekf::update_dsn` (measurement update in Hill frame)
//!
//! The Bennu ephemeris uncertainty (~5 km) enters only at the proximity uplink
//! step: converting the ground's heliocentric OD fix to a Hill-frame position
//! requires subtracting Bennu's nominal position, which is only known to ~5 km.
//! The cruise phase is unaffected — the spacecraft tracks its own heliocentric
//! position and the Bennu target is handled separately.

use nalgebra::Vector3;
use rand_distr::{Distribution, Normal};
use orbital_models::constants::{MU_SUN, AU, P_SRP};

// ── Public constants ──────────────────────────────────────────────────────────

/// Two-way range measurement noise 1-σ [m].
pub const RNG_NOISE_M: f64 = 2.0;
/// Range-rate (Doppler) measurement noise 1-σ [m/s].
pub const RRATE_NOISE_MPS: f64 = 1e-4;
/// Delta-DOR transverse angular noise 1-σ [rad]  (~10 nrad; OSIRIS-REx achieved ~3 nrad).
pub const DDOR_NOISE_RAD: f64 = 10e-9;

/// OD predict step size [s] — 1 hour.
pub const OD_STEP_S: f64 = 3_600.0;
/// Range / range-rate measurement interval [s] — every 2 OD steps = 2 hours.
pub const OD_MEAS_HZ: usize = 2;
/// Delta-DOR interval [OD steps] — once per 24 OD steps = daily.
pub const OD_DDOR_HZ: usize = 24;

// ── Internal types ────────────────────────────────────────────────────────────

type V3 = [f64; 3];
type S7 = [f64; 7];
type M7 = [[f64; 7]; 7];

// ── DSN uplink packet ─────────────────────────────────────────────────────────

/// Heliocentric state vector uplinked from ground OD to the spacecraft.
///
/// In the proximity phase this is converted to Hill-frame by subtracting
/// Bennu's nominal heliocentric position; the Bennu ephemeris uncertainty
/// (~5 km) is the dominant noise source at that conversion step.
#[derive(Clone, Debug)]
pub struct DsnUplink {
    /// OD heliocentric spacecraft position estimate [m, ecliptic J2000].
    pub r_helio_m:   Vector3<f64>,
    /// OD heliocentric spacecraft velocity estimate [m/s].
    pub v_helio_mps: Vector3<f64>,
    /// 1-σ position accuracy of the OD solution [m].
    pub sigma_r_m:   f64,
    /// 1-σ velocity accuracy of the OD solution [m/s].
    pub sigma_v_mps: f64,
}

// ── Ground OD EKF ─────────────────────────────────────────────────────────────

/// 7-state heliocentric ground orbit-determination EKF.
///
/// State: **x** = [r_x, r_y, r_z, v_x, v_y, v_z, C_R]
/// Covariance: **P** (7×7)
///
/// All quantities in SI (m, m/s) and ecliptic J2000.
///
/// The C_R column of the STM is computed analytically because finite-difference
/// computation has catastrophic cancellation (∂r/∂C_R ~ 1 m over 1 h against
/// r ~ 1.5×10¹¹ m, below double-precision noise for any reasonable step size).
pub struct GroundOdEkf {
    x:              S7,
    p:              M7,
    /// SC effective area / mass ratio for SRP [m²/kg].
    /// Set at construction so the same struct handles cruise and proximity configurations.
    area_over_mass: f64,
}

impl GroundOdEkf {
    /// Construct with known initial state and diagonal initial covariance.
    pub fn new(
        r:              Vector3<f64>,
        v:              Vector3<f64>,
        cr:             f64,
        sig_r:          f64,
        sig_v:          f64,
        sig_cr:         f64,
        area_over_mass: f64,
    ) -> Self {
        let x = [r[0], r[1], r[2], v[0], v[1], v[2], cr];
        let p = diag7([
            sig_r*sig_r, sig_r*sig_r, sig_r*sig_r,
            sig_v*sig_v, sig_v*sig_v, sig_v*sig_v,
            sig_cr*sig_cr,
        ]);
        Self { x, p, area_over_mass }
    }

    /// Current heliocentric position estimate [m].
    pub fn r(&self) -> Vector3<f64> { Vector3::new(self.x[0], self.x[1], self.x[2]) }
    /// Current heliocentric velocity estimate [m/s].
    pub fn v(&self) -> Vector3<f64> { Vector3::new(self.x[3], self.x[4], self.x[5]) }
    /// Current C_R estimate.
    pub fn cr(&self) -> f64 { self.x[6] }
    /// Position 1-σ from covariance diagonal [m] — RMS of individual axes.
    pub fn sigma_r_m(&self) -> f64 { ((self.p[0][0]+self.p[1][1]+self.p[2][2])/3.0).max(0.0).sqrt() }
    /// Velocity 1-σ [m/s].
    pub fn sigma_v_mps(&self) -> f64 { ((self.p[3][3]+self.p[4][4]+self.p[5][5])/3.0).max(0.0).sqrt() }

    /// Build a `DsnUplink` packet from the current OD state (ready for uplink to spacecraft).
    pub fn uplink(&self) -> DsnUplink {
        DsnUplink {
            r_helio_m:   self.r(),
            v_helio_mps: self.v(),
            sigma_r_m:   self.sigma_r_m(),
            sigma_v_mps: self.sigma_v_mps(),
        }
    }

    // ── Predict ───────────────────────────────────────────────────────────────

    /// Full STM propagation — used by the **ground OD** filter.
    ///
    /// Covariance is propagated via the 7×7 state transition matrix; process
    /// noise represents unmodelled accelerations and C_R drift.
    pub fn predict(&mut self, dt: f64) {
        let phi = stm7(self.x, dt, self.area_over_mass);
        self.x  = propagate7(self.x, dt, OD_STEP_S.min(dt), self.area_over_mass);

        let sc = dt / 86_400.0;
        let mut q = zero7();
        for i in 0..3 { q[i][i] = 1e2  * sc; }  // σ_r ~ 10 m/day
        for i in 3..6 { q[i][i] = 1e-8 * sc; }  // σ_v ~ 0.1 µm/s/day
        q[6][6]        = 1e-7  * sc;              // C_R slow drift

        self.p = m7_add(m7_mul(m7_mul(phi, self.p), m7_t(phi)), q);
    }

    /// Diagonal process-noise-only propagation — used by the **onboard EKF** in
    /// the cruise phase, where daily uplinks make full STM propagation unnecessary.
    pub fn predict_simple(&mut self, dt: f64) {
        self.x = propagate7(self.x, dt, OD_STEP_S.min(dt), self.area_over_mass);
        let sc = dt / 86_400.0;
        for i in 0..3 { self.p[i][i] += 1e3  * sc; } // σ_r ~ 32 m/day
        for i in 3..6 { self.p[i][i] += 1e-8 * sc; }
        self.p[6][6]              += 1e-8 * sc;
    }

    // ── Measurement updates ───────────────────────────────────────────────────

    /// Two-way range update: `z = |r_sc − r_earth| + noise`.
    pub fn update_range(&mut self, r_earth: &Vector3<f64>, z: f64) {
        let re = v3(r_earth);
        let dr = sub(rv3(self.x), re);
        let rng = norm(dr);
        if rng < 1.0 { return; }
        let hr = scale(dr, 1.0/rng);
        let h  = [hr[0], hr[1], hr[2], 0.0, 0.0, 0.0, 0.0];
        let inn = z - rng;
        scalar_update(&mut self.x, &mut self.p, h, inn, RNG_NOISE_M * RNG_NOISE_M);
        self.clamp_cr();
    }

    /// Range-rate (Doppler) update: `z = dr⃗·dv⃗ / |dr⃗| + noise`.
    pub fn update_rrate(
        &mut self,
        r_earth: &Vector3<f64>,
        v_earth: &Vector3<f64>,
        z: f64,
    ) {
        let re = v3(r_earth);
        let ve = v3(v_earth);
        let dr = sub(rv3(self.x), re);
        let dv = sub(vv3(self.x), ve);
        let rng = norm(dr);
        if rng < 1.0 { return; }
        let rdotv = dot(dr, dv);
        let h = [
            dv[0]/rng - rdotv*dr[0]/rng.powi(3),
            dv[1]/rng - rdotv*dr[1]/rng.powi(3),
            dv[2]/rng - rdotv*dr[2]/rng.powi(3),
            dr[0]/rng, dr[1]/rng, dr[2]/rng, 0.0,
        ];
        let inn = z - rdotv/rng;
        scalar_update(&mut self.x, &mut self.p, h, inn, RRATE_NOISE_MPS * RRATE_NOISE_MPS);
        self.clamp_cr();
    }

    /// Delta-DOR transverse angular update (two orthogonal measurements per pass).
    ///
    /// The two transverse unit vectors t1, t2 are fixed to the OD's current
    /// line-of-sight so the predicted measurement is identically zero and the
    /// innovation equals the true transverse angle directly.
    pub fn update_ddor(&mut self, r_earth: &Vector3<f64>, z1: f64, z2: f64) {
        let re = v3(r_earth);
        let dr  = sub(rv3(self.x), re);
        let rng = norm(dr);
        if rng < 1.0 { return; }
        let rhat = scale(dr, 1.0/rng);
        let up: V3 = if rhat[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        let t1 = normalize(cross(rhat, up));
        let t2 = cross(rhat, t1);
        let var = DDOR_NOISE_RAD * DDOR_NOISE_RAD;
        let h1: S7 = [t1[0]/rng, t1[1]/rng, t1[2]/rng, 0.0, 0.0, 0.0, 0.0];
        let h2: S7 = [t2[0]/rng, t2[1]/rng, t2[2]/rng, 0.0, 0.0, 0.0, 0.0];
        scalar_update(&mut self.x, &mut self.p, h1, z1, var);
        scalar_update(&mut self.x, &mut self.p, h2, z2, var);
        self.clamp_cr();
    }

    // ── Convenience: simulate raw observables and update in one call ──────────

    /// Simulate raw DSN observables from truth and immediately update the OD
    /// filter.  Provides the same interface for both cruise and proximity loops.
    ///
    /// `do_ddor`: pass `true` on the steps where the daily Delta-DOR measurement
    /// is available (typically once every `OD_DDOR_HZ` OD steps).
    pub fn update_from_truth<R: rand::Rng>(
        &mut self,
        r_sc_true: &Vector3<f64>,
        v_sc_true: &Vector3<f64>,
        r_earth:   &Vector3<f64>,
        v_earth:   &Vector3<f64>,
        do_ddor:   bool,
        rng:       &mut R,
    ) {
        let re = v3(r_earth);
        let ve = v3(v_earth);
        let dr_true = sub(v3(r_sc_true), re);
        let dv_true = sub(v3(v_sc_true), ve);
        let rng_true   = norm(dr_true);
        let rrate_true = dot(dr_true, dv_true) / rng_true;

        let nr = Normal::new(0.0_f64, RNG_NOISE_M).unwrap();
        let nv = Normal::new(0.0_f64, RRATE_NOISE_MPS).unwrap();

        self.update_range(r_earth, rng_true   + nr.sample(rng));
        self.update_rrate(r_earth, v_earth, rrate_true + nv.sample(rng));

        if do_ddor {
            let (z1, z2) = sim_ddor_inner(&dr_true, rng_true, &rv3(self.x), re, rng);
            self.update_ddor(r_earth, z1, z2);
        }
    }

    // ── Uplink handling ───────────────────────────────────────────────────────

    /// Add a commanded ΔV directly to the filter state velocity.
    ///
    /// Called after each burn step so the filter tracks the known commanded
    /// velocity change without waiting for post-burn ranging reconstruction.
    pub fn add_dv(&mut self, dv: &Vector3<f64>) {
        self.x[3] += dv[0];
        self.x[4] += dv[1];
        self.x[5] += dv[2];
    }

    /// Hard state injection used by the **cruise onboard EKF** when it receives
    /// a ground uplink.  Resets the r,v block of P to the uplink accuracy;
    /// C_R and its covariances are deliberately preserved (not uplinked).
    pub fn inject_uplink(&mut self, uplink: &DsnUplink, sigma_r: f64, sigma_v: f64) {
        for i in 0..3 {
            self.x[i]   = uplink.r_helio_m[i];
            self.x[3+i] = uplink.v_helio_mps[i];
        }
        let sr2 = sigma_r * sigma_r;
        let sv2 = sigma_v * sigma_v;
        for i in 0..6 { for j in 0..6 { self.p[i][j] = 0.0; } }
        for i in 0..3 { self.p[i][i] = sr2; }
        for i in 3..6 { self.p[i][i] = sv2; }
    }

    // ── C_R management ────────────────────────────────────────────────────────

    /// Freeze C_R after a burn and inflate velocity uncertainty by `sigma_dv`.
    ///
    /// Post-burn, the execution error (unknown ΔV residual) must be reconstructed
    /// from ranging before C_R estimation can resume.  Setting P[6,6] → 0 drives
    /// the Kalman gain K[6] → 0, preventing the filter from absorbing execution
    /// errors into the reflectivity estimate.
    pub fn freeze_cr_for_burn_recon(&mut self, sigma_dv: f64) {
        let dv2 = sigma_dv * sigma_dv;
        for i in 3..6 { if self.p[i][i] < dv2 { self.p[i][i] = dv2; } }
        self.p[6][6] = 1e-10;
        for i in 0..3 { for j in 3..7 { self.p[i][j] = 0.0; self.p[j][i] = 0.0; } }
        for i in 3..6 { self.p[i][6] = 0.0; self.p[6][i] = 0.0; }
    }

    /// Re-open C_R estimation after burn reconstruction is complete.
    pub fn thaw_cr_for_reestimation(&mut self, sigma_cr: f64) {
        self.p[6][6] = sigma_cr * sigma_cr;
    }

    fn clamp_cr(&mut self) {
        if self.x[6] < 0.5 { self.x[6] = 0.5; }
        if self.x[6] > 2.5 { self.x[6] = 2.5; }
    }
}

// ── Earth ephemeris ───────────────────────────────────────────────────────────

/// Earth heliocentric state (circular orbit approximation) at `t` seconds
/// from J2000 [m, m/s, ecliptic J2000].
pub fn earth_rv(t_s: f64) -> (Vector3<f64>, Vector3<f64>) {
    let r_e  = AU;
    let v_e  = (MU_SUN / r_e).sqrt();
    let om_e = v_e / r_e;
    let th   = om_e * t_s;
    (
        Vector3::new(r_e * th.cos(), r_e * th.sin(), 0.0),
        Vector3::new(v_e * (-th.sin()), v_e * th.cos(), 0.0),
    )
}

// ── Private: Delta-DOR observable simulation ─────────────────────────────────

fn sim_ddor_inner<R: rand::Rng>(
    dr_true: &V3,
    rng_true: f64,
    r_od:    &V3,
    r_earth: V3,
    rng:     &mut R,
) -> (f64, f64) {
    let nd = Normal::new(0.0_f64, DDOR_NOISE_RAD).unwrap();
    // Use OD's line-of-sight as the baseline for the transverse frame.
    let dr_od  = sub(*r_od, r_earth);
    let rhat_od = normalize(dr_od);
    let up: V3 = if rhat_od[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let t1 = normalize(cross(rhat_od, up));
    let t2 = cross(rhat_od, t1);
    let z1 = dot(*dr_true, t1) / rng_true + nd.sample(rng);
    let z2 = dot(*dr_true, t2) / rng_true + nd.sample(rng);
    (z1, z2)
}

// ── Private: 7-state heliocentric dynamics ────────────────────────────────────

fn srp_acc_aom(r: V3, c_r: f64, area_over_mass: f64) -> V3 {
    let r_au = norm(r) / AU;
    scale(normalize(r), P_SRP * c_r * area_over_mass / (r_au * r_au))
}

fn deriv7(x: S7, aom: f64) -> S7 {
    let r = [x[0], x[1], x[2]];
    let v = [x[3], x[4], x[5]];
    let cr = x[6];
    let rn = norm(r);
    let ag = scale(r, -MU_SUN / rn.powi(3));
    let a  = addv(ag, srp_acc_aom(r, cr, aom));
    [v[0], v[1], v[2], a[0], a[1], a[2], 0.0]
}

fn rk4_7_step(x: S7, dt: f64, aom: f64) -> S7 {
    let k1 = deriv7(x, aom);
    let k2 = deriv7(s7_add(x, s7_scale(k1, dt*0.5)), aom);
    let k3 = deriv7(s7_add(x, s7_scale(k2, dt*0.5)), aom);
    let k4 = deriv7(s7_add(x, s7_scale(k3, dt)), aom);
    s7_add(x, s7_scale(
        s7_add(s7_add(s7_add(k1, s7_scale(k2, 2.0)), s7_scale(k3, 2.0)), k4),
        dt/6.0,
    ))
}

fn propagate7(x0: S7, total: f64, step: f64, aom: f64) -> S7 {
    let n  = (total / step).ceil() as usize;
    let dt = total / n as f64;
    let mut x = x0;
    for _ in 0..n { x = rk4_7_step(x, dt, aom); }
    x
}

/// 7×7 state transition matrix via central finite differences (r, v columns)
/// and analytical formula (C_R column).
fn stm7(x0: S7, dt: f64, aom: f64) -> M7 {
    let mut phi = diag7([1.0f64; 7]);
    let step = OD_STEP_S.min(dt);

    for j in 0..3 {
        let h = 1e3_f64;
        let mut xp = x0; xp[j] += h;
        let mut xm = x0; xm[j] -= h;
        let yp = propagate7(xp, dt, step, aom);
        let ym = propagate7(xm, dt, step, aom);
        for i in 0..7 { phi[i][j] = (yp[i] - ym[i]) / (2.0 * h); }
    }
    for j in 3..6 {
        let h = 0.01_f64;
        let mut xp = x0; xp[j] += h;
        let mut xm = x0; xm[j] -= h;
        let yp = propagate7(xp, dt, step, aom);
        let ym = propagate7(xm, dt, step, aom);
        for i in 0..7 { phi[i][j] = (yp[i] - ym[i]) / (2.0 * h); }
    }

    // C_R column: analytical (FD has catastrophic cancellation at heliocentric scales).
    let r     = [x0[0], x0[1], x0[2]];
    let r_au  = norm(r) / AU;
    let srp_base = P_SRP * aom / (r_au * r_au);
    let rhat  = normalize(r);
    for i in 0..3 {
        phi[i][6]   = 0.5 * srp_base * rhat[i] * dt * dt;
        phi[3+i][6] =       srp_base * rhat[i] * dt;
    }
    phi
}

// ── Private: scalar EKF update ────────────────────────────────────────────────

fn scalar_update(x: &mut S7, p: &mut M7, h: S7, inn: f64, r_var: f64) {
    let mut ph = [0.0f64; 7];
    for i in 0..7 { for j in 0..7 { ph[i] += p[i][j] * h[j]; } }
    let s: f64 = h.iter().zip(ph.iter()).map(|(a, b)| a * b).sum::<f64>() + r_var;
    if s.abs() < 1e-30 { return; }
    let k: S7 = ph.map(|v| v / s);
    for i in 0..7 { x[i] += k[i] * inn; }
    // Joseph form for numerical stability
    let mut kh = zero7();
    for i in 0..7 { for j in 0..7 { kh[i][j] = k[i] * h[j]; } }
    let mut ikh = diag7([1.0; 7]);
    for i in 0..7 { for j in 0..7 { ikh[i][j] -= kh[i][j]; } }
    let ikht = m7_t(ikh);
    let mut krkt = zero7();
    for i in 0..7 { for j in 0..7 { krkt[i][j] = k[i] * k[j] * r_var; } }
    *p = m7_add(m7_mul(m7_mul(ikh, *p), ikht), krkt);
}

// ── Private: V3/S7/M7 helpers ─────────────────────────────────────────────────

#[inline] fn dot(a: V3, b: V3) -> f64 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
#[inline] fn norm(a: V3) -> f64 { dot(a, a).sqrt() }
#[inline] fn sub(a: V3, b: V3) -> V3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
#[inline] fn addv(a: V3, b: V3) -> V3 { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
#[inline] fn scale(a: V3, s: f64) -> V3 { [a[0]*s, a[1]*s, a[2]*s] }
#[inline] fn normalize(a: V3) -> V3 { scale(a, 1.0/norm(a)) }
#[inline] fn cross(a: V3, b: V3) -> V3 {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
#[inline] fn s7_add(a: S7, b: S7) -> S7 {
    [a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]+b[3],a[4]+b[4],a[5]+b[5],a[6]+b[6]]
}
#[inline] fn s7_scale(a: S7, s: f64) -> S7 {
    [a[0]*s,a[1]*s,a[2]*s,a[3]*s,a[4]*s,a[5]*s,a[6]*s]
}
/// Extract position sub-vector from a 7-state vector.
#[inline] fn rv3(x: S7) -> V3 { [x[0], x[1], x[2]] }
/// Extract velocity sub-vector from a 7-state vector.
#[inline] fn vv3(x: S7) -> V3 { [x[3], x[4], x[5]] }
/// Convert nalgebra Vector3 to raw V3.
#[inline] fn v3(v: &Vector3<f64>) -> V3 { [v[0], v[1], v[2]] }

fn zero7() -> M7 { [[0.0; 7]; 7] }
fn diag7(d: S7) -> M7 { let mut m = zero7(); for i in 0..7 { m[i][i] = d[i]; } m }
fn m7_add(a: M7, b: M7) -> M7 {
    let mut c = zero7();
    for i in 0..7 { for j in 0..7 { c[i][j] = a[i][j] + b[i][j]; } }
    c
}
fn m7_mul(a: M7, b: M7) -> M7 {
    let mut c = zero7();
    for i in 0..7 { for j in 0..7 { for k in 0..7 { c[i][j] += a[i][k] * b[k][j]; } } }
    c
}
fn m7_t(a: M7) -> M7 {
    let mut c = zero7();
    for i in 0..7 { for j in 0..7 { c[i][j] = a[j][i]; } }
    c
}
