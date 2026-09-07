//! CRTBP (Circular Restricted Three-Body Problem) in normalized units.
//!
//! # Coordinate system (rotating frame)
//! - Primary   (Earth): position (-μ, 0)
//! - Secondary (Moon):  position (1-μ, 0)
//!
//! # Normalized units
//! - Length    L* = Earth-Moon mean distance ≈ 384 400 km
//! - Time      T* = 1 / n  where  n = mean_motion = sqrt((μ_E+μ_M)/L*³)
//! - Velocity  V* = L* / T*

use orbital_models::constants::{MU_EARTH, MU_MOON, EARTH_MOON_DISTANCE};

// ─── Physical scaling ─────────────────────────────────────────────────────────

/// Dimensional scaling constants for the Earth-Moon CRTBP.
#[derive(Clone, Debug)]
pub struct CrtbpParams {
    /// Mass parameter  μ = M_moon / (M_earth + M_moon)
    pub mu: f64,
    /// Characteristic length L* [m]  (Earth-Moon mean distance)
    pub l_star: f64,
    /// Characteristic time   T* [s]  (1 / mean_motion)
    pub t_star: f64,
    /// Characteristic velocity V* [m/s]
    pub v_star: f64,
}

impl CrtbpParams {
    pub fn earth_moon() -> Self {
        let mu_total = MU_EARTH + MU_MOON;
        let t_star   = (EARTH_MOON_DISTANCE.powi(3) / mu_total).sqrt();
        Self {
            mu:     MU_MOON / mu_total,
            l_star: EARTH_MOON_DISTANCE,
            t_star,
            v_star: EARTH_MOON_DISTANCE / t_star,
        }
    }

    /// Normalized → dimensional position [m]
    pub fn dim_pos(&self, x_nd: f64) -> f64 { x_nd * self.l_star }
    /// Dimensional → normalized position
    pub fn nd_pos(&self, x_m: f64)  -> f64 { x_m  / self.l_star }
    /// Normalized → dimensional velocity [m/s]
    pub fn dim_vel(&self, v_nd: f64) -> f64 { v_nd * self.v_star }
    /// Dimensional → normalized velocity
    pub fn nd_vel(&self, v_ms: f64)  -> f64 { v_ms  / self.v_star }
    /// Normalized → dimensional time [s]
    pub fn dim_time(&self, t_nd: f64) -> f64 { t_nd * self.t_star }
    /// Normalized → dimensional time [days]
    pub fn dim_time_days(&self, t_nd: f64) -> f64 { t_nd * self.t_star / 86_400.0 }
}

// ─── Equations of motion (planar, 4-state) ───────────────────────────────────

/// Planar CRTBP equations of motion.
///
/// `state` = [x, y, vx, vy] in normalized units.
///
/// Returns [ẋ, ẏ, ẍ, ÿ].
///
/// Equivalent to `EoM_SS_circular.m` / `EoM_SS_theta.m` with β=0 (no sail)
/// and e=0 (circular, so 1/(1+e·cosθ) = 1).  Primary is Earth at (−μ, 0),
/// secondary is Moon at (1−μ, 0); adapt from the star/planet MATLAB convention
/// by swapping body roles: m1=Earth (1−μ), m2=Moon (μ).
///
/// MATLAB naming correspondence:
///   R1  = sqrt((x+μ)²+y²)   — distance to Earth (primary)
///   R2  = sqrt((x−(1−μ))²+y²) — distance to Moon (secondary)
///   r1n3 = R1³, r2n3 = R2³
#[inline]
pub fn eom_2d(mu: f64, state: &[f64; 4]) -> [f64; 4] {
    let (x, y, vx, vy) = (state[0], state[1], state[2], state[3]);

    // R1, R2: distances to Earth and Moon (cf. MATLAB r1, r2)
    let r1_sq = (x + mu).powi(2) + y * y;          // R1²
    let r2_sq = (x - (1.0 - mu)).powi(2) + y * y;  // R2²
    let r1_3  = r1_sq.powf(1.5);                    // R1³  (MATLAB: r1n3)
    let r2_3  = r2_sq.powf(1.5);                    // R2³  (MATLAB: r2n3)

    // x+μ  = x-component of r⃗_1 (Earth→spacecraft)
    // x−(1−μ) = x-component of r⃗_2 (Moon→spacecraft)
    let ax = 2.0 * vy + x
        - (1.0 - mu) * (x + mu)       / r1_3
        - mu         * (x - (1.0-mu)) / r2_3;
    let ay = -2.0 * vx + y
        - (1.0 - mu) * y / r1_3
        - mu         * y / r2_3;

    [vx, vy, ax, ay]
}

// ─── Jacobi constant ─────────────────────────────────────────────────────────

/// Jacobi constant  C = 2Ω − v²  (conserved quantity of the CRTBP).
pub fn jacobi_constant(mu: f64, state: &[f64; 4]) -> f64 {
    let (x, y, vx, vy) = (state[0], state[1], state[2], state[3]);
    let r1 = ((x + mu).powi(2)        + y * y).sqrt();
    let r2 = ((x - (1.0-mu)).powi(2)  + y * y).sqrt();
    let omega = 0.5 * (x*x + y*y) + (1.0-mu)/r1 + mu/r2;
    2.0 * omega - (vx*vx + vy*vy)
}

// ─── Lagrange points ─────────────────────────────────────────────────────────

/// x-coordinate of collinear Lagrange point L1 or L2 via Newton's method.
///
/// `point`: `1` → L1 (between Earth and Moon), `2` → L2 (beyond Moon).
///
/// Matches `lagrangeFind.m`: Newton iteration on
///   eq  = x − (1−μ)(x+μ)/R1³ − μ(x−(1−μ))/R2³  (y=0 on the x-axis)
///   der = 1 + 2(1−μ)/R1³ + 2μ/R2³               (simplified at y=0)
pub fn lagrange_x(mu: f64, point: u8) -> f64 {
    let gamma0 = (mu / 3.0_f64).cbrt();
    let mut x = match point {
        1 => 1.0 - mu - gamma0,
        2 => 1.0 - mu + gamma0,
        _ => panic!("only L1 (1) and L2 (2) are supported"),
    };

    for _ in 0..60 {
        let d1  = x + mu;           // x-component of r⃗_1 (MATLAB: mu + x)
        let d2  = x - (1.0 - mu);  // x-component of r⃗_2 (MATLAB: x - (1-mu))
        let r1_3 = (d1 * d1).powf(1.5);  // R1³  (MATLAB: r1n3)
        let r2_3 = (d2 * d2).powf(1.5);  // R2³  (MATLAB: r2n3)

        let eq  = x - (1.0-mu)*d1/r1_3 - mu*d2/r2_3;  // MATLAB: eq
        let der = 1.0 + 2.0*(1.0-mu)/r1_3 + 2.0*mu/r2_3;  // MATLAB: der

        let dx = -eq / der;
        x += dx;
        if dx.abs() < 1e-14 { break; }
    }
    x
}

// ─── Linearised dynamics at a collinear Lagrange point ───────────────────────

/// Second partial derivatives of the effective potential Ω at (x_l, 0).
///
/// Returns (Ωxx, Ωyy).  Ωxy = 0 by symmetry at y = 0.
pub fn omega_partials(mu: f64, x_l: f64) -> (f64, f64) {
    let d1 = x_l + mu;
    let d2 = x_l - (1.0 - mu);
    let r1_3 = (d1*d1).powf(1.5);
    let r2_3 = (d2*d2).powf(1.5);
    let r1_5 = (d1*d1).powf(2.5);
    let r2_5 = (d2*d2).powf(2.5);

    let oxx = 1.0
        - (1.0-mu)/r1_3 + 3.0*(1.0-mu)*d1*d1/r1_5
        - mu/r2_3       + 3.0*mu*d2*d2/r2_5;
    // At y=0: Ωyy = 1 - (1-μ)/r1³ - μ/r2³
    let oyy = 1.0 - (1.0-mu)/r1_3 - mu/r2_3;

    (oxx, oyy)
}

/// Compute the linearised Jacobian A at state (x_l, y, vx, vy).
///
/// A is the 4×4 matrix such that  d(δstate)/dt = A · δstate.
pub fn jacobian_2d(mu: f64, state: &[f64; 4]) -> [[f64; 4]; 4] {
    let (x, y, vx, vy) = (state[0], state[1], state[2], state[3]);
    let _ = (vx, vy); // only position enters A

    let d1 = x + mu;
    let d2 = x - (1.0 - mu);
    let r1_sq = d1*d1 + y*y;
    let r2_sq = d2*d2 + y*y;
    let r1_3 = r1_sq.powf(1.5);
    let r2_3 = r2_sq.powf(1.5);
    let r1_5 = r1_sq.powf(2.5);
    let r2_5 = r2_sq.powf(2.5);

    let oxx = 1.0
        - (1.0-mu)/r1_3 + 3.0*(1.0-mu)*d1*d1/r1_5
        - mu/r2_3       + 3.0*mu*d2*d2/r2_5;
    let oyy = 1.0
        - (1.0-mu)/r1_3 + 3.0*(1.0-mu)*y*y/r1_5
        - mu/r2_3       + 3.0*mu*y*y/r2_5;
    let oxy = 3.0*(1.0-mu)*d1*y/r1_5 + 3.0*mu*d2*y/r2_5;

    // rows: [x, y, vx, vy]
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [oxx, oxy, 0.0, 2.0],
        [oxy, oyy,-2.0, 0.0],
    ]
}

/// In-plane Lyapunov frequency ω at a collinear Lagrange point.
///
/// Solves the characteristic equation of the linearised in-plane dynamics:
/// λ⁴ + (4 − Ωxx − Ωyy)λ² + Ωxx·Ωyy = 0
/// The imaginary root ±iω is the center eigenvalue.
pub fn lyapunov_frequency(mu: f64, x_l: f64) -> f64 {
    let (oxx, oyy) = omega_partials(mu, x_l);
    // Quadratic in λ²:  λ⁴ + b·λ² + c = 0
    let b    = 4.0 - oxx - oyy;
    let c    = oxx * oyy;
    let disc = b*b - 4.0*c;
    // One root λ² < 0 → imaginary pair ±iω
    let lam_sq_neg = (-b - disc.sqrt()) / 2.0; // the negative root
    (-lam_sq_neg).sqrt()
}

/// Linearised initial vy for a Lyapunov orbit of x-amplitude `ax` around `x_l`.
///
/// Derived from the center eigenvector of the linearised flow.
pub fn lyapunov_vy0_approx(mu: f64, x_l: f64, ax: f64) -> f64 {
    let (oxx, _) = omega_partials(mu, x_l);
    let omega    = lyapunov_frequency(mu, x_l);
    // vy0 = -ax · (ω² + Ωxx) / 2
    -ax * (omega * omega + oxx) / 2.0
}

// ─── 3-D equations of motion ─────────────────────────────────────────────────

/// 3-D planar CRTBP equations of motion.
///
/// `state` = [x, y, z, vx, vy, vz] in normalized units.
/// Returns [ẋ, ẏ, ż, ẍ, ÿ, z̈].
///
/// The out-of-plane equation is:  z̈ = −(1−μ)z/R1³ − μz/R2³
/// (z decouples from xy for a planar orbit; R1, R2 now include z).
#[inline]
pub fn eom_3d(mu: f64, state: &[f64; 6]) -> [f64; 6] {
    let (x, y, z)    = (state[0], state[1], state[2]);
    let (vx, vy, vz) = (state[3], state[4], state[5]);

    let r1 = ((x + mu).powi(2) + y*y + z*z).sqrt();   // R1
    let r2 = ((x - (1.0-mu)).powi(2) + y*y + z*z).sqrt(); // R2
    let r1_3 = r1.powi(3);  // R1³
    let r2_3 = r2.powi(3);  // R2³

    let ax = 2.0*vy + x - (1.0-mu)*(x+mu)/r1_3 - mu*(x-(1.0-mu))/r2_3;
    let ay = -2.0*vx + y - (1.0-mu)*y/r1_3 - mu*y/r2_3;
    let az = -(1.0-mu)*z/r1_3 - mu*z/r2_3;

    [vx, vy, vz, ax, ay, az]
}

/// 3-D linearised Jacobian A at `state`.
///
/// State ordering: [x, y, z, vx, vy, vz].
/// A is the 6×6 matrix  d(δstate)/dt = A · δstate.
///
/// Note: Ωzz has no "+1" term (the centrifugal potential (x²+y²)/2 has no z part).
pub fn jacobian_3d(mu: f64, state: &[f64; 6]) -> [[f64; 6]; 6] {
    let (x, y, z) = (state[0], state[1], state[2]);

    let d1   = x + mu;
    let d2   = x - (1.0 - mu);
    let r1sq = d1*d1 + y*y + z*z;
    let r2sq = d2*d2 + y*y + z*z;
    let r1_3 = r1sq.powf(1.5);
    let r2_3 = r2sq.powf(1.5);
    let r1_5 = r1sq.powf(2.5);
    let r2_5 = r2sq.powf(2.5);

    let oxx = 1.0
        - (1.0-mu)/r1_3 + 3.0*(1.0-mu)*d1*d1/r1_5
        - mu/r2_3       + 3.0*mu*d2*d2/r2_5;
    let oyy = 1.0
        - (1.0-mu)/r1_3 + 3.0*(1.0-mu)*y*y/r1_5
        - mu/r2_3       + 3.0*mu*y*y/r2_5;
    let ozz =            // no "+1": centrifugal term is only in xy
        - (1.0-mu)/r1_3 + 3.0*(1.0-mu)*z*z/r1_5
        - mu/r2_3       + 3.0*mu*z*z/r2_5;
    let oxy = 3.0*(1.0-mu)*d1*y/r1_5 + 3.0*mu*d2*y/r2_5;
    let oxz = 3.0*(1.0-mu)*d1*z/r1_5 + 3.0*mu*d2*z/r2_5;
    let oyz = 3.0*(1.0-mu)*y*z/r1_5  + 3.0*mu*y*z/r2_5;

    // rows: [x, y, z, vx, vy, vz]
    [
        [0.0, 0.0, 0.0,  1.0,  0.0, 0.0],
        [0.0, 0.0, 0.0,  0.0,  1.0, 0.0],
        [0.0, 0.0, 0.0,  0.0,  0.0, 1.0],
        [oxx, oxy, oxz,  0.0,  2.0, 0.0],
        [oxy, oyy, oyz, -2.0,  0.0, 0.0],
        [oxz, oyz, ozz,  0.0,  0.0, 0.0],
    ]
}

// ─── Frame transformation ─────────────────────────────────────────────────────

/// Convert a CRTBP rotating-frame state to inertial coordinates at angle `theta`.
///
/// In the normalized CRTBP the rotating frame rotates at unit angular rate
/// (θ̇ = 1), so `theta = t` (normalized time) for a simulation starting at θ₀ = 0.
///
/// Equivalent to `rotationMatrix3.m` case 0 (pulsating→inertial) with e = 0,
/// which gives ρ = 1, ρ̇ = 0, θ̇ = 1.  The transform is:
///
/// ```text
/// x_i  =  cos θ · x  − sin θ · y
/// y_i  =  sin θ · x  + cos θ · y
/// vx_i =  cos θ · (vx − y) − sin θ · (vy + x)
/// vy_i =  sin θ · (vx − y) + cos θ · (vy + x)
/// ```
///
/// where (vx − y, vy + x) is the inertial velocity expressed in rotating-frame
/// coordinates: v_inertial = v_rot + ω × r with ω = ẑ (unit).
/// 2-D rotating → inertial transform (kept for reference; prefer the 3-D version).
pub fn rotating_to_inertial(state: &[f64; 4], theta: f64) -> [f64; 4] {
    let (x, y, vx, vy) = (state[0], state[1], state[2], state[3]);
    let (c, s) = (theta.cos(), theta.sin());

    // Inertial velocity components in rotating-frame basis
    let vx_rot_abs = vx - y;   // v_x + (ω × r)_x  where (ω×r)_x = -y (ω_z=1)
    let vy_rot_abs = vy + x;   // v_y + (ω × r)_y  where (ω×r)_y = +x

    [
        c * x  - s * y,
        s * x  + c * y,
        c * vx_rot_abs - s * vy_rot_abs,
        s * vx_rot_abs + c * vy_rot_abs,
    ]
}

/// Convert a 3-D CRTBP rotating-frame state [x,y,z,vx,vy,vz] to inertial
/// coordinates at rotation angle `theta` (= normalized time for θ̇=1).
///
/// The rotation is about the z-axis; z and vz are unchanged.
/// Equivalent to `rotationMatrix3.m` case 0 with e=0.
pub fn rotating_to_inertial_3d(state: &[f64; 6], theta: f64) -> [f64; 6] {
    let (x, y, z)    = (state[0], state[1], state[2]);
    let (vx, vy, vz) = (state[3], state[4], state[5]);
    let (c, s) = (theta.cos(), theta.sin());

    // v_inertial expressed in rotating-frame basis:
    //   v_inertial = v_rot + ω×r  with ω=ẑ  →  (vx−y, vy+x, vz)
    let vx_abs = vx - y;
    let vy_abs = vy + x;

    [
        c*x - s*y,
        s*x + c*y,
        z,
        c*vx_abs - s*vy_abs,
        s*vx_abs + c*vy_abs,
        vz,
    ]
}
