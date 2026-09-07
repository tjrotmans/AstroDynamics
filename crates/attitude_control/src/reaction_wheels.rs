//! Reaction wheel cluster — geometry, allocation, and saturation management.
//!
//! `ReactionWheelCluster` is a generic struct parameterized over the wheel
//! properties.  Call [`ReactionWheelCluster::four_wheel_pyramid`] to build
//! the standard symmetric 4-wheel pyramid used in most small spacecraft.
//!
//! # Wheel geometry (four-wheel pyramid)
//!
//! Wheel axes form a symmetric pyramid with each axis tilted at
//! β = arctan(1/√2) ≈ 35.26° from the body +z axis, at azimuths
//! 0°, 90°, 180°, 270°.  This gives equal torque authority about all
//! three body axes and one redundant degree of freedom.
//!
//! # Sign convention
//! - Positive wheel speed Ω_i = counterclockwise when viewed along ŵ_i.
//! - Motor torque τ_m_i > 0 accelerates wheel i (same sense).
//! - Reaction on spacecraft = −τ_m_i about ŵ_i.

use nalgebra::Vector3;

// β = arctan(1/√2): sin β = 1/√3, cos β = √(2/3)
const S: f64 = 0.577_350_269_189_626;  // sin β
const C: f64 = 0.816_496_580_927_726;  // cos β

/// Reaction wheel cluster for a rigid spacecraft.
///
/// Stores geometry and hardware limits; individual wheel speeds are external state.
pub struct ReactionWheelCluster {
    /// Unit spin-axis vectors in body frame [w1, w2, w3, w4].
    pub axes: [[f64; 3]; 4],
    /// Minimum-norm pseudo-inverse A† (4×3), pre-computed from axes.
    apinv: [[f64; 3]; 4],
    /// Null-space direction of the axis matrix A (3×4): the one wheel-speed
    /// combination (up to scale) that produces ZERO net body-frame momentum
    /// or torque — 4 actuators for 3 rotational DOF means this cluster has
    /// exactly one such redundant direction. `[1,-1,1,-1]` for this
    /// symmetric pyramid geometry (verified: `Σ null_dir[i]·axes[i] = 0`).
    /// Used for null-motion (wheel-speed equalization) — see `docs/MP/
    /// MANUAL.md` §10.3.
    null_dir: [f64; 4],
    /// Wheel moment of inertia [kg·m²].
    pub wheel_inertia: f64,
    /// Maximum wheel speed [rad/s].
    pub max_speed: f64,
    /// Maximum wheel motor torque [N·m].
    pub max_torque: f64,
    /// Desaturation threshold as a fraction of max_speed (0–1).
    pub desat_fraction: f64,
}

impl ReactionWheelCluster {
    /// Build a symmetric 4-wheel pyramid cluster.
    ///
    /// Axes: tilted at β = arctan(1/√2) from body +z at azimuths 0°/90°/180°/270°.
    ///
    /// # Arguments
    /// * `wheel_inertia`  – Wheel axial moment of inertia [kg·m²]
    /// * `max_speed`      – Maximum wheel speed [rad/s]
    /// * `max_torque`     – Maximum motor torque [N·m]
    /// * `desat_fraction` – Fraction of max_speed at which desaturation fires
    pub fn four_wheel_pyramid(
        wheel_inertia:  f64,
        max_speed:      f64,
        max_torque:     f64,
        desat_fraction: f64,
    ) -> Self {
        // A (3×4) = [ŵ1 | ŵ2 | ŵ3 | ŵ4]
        // A·Aᵀ = diag(2S², 2S², 4C²)  with S²=1/3, C²=2/3
        //       = diag(2/3, 2/3, 8/3)
        // A† = Aᵀ·(A·Aᵀ)⁻¹  → each row i scaled by [3/2, 3/2, 3/8]
        let a3h = 1.5 * S;    // 3/2 × sin β
        let a3r = 0.375 * C;  // 3/8 × cos β
        Self {
            axes: [
                [ S,  0.0, C],
                [0.0,  S,  C],
                [-S,  0.0, C],
                [0.0, -S,  C],
            ],
            apinv: [
                [ a3h,  0.0, a3r],
                [ 0.0,  a3h, a3r],
                [-a3h,  0.0, a3r],
                [ 0.0, -a3h, a3r],
            ],
            null_dir: [1.0, -1.0, 1.0, -1.0],
            wheel_inertia,
            max_speed,
            max_torque,
            desat_fraction,
        }
    }

    /// Total wheel angular momentum in body frame [N·m·s].
    ///
    /// H_w = I_w · Σ_i (Ω_i · ŵ_i)
    pub fn total_momentum(&self, speeds: &[f64; 4]) -> Vector3<f64> {
        let mut h = Vector3::zeros();
        for i in 0..4 {
            let w = self.axes[i];
            h += self.wheel_inertia * speeds[i] * Vector3::new(w[0], w[1], w[2]);
        }
        h
    }

    /// Allocate a commanded body-frame torque to wheel motor torques [N·m].
    ///
    /// Minimum-norm solution: τ_motor = −A† · τ_cmd.
    /// Each motor torque is clamped to ±max_torque.
    pub fn allocate(&self, tau_cmd: &Vector3<f64>) -> [f64; 4] {
        let mut tau_m = [0.0_f64; 4];
        for i in 0..4 {
            let row = self.apinv[i];
            let dot = row[0]*tau_cmd[0] + row[1]*tau_cmd[1] + row[2]*tau_cmd[2];
            tau_m[i] = (-dot).clamp(-self.max_torque, self.max_torque);
        }
        tau_m
    }

    /// The UNCLAMPED minimum-norm motor-torque command `−A†·τ_cmd` [N·m] —
    /// review E3, purely additive telemetry alongside
    /// [`allocate`](Self::allocate): what the allocator WANTS each wheel to
    /// do before the ±`max_torque` clamp. When any wheel clamps, the
    /// delivered body torque (`body_torque(allocate(τ))`) differs from the
    /// commanded τ in DIRECTION, not just magnitude — which the speed-only
    /// `wheel_sat_frac` telemetry reports nothing about. Comparing this
    /// against `allocate`'s output (or against `max_torque`) makes torque
    /// saturation as visible as speed saturation. Never used for control —
    /// physics consumers keep calling `allocate`.
    pub fn allocate_unclamped(&self, tau_cmd: &Vector3<f64>) -> [f64; 4] {
        let mut tau_m = [0.0_f64; 4];
        for i in 0..4 {
            let row = self.apinv[i];
            let dot = row[0]*tau_cmd[0] + row[1]*tau_cmd[1] + row[2]*tau_cmd[2];
            tau_m[i] = -dot;
        }
        tau_m
    }

    /// Wheel speed derivatives [rad/s²] given motor torques.
    ///
    /// Ω̇_i = τ_motor_i / I_w
    pub fn speed_dots(&self, tau_motor: &[f64; 4]) -> [f64; 4] {
        tau_motor.map(|t| t / self.wheel_inertia)
    }

    /// Actual torque on spacecraft body from wheels [N·m].
    ///
    /// τ_on_body = −Σ_i (τ_motor_i · ŵ_i) = −A · τ_motor.
    /// May differ slightly from τ_cmd when wheels are saturated.
    pub fn body_torque(&self, tau_motor: &[f64; 4]) -> Vector3<f64> {
        let mut tau = Vector3::zeros();
        for i in 0..4 {
            let w = self.axes[i];
            tau -= tau_motor[i] * Vector3::new(w[0], w[1], w[2]);
        }
        tau
    }

    /// Per-wheel motor torque [N·m] that would produce the given cluster
    /// angular-momentum RATE dH/dt (NOT a commanded body torque — see the
    /// sign note below). Minimum-norm solution using the same pseudo-
    /// inverse `allocate` uses, WITHOUT `allocate`'s negation.
    ///
    /// Since H_wheel_i = I_w·Ω_i, dH_wheel_i/dt = τ_motor_i exactly (about
    /// that wheel's own axis) — so d(H_cluster)/dt = A·τ_motor directly
    /// (A = the axis matrix), giving τ_motor = A†·target = apinv·target,
    /// no sign flip. Contrast with `allocate`, which solves for a
    /// commanded BODY torque = −A·τ_motor (opposite sign, by Newton's
    /// third law — the body reacts against whatever the wheel does to its
    /// own momentum).
    ///
    /// Used by momentum-management (desaturation) laws: commanding wheel
    /// torque toward a target dH/dt (e.g. `−gain·(H − H_target)`) actively
    /// unloads the wheel, at the cost of an equal-and-opposite reaction on
    /// the body that some OTHER actuator (RCS, magnetorquer, SRP trim...)
    /// must cancel to avoid disturbing attitude tracking — see
    /// `sim_engine::control::MomentumManagementLaw` for how that pairing
    /// is enforced, and `docs/MP/MANUAL.md` §10.3 for the full
    /// derivation.
    pub fn dump_motor_torque(&self, dh_dt_target: &Vector3<f64>) -> [f64; 4] {
        let mut tau_m = [0.0_f64; 4];
        for i in 0..4 {
            let row = self.apinv[i];
            let dot = row[0] * dh_dt_target[0] + row[1] * dh_dt_target[1] + row[2] * dh_dt_target[2];
            tau_m[i] = dot.clamp(-self.max_torque, self.max_torque);
        }
        tau_m
    }

    /// Null-motion (wheel-speed equalization) torque request [N·m, per
    /// wheel] — UNCLAMPED; the caller combines this with any other torque
    /// request and clamps the total once, jointly (see `sim_engine::
    /// control::allocate`'s `WheelsPrimary` branch).
    ///
    /// Drives the cluster's one redundant/"internal" wheel-speed
    /// combination (the projection of `speeds` onto `null_dir`) toward
    /// zero, at rate `gain` [1/s] — by construction this produces EXACTLY
    /// zero net body torque and zero net change to `total_momentum`
    /// (`Σ null_dir[i]·axes[i] = 0`), so it is safe to apply without
    /// disturbing attitude tracking or interfering with total-momentum
    /// desaturation, in the common (unclamped) case. Found necessary
    /// a law that only damps `total_momentum` (this cluster's
    /// controlled 3-DOF subspace) leaves this redundant 4th DOF completely
    /// unmanaged — individual wheels can drift arbitrarily far apart even
    /// while the cluster's net momentum stays well-behaved, confirmed via
    /// `cruise_commander_demo`'s real telemetry (a real, secularly growing
    /// null-space component, unbounded absent this correction). See
    /// `docs/MP/MANUAL.md` §10.3 for the derivation and the measured
    /// numbers that motivated this.
    ///
    /// Derivation: with `z = (speeds · null_dir) / (null_dir · null_dir)`
    /// and `H_i = I_w·Ω_i` so `dH_i/dt = τ_motor,i` exactly (per wheel),
    /// commanding `τ_motor = c·null_dir` gives `dz/dt = c/I_w` (the
    /// `null_dir·null_dir` factors cancel). Setting `dz/dt = −gain·z`
    /// gives `c = −I_w·gain·z`.
    pub fn null_motion_torque(&self, speeds: &[f64; 4], gain: f64) -> [f64; 4] {
        let n = self.null_dir;
        let nn: f64 = n.iter().map(|x| x * x).sum();
        let z = (0..4).map(|i| speeds[i] * n[i]).sum::<f64>() / nn;
        let c = -self.wheel_inertia * gain * z;
        std::array::from_fn(|i| c * n[i])
    }

    /// Returns `true` when any wheel speed exceeds `desat_fraction × max_speed`.
    pub fn needs_desat(&self, speeds: &[f64; 4]) -> bool {
        let threshold = self.desat_fraction * self.max_speed;
        speeds.iter().any(|&s| s.abs() > threshold)
    }

    /// Desaturation torque direction for RCS [N·m, body frame].
    ///
    /// Returns `Some(τ)` when desaturation is needed.  Fire RCS to produce
    /// this torque, absorbing angular momentum from the wheels.
    pub fn desat_torque(&self, speeds: &[f64; 4]) -> Option<Vector3<f64>> {
        if self.needs_desat(speeds) {
            Some(-self.total_momentum(speeds))
        } else {
            None
        }
    }
}
