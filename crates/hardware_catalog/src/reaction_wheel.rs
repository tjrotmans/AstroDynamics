//! Reaction wheel catalog entries — five representative size classes,
//! cubesat-micro through large-GEO-bus scale (widened from the original
//! two classes).

/// A reaction-wheel unit spec (single wheel; cluster geometry is the caller's
/// concern — see `attitude_control::ReactionWheelCluster`).
#[derive(Clone, Copy, Debug)]
pub struct ReactionWheelSpec {
    pub name: &'static str,
    /// Wheel spin-axis inertia [kg·m²]
    pub inertia_kgm2: f64,
    /// Maximum wheel speed [rad/s]
    pub max_speed_rads: f64,
    /// Maximum motor torque [N·m]
    pub max_torque_nm: f64,
    /// Wheel mass [kg] (for power/mass budget rollups)
    pub mass_kg: f64,
    /// Nominal operating power draw [W] — representative class estimate
    /// (Wertz & Larson, *SMAD*-class figures), not a specific flight unit's
    /// datasheet value.
    pub power_w: f64,
}

impl ReactionWheelSpec {
    /// Maximum angular momentum storage per wheel [N·m·s]: H_max = I·ω_max.
    pub fn max_momentum_nms(&self) -> f64 {
        self.inertia_kgm2 * self.max_speed_rads
    }

    /// Micro class — genuine cubesat-scale, below `small()`'s momentum tier
    /// Cites a real, named, flight-
    /// heritage product family: Blue Canyon Technologies RWp015
    /// (momentum 0.015 N·m·s, mass 0.13 kg — both cross-checked against
    /// two independent public sources, satcatalog.com and the BCT reaction-
    /// wheel datasheet listing, in agreement). `max_torque_nm`/`power_w`
    /// are taken from the same BCT product literature (0.004 N·m,
    /// ~0.85 W average) but were only confirmed via a single source this
    /// session — treat those two figures as representative-for-this-class
    /// rather than independently cross-verified the way momentum/mass are.
    pub fn micro() -> Self {
        const MOMENTUM_NMS: f64 = 0.015;
        const MAX_SPEED_RADS: f64 = 628.3; // ~6000 RPM, same convention as every other class here
        Self {
            name: "RW-Micro",
            inertia_kgm2: MOMENTUM_NMS / MAX_SPEED_RADS,
            max_speed_rads: MAX_SPEED_RADS,
            max_torque_nm: 0.004,
            mass_kg: 0.13,
            power_w: 0.85,
        }
    }

    /// Small class — cubesat/microsat-scale (momentum ~0.1 N·m·s).
    pub fn small() -> Self {
        Self {
            name: "RW-Small",
            inertia_kgm2: 1.6e-4,
            max_speed_rads: 628.3, // ~6000 RPM
            max_torque_nm: 0.005,
            mass_kg: 0.5,
            power_w: 3.0,
        }
    }

    /// Medium class — small-satellite scale (matches the Bennu mission's
    /// existing Ithaco_B-class wheel, momentum ~7.5 N·m·s). Power is a
    /// representative class estimate for this momentum class (Ithaco/
    /// Honeywell HR-class wheels of this size typically draw ~6-10 W
    /// nominal running power).
    pub fn medium() -> Self {
        Self {
            name: "RW-Medium",
            inertia_kgm2: 0.012,
            max_speed_rads: 628.3,
            max_torque_nm: 0.12,
            mass_kg: 2.5,
            power_w: 8.0,
        }
    }

    /// Large class — larger science-spacecraft scale (momentum ~50 N·m·s).
    pub fn large() -> Self {
        Self {
            name: "RW-Large",
            inertia_kgm2: 0.08,
            max_speed_rads: 628.3,
            max_torque_nm: 0.5,
            mass_kg: 9.0,
            power_w: 20.0,
        }
    }

    /// Extra-large class — larger GEO-comsat/observatory-bus scale, above
    /// `large`'s momentum tier.
    /// Representative-class figure (Wertz & Larson, *SMAD*-class figures
    /// for a large-bus momentum wheel), same status as `medium()`/`large()`
    /// above — not a specific named flight unit's datasheet value.
    pub fn xlarge() -> Self {
        Self {
            name: "RW-XLarge",
            inertia_kgm2: 0.24,
            max_speed_rads: 628.3,
            max_torque_nm: 1.0,
            mass_kg: 16.0,
            power_w: 35.0,
        }
    }

    /// All catalog entries, smallest to largest — used by sizing routines
    /// that pick the smallest unit meeting a requirement.
    pub fn catalog() -> [Self; 5] {
        [Self::micro(), Self::small(), Self::medium(), Self::large(), Self::xlarge()]
    }
}
