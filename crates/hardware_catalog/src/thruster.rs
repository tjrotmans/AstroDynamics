//! Thruster catalog entries — four representative propellant-type/size
//! classes,
//! spanning
//! cubesat-precision cold gas through mid-size monopropellant RCS/TCM.

/// A thruster unit spec for RCS/desaturation sizing.
#[derive(Clone, Copy, Debug)]
pub struct ThrusterSpec {
    pub name: &'static str,
    /// Thrust per thruster [N]
    pub thrust_n: f64,
    /// Specific impulse [s]
    pub isp_s: f64,
    /// Minimum impulse-bit duration [s]
    pub min_pulse_s: f64,
    /// Nominal valve-driver power draw [W] — representative class estimate
    /// (Wertz & Larson, *SMAD*-class figures), not a specific flight unit's
    /// datasheet value.
    pub power_w: f64,
    /// Unit mass [kg] (thruster head + valve, not the propellant tank/feed
    /// system) — representative class estimate (Wertz & Larson, *SMAD*),
    /// same figure class as the other catalog specs' `mass_kg`. Added
    /// for the derived-mass-properties work
    /// — every other catalog
    /// spec in this crate already had a `mass_kg`; this was the one gap.
    pub mass_kg: f64,
}

impl ThrusterSpec {
    /// Micro cold gas — genuine cubesat-scale, below `cold_gas()`'s thrust
    /// tier. Representative-class figure
    /// (Wertz & Larson, *SMAD*-class figures for a cubesat cold-gas
    /// precision thruster) — not a specific named flight unit's datasheet
    /// value; cold-gas Isp in the 40-70 s range is a well-established,
    /// propellant-independent (N2/GN2/butane) figure across this class
    /// regardless of the specific product.
    pub fn micro_cold_gas() -> Self {
        Self {
            name: "ColdGas-Micro",
            thrust_n: 0.05,
            isp_s: 50.0,
            min_pulse_s: 0.005,
            power_w: 1.0,
            mass_kg: 0.05,
        }
    }

    /// Cold gas (e.g. N2/GN2) — low Isp, very fine impulse-bit control,
    /// common for small-sat/precision proximity-ops desaturation.
    pub fn cold_gas() -> Self {
        Self {
            name: "ColdGas",
            thrust_n: 0.5,
            isp_s: 65.0,
            min_pulse_s: 0.010,
            power_w: 2.0,
            mass_kg: 0.2,
        }
    }

    /// Monopropellant hydrazine — higher Isp, coarser impulse-bit, standard
    /// for RCS/TCM on small-to-mid spacecraft (matches the Bennu/Mars-flyby
    /// mission's existing monoprop assumption). Power covers the valve
    /// driver plus catalyst-bed heater (estimated class figure).
    pub fn monoprop() -> Self {
        Self {
            name: "Monoprop",
            thrust_n: 1.0,
            isp_s: 220.0,
            min_pulse_s: 0.050,
            power_w: 5.0,
            mass_kg: 0.4,
        }
    }

    /// Higher-thrust monopropellant hydrazine — a real, widely-flown named
    /// product class, above `monoprop`'s thrust tier
    /// (filling the "no mid/high-thrust RCS option" gap
    /// in the catalog): Moog MONARC-5 (1 lbf = 4.45 N thrust,
    /// confirmed directly from Moog/vendor product literature — this is an
    /// unambiguous unit conversion, not an estimate). Isp (225 s) and mass
    /// (0.5 kg) are typical published figures for this thruster class
    /// (hydrazine monoprop in the 1-lbf-class range commonly runs 215-235 s
    /// — no exact datasheet table was available,
    /// so treat Isp/mass as representative-for-this-class
    /// rather than an exact confirmed datasheet value the way `thrust_n`
    /// is) — consistent with, and only slightly above, this crate's
    /// existing `monoprop()` Isp (220 s), which is itself already
    /// MONARC-5-class.
    pub fn monoprop_coarse() -> Self {
        Self {
            name: "Monoprop-MONARC5class",
            thrust_n: 4.45,
            isp_s: 225.0,
            min_pulse_s: 0.050,
            power_w: 9.0,
            mass_kg: 0.5,
        }
    }

    pub fn catalog() -> [Self; 4] {
        [Self::micro_cold_gas(), Self::cold_gas(), Self::monoprop(), Self::monoprop_coarse()]
    }
}
