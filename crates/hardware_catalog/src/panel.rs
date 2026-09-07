//! Solar panel catalog entries — two representative structural classes.
//!
//! Unlike the sensor/wheel/thruster catalogs, panel mass genuinely scales
//! with AREA (a bigger panel isn't a "finer grade" of the same unit, it's
//! more of the same material) — so this catalog gives an AREAL DENSITY
//! [kg/m^2] per structural class rather than a fixed per-unit mass; the
//! caller multiplies by the panel's own configured area.

/// A solar panel structural class spec.
#[derive(Clone, Copy, Debug)]
pub struct PanelSpec {
    pub name: &'static str,
    /// Areal mass density [kg/m^2] — includes substrate, cells, and wiring
    /// (Wertz & Larson, *SMAD*-class figures), not a specific flight
    /// panel's datasheet value.
    pub areal_density_kgm2: f64,
    /// Representative electrical efficiency (0-1) for this class — the
    /// same default `HardwareItem::SolarPanel::efficiency` falls back to
    /// when a mission doesn't specify its own.
    pub efficiency: f64,
}

impl PanelSpec {
    /// Rigid (honeycomb-substrate) panel — standard small-to-mid
    /// spacecraft class, heavier but structurally simple and cheap.
    pub fn rigid() -> Self {
        Self { name: "Panel-Rigid", areal_density_kgm2: 3.0, efficiency: 0.28 }
    }

    /// Deployable/flexible (roll-out or fold-out blanket) array — lighter
    /// areal density, standard for mass-constrained or large-array
    /// missions; typically higher unit cost per m^2 than rigid.
    pub fn deployable() -> Self {
        Self { name: "Panel-Deployable", areal_density_kgm2: 1.0, efficiency: 0.30 }
    }

    /// Lightest to heaviest — sizing picks the lightest class that still
    /// meets a mass-budget constraint (mirrors the other catalogs'
    /// "coarsest that still meets the requirement" sizing convention,
    /// just ordered by mass instead of noise).
    pub fn catalog() -> [Self; 2] {
        [Self::deployable(), Self::rigid()]
    }
}
