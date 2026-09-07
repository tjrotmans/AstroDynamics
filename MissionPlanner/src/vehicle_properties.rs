//! Derived spacecraft mass properties.
//!
//! Computes TOTAL MASS, CENTER OF MASS, and the full (generally
//! non-diagonal) INERTIA TENSOR from the built configuration — bus box +
//! every placed/itemized `HardwareItem`, via the parallel-axis theorem —
//! instead of the hand-typed `spacecraft.inertia_diag_kgm2`. See
//! `docs/MP/MANUAL.md` §6.4 for the governing math and the frame
//! convention this module implements.
//!
//! **Deliberately scoped as a pure, additive computation.** This module
//! does NOT change what the simulation actually integrates —
//! `SpacecraftProperties::inertia_diag_kgm2` (`sim_engine::truth`) and
//! every existing caller (4 demo binaries, `simulate.rs`) are untouched.
//! Wiring the real derived tensor into the live 6DOF dynamics loop
//! (replacing the diagonal-only `Vector3` inertia `omega_dot`/
//! `gravity_gradient` currently take) is a separate, larger, and riskier
//! migration — correctly computing the tensor here first, independently
//! verified, is the deliberately-chosen first step before that.
//!
//! ## Frame convention
//!
//! The body-frame ORIGIN is the spacecraft's bus geometric center — every
//! `HardwareItem` placement field (`position_m`, `center_offset_m`, etc.)
//! is relative to that FIXED point, not to the center of mass. The true
//! CoM is a DERIVED output of this module (`VehicleProperties::com_m`, an
//! offset from that same origin) — this resolves what would otherwise be a
//! circular definition ("position relative to a mass distribution the
//! position itself helps determine").
//!
//! **This has a real, easy-to-get-wrong consequence for torque arms**: any
//! code computing a torque as `r × F` (SRP torque on a `Plate`, gravity-
//! gradient torque) needs `r` measured from the TRUE CoM, not from this
//! module's geometric-center origin. `Plate.center_body` (from
//! `simulate.rs::build_plates`) is built directly from the geometric-
//! center-relative config fields — for a bus-dominated spacecraft (CoM
//! close to the bus center) this was a small, previously-undocumented
//! approximation. **Fixed**: `build_plates` now
//! re-expresses every returned plate's `center_body` relative to this
//! module's derived `com_m` before returning, gated on the same
//! `spacecraft.derive_inertia_from_geometry` opt-in as this module's own
//! inertia wiring — see that field's doc comment and
//! `build_plates`'s own for the exact behavior. `gravity_gradient` torque
//! is unaffected (it already takes the spacecraft's position vector
//! directly, not a `Plate.center_body`-style component offset).

use hardware_catalog::{ImuSpec, LidarSpec, OpNavCameraSpec, PanelSpec, ReactionWheelSpec, StarTrackerSpec, ThrusterSpec};

use crate::config::{HardwareItem, MissionConfig, PanelArticulation};

/// Representative HGA-class antenna mass [kg] — no dedicated antenna
/// catalog exists yet (see `HardwareItem::CommAntenna::mass_kg`'s own doc
/// comment); a fixed placeholder, not a cited class figure like the other
/// catalog fallbacks.
const DEFAULT_COMM_ANTENNA_MASS_KG: f64 = 4.0;

/// Generic areal density [kg/m^2] fallback for an un-massed `CustomPlate`
/// — a rough placeholder (roughly between `PanelSpec::deployable()` and
/// `::rigid()`), since a `CustomPlate` represents an arbitrary shape with
/// no natural structural class the way a solar panel does.
const GENERIC_PLATE_AREAL_DENSITY_KGM2: f64 = 2.0;

/// One component's resolved (mass, position, own contribution to the total
/// inertia tensor about the TRUE center of mass) — the per-component
/// breakdown shown in the UI, so inertia is never an opaque number — the
/// display explains exactly how each placed component contributes.
#[derive(Debug, Clone)]
pub struct ComponentContribution {
    pub label: String,
    pub mass_kg: f64,
    /// Position relative to the geometric-center origin [m] (see module
    /// doc comment) — NOT relative to the derived CoM.
    pub position_m: [f64; 3],
    /// This component's own contribution to the total inertia tensor,
    /// ABOUT THE TRUE CoM [kg*m^2], full symmetric 3x3, row-major
    /// (`[[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]]`). For every
    /// component except the bus this is the point-mass parallel-axis form
    /// `m*(|d|^2*I3 - d(x)d)`, `d = position_m - com_m` — a real, generally
    /// non-diagonal tensor once the component sits off any principal axis.
    pub inertia_about_com_kgm2: [[f64; 3]; 3],
}

/// Full derived-mass-properties result.
#[derive(Debug, Clone)]
pub struct VehicleProperties {
    pub total_mass_kg: f64,
    /// Center of mass, relative to the geometric-center origin [m].
    pub com_m: [f64; 3],
    /// Full inertia tensor about the TRUE CoM [kg*m^2] (symmetric 3x3,
    /// row-major) — the sum of every `ComponentContribution`'s own term.
    pub inertia_kgm2: [[f64; 3]; 3],
    pub contributions: Vec<ComponentContribution>,
    /// Non-fatal issues found while resolving the configuration (e.g. the
    /// sum of itemized hardware masses exceeding the spacecraft's total
    /// mass) — surfaced to the caller rather than silently clamped away.
    pub warnings: Vec<String>,
}

type Mat3 = [[f64; 3]; 3];

fn mat3_zero() -> Mat3 {
    [[0.0; 3]; 3]
}

fn mat3_add(a: Mat3, b: &Mat3) -> Mat3 {
    let mut out = a;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] += b[i][j];
        }
    }
    out
}

/// Point-mass parallel-axis contribution to the inertia tensor about the
/// origin `d` is measured from: `I = m*(|d|^2*I3 - d ⊗ d)` — the standard
/// closed form (e.g. Wertz & Larson, *SMAD*; any classical-mechanics
/// reference). Full 3x3, not per-axis scalars, so an off-axis placement
/// produces the real cross-coupling (Ixy/Ixz/Iyz) terms a diagonal-only
/// model cannot represent.
fn point_mass_inertia(mass_kg: f64, d: [f64; 3]) -> Mat3 {
    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let mut i = mat3_zero();
    for a in 0..3 {
        for b in 0..3 {
            let delta = if a == b { 1.0 } else { 0.0 };
            i[a][b] = mass_kg * (d2 * delta - d[a] * d[b]);
        }
    }
    i
}

/// Uniform rectangular box inertia about its OWN centroid (SMAD/standard
/// closed form): `Ixx = m/12*(ly^2+lz^2)`, etc. — diagonal by construction
/// (a box's own centroid is always a principal axis for its own uniform
/// mass distribution).
fn box_inertia_about_own_centroid(mass_kg: f64, dims_m: [f64; 3]) -> Mat3 {
    let (lx, ly, lz) = (dims_m[0], dims_m[1], dims_m[2]);
    let f = mass_kg / 12.0;
    [
        [f * (ly * ly + lz * lz), 0.0, 0.0],
        [0.0, f * (lx * lx + lz * lz), 0.0],
        [0.0, 0.0, f * (lx * lx + ly * ly)],
    ]
}

/// Resolve one hardware item's (mass_kg, position_m, label) — manual
/// override if set, else catalog fallback (matched by `model` where the
/// variant has one, else the medium/representative grade), else a fixed
/// placeholder for the two variants with no catalog (`CommAntenna`,
/// `CustomPlate`). Position defaults to the geometric-center origin
/// (`[0,0,0]`) for every variant without a placement field, or an unplaced
/// entry of a variant that has one — see the module doc comment for why
/// that's a reasonable simplification for wheels/aggregate RCS
/// specifically.
fn resolve_item(h: &HardwareItem, index: usize) -> (String, f64, [f64; 3]) {
    let zero = [0.0, 0.0, 0.0];
    match h {
        HardwareItem::ReactionWheelCluster { model, count, mass_kg, .. } => {
            let unit_mass = mass_kg.unwrap_or_else(|| {
                ReactionWheelSpec::catalog()
                    .iter()
                    .find(|s| Some(s.name) == model.as_deref())
                    .unwrap_or(&ReactionWheelSpec::medium())
                    .mass_kg
            });
            (format!("ReactionWheelCluster #{index} (x{count})"), unit_mass * (*count as f64), zero)
        }
        HardwareItem::RCS { count, mass_kg, .. } => {
            let unit_mass = mass_kg.unwrap_or_else(|| ThrusterSpec::monoprop().mass_kg);
            let n = count.unwrap_or(1) as f64;
            (format!("RCS #{index} (x{})", count.unwrap_or(1)), unit_mass * n, zero)
        }
        HardwareItem::RcsThruster { position_m, mass_kg, .. } => {
            let m = mass_kg.unwrap_or_else(|| ThrusterSpec::monoprop().mass_kg);
            (format!("RcsThruster #{index}"), m, *position_m)
        }
        HardwareItem::StarTracker { model, mass_kg, position_m, .. } => {
            let m = mass_kg.unwrap_or_else(|| {
                StarTrackerSpec::catalog()
                    .iter()
                    .find(|s| Some(s.name) == model.as_deref())
                    .unwrap_or(&StarTrackerSpec::medium())
                    .mass_kg
            });
            (format!("StarTracker #{index}"), m, position_m.unwrap_or(zero))
        }
        HardwareItem::IMU { mass_kg, .. } => {
            (format!("IMU #{index}"), mass_kg.unwrap_or_else(|| ImuSpec::medium().mass_kg), zero)
        }
        HardwareItem::OpNavCamera { mass_kg, position_m, .. } => {
            let m = mass_kg.unwrap_or_else(|| OpNavCameraSpec::medium().mass_kg);
            (format!("OpNavCamera #{index}"), m, position_m.unwrap_or(zero))
        }
        HardwareItem::Lidar { mass_kg, position_m, .. } => {
            let m = mass_kg.unwrap_or_else(|| LidarSpec::medium().mass_kg);
            (format!("Lidar #{index}"), m, position_m.unwrap_or(zero))
        }
        HardwareItem::CommAntenna { mass_kg, position_m, .. } => {
            let m = mass_kg.unwrap_or(DEFAULT_COMM_ANTENNA_MASS_KG);
            (format!("CommAntenna #{index}"), m, position_m.unwrap_or(zero))
        }
        HardwareItem::SolarPanel { area_m2, position_m, width_m, height_m, mass_kg, .. } => {
            let area = match (width_m, height_m) {
                (Some(w), Some(hh)) => w * hh,
                _ => *area_m2,
            };
            let m = mass_kg.unwrap_or_else(|| area * PanelSpec::rigid().areal_density_kgm2);
            (format!("SolarPanel #{index}"), m, position_m.unwrap_or(zero))
        }
        HardwareItem::CustomPlate { area_m2, center_offset_m, mass_kg, .. } => {
            let m = mass_kg.unwrap_or_else(|| area_m2 * GENERIC_PLATE_AREAL_DENSITY_KGM2);
            (format!("CustomPlate #{index}"), m, *center_offset_m)
        }
    }
}

/// The panel's own configured gimbal axis/axes, if any — exposed
/// separately (not folded into `resolve_item`) since a future consumer
/// (the derived-mass-properties endpoint, or a display layer) may want to
/// flag an articulated panel distinctly, without re-matching the whole
/// `HardwareItem` enum.
pub fn panel_articulation(h: &HardwareItem) -> Option<&PanelArticulation> {
    match h {
        HardwareItem::SolarPanel { articulation, .. } => articulation.as_ref(),
        _ => None,
    }
}

/// Compute [`VehicleProperties`] for a [`MissionConfig`] — total mass,
/// center of mass, full inertia tensor, and a per-component breakdown.
/// See the module doc comment for the frame convention and scope.
pub fn compute_vehicle_properties(cfg: &MissionConfig) -> VehicleProperties {
    let mut warnings = Vec::new();

    // ── Resolve every itemized hardware component's (mass, position) ──────
    let items: Vec<(String, f64, [f64; 3])> = cfg
        .spacecraft
        .hardware
        .iter()
        .enumerate()
        .map(|(i, h)| resolve_item(h, i))
        .collect();
    let itemized_mass_kg: f64 = items.iter().map(|(_, m, _)| m).sum();

    // ── Bus/unmodeled remainder: total spacecraft mass minus everything
    // itemized above, sitting at the geometric-center origin by definition
    // (see module doc comment). `spacecraft.mass_kg` already equals
    // dry_mass_kg + propellant_mass_kg (enforced by check_config), so this
    // remainder implicitly includes the propellant and any subsystem mass
    // not broken out into `hardware` — a documented simplification, not an
    // oversight (no per-tank position field exists to do better). ────────
    let total_mass_kg = cfg.spacecraft.mass_kg;
    let bus_mass_kg = (total_mass_kg - itemized_mass_kg).max(0.0);
    if itemized_mass_kg > total_mass_kg + 1e-6 {
        warnings.push(format!(
            "Itemized hardware mass ({itemized_mass_kg:.2} kg) exceeds spacecraft.mass_kg \
             ({total_mass_kg:.2} kg) — the bus/structure remainder was clamped to 0 kg rather \
             than going negative. Increase spacecraft.mass_kg or the hardware masses are \
             double-counting something."
        ));
    }

    // ── Center of mass, relative to the geometric-center origin ───────────
    let mut com = [0.0_f64; 3];
    for (_, m, pos) in &items {
        for k in 0..3 {
            com[k] += m * pos[k];
        }
    }
    // bus contributes zero (it sits AT the origin by definition)
    if total_mass_kg > 1e-9 {
        for k in 0..3 {
            com[k] /= total_mass_kg;
        }
    }

    // ── Inertia tensor about the true CoM: bus box (own centroid = origin,
    // shifted to CoM) + each item's point-mass parallel-axis term ─────────
    let mut inertia = mat3_zero();
    let bus_d = [-com[0], -com[1], -com[2]]; // origin, relative to CoM
    let bus_own = box_inertia_about_own_centroid(bus_mass_kg, cfg.spacecraft.bus_dims_m);
    let bus_shift = point_mass_inertia(bus_mass_kg, bus_d);
    let bus_total = mat3_add(bus_own, &bus_shift);
    inertia = mat3_add(inertia, &bus_total);

    let mut contributions = vec![ComponentContribution {
        label: "Bus/structure (+ propellant, unmodeled subsystems)".to_string(),
        mass_kg: bus_mass_kg,
        position_m: [0.0, 0.0, 0.0],
        inertia_about_com_kgm2: bus_total,
    }];

    for (label, m, pos) in &items {
        let d = [pos[0] - com[0], pos[1] - com[1], pos[2] - com[2]];
        let i_comp = point_mass_inertia(*m, d);
        inertia = mat3_add(inertia, &i_comp);
        contributions.push(ComponentContribution {
            label: label.clone(),
            mass_kg: *m,
            position_m: *pos,
            inertia_about_com_kgm2: i_comp,
        });
    }

    VehicleProperties { total_mass_kg, com_m: com, inertia_kgm2: inertia, contributions, warnings }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mission_toml_with_hardware(hardware_block: &str) -> String {
        format!(
            r#"
[mission]
name = "Test"
objective = "Orbit"

[target_body]
name = "Bennu"
ephemeris = "Keplerian"

[spacecraft]
mass_kg = 1000.0
dry_mass_kg = 800.0
propellant_mass_kg = 200.0
bus_dims_m = [2.0, 2.0, 0.63]
inertia_diag_kgm2 = [366.67, 366.67, 666.67]
srp_model = "FlatPlate"

{hardware_block}

[trajectory]
phases = ["Cruise"]
solver = "Hohmann"
departure_body = "Earth"

[gnc]
navigation_filter = "EKF"
pointing_mode = "Nadir"
attitude_controller = "ReactionWheelPD"

[simulation]
integrator = "DormandPrince45"
rtol = 1.0e-9
atol = 1.0e-7
dt_truth_s = 10.0
dt_meas_s = 120.0
monte_carlo_runs = 0
output_dir = "out/test/"
"#
        )
    }

    fn parse(hardware_block: &str) -> MissionConfig {
        toml::from_str(&mission_toml_with_hardware(hardware_block)).expect("should parse")
    }

    /// Point-mass parallel-axis, hand-computed reference: a single 10 kg
    /// item offset purely along +y at 2 m. `d = (0,2,0)`, `|d|^2 = 4`.
    /// `I = m*(|d|^2*I3 - d(x)d)` gives a diagonal result here (d has only
    /// one nonzero component): Ixx = Izz = m*|d|^2 = 40, Iyy = m*(4 - 4) = 0,
    /// all off-diagonal terms involving y are also individually zero
    /// because d_x = d_z = 0 (the cross terms d_i*d_j vanish whenever
    /// either index is a zero component).
    #[test]
    fn point_mass_inertia_matches_hand_computation() {
        let i = point_mass_inertia(10.0, [0.0, 2.0, 0.0]);
        assert!((i[0][0] - 40.0).abs() < 1e-9, "Ixx: {i:?}");
        assert!((i[1][1] - 0.0).abs() < 1e-9, "Iyy: {i:?}");
        assert!((i[2][2] - 40.0).abs() < 1e-9, "Izz: {i:?}");
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            assert!(i[a][b].abs() < 1e-9, "off-diagonal [{a}][{b}] should be 0: {i:?}");
        }
    }

    /// The real frame-consistency check, at the level where it's actually
    /// meaningful: `compute_vehicle_properties` structurally pins the bus
    /// to the origin BY DEFINITION (that's the whole point of the
    /// geometric-center-origin convention — the bus's position isn't a
    /// free parameter), so "shift every hardware item's stated position
    /// but leave the bus at [0,0,0]" is NOT a rigid translation of the
    /// whole physical configuration — it's a genuine reconfiguration
    /// (components moved relative to the bus), which correctly SHOULD
    /// change the inertia tensor, and an earlier version of this test
    /// wrongly expected it not to.
    ///
    /// The actual invariant worth checking — "inertia about the true CoM
    /// doesn't depend on which point you called the origin" — is tested
    /// directly on the parallel-axis assembly math instead: take an
    /// arbitrary point-mass system, compute its inertia about its own CoM
    /// from one coordinate origin, then re-express every position in a
    /// SECOND coordinate system offset by an arbitrary constant shift
    /// (recomputing CoM in that new system too) and confirm the resulting
    /// inertia-about-CoM is identical — the invariant that proves the
    /// frame convention is applied consistently.
    #[test]
    fn parallel_axis_assembly_is_origin_choice_invariant() {
        let masses = [(3.0_f64, [1.0_f64, 0.5, -0.2]), (7.0, [-2.0, 1.0, 0.8]), (2.0, [0.3, -1.5, 1.1])];
        let total_m: f64 = masses.iter().map(|(m, _)| m).sum();

        let com_of = |pts: &[(f64, [f64; 3])]| -> [f64; 3] {
            let mut c = [0.0; 3];
            for (m, p) in pts {
                for k in 0..3 {
                    c[k] += m * p[k];
                }
            }
            for k in 0..3 {
                c[k] /= total_m;
            }
            c
        };
        let inertia_of = |pts: &[(f64, [f64; 3])], com: [f64; 3]| -> Mat3 {
            pts.iter().fold(mat3_zero(), |acc, (m, p)| {
                let d = [p[0] - com[0], p[1] - com[1], p[2] - com[2]];
                mat3_add(acc, &point_mass_inertia(*m, d))
            })
        };

        let com_a = com_of(&masses);
        let inertia_a = inertia_of(&masses, com_a);

        // Re-express in a second coordinate system, offset by an arbitrary
        // shift — a genuine relabeling of the same physical configuration,
        // not a reconfiguration.
        let shift = [10.0, -5.0, 2.0];
        let masses_shifted: Vec<(f64, [f64; 3])> = masses
            .iter()
            .map(|(m, p)| (*m, [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]))
            .collect();
        let com_b = com_of(&masses_shifted);
        let inertia_b = inertia_of(&masses_shifted, com_b);

        for k in 0..3 {
            assert!(
                (com_b[k] - (com_a[k] + shift[k])).abs() < 1e-9,
                "CoM should shift by the same constant: {com_a:?} -> {com_b:?}, shift {shift:?}"
            );
        }
        for a in 0..3 {
            for b in 0..3 {
                let diff = (inertia_a[a][b] - inertia_b[a][b]).abs();
                assert!(diff < 1e-9, "inertia[{a}][{b}] should be origin-choice-invariant: {} vs {} (diff={diff})", inertia_a[a][b], inertia_b[a][b]);
            }
        }
    }

    /// A configuration symmetric about all three principal planes (mirrored
    /// pairs of identical components on +/-x, +/-y, +/-z) should produce a
    /// purely diagonal inertia tensor — the "only for a block" case the
    /// user described: general placement gives real off-diagonal terms,
    /// but a genuinely symmetric layout reduces back to the diagonal
    /// approximation every existing sim demo already assumes.
    #[test]
    fn symmetric_placement_reduces_to_diagonal_inertia() {
        let cfg = parse(
            r#"[[spacecraft.hardware]]
type = "CustomPlate"
normal = [1.0, 0.0, 0.0]
area_m2 = 1.0
center_offset_m = [1.5, 0.0, 0.0]
mass_kg = 5.0

[[spacecraft.hardware]]
type = "CustomPlate"
normal = [-1.0, 0.0, 0.0]
area_m2 = 1.0
center_offset_m = [-1.5, 0.0, 0.0]
mass_kg = 5.0

[[spacecraft.hardware]]
type = "CustomPlate"
normal = [0.0, 1.0, 0.0]
area_m2 = 1.0
center_offset_m = [0.0, 1.2, 0.0]
mass_kg = 3.0

[[spacecraft.hardware]]
type = "CustomPlate"
normal = [0.0, -1.0, 0.0]
area_m2 = 1.0
center_offset_m = [0.0, -1.2, 0.0]
mass_kg = 3.0"#,
        );
        let vp = compute_vehicle_properties(&cfg);
        // CoM should sit at the origin (perfectly symmetric mass distribution).
        for k in 0..3 {
            assert!(vp.com_m[k].abs() < 1e-9, "CoM should be at the origin: {:?}", vp.com_m);
        }
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            assert!(
                vp.inertia_kgm2[a][b].abs() < 1e-6,
                "off-diagonal [{a}][{b}] should vanish for a symmetric layout: {:?}",
                vp.inertia_kgm2
            );
        }
    }

    /// Itemized hardware mass exceeding the spacecraft's total mass must be
    /// surfaced as a warning, not silently clamped away with no trace.
    #[test]
    fn overweight_hardware_produces_a_warning() {
        let cfg = parse(
            r#"[[spacecraft.hardware]]
type = "SolarPanel"
area_m2 = 2.0
mass_kg = 5000.0"#,
        );
        let vp = compute_vehicle_properties(&cfg);
        assert!(
            vp.warnings.iter().any(|w| w.contains("exceeds")),
            "expected an overweight warning, got: {:?}", vp.warnings
        );
        // Bus/structure remainder should be clamped to 0, not negative.
        assert_eq!(vp.contributions[0].mass_kg, 0.0);
    }

    /// No hardware at all: total mass should equal spacecraft.mass_kg
    /// exactly (all of it attributed to the bus/structure remainder), CoM
    /// at the origin, and inertia should match the plain bus-box formula
    /// (no parallel-axis shift needed since the bus already sits at the
    /// origin and the origin equals the CoM in this case).
    #[test]
    fn no_hardware_matches_plain_bus_box_inertia() {
        let cfg = parse("");
        let vp = compute_vehicle_properties(&cfg);
        assert!((vp.total_mass_kg - 1000.0).abs() < 1e-9);
        for k in 0..3 {
            assert!(vp.com_m[k].abs() < 1e-9);
        }
        let expected = box_inertia_about_own_centroid(1000.0, [2.0, 2.0, 0.63]);
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    (vp.inertia_kgm2[a][b] - expected[a][b]).abs() < 1e-6,
                    "[{a}][{b}]: got {}, expected {}", vp.inertia_kgm2[a][b], expected[a][b]
                );
            }
        }
    }
}
