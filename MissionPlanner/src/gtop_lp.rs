//! GTOP-style low-precision analytic planetary ephemeris (Phase 9w-ii).
//!
//! Keplerian elements + linear centennial rates, heliocentric ecliptic J2000
//! (Standish, "Keplerian Elements for Approximate Positions of the Major
//! Planets", JPL, 1800 AD–2050 AD table — the same family of low-precision
//! elements GTOP's `pleph_an` / pykep's `jpl_lp` use).
//!
//! Purpose: evaluate published GTOP benchmark chromosomes against the SAME
//! KIND of ephemeris they were optimized on, removing the DE440s-vs-analytic
//! confound (worth up to ±2.9 km/s on resonance-sensitive legs — see
//! the design notes Phase 9v-v). Select at runtime with `GTOP_LP=1` in the check
//! binaries; DE440s (ANISE) remains the default everywhere else.
//!
//! Note: `bin/gtop_cassini2_check.rs` predates this module and carries its
//! own identical inline copy of these elements — left untouched per the
//! don't-refactor-validated-code rule; consolidation is a separate, approved
//! change if wanted.

use ephemeris::Body;
use nalgebra::Vector3;

/// Mean obliquity of the ecliptic at J2000 [rad] — IAU 2006 (Hilton et al.).
/// Same constant `plot/mga_frame_utils.py` uses for the inverse rotation.
pub const OBLIQUITY_J2000_RAD: f64 = 23.439_291_11_f64 * std::f64::consts::PI / 180.0;

/// Rotate a vector from heliocentric ecliptic J2000 into equatorial ICRF
/// (rotation by +ε about the x-axis).
pub fn ecliptic_to_icrf(v: Vector3<f64>) -> Vector3<f64> {
    let (s, c) = OBLIQUITY_J2000_RAD.sin_cos();
    Vector3::new(v.x, c * v.y - s * v.z, s * v.y + c * v.z)
}

/// Elements per body: [a_AU, e, I_deg, L_deg, peri_deg (ϖ), node_deg (Ω)]
/// and their per-Julian-century rates (Standish table, see module docs).
pub struct LpElements {
    pub kep0: [f64; 6],
    pub rate: [f64; 6],
}

/// Low-precision elements for the bodies the GTOP benchmarks visit.
/// Panics for bodies not in the table — the check binaries are the only
/// intended callers and their sequences are fixed.
pub fn lp_elements(body: Body) -> LpElements {
    match body {
        Body::Venus => LpElements {
            kep0: [0.723_335_66, 0.006_776_72, 3.394_676_05, 181.979_099_50, 131.602_467_18, 76.679_842_55],
            rate: [0.000_003_90, -0.000_041_07, -0.000_788_90, 58_517.815_387_29, 0.002_683_29, -0.277_694_18],
        },
        Body::Earth => LpElements {
            // Earth-Moon barycenter — the same simplification GTOP makes.
            kep0: [1.000_002_61, 0.016_711_23, -0.000_015_31, 100.464_571_66, 102.937_681_93, 0.0],
            rate: [0.000_005_62, -0.000_043_92, -0.012_946_68, 35_999.372_449_81, 0.323_273_64, 0.0],
        },
        Body::Jupiter => LpElements {
            kep0: [5.202_887_00, 0.048_386_24, 1.304_396_95, 34.396_440_51, 14.728_479_83, 100.473_909_09],
            rate: [-0.000_116_07, -0.000_132_53, -0.001_837_14, 3_034.746_127_75, 0.212_526_68, 0.204_691_06],
        },
        Body::Saturn => LpElements {
            kep0: [9.536_675_94, 0.053_861_79, 2.485_991_87, 49.954_244_23, 92.598_878_31, 113.662_424_48],
            rate: [-0.001_250_60, -0.000_509_91, 0.001_936_09, 1_222.493_622_01, -0.418_972_16, -0.288_677_94],
        },
        other => panic!("lp_elements: no low-precision elements for {other:?}"),
    }
}

/// Heliocentric ecliptic-J2000 position [m] from the low-precision elements
/// at Julian date `jd` (TDB ≈ TT — sub-minute timescale differences are far
/// below the accuracy of these elements).
pub fn lp_position_ecliptic(el: &LpElements, jd: f64) -> Vector3<f64> {
    const AU_M: f64 = 1.495_978_707e11;
    let t_cy = (jd - 2_451_545.0) / 36_525.0;
    let a = (el.kep0[0] + el.rate[0] * t_cy) * AU_M;
    let e = el.kep0[1] + el.rate[1] * t_cy;
    let i = (el.kep0[2] + el.rate[2] * t_cy).to_radians();
    let l = (el.kep0[3] + el.rate[3] * t_cy).to_radians();
    let peri = (el.kep0[4] + el.rate[4] * t_cy).to_radians();
    let node = (el.kep0[5] + el.rate[5] * t_cy).to_radians();

    let omega = peri - node;        // argument of periapsis
    let m = (l - peri) % (2.0 * std::f64::consts::PI); // mean anomaly

    // Kepler solve (Newton).
    let mut ea = m;
    for _ in 0..30 {
        let f = ea - e * ea.sin() - m;
        ea -= f / (1.0 - e * ea.cos());
    }
    let x_pf = a * (ea.cos() - e);
    let y_pf = a * (1.0 - e * e).sqrt() * ea.sin();

    let (so, co) = omega.sin_cos();
    let (sn, cn) = node.sin_cos();
    let (si, ci) = i.sin_cos();
    Vector3::new(
        (co * cn - so * sn * ci) * x_pf + (-so * cn - co * sn * ci) * y_pf,
        (co * sn + so * cn * ci) * x_pf + (-so * sn + co * cn * ci) * y_pf,
        (so * si) * x_pf + (co * si) * y_pf,
    )
}

/// Low-precision body state in the frame the rest of the MGA code uses
/// (ICRF equatorial, matching ANISE): analytic ecliptic position rotated to
/// ICRF, velocity by central finite difference (±60 s).
pub fn lp_state_icrf(body: Body, jd: f64) -> (Vector3<f64>, Vector3<f64>) {
    let el = lp_elements(body);
    let r = ecliptic_to_icrf(lp_position_ecliptic(&el, jd));
    let dt_days = 60.0 / 86_400.0;
    let r_m = ecliptic_to_icrf(lp_position_ecliptic(&el, jd - dt_days));
    let r_p = ecliptic_to_icrf(lp_position_ecliptic(&el, jd + dt_days));
    let v = (r_p - r_m) / 120.0;
    (r, v)
}

// ── GTOP's OWN analytic ephemeris (exact reproduction) ──────────────────────
// Transcribed verbatim from GTOPX `gtopx.cpp::Planet_Ephemerides_Analytical`
// (Schlueter et al., https://www.midaco-solver.com — GPL; original ESA ACT
// GTOP code, © 2004-2007 European Space Agency; retrieved).
// These are 1900-epoch mean-element polynomials (T = Julian centuries since
// 1900.0 = (MJD2000 + 36525)/36525), NOT the Standish 1800-2050 table above —
// GTOP benchmark objectives are only bit-reproducible against THIS element
// set (the resonant V-V leg in Cassini-1 amplifies ephemeris differences ~7×
// into the flyby geometry, so "any reasonable ephemeris" is not enough).

/// Sun's gravitational parameter used by ALL GTOP/GTOPX benchmarks [m³/s²]
/// (their `MU[0] = 1.32712428e11` km³/s² — deliberately NOT the DE430 value).
pub const GTOP_MU_SUN_M3S2: f64 = 1.327_124_28e20;

/// GTOP's astronomical unit [m] (`AU = 149597870.66` km in gtopx.cpp).
pub const GTOP_AU_M: f64 = 1.495_978_706_6e11;

/// GTOP planetary gravitational parameters [m³/s²], indexed like their
/// `MU[9]` table (1 = Mercury … 8 = Neptune; values are km³/s² × 1e9).
pub fn gtop_mu_planet_m3s2(body: Body) -> f64 {
    match body {
        Body::Mercury => 2.232_1e13,
        Body::Venus => 3.248_60e14,
        Body::Earth => 3.986_011_9e14,
        Body::Mars => 4.282_83e13,
        Body::Jupiter => 1.267e17,
        Body::Saturn => 3.79e16,
        Body::Uranus => 5.78e15,
        Body::Neptune => 6.8e15,
        other => panic!("gtop_mu_planet_m3s2: no GTOP constant for {other:?}"),
    }
}

/// GTOP mean elements at time `t_cy` (Julian centuries since 1900.0):
/// `[a_AU, e, i_deg, node_deg (Ω), peri_deg (ω), mean_anomaly_deg]`.
/// Polynomials transcribed verbatim from `Planet_Ephemerides_Analytical`.
fn gtop_elements(body: Body, t: f64) -> [f64; 6] {
    match body {
        Body::Mercury => [
            0.387_098_60,
            0.205_614_21 + 0.000_020_46 * t - 0.000_000_03 * t * t,
            7.002_880_555_555_555_56 + 1.860_833_333_333_333_33e-3 * t - 1.833_333_333_333_333_33e-5 * t * t,
            4.714_594_444_444_444_44e1 + 1.185_208_333_333_333_33 * t + 1.738_888_888_888_888_89e-4 * t * t,
            2.875_375_277_777_777_78e1 + 3.702_805_555_555_555_56e-1 * t + 1.208_333_333_333_333_33e-4 * t * t,
            1.022_793_805_555_555_56e2 + (1.494_725_152_888_888_89e5 + 6.388_888_888_888_888_89e-6 * t) * t,
        ],
        Body::Venus => [
            0.723_331_60,
            0.006_820_69 - 0.000_047_74 * t + 0.000_000_091 * t * t,
            3.393_630_555_555_555_56 + 1.005_833_333_333_333_33e-3 * t - 9.722_222_222_222_222_22e-7 * t * t,
            7.577_964_722_222_222_22e1 + 8.998_5e-1 * t + 4.1e-4 * t * t,
            5.438_418_611_111_111_11e1 + 5.081_861_111_111_111_11e-1 * t - 1.386_388_888_888_888_89e-3 * t * t,
            2.126_032_194_444_444_44e2 + (5.851_780_387_5e4 + 1.286_055_555_555_555_56e-3 * t) * t,
        ],
        Body::Earth => [
            1.000_000_23,
            0.016_751_04 - 0.000_041_80 * t - 0.000_000_126 * t * t,
            0.0,
            0.0,
            1.012_208_333_333_333_33e2 + 1.719_175_0 * t + 4.527_777_777_777_777_78e-4 * t * t + 3.333_333_333_333_333_33e-6 * t * t * t,
            3.584_758_444_444_444_44e2 + (3.599_904_975e4 - 1.502_777_777_777_777_78e-4 * t - 3.333_333_333_333_333_33e-6 * t * t) * t,
        ],
        Body::Mars => [
            1.523_688_399,
            0.093_312_90 + 0.000_092_064 * t - 0.000_000_077 * t * t,
            1.850_333_333_333_333_33 - 6.75e-4 * t + 1.261_111_111_111_111_11e-5 * t * t,
            4.878_644_166_666_666_67e1 + 7.709_916_666_666_666_67e-1 * t - 1.388_888_888_888_888_89e-6 * t * t - 5.333_333_333_333_333_33e-6 * t * t * t,
            2.854_317_611_111_111_11e2 + 1.069_766_666_666_666_67 * t + 1.312_5e-4 * t * t + 4.138_888_888_888_888_89e-6 * t * t * t,
            3.195_294_25e2 + (1.913_985_85e4 + 1.808_055_555_555_555_56e-4 * t + 1.194_444_444_444_444_44e-6 * t * t) * t,
        ],
        Body::Jupiter => [
            5.202_561_0,
            0.048_334_75 + 0.000_164_18 * t - 0.000_000_467_6 * t * t - 0.000_000_001_7 * t * t * t,
            1.308_736_111_111_111_11 - 5.696_111_111_111_111_11e-3 * t + 3.888_888_888_888_888_89e-6 * t * t,
            9.944_338_611_111_111_11e1 + 1.010_53 * t + 3.522_222_222_222_222_22e-4 * t * t - 8.511_111_111_111_111_11e-6 * t * t * t,
            2.732_775_416_666_666_67e2 + 5.994_316_666_666_666_67e-1 * t + 7.040_5e-4 * t * t + 5.077_777_777_777_777_78e-6 * t * t * t,
            2.253_283_277_777_777_78e2 + (3.034_692_023_888_888_89e3 - 7.215_888_888_888_888_89e-4 * t + 1.784_444_444_444_444_44e-6 * t * t) * t,
        ],
        Body::Saturn => [
            9.554_747_0,
            0.055_892_32 - 0.000_345_5 * t - 0.000_000_728 * t * t + 0.000_000_000_74 * t * t * t,
            2.492_519_444_444_444_44 - 3.918_888_888_888_888_89e-3 * t - 1.548_888_888_888_888_89e-5 * t * t + 4.444_444_444_444_444_44e-8 * t * t * t,
            1.127_903_888_888_888_89e2 + 8.731_951_388_888_888_89e-1 * t - 1.521_805_555_555_555_56e-4 * t * t - 5.305_555_555_555_555_56e-6 * t * t * t,
            3.383_077_722_222_222_22e2 + 1.085_220_694_444_444_44 * t + 9.785_416_666_666_666_67e-4 * t * t + 9.916_666_666_666_666_67e-6 * t * t * t,
            1.754_662_166_666_666_67e2 + (1.221_551_467_777_777_78e3 - 5.018_194_444_444_444_44e-4 * t - 5.194_444_444_444_444_44e-6 * t * t) * t,
        ],
        Body::Uranus => [
            19.218_14,
            0.046_344_4 - 0.000_026_58 * t + 0.000_000_077 * t * t,
            7.724_638_888_888_888_89e-1 + 6.252_777_777_777_777_78e-4 * t + 3.95e-5 * t * t,
            7.347_709_722_222_222_22e1 + 4.986_677_777_777_777_78e-1 * t + 1.311_666_666_666_666_67e-3 * t * t,
            9.807_155_277_777_777_78e1 + 9.857_65e-1 * t - 1.074_472_222_222_222_22e-3 * t * t - 6.055_555_555_555_555_56e-7 * t * t * t,
            7.264_881_944_444_444_44e1 + (4.283_791_130_555_555_56e2 + 7.884_444_444_444_444_44e-5 * t + 1.111_111_111_111_111_11e-9 * t * t) * t,
        ],
        Body::Neptune => [
            30.109_57,
            0.008_997_04 + 0.000_006_33 * t - 0.000_000_002 * t * t,
            1.779_241_666_666_666_67 - 9.543_611_111_111_111_11e-3 * t - 9.111_111_111_111_111_11e-6 * t * t,
            1.306_813_583_333_333_33e2 + 1.098_935_0 * t + 2.498_666_666_666_666_67e-4 * t * t - 4.717_777_777_777_777_78e-6 * t * t * t,
            2.760_459_666_666_666_67e2 + 3.256_394_444_444_444_44e-1 * t + 1.409_5e-4 * t * t + 4.113_333_333_333_333_33e-6 * t * t * t,
            3.773_066_944_444_444_44e1 + (2.184_613_397_222_222_22e2 - 7.033_333_333_333_333_33e-5 * t) * t,
        ],
        other => panic!("gtop_elements: no GTOP elements for {other:?}"),
    }
}

/// GTOP's exact analytic planetary state at `mjd2000` [days], in GTOP's
/// heliocentric "j2000" frame (ecliptic) rotated to ICRF for consistency
/// with the rest of this codebase. Position [m], velocity [m/s].
///
/// Faithful reimplementation of `Planet_Ephemerides_Analytical` +
/// `Conversion` from gtopx.cpp: Kepler solve by Newton (initial guess
/// `E = M + e·cos M`, tolerance 1e-13), perifocal state from eccentric
/// anomaly, rotation R3(−Ω)·R1(−i)·R3(−ω).
pub fn gtop_state_icrf(body: Body, mjd2000: f64) -> (Vector3<f64>, Vector3<f64>) {
    let t = (mjd2000 + 36_525.0) / 36_525.0;
    let el = gtop_elements(body, t);

    let a = el[0] * GTOP_AU_M;
    let e = el[1];
    let i = el[2].to_radians();
    let node = el[3].to_radians();
    let peri = el[4].to_radians();
    let m = el[5].to_radians() % (2.0 * std::f64::consts::PI);

    // Kepler solve, exactly GTOP's `Mean2Eccentric` (Newton, guess M + e·cosM).
    let mut ea = m + e * m.cos();
    for _ in 0..100 {
        let ea_new = ea - (ea - e * ea.sin() - m) / (1.0 - e * ea.cos());
        let err = (ea - ea_new).abs();
        ea = ea_new;
        if err <= 1e-13 {
            break;
        }
    }

    let b = a * (1.0 - e * e).sqrt();
    let n = (GTOP_MU_SUN_M3S2 / (a * a * a)).sqrt();
    let (s_ea, c_ea) = ea.sin_cos();
    let x_pf = a * (c_ea - e);
    let y_pf = b * s_ea;
    let xd_pf = -(a * n * s_ea) / (1.0 - e * c_ea);
    let yd_pf = (b * n * c_ea) / (1.0 - e * c_ea);

    let (so, co) = peri.sin_cos();
    let (sn, cn) = node.sin_cos();
    let (si, ci) = i.sin_cos();
    let rot = |x: f64, y: f64| -> Vector3<f64> {
        Vector3::new(
            (cn * co - sn * so * ci) * x + (-cn * so - sn * co * ci) * y,
            (sn * co + cn * so * ci) * x + (-sn * so + cn * co * ci) * y,
            (so * si) * x + (co * si) * y,
        )
    };
    (ecliptic_to_icrf(rot(x_pf, y_pf)), ecliptic_to_icrf(rot(xd_pf, yd_pf)))
}
