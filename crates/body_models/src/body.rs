//! Target body definition and named constructors for well-known solar system bodies.

use crate::{AtmosphereModel, GravityModel};

/// Astronomical unit [m] — IAU 2012 definition (exact): 1.495 978 707 × 10¹¹ m.
/// Used to keep the source tables' AU-denominated semi-major axes verbatim in
/// the constructors below (the multiplication happens in code, not by hand).
const AU_M: f64 = 1.495_978_707e11;

/// Heliocentric Keplerian orbital elements at a stated epoch, referred to the
/// mean ecliptic and equinox of J2000 — the native frame of both source
/// tables (Standish 1992 for the planets, JPL Small-Body Database for the
/// asteroids). NOT the equatorial ICRF frame ANISE state vectors use — a
/// consumer mixing the two must rotate by the J2000 obliquity (23.439 291 11°).
///
/// Purpose: a real catalog-served orbit shape
/// for every heliocentric body, so display consumers can draw true elliptical
/// orbit rings — including for small bodies with no ephemeris-kernel coverage
/// — instead of hardcoding a client-side element table. These are MEAN/
/// osculating elements at one epoch with no secular rates: fine for orbit
/// rings and rough position estimates, not a substitute for real ephemeris
/// (`/api/bodies/{name}/state`) where one exists.
#[derive(Clone, Copy, Debug)]
pub struct OrbitalElements {
    /// Semi-major axis [m].
    pub sma_m: f64,
    /// Eccentricity [-].
    pub eccentricity: f64,
    /// Inclination to the J2000 ecliptic [deg].
    pub inclination_deg: f64,
    /// Longitude of the ascending node Ω [deg].
    pub raan_deg: f64,
    /// Argument of periapsis ω [deg]. For the planets this is derived in-code
    /// as ϖ − Ω from Standish's (L, ϖ, Ω) parameterization — the source
    /// values stay verbatim in the expression.
    pub arg_periapsis_deg: f64,
    /// Mean anomaly at `epoch_jd` [deg]. For the planets: L − ϖ (Standish).
    pub mean_anomaly_deg: f64,
    /// Epoch of the elements [Julian Date, TDB]. J2000.0 (2451545.0) for the
    /// planets; the JPL SBDB solution epoch for each small body.
    pub epoch_jd: f64,
}

/// All physical properties of a target body needed for mission design and simulation.
///
/// Build via a named constructor (`TargetBody::bennu()`, `TargetBody::mars()`, …)
/// or construct directly from a parsed TOML mission config (via `MissionPlanner`).
#[derive(Clone, Debug)]
pub struct TargetBody {
    pub name: String,
    /// Body classification for frontend grouping: "Planet", "Moon", "Asteroid", "Star".
    pub kind: &'static str,
    /// Gravitational parameter [m³/s²]
    pub mu_m3s2: f64,
    /// Mean equatorial radius [m]
    pub radius_m: f64,
    /// Gravity model — determines which perturbation accelerations are computed
    pub gravity: GravityModel,
    /// Atmosphere model — determines whether aerodynamic drag is computed
    pub atmosphere: AtmosphereModel,
    /// Body spin rate [rad/s], positive = prograde
    pub spin_rate_rads: f64,
    /// North pole right ascension [deg], ICRF/J2000 constant term. `None`
    /// when no citable pole orientation is available (currently: all small
    /// bodies in this catalog) — callers applying zonal-harmonic gravity
    /// must fall back to point-mass central fidelity and warn rather than
    /// guess an orientation. See `gravity::zonal_harmonics_body_oriented`.
    pub pole_ra_deg: Option<f64>,
    /// North pole declination [deg], ICRF/J2000 constant term. See `pole_ra_deg`.
    pub pole_dec_deg: Option<f64>,
    /// The body this one actually orbits, by catalog name — `None` means
    /// "orbits the Sun directly" (true for every planet/asteroid in this
    /// catalog, and trivially true for the Sun itself). `Some("Earth")` for
    /// the Moon, `Some("Jupiter")` for Europa, `Some("Saturn")` for Titan,
    /// `Some("Mars")` for Phobos/Deimos — i.e. every moon. This matters
    /// because the Laplace sphere-of-influence formula (`R_soi = a ·
    /// (m_body/m_primary)^(2/5)`) needs the body's *actual* primary, not
    /// the Sun, to give a physically meaningful radius — using the Sun for
    /// a moon (its heliocentric distance/mass ratio) computes a number,
    /// just not its real SOI. See the design notes Phase 7k/8h.
    pub primary: Option<&'static str>,
    /// Mean orbital semi-major axis around `primary` (or around the Sun when
    /// `primary` is `None`) [m]. `None` for small bodies whose orbits are
    /// epoch-dependent and not catalogued here (Bennu, Apophis, Ryugu, Eros,
    /// Didymos). Used by the Tisserand-graph beam search (Phase 9j) for the
    /// circular-orbit approximation — see `sequence_search.rs`.
    ///
    /// Source (planets): JPL Planetary Fact Sheet, Williams (2021),
    /// ssd.jpl.nasa.gov/planets/phys_par.html.
    /// Source (moons): JPL Solar System Dynamics satellite physical parameters,
    /// ssd.jpl.nasa.gov/sats/phys_par.html.
    pub sma_m: Option<f64>,
    /// Full heliocentric orbital elements (see [`OrbitalElements`]).
    /// `Some` for the 8 planets (Standish 1992 J2000 mean elements) and the
    /// 5 catalogued small bodies (JPL SBDB osculating elements at each
    /// body's stated solution epoch). `None` for the Sun (frame origin) and
    /// for every moon — a moon's elements are PRIMARY-relative, a different
    /// contract than this heliocentric field; add a separate field if that's
    /// ever needed rather than overloading this one.
    pub orbital_elements: Option<OrbitalElements>,
}

impl TargetBody {
    /// Bennu (101955 Bennu).
    /// μ: Chesley et al. 2020 (Icarus). R: Lauretta et al. 2019 (Science).
    /// J2–J4: Scheeres et al. 2020 (Science Advances). Spin: Nolan et al. 2013 (Icarus).
    pub fn bennu() -> Self {
        Self {
            name: "Bennu".into(),
            kind: "Asteroid",
            mu_m3s2: 4.89,
            radius_m: 262.0,
            gravity: GravityModel::J2J3J4 {
                j2: 1.926_101e-2,
                j3: -1.221_940e-3,
                j4: -6.496_002e-3,
            },
            atmosphere: AtmosphereModel::None,
            // 4.297 h rotation period (Nolan et al. 2013)
            spin_rate_rads: 4.0628e-4,
            // Pole orientation deliberately not modeled for small bodies in
            // this catalog yet — callers fall back to point-mass central
            // fidelity with a warning. See `pole_ra_deg` doc comment.
            pole_ra_deg: None,
            pole_dec_deg: None,
            primary: None,
            sma_m: None,  // epoch-dependent, not catalogued here
            // JPL SBDB osculating elements, epoch JD 2455562.5 (2011-01-01);
            // verified.
            orbital_elements: Some(OrbitalElements {
                sma_m: 1.126 * AU_M,
                eccentricity: 0.2037,
                inclination_deg: 6.035,
                raan_deg: 2.061,
                arg_periapsis_deg: 66.22,
                mean_anomaly_deg: 101.70,
                epoch_jd: 2_455_562.5,
            }),
        }
    }

    /// Mercury. μ, R: Anderson et al. 2012 (Science 336, MESSENGER flyby gravity).
    /// Atmosphere: effectively none (exosphere, negligible for trajectory design).
    /// Pole: Archinal et al. 2018 (IAU WGCCRE 2015, CeMDA 130:22), constant terms only.
    /// SMA: JPL Williams 2021 (planets fact sheet).
    pub fn mercury() -> Self {
        Self {
            name: "Mercury".into(),
            kind: "Planet",
            mu_m3s2: 2.203_208_863_5e13,  // Anderson et al. 2012
            radius_m: 2_439_400.0,          // MESSENGER (Smith et al. 2012)
            gravity: GravityModel::PointMass, // J2 ≈ 5.0e-5 (Margot et al. 2012), negligible
            atmosphere: AtmosphereModel::None,
            spin_rate_rads: 1.240_013_5e-6,  // 58.646 d sidereal period
            pole_ra_deg: Some(281.0103),
            pole_dec_deg: Some(61.4155),
            primary: None,
            sma_m: Some(5.791e10),           // 0.387 AU (JPL Williams 2021)
            // Standish (1992) J2000 mean elements: a=0.38709927 AU,
            // e=0.20563593, i=7.00497902°, L=252.25032350°, ϖ=77.45779628°,
            // Ω=48.33076593°.
            orbital_elements: Some(OrbitalElements {
                sma_m: 0.387_099_27 * AU_M,
                eccentricity: 0.205_635_93,
                inclination_deg: 7.004_979_02,
                raan_deg: 48.330_765_93,
                arg_periapsis_deg: 77.457_796_28 - 48.330_765_93,
                mean_anomaly_deg: 252.250_323_50 - 77.457_796_28,
                epoch_jd: 2_451_545.0,
            }),
        }
    }

    /// Venus. μ, R: Konopliv et al. 1999 (Icarus 139, Magellan gravity).
    /// Atmosphere: dense CO₂ atmosphere; scale height ≈ 15.9 km at cloud top ~60 km
    /// (Seiff et al. 1985) — modelled with effective surface density scaled to
    /// be consistent with the 8.5 km lower-troposphere scale height used for
    /// trajectory-planning drag estimates (surface, not 60 km).
    /// Pole: Archinal et al. 2018 (IAU WGCCRE 2015, CeMDA 130:22), Venus pole.
    /// Note: Venus rotates retrograde (spin_rate < 0 by convention).
    /// SMA: JPL Williams 2021.
    pub fn venus() -> Self {
        Self {
            name: "Venus".into(),
            kind: "Planet",
            mu_m3s2: 3.248_585_920_790e14, // Konopliv et al. 1999
            radius_m: 6_051_800.0,           // IAU 2009 Working Group (Archinal et al. 2011)
            gravity: GravityModel::PointMass, // J2 ≈ 4.458e-6 (Konopliv & Sjogren 1996) — negligible
            atmosphere: AtmosphereModel::Exponential {
                scale_height_m: 15_900.0,    // Seiff et al. 1985, at ~60 km altitude
                rho0_kg_m3: 65.0,            // approximate Venus cloud-top density [kg/m³]
            },
            spin_rate_rads: -2.992_2e-7,     // 243.025 d retrograde (IAU 2009)
            pole_ra_deg: Some(272.76),
            pole_dec_deg: Some(67.16),
            primary: None,
            sma_m: Some(1.082e11),           // 0.723 AU (JPL Williams 2021)
            // Standish (1992) J2000 mean elements: a=0.72333566 AU,
            // e=0.00677672, i=3.39467605°, L=181.97909950°, ϖ=131.60246718°,
            // Ω=76.67984255°.
            orbital_elements: Some(OrbitalElements {
                sma_m: 0.723_335_66 * AU_M,
                eccentricity: 0.006_776_72,
                inclination_deg: 3.394_676_05,
                raan_deg: 76.679_842_55,
                arg_periapsis_deg: 131.602_467_18 - 76.679_842_55,
                mean_anomaly_deg: 181.979_099_50 - 131.602_467_18,
                epoch_jd: 2_451_545.0,
            }),
        }
    }

    /// Earth. μ, R: IERS Conventions 2010. J2–J4: EGM96 (Lemoine et al. 1998).
    /// Atmosphere: standard sea-level density, 8.5 km scale height (US Standard 1976).
    /// Pole: ICRF/J2000 is defined by Earth's mean equator and pole at J2000.0
    /// (IERS Conventions 2010) — RA=0/Dec=90° is exact by construction, not a
    /// measured constant like the other bodies below.
    pub fn earth() -> Self {
        Self {
            name: "Earth".into(),
            kind: "Planet",
            mu_m3s2: 3.986_004_418e14,
            radius_m: 6_378_137.0,
            gravity: GravityModel::J2J3J4 {
                j2: 1.082_626_68e-3,
                j3: -2.532_4e-6,
                j4: -1.619_8e-6,
            },
            atmosphere: AtmosphereModel::Exponential {
                scale_height_m: 8_500.0,
                rho0_kg_m3: 1.225,
            },
            spin_rate_rads: 7.292_115e-5,
            pole_ra_deg: Some(0.0),
            pole_dec_deg: Some(90.0),
            primary: None,
            // 1 AU — IAU 2012 definition: exactly 1.495 978 707 × 10¹¹ m
            sma_m: Some(1.495_978_707e11),
            // Standish (1992) J2000 mean elements for the Earth–Moon
            // barycenter: a=1.00000261 AU, e=0.01671123, i=-0.00001531°,
            // L=100.46457166°, ϖ=102.93768193°, Ω=0.0°. M normalized +360
            // (L − ϖ is slightly negative at J2000).
            orbital_elements: Some(OrbitalElements {
                sma_m: 1.000_002_61 * AU_M,
                eccentricity: 0.016_711_23,
                inclination_deg: -0.000_015_31,
                raan_deg: 0.0,
                arg_periapsis_deg: 102.937_681_93,
                mean_anomaly_deg: 100.464_571_66 - 102.937_681_93 + 360.0,
                epoch_jd: 2_451_545.0,
            }),
        }
    }

    /// Moon. μ, R: DE430, Folkner et al. 2014. J2–J4: GRAIL (Zuber et al. 2013).
    /// Pole: mean pole, Archinal et al. 2018 ("Report of the IAU WGCCRE: 2015",
    /// CeMDA 130:22), lunar pole table, constant term only. Caveat: unlike the
    /// other bodies here, the Moon's true pole precesses on the ~18.6-yr lunar
    /// nodal (Cassini state) cycle with periodic terms of several tenths of a
    /// degree — dropped here. Fine for Artemis-class missions of ~10 days;
    /// degrades faster with mission duration than for the other bodies below.
    pub fn moon() -> Self {
        Self {
            name: "Moon".into(),
            kind: "Moon",
            mu_m3s2: 4.902_800_066e12,
            radius_m: 1_737_400.0,
            gravity: GravityModel::J2J3J4 {
                j2: 2.032_3e-4,
                j3: -8.468e-6,
                j4: -9.009e-6,
            },
            atmosphere: AtmosphereModel::None,
            // Synchronous rotation ≈ 27.321 d
            spin_rate_rads: 2.661_7e-6,
            pole_ra_deg: Some(269.9949),
            pole_dec_deg: Some(66.5392),
            primary: Some("Earth"),
            // Mean Earth–Moon distance (JPL, Williams 2021)
            sma_m: Some(3.844e8),
            orbital_elements: None,  // moon — primary-relative, see field doc
        }
    }

    /// Mars. μ, R: Konopliv et al. 2011 (MRO gravity). J2–J4: MRO/MGS solution.
    /// Atmosphere: Zurek & Smrekar 2007 (approximate exponential fit at low altitude).
    /// Pole: Archinal et al. 2018 (IAU WGCCRE 2015, CeMDA 130:22), Mars pole
    /// table, constant term only (secular rate ~-0.00061°/century — negligible
    /// over a single mission).
    pub fn mars() -> Self {
        Self {
            name: "Mars".into(),
            kind: "Planet",
            mu_m3s2: 4.282_837_362_069_909e13,
            radius_m: 3_396_200.0,
            gravity: GravityModel::J2J3J4 {
                j2: 1.960_45e-3,
                j3: 3.142_5e-5,
                j4: -1.538_5e-5,
            },
            atmosphere: AtmosphereModel::Exponential {
                scale_height_m: 11_100.0,
                rho0_kg_m3: 0.020,
            },
            spin_rate_rads: 7.088e-5,
            pole_ra_deg: Some(317.269202),
            pole_dec_deg: Some(54.432516),
            primary: None,
            // 1.524 AU (JPL, Williams 2021)
            sma_m: Some(2.279_368_7e11),
            // Standish (1992) J2000 mean elements: a=1.52371034 AU,
            // e=0.09339410, i=1.84969142°, L=-4.55343205°, ϖ=-23.94362959°,
            // Ω=49.55953891°. ω normalized +360 (ϖ − Ω is negative).
            orbital_elements: Some(OrbitalElements {
                sma_m: 1.523_710_34 * AU_M,
                eccentricity: 0.093_394_10,
                inclination_deg: 1.849_691_42,
                raan_deg: 49.559_538_91,
                arg_periapsis_deg: -23.943_629_59 - 49.559_538_91 + 360.0,
                mean_anomaly_deg: -4.553_432_05 - -23.943_629_59,
                epoch_jd: 2_451_545.0,
            }),
        }
    }

    /// Apophis (99942 Apophis). μ: Brozović et al. 2018 (Icarus). R: Pravec et al. 2014.
    /// Gravity: point mass (J2 not yet characterised pre-flyby).
    pub fn apophis() -> Self {
        Self {
            name: "Apophis".into(),
            kind: "Asteroid",
            mu_m3s2: 2.646e-3,
            radius_m: 185.0,
            gravity: GravityModel::PointMass,
            atmosphere: AtmosphereModel::None,
            // ~30.56 h rotation period (Pravec et al. 2014)
            spin_rate_rads: 5.714e-5,
            pole_ra_deg: None,
            pole_dec_deg: None,
            primary: None,
            sma_m: None,
            // JPL Horizons/SBDB osculating elements, epoch JD 2460800.5
            // (2025-05-05, post-uncertainty-collapse solution from the 2021
            // radar campaign); verified. Note Apophis's 2029-04-13
            // Earth flyby substantially changes these elements — a mission
            // spanning that date needs a post-flyby solution instead.
            orbital_elements: Some(OrbitalElements {
                sma_m: 0.9224 * AU_M,
                eccentricity: 0.1911,
                inclination_deg: 3.341,
                raan_deg: 203.9,
                arg_periapsis_deg: 126.7,
                mean_anomaly_deg: 90.28,
                epoch_jd: 2_460_800.5,
            }),
        }
    }

    /// Ryugu (162173 Ryugu). μ: Watanabe et al. 2019 (Science). R: Sugita et al. 2019.
    /// J2–J4: Scheeres et al. 2019 (Science). Pole orientation (Watanabe et al. 2019)
    /// deliberately not wired up here — small-body poles are out of scope for this
    /// catalog's zonal-harmonic fidelity until central-body propagation actually
    /// needs one; central-body gravity falls back to point-mass with a warning.
    pub fn ryugu() -> Self {
        Self {
            name: "Ryugu".into(),
            kind: "Asteroid",
            mu_m3s2: 30.03,
            radius_m: 448.0,
            gravity: GravityModel::J2J3J4 {
                j2: 6.75e-2,
                j3: 0.0,
                j4: -4.78e-2,
            },
            atmosphere: AtmosphereModel::None,
            // 7.632 h rotation period (Watanabe et al. 2019)
            spin_rate_rads: 2.285e-4,
            pole_ra_deg: None,
            pole_dec_deg: None,
            primary: None,
            sma_m: None,
            // JPL SBDB osculating elements, epoch JD 2455907.5 (2011-12-12);
            // verified.
            orbital_elements: Some(OrbitalElements {
                sma_m: 1.1896 * AU_M,
                eccentricity: 0.1902,
                inclination_deg: 5.8837,
                raan_deg: 251.62,
                arg_periapsis_deg: 211.43,
                mean_anomaly_deg: 3.9832,
                epoch_jd: 2_455_907.5,
            }),
        }
    }

    /// Sun. μ, R: IAU 2015 Working Group on Nominal Solar and Planetary Radii (Mamajek et al. 2015).
    /// Modelled as point mass for third-body perturbation use; J2 = 2.21e-7 (Pireaux & Rozelot 2003)
    /// is negligible for all but Mercury-like orbits.
    pub fn sun() -> Self {
        Self {
            name: "Sun".into(),
            kind: "Star",
            mu_m3s2: 1.327_124_400_18e20,
            radius_m: 695_700_000.0,
            gravity: GravityModel::PointMass,
            atmosphere: AtmosphereModel::None,
            // Carrington rotation ~25.38 d (equatorial sidereal, Snodgrass & Ulrich 1990)
            spin_rate_rads: 2.865e-6,
            pole_ra_deg: None,
            pole_dec_deg: None,
            primary: None,
            sma_m: None,  // Sun is the reference frame origin, not an orbiting body
            orbital_elements: None,
        }
    }

    /// Jupiter. μ, R: Folkner et al. 2014 (JPL DE430).
    /// J2, J4: Iess et al. 2018 (Nature 555, 220) from Juno gravity science.
    /// J3 is consistent with zero within Juno measurement uncertainty.
    /// Atmosphere not modelled — drag is negligible at orbital altitudes above Jupiter.
    /// Pole: Archinal et al. 2018 (IAU WGCCRE 2015, CeMDA 130:22), Jupiter pole
    /// table, constant term only (secular rate -0.006499°/century — negligible).
    pub fn jupiter() -> Self {
        Self {
            name: "Jupiter".into(),
            kind: "Planet",
            mu_m3s2: 1.267_127_678e17,
            radius_m: 71_492_000.0,
            gravity: GravityModel::J2J3J4 {
                j2: 1.469_6e-2,
                j3: 0.0,
                j4: -5.871_4e-4,
            },
            atmosphere: AtmosphereModel::None,
            // 9.925 h rotation period — System III (Seidelmann et al. 2007 IAU report)
            spin_rate_rads: 1.758_5e-4,
            pole_ra_deg: Some(268.056595),
            pole_dec_deg: Some(64.495303),
            primary: None,
            // 5.203 AU (JPL, Williams 2021)
            sma_m: Some(7.783_286_4e11),
            // Standish (1992) J2000 mean elements: a=5.20288700 AU,
            // e=0.04838624, i=1.30439695°, L=34.39644051°, ϖ=14.72847983°,
            // Ω=100.47390909°. ω normalized +360.
            orbital_elements: Some(OrbitalElements {
                sma_m: 5.202_887_00 * AU_M,
                eccentricity: 0.048_386_24,
                inclination_deg: 1.304_396_95,
                raan_deg: 100.473_909_09,
                arg_periapsis_deg: 14.728_479_83 - 100.473_909_09 + 360.0,
                mean_anomaly_deg: 34.396_440_51 - 14.728_479_83,
                epoch_jd: 2_451_545.0,
            }),
        }
    }

    /// Saturn. μ: Jacobson et al. (SAT441), via JPL Solar System Dynamics
    /// Planetary Physical Parameters (ssd.jpl.nasa.gov/planets/phys_par.html).
    /// R: Archinal et al. 2018 (IAU WGCCRE 2015). Added only as Titan's real
    /// orbital primary (Phase 7k/8h's SOI-primary fix) — gravity fidelity
    /// not modeled (no cited J2/J3/J4/pole wired up here yet); falls back to
    /// point-mass central fidelity with a warning if ever needed as central body.
    pub fn saturn() -> Self {
        Self {
            name: "Saturn".into(),
            kind: "Planet",
            mu_m3s2: 3.793_120_6e16,
            radius_m: 60_268_000.0,
            gravity: GravityModel::PointMass,
            atmosphere: AtmosphereModel::None,
            // 10.656 h rotation period — System III (Seidelmann et al. 2007 IAU report)
            spin_rate_rads: 1.637_8e-4,
            pole_ra_deg: None,
            pole_dec_deg: None,
            primary: None,
            // 9.537 AU (JPL, Williams 2021)
            sma_m: Some(1.426_666_4e12),
            // Standish (1992) J2000 mean elements: a=9.53667594 AU,
            // e=0.05386179, i=2.48599187°, L=49.95424423°, ϖ=92.59887831°,
            // Ω=113.66242448°. ω and M normalized +360.
            orbital_elements: Some(OrbitalElements {
                sma_m: 9.536_675_94 * AU_M,
                eccentricity: 0.053_861_79,
                inclination_deg: 2.485_991_87,
                raan_deg: 113.662_424_48,
                arg_periapsis_deg: 92.598_878_31 - 113.662_424_48 + 360.0,
                mean_anomaly_deg: 49.954_244_23 - 92.598_878_31 + 360.0,
                epoch_jd: 2_451_545.0,
            }),
        }
    }

    /// Europa (moon of Jupiter). μ: Anderson et al. 1998 (Icarus 135, 390), Galileo gravity science.
    /// R: IAU 2015. J2: Anderson et al. 1998.
    /// Atmosphere: trace oxygen exosphere — aerodynamic drag is negligible.
    /// Pole: Archinal et al. 2018 (IAU WGCCRE 2015), satellite table — Europa's
    /// pole is reported near-coincident with Jupiter's; small periodic Laplace-
    /// plane terms (<0.1°) dropped, consistent with this catalog's fast-over-
    /// precise approximation elsewhere.
    pub fn europa() -> Self {
        Self {
            name: "Europa".into(),
            kind: "Moon",
            mu_m3s2: 3.202_738_774_922_892e12,
            radius_m: 1_560_800.0,
            gravity: GravityModel::J2 { j2: 4.35e-4 },
            atmosphere: AtmosphereModel::None,
            // Synchronous rotation: orbital period 3.5512 d (Seidelmann et al. 2007)
            spin_rate_rads: 2.048e-5,
            pole_ra_deg: Some(268.08),
            pole_dec_deg: Some(64.51),
            primary: Some("Jupiter"),
            // 671 100 km from Jupiter (JPL Solar System Dynamics)
            sma_m: Some(6.711e8),
            orbital_elements: None,  // moon — primary-relative, see field doc
        }
    }

    /// Titan (moon of Saturn). μ: Iess et al. 2012 (Science 337, 457), Cassini gravity.
    /// R: IAU 2015. J2: Iess et al. 2010 (Science 327, 1367).
    /// Atmosphere: dense nitrogen, ~1.5 bar surface (Fulchignoni et al. 2005).
    /// Pole: Archinal et al. 2018 (IAU WGCCRE 2015), Saturn-satellite table,
    /// constant term only — librational/nodal terms from Saturn's satellite
    /// theory dropped, small relative to a single mission's duration.
    pub fn titan() -> Self {
        Self {
            name: "Titan".into(),
            kind: "Moon",
            mu_m3s2: 8.978_138_376_618_386e12,
            radius_m: 2_575_500.0,
            gravity: GravityModel::J2 { j2: 3.18e-5 },
            // Scale height H = RT/(Mg) ≈ 21 km at 94 K, N₂, g = 1.35 m/s² (Fulchignoni 2005)
            // Sea-level density ≈ 5.3 kg/m³ at 94 K and 1.467 bar
            atmosphere: AtmosphereModel::Exponential {
                scale_height_m: 21_000.0,
                rho0_kg_m3: 5.3,
            },
            // Synchronous rotation: orbital period 15.945 d (Seidelmann et al. 2007)
            spin_rate_rads: 4.560_6e-6,
            pole_ra_deg: Some(39.4827),
            pole_dec_deg: Some(83.4279),
            primary: Some("Saturn"),
            // 1 221 870 km from Saturn (JPL Solar System Dynamics)
            sma_m: Some(1.221_870e9),
            orbital_elements: None,  // moon — primary-relative, see field doc
        }
    }

    /// Phobos (inner moon of Mars). μ: Jacobson et al. 2014 (AJ 148, 76), Mars Express flybys.
    /// R: Willner et al. 2010 (Planetary and Space Science 58, 1870).
    /// J2 not well-constrained from remote sensing; modelled as point mass.
    pub fn phobos() -> Self {
        Self {
            name: "Phobos".into(),
            kind: "Moon",
            mu_m3s2: 7.087_546e5,
            radius_m: 11_100.0,
            gravity: GravityModel::PointMass,
            atmosphere: AtmosphereModel::None,
            // Synchronous rotation: orbital period 7.6540 h (Seidelmann et al. 2007)
            spin_rate_rads: 2.279e-4,
            pole_ra_deg: None,
            pole_dec_deg: None,
            primary: Some("Mars"),
            // 9 376 km from Mars centre (JPL Solar System Dynamics)
            sma_m: Some(9.376e6),
            orbital_elements: None,  // moon — primary-relative, see field doc
        }
    }

    /// Deimos (outer moon of Mars). μ: Jacobson 2010 (AJ 139, 668).
    /// R: Thomas 1993 (Icarus 105, 326).
    pub fn deimos() -> Self {
        Self {
            name: "Deimos".into(),
            kind: "Moon",
            mu_m3s2: 9.615_9e4,
            radius_m: 6_200.0,
            gravity: GravityModel::PointMass,
            atmosphere: AtmosphereModel::None,
            // Synchronous rotation: orbital period 30.312 h (Seidelmann et al. 2007)
            spin_rate_rads: 5.757e-5,
            pole_ra_deg: None,
            pole_dec_deg: None,
            primary: Some("Mars"),
            // 23 463.2 km from Mars centre (JPL Solar System Dynamics)
            sma_m: Some(2.346_32e7),
            orbital_elements: None,  // moon — primary-relative, see field doc
        }
    }

    /// 433 Eros. μ: Yeomans et al. 2000 (Science 289, 2085), NEAR Shoemaker radio science.
    /// R: Thomas et al. 2002 (Icarus 155, 18).
    /// J2–J4: Miller et al. 2002 (Icarus 155, 3) from NEAR Shoemaker gravity science in orbit.
    /// Pole orientation (also Yeomans et al. 2000) deliberately not wired up — see
    /// the Ryugu doc comment above for why small-body poles are out of scope here.
    pub fn eros() -> Self {
        Self {
            name: "Eros".into(),
            kind: "Asteroid",
            mu_m3s2: 4.463e5,
            radius_m: 8_416.0,
            gravity: GravityModel::J2J3J4 {
                j2: 1.126e-1,
                j3: -3.01e-2,
                j4: 3.86e-2,
            },
            atmosphere: AtmosphereModel::None,
            // 5.270 h rotation period (Yeomans et al. 2000)
            spin_rate_rads: 3.313e-4,
            pole_ra_deg: None,
            pole_dec_deg: None,
            primary: None,
            sma_m: None,
            // JPL SBDB osculating elements, epoch JD 2458000.5 (2017-09-04);
            // verified.
            orbital_elements: Some(OrbitalElements {
                sma_m: 1.4579 * AU_M,
                eccentricity: 0.2226,
                inclination_deg: 10.828,
                raan_deg: 304.32,
                arg_periapsis_deg: 178.82,
                mean_anomaly_deg: 71.280,
                epoch_jd: 2_458_000.5,
            }),
        }
    }

    /// 65803 Didymos primary. μ (system): Scheirich et al. 2022 (PSJ 3, 160), post-DART.
    /// R: Naidu et al. 2020 (Icarus 348, 113777) for the primary.
    /// J2 not characterised — modelled as point mass; Dimorphos treated separately.
    pub fn didymos() -> Self {
        Self {
            name: "Didymos".into(),
            kind: "Asteroid",
            mu_m3s2: 41.0,
            radius_m: 390.0,
            gravity: GravityModel::PointMass,
            atmosphere: AtmosphereModel::None,
            // 2.2600 h rotation period (Naidu et al. 2020)
            spin_rate_rads: 7.716e-4,
            pole_ra_deg: None,
            pole_dec_deg: None,
            primary: None,
            sma_m: None,
            // JPL SBDB osculating elements, epoch JD 2459600.5 (2022-01-21,
            // pre-DART; the DART impact's ~mm/s heliocentric ΔV is far below
            // this catalog's precision); verified.
            orbital_elements: Some(OrbitalElements {
                sma_m: 1.6442 * AU_M,
                eccentricity: 0.38385,
                inclination_deg: 3.4079,
                raan_deg: 73.196,
                arg_periapsis_deg: 319.32,
                mean_anomaly_deg: 232.01,
                epoch_jd: 2_459_600.5,
            }),
        }
    }

    /// Uranus. μ: Jacobson (2014), AJ 148, 76.
    /// R: Archinal et al. (2018), IAU WGCCRE 2015, CeMDA 130, 22.
    pub fn uranus() -> Self {
        Self {
            name: "Uranus".into(),
            kind: "Planet",
            mu_m3s2: 5.793_951_3e15,
            radius_m: 25_559_000.0,
            gravity: GravityModel::PointMass,
            atmosphere: AtmosphereModel::None,
            // 17.24 h (retrograde — Archinal et al. 2018)
            spin_rate_rads: -1.012e-4,
            pole_ra_deg: None,
            pole_dec_deg: None,
            primary: None,
            // 19.19 AU (JPL Williams 2021)
            sma_m: Some(2.870_658_2e12),
            // Standish (1992) J2000 mean elements: a=19.18916464 AU,
            // e=0.04725744, i=0.77263783°, L=313.23810451°, ϖ=170.95427630°,
            // Ω=74.01692503°.
            orbital_elements: Some(OrbitalElements {
                sma_m: 19.189_164_64 * AU_M,
                eccentricity: 0.047_257_44,
                inclination_deg: 0.772_637_83,
                raan_deg: 74.016_925_03,
                arg_periapsis_deg: 170.954_276_30 - 74.016_925_03,
                mean_anomaly_deg: 313.238_104_51 - 170.954_276_30,
                epoch_jd: 2_451_545.0,
            }),
        }
    }

    /// Neptune. μ: Jacobson (2009), AJ 137, 4322.
    /// R: Archinal et al. (2018), IAU WGCCRE 2015, CeMDA 130, 22.
    pub fn neptune() -> Self {
        Self {
            name: "Neptune".into(),
            kind: "Planet",
            mu_m3s2: 6.836_527_1e15,
            radius_m: 24_764_000.0,
            gravity: GravityModel::PointMass,
            atmosphere: AtmosphereModel::None,
            // 16.11 h (Archinal et al. 2018)
            spin_rate_rads: 1.083e-4,
            pole_ra_deg: None,
            pole_dec_deg: None,
            primary: None,
            // 30.07 AU (JPL Williams 2021)
            sma_m: Some(4.498_396_4e12),
            // Standish (1992) J2000 mean elements: a=30.06992276 AU,
            // e=0.00859048, i=1.77004347°, L=-55.12002969°, ϖ=44.96476227°,
            // Ω=131.78422574°. ω and M normalized +360.
            orbital_elements: Some(OrbitalElements {
                sma_m: 30.069_922_76 * AU_M,
                eccentricity: 0.008_590_48,
                inclination_deg: 1.770_043_47,
                raan_deg: 131.784_225_74,
                arg_periapsis_deg: 44.964_762_27 - 131.784_225_74 + 360.0,
                mean_anomaly_deg: -55.120_029_69 - 44.964_762_27 + 360.0,
                epoch_jd: 2_451_545.0,
            }),
        }
    }

    /// All catalog presets, for name-based lookup (see `by_name`).
    pub fn catalog() -> Vec<Self> {
        vec![
            Self::sun(),
            Self::mercury(),
            Self::venus(),
            Self::bennu(),
            Self::earth(),
            Self::moon(),
            Self::mars(),
            Self::apophis(),
            Self::ryugu(),
            Self::jupiter(),
            Self::saturn(),
            Self::europa(),
            Self::titan(),
            Self::phobos(),
            Self::deimos(),
            Self::eros(),
            Self::didymos(),
            Self::uranus(),
            Self::neptune(),
        ]
    }

    /// Looks up a catalog preset by name, case-insensitive.
    ///
    /// Used by `MissionPlanner::config::TargetBodyConfig` to resolve a TOML
    /// `target_body.name` into physical constants when the mission file
    /// omits `mu_m3s2`/`radius_m`/`gravity_model`/`atmosphere` — letting a
    /// frontend (or a TOML author) specify a body by name alone.
    pub fn by_name(name: &str) -> Option<Self> {
        Self::catalog().into_iter().find(|b| b.name.eq_ignore_ascii_case(name))
    }

    /// Sphere-of-influence radius using the Laplace SOI formula.
    ///
    /// # Arguments
    /// * `mu_central_m3s2` — gravitational parameter of the central body [m³/s²]
    /// * `orbit_radius_m` — orbital radius of this body around the central body [m]
    pub fn soi_radius_m(&self, mu_central_m3s2: f64, orbit_radius_m: f64) -> f64 {
        orbit_radius_m * (self.mu_m3s2 / mu_central_m3s2).powf(2.0 / 5.0)
    }

    /// Hill sphere radius (approximate).
    ///
    /// R_Hill = a · (m / 3M)^(1/3), where a is the orbital semi-major axis.
    pub fn hill_radius_m(&self, mu_central_m3s2: f64, orbit_radius_m: f64) -> f64 {
        orbit_radius_m * (self.mu_m3s2 / (3.0 * mu_central_m3s2)).powf(1.0 / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn by_name_is_case_insensitive() {
        let a = TargetBody::by_name("Apophis").unwrap();
        let b = TargetBody::by_name("apophis").unwrap();
        let c = TargetBody::by_name("APOPHIS").unwrap();
        assert_eq!(a.mu_m3s2, TargetBody::apophis().mu_m3s2);
        assert_eq!(a.mu_m3s2, b.mu_m3s2);
        assert_eq!(a.mu_m3s2, c.mu_m3s2);
    }

    #[test]
    fn by_name_unknown_returns_none() {
        assert!(TargetBody::by_name("Planet Nine").is_none());
    }

    #[test]
    fn catalog_contains_all_named_constructors() {
        let names: Vec<String> = TargetBody::catalog().into_iter().map(|b| b.name).collect();
        for expected in [
            "Sun", "Bennu", "Earth", "Moon", "Mars", "Apophis", "Ryugu",
            "Jupiter", "Saturn", "Europa", "Titan", "Phobos", "Deimos", "Eros", "Didymos",
        ] {
            assert!(names.contains(&expected.to_string()), "catalog missing {expected}");
        }
    }
}
