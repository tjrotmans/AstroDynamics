//! Bennu state-vector ephemeris from a JPL Horizons text file.
//!
//! The file must be "GEOMETRIC cartesian states" (output format 3),
//! **Ecliptic of J2000.0** reference frame, Sun (10) as center body, units KM-S.
//! Positions are stored in meters, velocities in m/s, already in the ecliptic
//! frame that matches `earth_rv` (Earth in the ecliptic X-Y plane).
//!
//! Cubic Hermite interpolation uses the tabulated velocities as endpoint
//! derivatives, giving sub-km positional accuracy between 10-day samples.

use orbital_math::hermite_track::HermiteTrack;

type V3 = [f64; 3];

/// Bennu state-vector ephemeris loaded from a JPL Horizons text file.
pub struct BennuEphem {
    track: HermiteTrack,
}

impl BennuEphem {
    /// Load from a Horizons state-vector file.
    ///
    /// `filename` is searched for in `kernels/<filename>` starting from the
    /// current working directory and walking up through parent directories,
    /// so it works whether run from the package or workspace root.
    pub fn load(filename: &str) -> Self {
        let path = ephemeris::find_kernel(filename);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Cannot read '{}': {}", path, e));

        let mut t_vec: Vec<f64>      = Vec::new();
        let mut r_vec: Vec<[f64; 3]> = Vec::new();
        let mut v_vec: Vec<[f64; 3]> = Vec::new();

        let mut iter = text.lines();

        // Advance to the $$SOE marker
        for line in iter.by_ref() {
            if line.trim() == "$$SOE" { break; }
        }

        // Parse records between $$SOE and $$EOE.
        // Each record is four lines:
        //   1.  JD epoch  (first token = Julian Date TDB)
        //   2.  X Y Z     (position in km)
        //   3.  VX VY VZ  (velocity in km/s)
        //   4.  LT RG RR  (light-time / range — skipped)
        'records: loop {
            let epoch_line: &str = 'find: loop {
                match iter.next() {
                    None                                            => break 'records,
                    Some(l) if l.trim() == "$$EOE"                 => break 'records,
                    Some(l) if l.trim()
                                .starts_with(|c: char| c.is_ascii_digit()) => break 'find l,
                    _ => {}
                }
            };

            let jd: f64 = epoch_line.trim()
                .split_whitespace().next().unwrap()
                .parse()
                .unwrap_or_else(|_| panic!("Cannot parse JD from: {:?}", epoch_line));
            let t_sec = (jd - 2_451_545.0) * 86_400.0;

            let xyz_line  = iter.next().expect("File ended after JD line");
            let vxyz_line = iter.next().expect("File ended after XYZ line");
            iter.next(); // skip LT / RG / RR

            let x  = parse_labeled(xyz_line,  "X");
            let y  = parse_labeled(xyz_line,  "Y");
            let z  = parse_labeled(xyz_line,  "Z");
            let vx = parse_labeled(vxyz_line, "VX");
            let vy = parse_labeled(vxyz_line, "VY");
            let vz = parse_labeled(vxyz_line, "VZ");

            t_vec.push(t_sec);
            r_vec.push([x * 1_000.0, y * 1_000.0, z * 1_000.0]);
            v_vec.push([vx * 1_000.0, vy * 1_000.0, vz * 1_000.0]);
        }

        assert!(!t_vec.is_empty(), "No records parsed from '{}'", path);
        println!(
            "  Loaded {} Bennu records  [{:.0}–{:.0} days from J2000]",
            t_vec.len(),
            t_vec.first().unwrap() / 86_400.0,
            t_vec.last().unwrap()  / 86_400.0,
        );

        BennuEphem { track: HermiteTrack::new(t_vec, r_vec, v_vec) }
    }

    /// Interpolated Bennu state at `t_sec` (TDB seconds from J2000).
    ///
    /// Uses cubic Hermite interpolation between the two bracketing samples.
    /// Panics if `t_sec` is outside the file's time coverage.
    pub fn query(&self, t_sec: f64) -> (V3, V3) {
        self.track.query(t_sec)
    }

    /// File coverage as `(t_start, t_end)` in TDB seconds from J2000.
    pub fn coverage(&self) -> (f64, f64) {
        self.track.coverage()
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Extract the numeric value following a label in a Horizons state-vector line.
fn parse_labeled(line: &str, label: &str) -> f64 {
    let with_space = format!("{} =", label);
    let no_space   = format!("{}=",  label);

    let after_eq = if let Some(pos) = line.find(&with_space) {
        &line[pos + with_space.len()..]
    } else if let Some(pos) = line.find(&no_space) {
        &line[pos + no_space.len()..]
    } else {
        panic!("Label '{}' not found in Horizons line: {:?}", label, line)
    };

    after_eq
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("No value after '{}' in: {:?}", label, line))
        .parse()
        .unwrap_or_else(|_| panic!("Cannot parse value after '{}' in: {:?}", label, line))
}
