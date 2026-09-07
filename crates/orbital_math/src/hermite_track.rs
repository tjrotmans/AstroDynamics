//! Cubic Hermite interpolation table for body state vectors.
//!
//! Stores a time-series of position+velocity samples and interpolates between
//! them using cubic Hermite basis functions, treating the tabulated velocities
//! as endpoint derivatives.  Sub-km accuracy is typical for 10-day sample spacing.

/// State-vector ephemeris with cubic Hermite interpolation.
///
/// Stores positions [m] and velocities [m/s] at uniformly or non-uniformly
/// spaced epochs (TDB seconds from J2000).  Call [`HermiteTrack::query`] to get
/// an interpolated state at any epoch within the table coverage.
pub struct HermiteTrack {
    t: Vec<f64>,
    r: Vec<[f64; 3]>,
    v: Vec<[f64; 3]>,
}

impl HermiteTrack {
    /// Construct from pre-parsed time, position, and velocity vectors.
    ///
    /// All three slices must have the same length and `t` must be monotonically
    /// increasing.
    pub fn new(t: Vec<f64>, r: Vec<[f64; 3]>, v: Vec<[f64; 3]>) -> Self {
        assert_eq!(t.len(), r.len(), "HermiteTrack: t and r must have equal length");
        assert_eq!(t.len(), v.len(), "HermiteTrack: t and v must have equal length");
        Self { t, r, v }
    }

    /// Cubic Hermite interpolation at `t_sec` (TDB seconds from J2000).
    ///
    /// Returns `(position [m; 3], velocity [m/s; 3])`.
    ///
    /// # Panics
    /// Panics if `t_sec` is outside the table coverage.
    pub fn query(&self, t_sec: f64) -> ([f64; 3], [f64; 3]) {
        let n = self.t.len();
        assert!(
            t_sec >= self.t[0] && t_sec <= self.t[n - 1],
            "HermiteTrack: query {:.1} d from J2000 is outside coverage [{:.1}, {:.1}] d",
            t_sec / 86_400.0,
            self.t[0] / 86_400.0,
            self.t[n - 1] / 86_400.0,
        );

        let i = self.t.partition_point(|&ts| ts <= t_sec)
            .saturating_sub(1)
            .min(n - 2);

        let (t0, t1) = (self.t[i], self.t[i + 1]);
        let (r0, r1) = (self.r[i], self.r[i + 1]);
        let (v0, v1) = (self.v[i], self.v[i + 1]);

        let h  = t1 - t0;
        let s  = (t_sec - t0) / h;
        let s2 = s * s;
        let s3 = s2 * s;

        // Cubic Hermite basis functions for position
        let h00 =  2.0 * s3 - 3.0 * s2 + 1.0;
        let h10 =        s3 - 2.0 * s2 + s;
        let h01 = -2.0 * s3 + 3.0 * s2;
        let h11 =        s3 - s2;

        // Derivative basis functions for velocity (d/dt of the above, divided by h)
        let d00 = ( 6.0 * s2 - 6.0 * s) / h;
        let d10 =   3.0 * s2 - 4.0 * s + 1.0;
        let d01 = (-6.0 * s2 + 6.0 * s) / h;
        let d11 =   3.0 * s2 - 2.0 * s;

        let mut r_out = [0.0f64; 3];
        let mut v_out = [0.0f64; 3];
        for k in 0..3 {
            r_out[k] = h00 * r0[k] + h10 * h * v0[k] + h01 * r1[k] + h11 * h * v1[k];
            v_out[k] = d00 * r0[k] + d10 * v0[k]     + d01 * r1[k] + d11 * v1[k];
        }
        (r_out, v_out)
    }

    /// Coverage as `(t_start_s, t_end_s)` in TDB seconds from J2000.
    pub fn coverage(&self) -> (f64, f64) {
        (*self.t.first().unwrap(), *self.t.last().unwrap())
    }

    /// Number of tabulated samples.
    pub fn len(&self) -> usize { self.t.len() }
}
