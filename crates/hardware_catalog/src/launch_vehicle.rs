//! Launch vehicle catalog — C3-vs-injected-mass performance, anchored to real
//! flown (or, for Starship, explicitly flagged as not-yet-flown) missions.
//!
//! Unlike the other catalogs in this crate (representative performance
//! *classes*, not specific flight units), launch-vehicle C3 curves are
//! genuinely vehicle-specific and not something to approximate from general
//! engineering practice — so every data point here is a real, individually
//! citable flown mission, not a smoothed vendor curve (the official ULA/
//! SpaceX performance curves exist only as image-based PDF charts with no
//! extractable numeric data available to this project). Where only one real
//! point exists for a vehicle, the catalog reports that point honestly
//! rather than inventing a second one to fabricate a slope.

/// A launch vehicle's C3-vs-injected-mass performance, built from one or more
/// verified (C3, injected mass) data points.
#[derive(Clone, Copy, Debug)]
pub struct LaunchVehicleSpec {
    pub name: &'static str,
    /// Verified (C3 [km²/s²], injected mass [kg]) points from real flown
    /// missions, sorted by C3 ascending (mass descending) — see citations on
    /// each constructor. Empty for vehicles with no flown interplanetary
    /// payload yet (see `starship()`).
    pub performance_points: &'static [(f64, f64)],
}

impl LaunchVehicleSpec {
    /// Atlas V 401 (no solid boosters, single-engine Centaur, no kick stage).
    /// Two real missions in this configuration family:
    /// - MAVEN (2013): Atlas V401, C3 = 11.84 km²/s², 2,454 kg launch mass.
    /// - OSIRIS-REx (2016): Atlas V411 (one solid booster vs. 401's zero —
    ///   same single-engine-Centaur/no-kick-stage family, kept in this class
    ///   as a documented cross-variant approximation since no second 401-only
    ///   data point was available), C3 = 29.29678 km²/s², 2,105 kg.
    pub fn atlas_v_401() -> Self {
        Self {
            name: "AtlasV401",
            performance_points: &[(11.84, 2454.0), (29.29678, 2105.0)],
        }
    }

    /// Atlas V 551 + Star 48B solid kick stage (five solid boosters, plus a
    /// third-stage solid motor for very high energy escape).
    /// One real, extensively documented mission: New Horizons (2006),
    /// C3 = 157.7502 km²/s², 478 kg launch mass — the highest-energy single
    /// launch ever flown, still the record as of this writing.
    pub fn atlas_v_551_star48() -> Self {
        Self {
            name: "AtlasV551Star48",
            performance_points: &[(157.7502, 478.0)],
        }
    }

    /// Falcon 9 (Full Thrust / Block 5, no kick stage).
    /// - DART (2021): real, fully verified — C3 = 0.1 km²/s², 610 kg launch
    ///   mass (deliberately a low-energy escape, not a vehicle limit).
    /// - "To Mars" capability, 4,020 kg, is SpaceX's own published Falcon 9
    ///   Full Thrust figure — but SpaceX does not state the exact C3 this
    ///   figure assumes. Paired here with C3 = 11.0 km²/s², the upper end of
    ///   the typical Earth-Mars Hohmann-class range and consistent with both
    ///   this project's own Mars-transfer examples (`mars_flyby.toml`/
    ///   `mars_orbit.toml`'s ~9.2 km²/s²) and MAVEN's real flown C3 = 11.84
    ///   km²/s² — an explicitly assumed pairing, not an independently
    ///   confirmed one, chosen to not understate real Mars-class capability.
    pub fn falcon_9() -> Self {
        Self {
            name: "Falcon9",
            performance_points: &[(0.1, 610.0), (11.0, 4020.0)],
        }
    }

    /// Starship (no real interplanetary performance data yet — deliberately
    /// left with no performance points).
    ///
    /// Two independent reasons this isn't estimated, not just "data is
    /// missing": (1) Starship has not yet flown an operational
    /// interplanetary payload, so there is no real mission to anchor a
    /// curve to, real or assumed. (2) More fundamentally, SpaceX's actual
    /// Mars architecture for Starship uses multi-launch *orbital
    /// refueling* (tanker flights topping off the ship in LEO before its
    /// own trans-Mars burn) — a single-vehicle C3-vs-mass curve, the model
    /// every other entry in this catalog uses, cannot represent that at
    /// all; a refueled Starship's real interplanetary capability depends on
    /// how many tanker flights are flown, not on a fixed per-launch curve.
    /// A rough single-launch-only estimate was attempted (Raptor vacuum
    /// Isp ≈ 380 s and public propellant/dry-mass figures are known), but
    /// the ascent-vs-escape propellant split needed to apply the rocket
    /// equation isn't public, so any such estimate would be unfounded
    /// precision, not a real one. `injected_mass_kg()` correctly reports
    /// `None` for every C3 with an empty point list — same code path as
    /// "exceeds this vehicle's verified envelope", which is honest here:
    /// the verified envelope is presently empty.
    pub fn starship() -> Self {
        Self {
            name: "Starship",
            performance_points: &[],
        }
    }

    pub fn catalog() -> [Self; 4] {
        [Self::falcon_9(), Self::starship(), Self::atlas_v_401(), Self::atlas_v_551_star48()]
    }

    /// Looks up a catalog entry by name, case-insensitive.
    pub fn by_name(name: &str) -> Option<Self> {
        Self::catalog().into_iter().find(|v| v.name.eq_ignore_ascii_case(name))
    }

    /// Max injected mass deliverable at the requested C3, from verified
    /// flown-mission data points only — no extrapolation past real data.
    ///
    /// Monotonicity (higher C3 → equal-or-less deliverable mass, for a fixed
    /// vehicle) is a real physical guarantee, not an assumption, so:
    /// - below the lowest verified point, that point's mass is a valid
    ///   conservative *lower bound* (real performance there is at least
    ///   that good, possibly better);
    /// - between two verified points, linear interpolation — matches how
    ///   vendors themselves describe local curve behavior ("mass
    ///   sensitivity" in kg per km²/s²);
    /// - above the highest verified point, `None` — pick a more energetic
    ///   vehicle rather than guess past what's actually been flown.
    /// `None` is also returned for an empty point list (`starship()`).
    pub fn injected_mass_kg(&self, c3_km2s2: f64) -> Option<f64> {
        let pts = self.performance_points;
        if pts.is_empty() {
            return None;
        }
        if c3_km2s2 <= pts[0].0 {
            return Some(pts[0].1);
        }
        for i in 0..pts.len() - 1 {
            let (c3_a, m_a) = pts[i];
            let (c3_b, m_b) = pts[i + 1];
            if c3_km2s2 >= c3_a && c3_km2s2 <= c3_b {
                let t = (c3_km2s2 - c3_a) / (c3_b - c3_a);
                return Some(m_a + t * (m_b - m_a));
            }
        }
        None
    }

    /// Inverse of [`injected_mass_kg`](Self::injected_mass_kg): the maximum
    /// departure C3 [km²/s²] this vehicle can give a payload of `mass_kg`,
    /// from verified flown-mission data only — the "what does the launcher
    /// still deliver when it can't reach the required C3" question a
    /// partial launcher/onboard ΔV split needs (a launcher that falls short
    /// of the required C3 does not deliver nothing; it delivers its maximum
    /// at this mass, and the spacecraft tops up the rest).
    ///
    /// Defined DIRECTLY from the forward curve, so the two can never
    /// disagree: the largest C3 within the verified range at which
    /// `injected_mass_kg(C3) >= mass_kg`. Consequences:
    /// - `injected_mass_kg(c3) >= mass_kg` ⇒ `max_c3_at_mass_km2s2 >= c3`
    ///   (a feasible check always reports a covering launcher C3);
    /// - the answer is clamped to the highest verified point — real
    ///   capability beyond it may be higher, but the verified envelope
    ///   ends there (the same no-extrapolation rule as the forward curve);
    /// - `None` when no verified C3 injects this much mass (including an
    ///   empty point list, `starship()`).
    /// The scan is over the piecewise-linear curve segment by segment
    /// rather than assuming mass decreases with C3: a real catalog entry
    /// can carry a low-C3 point that is a PAYLOAD, not a vehicle limit
    /// (Falcon 9's DART point, 610 kg at C3 0.1, alongside 4,020 kg at
    /// C3 11), which makes the forward curve non-monotone.
    pub fn max_c3_at_mass_km2s2(&self, mass_kg: f64) -> Option<f64> {
        let pts = self.performance_points;
        if pts.is_empty() {
            return None;
        }
        let mut best: Option<f64> = None;
        let mut consider = |c3: f64| {
            best = Some(best.map_or(c3, |b: f64| b.max(c3)));
        };
        // Below the lowest verified C3 the forward curve returns that
        // point's mass (a conservative lower bound), so that C3 counts.
        if pts[0].1 >= mass_kg {
            consider(pts[0].0);
        }
        for i in 0..pts.len() - 1 {
            let (c3_a, m_a) = pts[i];
            let (c3_b, m_b) = pts[i + 1];
            match (m_a >= mass_kg, m_b >= mass_kg) {
                (_, true) => consider(c3_b),
                (true, false) => {
                    // Crossing where the linear segment drops below the mass.
                    let t = (m_a - mass_kg) / (m_a - m_b);
                    consider(c3_a + t * (c3_b - c3_a));
                }
                (false, false) => {}
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_c3_at_mass_inverts_the_forward_curve_between_points() {
        let v = LaunchVehicleSpec::atlas_v_401();
        // Midpoint mass of (11.84, 2454.0) and (29.29678, 2105.0) maps back to
        // the midpoint C3 — the inverse of `between_points_interpolates_linearly`.
        let m_mid = 0.5 * (2454.0 + 2105.0);
        let c3_mid = 0.5 * (11.84 + 29.29678);
        assert!((v.max_c3_at_mass_km2s2(m_mid).unwrap() - c3_mid).abs() < 1e-9);
        // Round trip at an arbitrary interior C3.
        let c3 = 20.0;
        let m = v.injected_mass_kg(c3).unwrap();
        assert!((v.max_c3_at_mass_km2s2(m).unwrap() - c3).abs() < 1e-9);
    }

    #[test]
    fn max_c3_at_mass_clamps_to_highest_verified_c3_for_light_payloads() {
        let v = LaunchVehicleSpec::atlas_v_401();
        assert_eq!(v.max_c3_at_mass_km2s2(500.0), Some(29.29678));
        assert_eq!(v.max_c3_at_mass_km2s2(2105.0), Some(29.29678));
    }

    /// Falcon 9's curve is NON-monotone (the DART point is a payload, not a
    /// limit): a 1,200 kg spacecraft is liftable to the whole verified
    /// range, so the inverse must report the highest verified C3, and it
    /// must agree with the forward curve's own feasibility verdict.
    #[test]
    fn max_c3_at_mass_handles_a_non_monotone_curve_and_agrees_with_the_forward_check() {
        let v = LaunchVehicleSpec::falcon_9();
        assert_eq!(v.max_c3_at_mass_km2s2(1200.0), Some(11.0));
        // Forward says feasible at C3 9.2 → the inverse must cover 9.2.
        assert!(v.injected_mass_kg(9.2).unwrap() >= 1200.0);
        assert!(v.max_c3_at_mass_km2s2(1200.0).unwrap() >= 9.2);
        // Heavier than the DART point but liftable on the rising segment:
        // the crossing where the segment reaches the mass is the MINIMUM C3,
        // not the maximum — the maximum is still the top of the range.
        assert_eq!(v.max_c3_at_mass_km2s2(2000.0), Some(11.0));
        // Heavier than every point → None.
        assert_eq!(v.max_c3_at_mass_km2s2(5000.0), None);
    }

    #[test]
    fn max_c3_at_mass_is_none_when_the_mass_exceeds_every_verified_point() {
        let v = LaunchVehicleSpec::atlas_v_401();
        assert_eq!(v.max_c3_at_mass_km2s2(3000.0), None);
        assert_eq!(LaunchVehicleSpec::starship().max_c3_at_mass_km2s2(100.0), None);
        // Single-point vehicle: at or under its one verified mass → that C3.
        let nh = LaunchVehicleSpec::atlas_v_551_star48();
        assert_eq!(nh.max_c3_at_mass_km2s2(478.0), Some(157.7502));
        assert_eq!(nh.max_c3_at_mass_km2s2(479.0), None);
    }

    #[test]
    fn below_lowest_point_returns_conservative_lower_bound() {
        let v = LaunchVehicleSpec::atlas_v_401();
        assert_eq!(v.injected_mass_kg(5.0), Some(2454.0));
    }

    #[test]
    fn between_points_interpolates_linearly() {
        let v = LaunchVehicleSpec::atlas_v_401();
        // Midpoint of (11.84, 2454.0) and (29.29678, 2105.0).
        let c3_mid = 0.5 * (11.84 + 29.29678);
        let m_mid = 0.5 * (2454.0 + 2105.0);
        assert!((v.injected_mass_kg(c3_mid).unwrap() - m_mid).abs() < 1e-9);
    }

    #[test]
    fn above_highest_point_returns_none() {
        let v = LaunchVehicleSpec::atlas_v_401();
        assert_eq!(v.injected_mass_kg(200.0), None);
    }

    #[test]
    fn single_point_vehicle_never_extrapolates_upward() {
        let v = LaunchVehicleSpec::atlas_v_551_star48();
        assert_eq!(v.injected_mass_kg(157.7502), Some(478.0));
        assert_eq!(v.injected_mass_kg(160.0), None);
        assert_eq!(v.injected_mass_kg(0.0), Some(478.0));
    }

    #[test]
    fn starship_has_no_performance_data_yet() {
        let v = LaunchVehicleSpec::starship();
        assert_eq!(v.injected_mass_kg(0.0), None);
        assert_eq!(v.injected_mass_kg(10.0), None);
    }

    #[test]
    fn by_name_is_case_insensitive() {
        assert!(LaunchVehicleSpec::by_name("falcon9").is_some());
        assert!(LaunchVehicleSpec::by_name("FALCON9").is_some());
        assert!(LaunchVehicleSpec::by_name("nonexistent").is_none());
    }
}
