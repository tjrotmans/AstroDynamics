//! Priority-ordered multi-rule attitude commander
//! — the mission-level generalization of `orbital_models::
//! guidance::pointing::triad_quat`/`align_vector_with` from raw vector pairs
//! to an ordered list of named pointing rules. See `docs/MP/MANUAL.md`
//! §9.4 for the governing physics (why only 2 of N rules can ever control
//! the attitude, and why that's a hard DOF limit, not a modeling choice).
//!
//! This module is deliberately still mission-context-free: a [`ResolvedRule`]
//! already carries fully-resolved body-frame and inertial-frame unit
//! vectors — resolving those from `MissionConfig` (hardware boresight/
//! normal fields, Sun/Earth/target-body/velocity/inertial targets) is
//! `MissionPlanner::cruise`'s job, per this crate's existing convention of
//! keeping `sim_engine` free of TOML/config-schema knowledge.

use nalgebra::{Vector3, Vector4};
use orbital_models::guidance::pointing::{align_vector_with, triad_quat};

/// One fully-resolved pointing constraint: a body-frame direction that
/// should point at an inertial-frame direction, both already unit-length
/// (or close to it — resolvers should normalize before constructing this).
#[derive(Clone, Debug)]
pub struct ResolvedRule {
    pub body_vec: Vector3<f64>,
    pub target_vec: Vector3<f64>,
    /// Human-readable identifier for reporting (e.g. "CommAntenna -> Earth").
    pub label: String,
}

/// Outcome of evaluating one rule against the attitude actually solved for.
#[derive(Clone, Debug)]
pub struct RuleOutcome {
    pub label: String,
    /// Angle [deg] between the achieved body-vector direction and the
    /// rule's target — 0 (to numerical precision) for the primary rule,
    /// generally small-but-nonzero for the secondary, and whatever the
    /// geometry dictates for every lower-priority rule (§9.4: they cannot
    /// change the attitude, only be measured against it).
    pub achieved_error_deg: f64,
    /// `true` only for the (up to) 2 highest-priority rules that actually
    /// determined the solved attitude; `false` for every rule beyond that,
    /// which are report-only.
    pub is_controlling: bool,
}

/// Solve for the attitude quaternion [w,x,y,z] satisfying as many of
/// `rules` (in priority order, index 0 = highest) as the 3 rotational DOF
/// allow, and report how well every rule — controlling or not — is
/// actually satisfied by that attitude.
///
/// - 0 rules: identity quaternion, empty outcome list (nothing commanded).
/// - 1 rule: [`align_vector_with`] — exact primary, roll free.
/// - 2+ rules: [`triad_quat`] on the top 2 (exact primary, best-effort
///   secondary); every rule from index 2 onward is evaluated against the
///   resulting attitude only, per §9.4's DOF-budget argument.
pub fn solve_prioritized_attitude(rules: &[ResolvedRule]) -> (Vector4<f64>, Vec<RuleOutcome>) {
    let q = match rules.len() {
        0 => Vector4::new(1.0, 0.0, 0.0, 0.0),
        1 => align_vector_with(&rules[0].body_vec, &rules[0].target_vec),
        _ => triad_quat(&rules[0].body_vec, &rules[0].target_vec, &rules[1].body_vec, &rules[1].target_vec),
    };

    let outcomes = rules
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let achieved = orbital_models::attitude::body_to_inertial(&q, &r.body_vec.normalize());
            let target = r.target_vec.normalize();
            let dot = achieved.dot(&target).clamp(-1.0, 1.0);
            RuleOutcome {
                label: r.label.clone(),
                achieved_error_deg: dot.acos().to_degrees(),
                is_controlling: i < 2,
            }
        })
        .collect();

    (q, outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(body: Vector3<f64>, target: Vector3<f64>, label: &str) -> ResolvedRule {
        ResolvedRule { body_vec: body, target_vec: target, label: label.to_string() }
    }

    #[test]
    fn no_rules_returns_identity_and_no_outcomes() {
        let (q, outcomes) = solve_prioritized_attitude(&[]);
        assert_eq!(q, Vector4::new(1.0, 0.0, 0.0, 0.0));
        assert!(outcomes.is_empty());
    }

    #[test]
    fn single_rule_is_exact_and_controlling() {
        let rules = vec![rule(Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), "cam")];
        let (_q, outcomes) = solve_prioritized_attitude(&rules);
        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].achieved_error_deg < 1e-6);
        assert!(outcomes[0].is_controlling);
    }

    #[test]
    fn third_rule_is_evaluated_but_never_controlling() {
        // Three mutually incompatible constraints on 3 body axes -- only
        // the first two can drive the solve; the third is pure reporting.
        let rules = vec![
            rule(Vector3::new(1.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0), "primary"),
            rule(Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 1.0, 0.0), "secondary"),
            rule(Vector3::new(0.0, 0.0, 1.0), Vector3::new(1.0, 0.0, 0.0), "tertiary (impossible)"),
        ];
        let (_q, outcomes) = solve_prioritized_attitude(&rules);
        assert_eq!(outcomes.len(), 3);
        assert!(outcomes[0].achieved_error_deg < 1e-6, "primary should be exact");
        assert!(outcomes[0].is_controlling);
        assert!(outcomes[1].is_controlling);
        assert!(!outcomes[2].is_controlling);
        // The third rule wants body +z at (1,0,0), but body +z is already
        // forced perpendicular to both +x and +y's exact alignments -- it
        // must end up perpendicular to (1,0,0) too, i.e. a 90 deg error.
        assert!((outcomes[2].achieved_error_deg - 90.0).abs() < 1e-6, "got {}", outcomes[2].achieved_error_deg);
    }

    #[test]
    fn compatible_secondary_is_also_exact() {
        // Orthogonal body vectors, orthogonal targets -- fully compatible,
        // so both primary and secondary should come out exact.
        let rules = vec![
            rule(Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0), "a"),
            rule(Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 0.0, 0.0), "b"),
        ];
        let (_q, outcomes) = solve_prioritized_attitude(&rules);
        assert!(outcomes[0].achieved_error_deg < 1e-6);
        assert!(outcomes[1].achieved_error_deg < 1e-6);
    }
}
