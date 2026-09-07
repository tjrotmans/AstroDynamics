//! Apples-to-apples check: how much does OUR OWN MbhSolver,
//! configured with pagmo's real verified default settings (HookeJeeves
//! local descent matching pagmo's compass_search{} default, kick_scale=
//! 0.01, perturb_fraction=1.0, stop_after=5 -- see mbh.rs/hooke_jeeves.rs
//! doc comments for the source verification), improve on the SAME raw
//! pre-MBH pruning/backfit seed a real pagmo run was also seeded with?
//! Uses our own real fitness function (`phase2_fitness`, widened to `pub`
//! for this exact purpose) -- zero risk of comparing against a
//! reimplemented/divergent objective.
//!
//! Usage: cargo run -p mission_planner --bin our_mbh_seeded_check --release
//!   -- <config.toml> <seed_chromosome_csv_path> <n_runs>

use ephemeris::Almanac;
use mission_planner::config::MissionConfig;
use mission_planner::mga::{build_bounds, phase2_fitness};
use trajectory_solver::{HookeJeevesSearch, MbhSolver};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: our_mbh_seeded_check <config.toml> <seed_chromosome_csv_path> <n_runs>");
        std::process::exit(1);
    }
    let cfg_path = &args[1];
    let seed_csv_path = &args[2];
    let n_runs: usize = args[3].parse().expect("bad n_runs");

    let cfg_str = std::fs::read_to_string(cfg_path).expect("could not read config");
    let cfg: MissionConfig = toml::from_str(&cfg_str).expect("could not parse config");

    let almanac = Almanac::new("kernels/de440s.bsp")
        .or_else(|_| Almanac::new("../kernels/de440s.bsp"))
        .expect("de440s.bsp not found");

    let opt = cfg.optimization.as_ref().expect("no [optimization] section");
    let mga = opt.mga.as_ref().expect("no [optimization.mga] section");
    let flyby_bodies = mga.flyby_bodies.clone();
    let bounds = build_bounds(&cfg, &flyby_bodies).expect("could not build bounds");

    let dep_epoch_str = opt.departure_epoch.as_deref().expect("no departure_epoch");
    let dep_epoch = mission_planner::design::parse_epoch(dep_epoch_str).expect("bad departure_epoch");
    let dep_jd_base = mission_planner::design::epoch_to_jd(dep_epoch);

    // Load the seed chromosome (same format as mga_best_chromosome.csv /
    // elite_seed_pre_mbh.csv: header row, then `fitness,p0,p1,...` or
    // `flyby_bodies,p0,p1,...` data row).
    let csv = std::fs::read_to_string(seed_csv_path).expect("could not read seed csv");
    let mut lines = csv.lines();
    let _header = lines.next().expect("empty seed csv");
    let data_row = lines.next().expect("no data row in seed csv");
    let fields: Vec<&str> = data_row.split(',').collect();
    let seed_x: Vec<f64> = fields[1..].iter().map(|s| s.parse().expect("bad seed value")).collect();

    let seed_fitness = phase2_fitness(&seed_x, &cfg, &almanac, dep_jd_base, &flyby_bodies)
        .expect("seed chromosome infeasible under our own evaluator");
    println!("Raw pre-MBH seed fitness (our own evaluator): {seed_fitness:.2} m/s");

    let mut best_f = seed_fitness;
    for i in 0..n_runs {
        let mbh = MbhSolver {
            hops: 200,
            perturb_fraction: 1.0,
            kick_scale: 0.01,
            local: HookeJeevesSearch::default().into(),
            seed: 5000 + i as u64,
            stop_after: Some(5),
            global_stall: None,
            extra_random_chains: 0,
            migration: None,
        };
        let result = mbh.run(
            &bounds,
            &[seed_x.clone()],
            |p| phase2_fitness(p, &cfg, &almanac, dep_jd_base, &flyby_bodies),
        );
        let tag = if result.best_fitness < best_f { best_f = result.best_fitness; "  <- new best" } else { "" };
        println!(
            "  run {i}: {:.2} m/s (delta: {:+.2}){tag}",
            result.best_fitness, result.best_fitness - seed_fitness
        );
    }

    println!(
        "\nSeed: {seed_fitness:.2} m/s -> Best after our mbh: {best_f:.2} m/s (improvement: {:.2} m/s, {:.3}%)",
        seed_fitness - best_f, 100.0 * (seed_fitness - best_f) / seed_fitness
    );
}
