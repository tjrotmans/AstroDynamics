//! Export DE440S Moon ECI positions to CSV.
//!
//! Samples Moon (and Sun) positions from DE440S over a user-defined window at
//! hourly intervals and writes them to out/moon_ephem.csv.  The Python plot
//! script can load this instead of using 2-body propagation for the Moon dot.
//!
//! Time column is seconds from TLI epoch (23:49:00 UTC), matching
//! the Artemis trajectory CSV convention.
//!
//! Usage:
//!   cargo run -p artemis --bin moon_ephem --release
//!
//! Optional env vars (all in days relative to TLI epoch):
//!   START_DAYS  (default: -150)
//!   END_DAYS    (default:  300)
//!   STEP_HOURS  (default:    1)

use std::fmt::Write as FmtWrite;

use hifitime::{Duration, Epoch};

use ephemeris::{Almanac, Body};

const TLI_EPOCH_GREG: (i32, u8, u8, u8, u8, u8, u32) = (2026, 4, 2, 23, 49, 0, 0);
const DEFAULT_START_DAYS: f64 = -150.0;
const DEFAULT_END_DAYS:   f64 =  300.0;
const DEFAULT_STEP_HOURS: f64 =    1.0;

fn main() {
    let start_days = env_f64("START_DAYS", DEFAULT_START_DAYS);
    let end_days   = env_f64("END_DAYS",   DEFAULT_END_DAYS);
    let step_hours = env_f64("STEP_HOURS", DEFAULT_STEP_HOURS);

    let tli_epoch = Epoch::from_gregorian_utc(
        TLI_EPOCH_GREG.0, TLI_EPOCH_GREG.1, TLI_EPOCH_GREG.2,
        TLI_EPOCH_GREG.3, TLI_EPOCH_GREG.4, TLI_EPOCH_GREG.5,
        TLI_EPOCH_GREG.6,
    );

    println!("Loading DE440S kernel ...");
    let almanac = Almanac::new(&ephemeris::find_kernel("de440s.bsp"))
        .expect("Failed to load de440s.bsp");

    let step_s     = step_hours * 3600.0;
    let n_steps    = ((end_days - start_days) * 86_400.0 / step_s).ceil() as usize + 1;
    println!("Sampling Moon+Sun: {n_steps} points  ({start_days:.0} to {end_days:.0} days from TLI)");

    let mut out = String::with_capacity(n_steps * 100);
    writeln!(out, "time_s,moon_x_m,moon_y_m,moon_z_m,sun_x_m,sun_y_m,sun_z_m").unwrap();

    for i in 0..n_steps {
        let t_s    = start_days * 86_400.0 + i as f64 * step_s;
        let epoch  = tli_epoch + Duration::from_seconds(t_s);

        let moon = almanac.body_state_eci(Body::Moon, epoch)
            .expect("Moon query failed").position.inner;
        let sun  = almanac.body_state_eci(Body::Sun, epoch)
            .expect("Sun query failed").position.inner;

        writeln!(out, "{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            t_s,
            moon[0], moon[1], moon[2],
            sun[0],  sun[1],  sun[2],
        ).unwrap();
    }

    std::fs::create_dir_all("out").unwrap();
    let path = "out/moon_ephem.csv";
    std::fs::write(path, &out).expect("Failed to write CSV");
    println!("Saved {n_steps} rows -> {path}");
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
