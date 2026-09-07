//! Cruise phase mission operations simulator (Earth -> Bennu).
//!
//! Three parallel tracks:
//!   Truth        — RK4 + SRP, C_R = 1.35 (unknown to the filters)
//!   Ground OD    — 7-state EKF [r,v,C_R], DSN range+range-rate (2 h cadence).
//!                  Covariance propagated via a non-dimensionally-scaled 7×7 STM;
//!                  C_R column is computed analytically (FD unusable — catastrophic
//!                  cancellation; ∂r/∂C_R ~ 1 m vs r ~ 1.5e11 m).
//!   On-board EKF — 7-state EKF [r,v,C_R], receives daily full-state uplink from
//!                  ground OD; process-noise-only covariance (no STM needed).
//!
//! TCM planning uses the ground OD state.
//!
//! Run:   cargo run -p autonomous_navigation --bin cruise_operations --release
//! Plot:  python plot/plot_cruise_ops.py

use autonomous_navigation::bennu_ephem::BennuEphem;
use autonomous_navigation::dynamics::bennu::bennu_heliocentric_pos as keplerian_bennu_pos;
use autonomous_navigation::sensors::dsn::{GroundOdEkf, RNG_NOISE_M, RRATE_NOISE_MPS, DDOR_NOISE_RAD};
use orbital_models::constants::{MU_SUN, AU, P_SRP};
use orbital_math::lambert::lambert_min_dv;
use nalgebra::Vector3;
use std::f64::consts::PI;

// -- Constants ----------------------------------------------------------------
const SC_AREA:   f64 = 10.5;            // m^2
const SC_MASS:   f64 = 950.0;           // kg
const C_R_TRUTH: f64 = 1.35;           // true reflectivity (unknown to filters)
const C_R_INIT:  f64 = 1.20;           // both filters start here

// Earth circular orbit
const R_E: f64 = AU;

// Arrival approach burn: target relative velocity w.r.t. Bennu at handoff [m/s]
const ARRIVAL_APPROACH_V_MPS: f64 = 5.0;

// Close-Approach Targeting burn: desired flyby periapsis [m]
// At this range OpNav SNR is high and the EKF receives the most informative updates.
const FLYBY_PERIAPSIS_M: f64 = 5_000.0;

// Mission ops
const TCM_THRESHOLD_KM: f64 = 500.0;
const MIN_TCM_MS:       f64 = 0.05;
const TCM_MAG_ERR:      f64 = 0.02;
const TCM_PT_ERR:       f64 = 0.0175;
const MIN_TCM_DAY:      f64 = 1.0;   // no TCMs before initial C_R calibration
const POST_TCM_FREEZE:    f64 = 30.0;  // total days from burn before next TCM allowed
const BURN_RECON_DAYS:    f64 =  15.0;  // days with C_R frozen (burn reconstruction)
// Remaining POST_TCM_FREEZE - BURN_RECON_DAYS = 7 days for C_R re-estimation
const POST_BURN_CR_SIGMA: f64 = 0.02;  // C_R sigma at thaw — kept small to avoid noise-driven
                                        // jumps (range-rate at 0.1 mm/s precision makes K[6] ∝ P[6,6];
                                        // σ=0.10 gives K≈190/ms, RMS walk ≈0.17 over 7 d; σ=0.02 → 0.007)

// DSN measurement noise — constants are in sensors::dsn (imported above).

// On-board uplink noise (OD accuracy when state is uplinked)
const UPL_SIGMA_R: f64 = 2_000.0;   // 2 km position uplink uncertainty
const UPL_SIGMA_V: f64 = 0.02;      // 2 cm/s velocity uplink uncertainty

// -- Types --------------------------------------------------------------------
type V3 = [f64; 3];
type S7 = [f64; 7];

// -- V3 helpers ---------------------------------------------------------------
#[inline] fn dot(a: V3, b: V3) -> f64 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
#[inline] fn norm(a: V3) -> f64 { dot(a,a).sqrt() }
#[inline] fn sub(a: V3, b: V3) -> V3 { [a[0]-b[0],a[1]-b[1],a[2]-b[2]] }
#[inline] fn add(a: V3, b: V3) -> V3 { [a[0]+b[0],a[1]+b[1],a[2]+b[2]] }
#[inline] fn scale(a: V3, s: f64) -> V3 { [a[0]*s,a[1]*s,a[2]*s] }
#[inline] fn normalize(a: V3) -> V3 { scale(a, 1.0/norm(a)) }
#[inline] fn cross(a: V3, b: V3) -> V3 {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
// -- S7 helpers ---------------------------------------------------------------
#[inline] fn s7_add(a: S7, b: S7) -> S7 {
    [a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]+b[3],a[4]+b[4],a[5]+b[5],a[6]+b[6]]
}
#[inline] fn s7_scale(a: S7, s: f64) -> S7 {
    [a[0]*s,a[1]*s,a[2]*s,a[3]*s,a[4]*s,a[5]*s,a[6]*s]
}
/// Convert a `nalgebra::Vector3` to a raw `V3` array for use with the local V3 helpers.
#[inline] fn v3_from(v: Vector3<f64>) -> V3 { [v[0], v[1], v[2]] }

// -- Ephemerides --------------------------------------------------------------
fn earth_rv(t: f64) -> (V3, V3) {
    let v_e  = (MU_SUN / R_E).sqrt();
    let om_e = v_e / R_E;
    let th   = om_e * t;
    ([R_E*th.cos(), R_E*th.sin(), 0.0], [v_e*(-th.sin()), v_e*th.cos(), 0.0])
}

// -- Dynamics -----------------------------------------------------------------
fn srp_acc(r: V3, c_r: f64) -> V3 {
    let r_au = norm(r) / AU;
    scale(normalize(r), P_SRP * c_r * SC_AREA / (SC_MASS * r_au * r_au))
}

// 7-state derivative: x = [r, v, C_R],  dC_R/dt = 0
fn deriv7(x: S7) -> S7 {
    let r=[x[0],x[1],x[2]]; let v=[x[3],x[4],x[5]]; let cr=x[6];
    let rn=norm(r);
    let ag=scale(r,-MU_SUN/rn.powi(3));
    let a=add(ag,srp_acc(r,cr));
    [v[0],v[1],v[2],a[0],a[1],a[2],0.0]
}
fn rk4_7(x: S7, dt: f64) -> S7 {
    let k1=deriv7(x);
    let k2=deriv7(s7_add(x,s7_scale(k1,dt*0.5)));
    let k3=deriv7(s7_add(x,s7_scale(k2,dt*0.5)));
    let k4=deriv7(s7_add(x,s7_scale(k3,dt)));
    s7_add(x,s7_scale(s7_add(s7_add(s7_add(k1,s7_scale(k2,2.0)),s7_scale(k3,2.0)),k4),dt/6.0))
}

// Truth propagation uses V3 interface (C_R fixed)
fn rk4_rv(r: V3, v: V3, dt: f64, cr: f64) -> (V3, V3) {
    let x=[r[0],r[1],r[2],v[0],v[1],v[2],cr];
    let y=rk4_7(x,dt);
    ([y[0],y[1],y[2]],[y[3],y[4],y[5]])
}
fn propagate_rv(r0: V3, v0: V3, total: f64, step: f64, cr: f64) -> (V3, V3) {
    let n=(total/step).ceil() as usize;
    let dt=total/n as f64;
    let (mut r,mut v)=(r0,v0);
    for _ in 0..n { (r,v)=rk4_rv(r,v,dt,cr); }
    (r,v)
}
fn propagate7(x0: S7, total: f64, step: f64) -> S7 {
    let n=(total/step).ceil() as usize;
    let dt=total/n as f64;
    let mut x=x0;
    for _ in 0..n { x=rk4_7(x,dt); }
    x
}

// -- SRP-corrected nominal velocity via Newton shooting ----------------------
// Finds v0 such that propagating r0 → r_target over tof with SRP (cr) lands
// within 1 km of r_target.  This accounts for the SRP drift that the pure
// Lambert solution ignores.
fn srp_shoot(r0: V3, r_tgt: V3, tof: f64, v_guess: V3, cr: f64) -> V3 {
    let mut v = v_guess;
    for iter in 0..20 {
        let (r_arr, _) = propagate_rv(r0, v, tof, 7_200.0, cr);
        let miss = sub(r_arr, r_tgt);
        let miss_km = norm(miss) / 1e3;
        if miss_km < 1.0 { break; }
        let jac = drr_dv(r0, v, cr, tof);
        let dv  = min_norm_dv(jac, scale(miss, -1.0));
        let dmag = norm(dv);
        // Limit step to avoid large oscillations on first iterations
        let step = if dmag > 50.0 { scale(dv, 50.0 / dmag) } else { dv };
        v = add(v, step);
        if iter == 19 {
            eprintln!("srp_shoot did not converge; residual {:.1} km", miss_km);
        }
    }
    v
}

// -- B-plane ------------------------------------------------------------------
fn bplane_miss(r_sc: V3, r_tgt: V3, v_sc: V3, v_tgt: V3) -> f64 {
    let r_miss = sub(r_sc, r_tgt);
    let v_inf  = sub(v_sc, v_tgt);
    let s = if norm(v_inf) > 1.0 { normalize(v_inf) } else { normalize(r_miss) };
    let rperp = sub(r_miss, scale(s, dot(r_miss, s)));
    norm(rperp)
}

// -- Numerical targeting Jacobian (3x3 position vs velocity) ------------------
fn drr_dv(r0: V3, v0: V3, cr: f64, dt_total: f64) -> [[f64;3];3] {
    let h=0.5_f64;
    let mut jac=[[0.0f64;3];3];
    for j in 0..3{
        let mut vp=v0; let mut vm=v0; vp[j]+=h; vm[j]-=h;
        let (rp,_)=propagate_rv(r0,vp,dt_total,3600.0,cr);
        let (rm,_)=propagate_rv(r0,vm,dt_total,3600.0,cr);
        for i in 0..3{jac[i][j]=(rp[i]-rm[i])/(2.0*h);}
    }
    jac
}
fn inv3(m: [[f64;3];3]) -> [[f64;3];3] {
    let d=m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
         -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
         +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
    let id=1.0/d;
    [[(m[1][1]*m[2][2]-m[1][2]*m[2][1])*id,(m[0][2]*m[2][1]-m[0][1]*m[2][2])*id,(m[0][1]*m[1][2]-m[0][2]*m[1][1])*id],
     [(m[1][2]*m[2][0]-m[1][0]*m[2][2])*id,(m[0][0]*m[2][2]-m[0][2]*m[2][0])*id,(m[0][2]*m[1][0]-m[0][0]*m[1][2])*id],
     [(m[1][0]*m[2][1]-m[1][1]*m[2][0])*id,(m[0][1]*m[2][0]-m[0][0]*m[2][1])*id,(m[0][0]*m[1][1]-m[0][1]*m[1][0])*id]]
}
fn min_norm_dv(j: [[f64;3];3], b: V3) -> V3 {
    let mut jjt=[[0.0f64;3];3];
    for i in 0..3{for k in 0..3{for l in 0..3{jjt[i][k]+=j[i][l]*j[k][l];}}}
    let jjt_inv=inv3(jjt);
    let y=[dot(jjt_inv[0],b),dot(jjt_inv[1],b),dot(jjt_inv[2],b)];
    let mut dv=[0.0f64;3];
    for l in 0..3{for i in 0..3{dv[l]+=j[i][l]*y[i];}}
    dv
}

// -- PRNG ---------------------------------------------------------------------
struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Self(s|1) }
    fn u64(&mut self) -> u64 { self.0^=self.0<<13;self.0^=self.0>>7;self.0^=self.0<<17;self.0 }
    fn uni(&mut self) -> f64 { (self.u64()>>11) as f64/(1u64<<53) as f64 }
    fn gauss(&mut self) -> f64 {
        let u=self.uni().max(1e-15); let v=self.uni();
        (-2.0*u.ln()).sqrt()*(2.0*PI*v).cos()
    }
}

// -- TCM record ---------------------------------------------------------------
struct Tcm {
    t_day:       f64,
    r:           V3,
    v_pre_burn:  V3,   // truth velocity BEFORE execution (for post-sim analysis)
    dv_plan:     V3,
    dv_exec:     V3,
    miss_before: f64,
    miss_after:  f64,
}

fn load_best_solution() -> (f64, f64) {
    let path = "out/cruise/best_solution.csv";
    let text = std::fs::read_to_string(path).unwrap_or_else(|_| panic!(
        "Cannot read '{}'. Run `cargo run --bin cruise_design --release` first.",
        path
    ));
    let line = text.lines().nth(1).unwrap_or_else(|| panic!(
        "best_solution.csv has no data row"
    ));
    let mut f = line.split(',');
    let dep: f64 = f.next().unwrap().trim().parse().expect("dep_day parse");
    let tof: f64 = f.next().unwrap().trim().parse().expect("tof_day parse");
    (dep, tof)
}

// -- Main ---------------------------------------------------------------------
fn main() {
    std::fs::create_dir_all("out/cruise_ops").unwrap();
    let mut rng=Rng::new(20260429);

    println!("Loading Bennu ephemeris...");
    let bennu_ephem = BennuEphem::load("horizons_results_bennu.txt");

    let (dep_days, tof_days) = load_best_solution();
    println!("  Loaded best_solution.csv: dep={:.1} d, TOF={:.1} d", dep_days, tof_days);

    let dep_s=dep_days*86400.0;
    let tof_s=tof_days*86400.0;
    let arr_s=dep_s+tof_s;

    let (r_e_dep,v_e_dep)=earth_rv(dep_s);
    let (r_b_arr,v_b_arr)=bennu_ephem.query(arr_s);
    let (v1_lam,_)=lambert_min_dv(r_e_dep,r_b_arr,tof_s,MU_SUN,v_e_dep,v_b_arr)
        .expect("Lambert failed");

    // Correct the departure velocity for SRP so the nominal trajectory
    // actually arrives at Bennu under truth SRP (C_R_TRUTH).
    // Without this the Lambert arc drifts ~70,000 km due to solar pressure.
    println!("Computing SRP-corrected nominal departure velocity...");
    let v0_nom = srp_shoot(r_e_dep, r_b_arr, tof_s, v1_lam, C_R_TRUTH);
    {
        let (r_check, _) = propagate_rv(r_e_dep, v0_nom, tof_s, 7_200.0, C_R_TRUTH);
        println!("  Lambert residual with SRP: {:.0} km",
                 norm(sub(r_check, r_b_arr)) / 1e3);
    }

    println!("=== Cruise Operations Simulator ===");
    println!("  Departure  : J2000 + {:.1} d", dep_days);
    println!("  TOF        : {:.1} d", tof_days);
    println!("  Truth C_R  = {:.2}  |  Both filters start at C_R = {:.2}", C_R_TRUTH, C_R_INIT);
    println!("  First TCM no earlier than Day {:.0}", MIN_TCM_DAY);

    // -- Initial conditions: Lambert + small injection dispersion -------------
    // Truth: SRP-corrected nominal + injection dispersion
    let r_truth: V3=[
        r_e_dep[0]+5_000.0*rng.gauss(),
        r_e_dep[1]+5_000.0*rng.gauss(),
        r_e_dep[2]+1_000.0*rng.gauss(),
    ];
    let v_truth: V3=[
        v0_nom[0]+1.0*rng.gauss(),
        v0_nom[1]+1.0*rng.gauss(),
        v0_nom[2]+0.5*rng.gauss(),
    ];
    let mut r_tr=r_truth; let mut v_tr=v_truth;

    // Ground OD: seeded with truth injection state (real practice: LV telemetry
    // gives the actual dispersed state within seconds of separation).
    // Tight initial uncertainty reflects launch vehicle state knowledge.
    let mut od = GroundOdEkf::new(
        Vector3::from(r_truth), Vector3::from(v_truth), C_R_TRUTH,
        1_000.0, 0.05, 0.05,     // 1 km, 5 cm/s, C_R ±0.05
        SC_AREA / SC_MASS,
    );

    // On-board EKF: also seeded with truth state but with initial C_R guess.
    // Gets daily [r,v] uplinks from ground OD. C_R won't converge from ranging alone.
    let mut ekf = GroundOdEkf::new(
        Vector3::from(r_truth), Vector3::from(v_truth), C_R_INIT,
        5_000.0, 0.2, 0.2,
        SC_AREA / SC_MASS,
    );

    // -- Simulation timing ----------------------------------------------------
    let dt_tr   = 1_800.0;  // 30-min truth step
    let dt_ekf  = 3_600.0;  // 1-h EKF/OD predict step
    let n_steps = (tof_s/dt_tr).ceil() as usize;
    let meas_hz = 2usize;   // DSN measurements every 2 EKF steps = 2h

    struct Row{t:f64,x:f64,y:f64,z:f64,vx:f64,vy:f64,vz:f64,cr:f64}
    let mut truth_log: Vec<Row>=Vec::new();
    let mut od_log:    Vec<Row>=Vec::new();
    let mut ekf_log:   Vec<Row>=Vec::new();
    let mut miss_log:  Vec<(f64,f64)>=Vec::new();
    let mut tcms:      Vec<Tcm>=Vec::new();
    let mut dv_total=0.0f64;

    let mut t_elapsed=0.0f64;
    let mut od_t_elapsed=0.0f64;  // OD's own clock — avoids overshoot at end
    let mut ekf_sub=0usize;
    let mut ekf_cnt=0usize;
    let log_every=24usize;  // log every 24 × 30 min = 12 h
    let mut log_cnt=0usize;
    let mut last_day_check=-1i64;
    let mut last_tcm_day: f64  = -POST_TCM_FREEZE; // initialised so Day 0 is not frozen
    let mut last_thaw_day: f64 = -1.0;             // tracks whether thaw already fired this burn
    let mut last_tcm_od_cr: f64 = C_R_INIT;        // OD C_R at time of most recent TCM (for end analysis)

    println!("\nRunning simulation ({:.0} days)...", tof_days);

    for _step in 0..n_steps {
        let actual_dt=dt_tr.min(tof_s-t_elapsed);

        // -- Truth propagation ------------------------------------------------
        (r_tr,v_tr)=rk4_rv(r_tr,v_tr,actual_dt,C_R_TRUTH);
        t_elapsed+=actual_dt;
        let t_abs=dep_s+t_elapsed;

        // -- Ground OD + on-board EKF predict (1h cadence) -------------------
        ekf_sub+=1;
        if ekf_sub*(dt_tr as usize) >= dt_ekf as usize {
            ekf_sub=0;
            // Clamp last step so OD clock matches truth clock exactly at end
            let od_dt = f64::min(dt_ekf, tof_s - od_t_elapsed);
            od_t_elapsed += od_dt;
            od.predict(od_dt);
            ekf.predict_simple(od_dt);
            ekf_cnt+=1;

            // DSN measurement update (ground OD only)
            if ekf_cnt % meas_hz == 0 {
                let (r_earth,v_earth)=earth_rv(t_abs);
                let ve3 = Vector3::from(r_earth); // nalgebra Vector3 for GroundOdEkf API
                let vve3 = Vector3::from(v_earth);
                let dr=sub(r_tr,r_earth);
                let rng_true=norm(dr);
                let rrate_true=dot(dr,sub(v_tr,v_earth))/rng_true;

                od.update_range(&ve3, rng_true   + RNG_NOISE_M    * rng.gauss());
                od.update_rrate(&ve3, &vve3,
                                rrate_true + RRATE_NOISE_MPS * rng.gauss());

                // Delta-DOR: daily transverse angular measurement.
                if ekf_cnt % 24 == 0 {
                    let dr_od  = sub(v3_from(od.r()), r_earth);
                    let rhat_od = normalize(dr_od);
                    let up: V3 = if rhat_od[2].abs() < 0.9 { [0.0, 0.0, 1.0] }
                                 else                       { [1.0, 0.0, 0.0] };
                    let t1 = normalize(cross(rhat_od, up));
                    let t2 = cross(rhat_od, t1);
                    let z1 = dot(dr, t1) / rng_true + DDOR_NOISE_RAD * rng.gauss();
                    let z2 = dot(dr, t2) / rng_true + DDOR_NOISE_RAD * rng.gauss();
                    od.update_ddor(&ve3, z1, z2);
                }
            }
        }

        // -- Logging (every 12 h) --------------------------------------------
        log_cnt+=1;
        if log_cnt>=log_every || t_elapsed>=tof_s {
            log_cnt=0;
            truth_log.push(Row{t:t_elapsed,x:r_tr[0],y:r_tr[1],z:r_tr[2],
                               vx:v_tr[0],vy:v_tr[1],vz:v_tr[2],cr:C_R_TRUTH});
            od_log.push(Row{t:t_elapsed,x:od.r()[0],y:od.r()[1],z:od.r()[2],
                            vx:od.v()[0],vy:od.v()[1],vz:od.v()[2],cr:od.cr()});
            ekf_log.push(Row{t:t_elapsed,x:ekf.r()[0],y:ekf.r()[1],z:ekf.r()[2],
                             vx:ekf.v()[0],vy:ekf.v()[1],vz:ekf.v()[2],cr:ekf.cr()});
        }

        // -- Daily operations -------------------------------------------------
        let day_now=(t_elapsed/86400.0).floor() as i64;
        let days_rem=(tof_s-t_elapsed)/86400.0;

        if day_now > last_day_check {
            last_day_check=day_now;

            // 1. Ground uplinks OD state to on-board EKF
            ekf.inject_uplink(&od.uplink(), UPL_SIGMA_R, UPL_SIGMA_V);

            // 2a. C_R thaw — fires once, BURN_RECON_DAYS after each burn.
            //     C_R was frozen at burn time (P[6][6] ≈ 0); re-opening it here
            //     lets predict_od() rebuild cross-correlations via the STM so the
            //     filter can estimate C_R from the post-reconstruction arc before
            //     the next TCM window opens.
            if last_tcm_day >= 0.0
                && (day_now as f64) >= last_tcm_day + BURN_RECON_DAYS
                && last_thaw_day < last_tcm_day
            {
                od.thaw_cr_for_reestimation(POST_BURN_CR_SIGMA);
                last_thaw_day = day_now as f64;
                // Burn reconstruction quality: velocity sigma from filter + actual error vs truth
                let sv = od.sigma_v_mps();
                let v_err = norm(sub(v3_from(od.v()), v_tr));
                println!("  Day {:>4}: C_R thawed  | burn recon: σ_v={:.4} m/s  |δv|={:.4} m/s  C_R_od={:.3}",
                         day_now, sv, v_err, od.cr());
            }

            // 2b. TCM check — gated by two independent freeze windows:
            //    a) MIN_TCM_DAY: initial C_R calibration period at cruise start
            //    b) POST_TCM_FREEZE: C_R re-estimation window after every burn
            //    Both must be satisfied before a new TCM is planned.
            let days_elapsed = t_elapsed / 86400.0;
            let in_calib_freeze = days_elapsed < MIN_TCM_DAY
                               || (day_now as f64) < last_tcm_day + POST_TCM_FREEZE;
            if days_rem > 10.0 && !in_calib_freeze {
                let dt_to_arr=tof_s-t_elapsed;
                let od_s7: S7 = [od.r()[0],od.r()[1],od.r()[2],
                                  od.v()[0],od.v()[1],od.v()[2],od.cr()];
                let od_arr=propagate7(od_s7, dt_to_arr, 3_600.0);
                let od_r_arr=[od_arr[0],od_arr[1],od_arr[2]];
                let od_v_arr=[od_arr[3],od_arr[4],od_arr[5]];
                // r_b_arr / v_b_arr: Bennu state at the fixed arrival epoch (arr_s).
                // Both the OD prediction (propagated for dt_to_arr) and the Bennu
                // target are evaluated at the same epoch, so the miss is consistent.
                let miss_km=bplane_miss(od_r_arr,r_b_arr,od_v_arr,v_b_arr)/1e3;

                miss_log.push((t_elapsed/86400.0, miss_km));

                if miss_km > TCM_THRESHOLD_KM {
                    let jac=drr_dv(v3_from(od.r()),v3_from(od.v()),od.cr(),dt_to_arr);
                    let dv_plan=min_norm_dv(jac,sub(r_b_arr,od_r_arr));
                    let dv_mag=norm(dv_plan);

                    if dv_mag > MIN_TCM_MS {
                        let merge=tcms.last()
                            .map(|p| day_now as f64 - p.t_day < 3.0)
                            .unwrap_or(false);

                        // Execution error
                        let dv_hat=normalize(dv_plan);
                        let perp1=normalize(cross(dv_hat,
                            if dv_hat[2].abs()<0.9{[0.0,0.0,1.0]}else{[1.0,0.0,0.0]}));
                        let perp2=cross(dv_hat,perp1);
                        let mag_s=1.0+TCM_MAG_ERR*rng.gauss();
                        let pt1=dv_mag*TCM_PT_ERR*rng.gauss();
                        let pt2=dv_mag*TCM_PT_ERR*rng.gauss();
                        let dv_exec=add(scale(dv_plan,mag_s),
                                        add(scale(perp1,pt1),scale(perp2,pt2)));

                        // Predicted miss after correction
                        let od_cr_now=od.cr();
                        let (r_ca,v_ca)=propagate_rv(v3_from(od.r()),add(v3_from(od.v()),dv_plan),
                                                      dt_to_arr,3_600.0,od_cr_now);
                        let miss_after=bplane_miss(r_ca,r_b_arr,v_ca,v_b_arr)/1e3;

                        // Save pre-burn truth velocity for post-sim C_R isolation analysis
                        let v_tr_pre_burn = v_tr;
                        last_tcm_od_cr = od_cr_now;

                        // Apply: truth gets exec error, OD and EKF get clean plan
                        v_tr=add(v_tr,dv_exec);
                        od.add_dv(&Vector3::from(dv_plan));
                        ekf.add_dv(&Vector3::from(dv_plan));
                        dv_total+=norm(dv_exec);
                        // Phase 1: freeze C_R and inflate velocity for burn reconstruction.
                        // OD knows dv_plan but not dv_exec; ranging over BURN_RECON_DAYS
                        // will reconstruct the residual ΔV before C_R re-estimation begins.
                        let dv_exec_sigma = dv_mag * TCM_MAG_ERR.hypot(TCM_PT_ERR);
                        od.freeze_cr_for_burn_recon(dv_exec_sigma);
                        last_tcm_day = day_now as f64;  // start post-burn C_R freeze

                        if merge && !tcms.is_empty() {
                            let n_tcm=tcms.len();
                            let prev=tcms.last_mut().unwrap();
                            prev.dv_plan=add(prev.dv_plan,dv_plan);
                            prev.dv_exec=add(prev.dv_exec,dv_exec);
                            prev.miss_after=miss_after;
                            let em=norm(prev.dv_exec); let mb=prev.miss_before; let ma=prev.miss_after;
                            println!("  Day {:>4}: TCM-{} augmented  |ΔV|={:.4} m/s  miss {:.0} -> {:.0} km",
                                day_now, n_tcm, em, mb, ma);
                        } else {
                            println!("  Day {:>4}: TCM-{}  planned   |ΔV|={:.4} m/s  miss {:.0} -> {:.0} km  C_R_od={:.3}",
                                day_now, tcms.len()+1, dv_mag, miss_km, miss_after, od.cr());
                            tcms.push(Tcm{t_day:day_now as f64,r:r_tr,v_pre_burn:v_tr_pre_burn,
                                dv_plan,dv_exec,miss_before:miss_km,miss_after});
                        }
                    }
                }
            }
        }

        if t_elapsed>=tof_s{break;}
    }

    // -- Arrival Approach Burn (AB) -------------------------------------------
    // Cancel the hyperbolic-excess velocity w.r.t. Bennu that the Lambert
    // intercept leaves at arrival, then set a small controlled approach velocity.
    let v_tr_pre_ab    = v_tr;                          // save for CAT targeting below
    let (r_b_fin,v_b_fin)=bennu_ephem.query(arr_s);
    let v_rel_arr      = sub(v_tr, v_b_fin);
    let v_rel_mag      = norm(v_rel_arr);
    let d_to_bennu     = sub(r_b_fin, r_tr);
    let d_mag          = norm(d_to_bennu);
    let toward_bennu   = if d_mag > 1.0 { scale(d_to_bennu, 1.0/d_mag) } else { [1.0,0.0,0.0] };
    // dv_ab: cancel v_rel, leave ARRIVAL_APPROACH_V_MPS toward Bennu
    let dv_ab          = sub(scale(toward_bennu, ARRIVAL_APPROACH_V_MPS), v_rel_arr);
    let dv_ab_mag      = norm(dv_ab);
    let _v_nom_post_ab = add(v_tr_pre_ab, dv_ab);      // nominal (no exec error) for CAT targeting
    // Truth gets a small execution error (0.1% mag — AB is a major planned maneuver,
    // not a small correction; applying TCM_MAG_ERR would give ~41 m/s error on a
    // 2 km/s burn and completely swamp the 5 m/s planned approach velocity).
    let dv_ab_truth    = scale(dv_ab, 1.0 + 0.001 * rng.gauss());
    v_tr               = add(v_tr, dv_ab_truth);
    od.add_dv(&Vector3::from(dv_ab));
    ekf.add_dv(&Vector3::from(dv_ab));
    dv_total          += dv_ab_mag;
    // -- Close-Approach Targeting (CAT) burn ----------------------------------
    // The AB burn leaves v_rel ≈ 5 m/s toward Bennu, but at 465 km (far outside
    // Bennu's ~31 km Hill sphere) solar tidal forces dominate the trajectory.
    // A small perpendicular kick sets the impact parameter so the spacecraft
    // actually reaches FLYBY_PERIAPSIS_M at closest approach.
    //
    // Strategy: bisect on the transverse ΔV magnitude.  Periapsis is monotonically
    // increasing with transverse speed (more sideways = larger miss distance).
    // The bisection propagates in heliocentric frame using the truth dynamics;
    // ~40 iterations converge the periapsis to < 0.1 m.

    // Helper: propagate from handoff state in Hill frame with extra transverse kick
    // and return the minimum range to Bennu (periapsis) over the next 3 days.
    //
    // Uses the same Keplerian Bennu position (keplerian_bennu_pos) and linearised
    // tidal dynamics as proximity_init::fast_approach so the targeted periapsis
    // matches what the proximity EKF sim will actually see.
    // Bisect using the TRUTH post-AB velocity (which includes AB execution error)
    // so the executed closest-approach distance matches the target.
    // OD/EKF receive the same commanded CAT burn; they won't know about the AB error.
    let r_hill_0  = sub(r_tr, r_b_fin);
    let r_hat     = normalize(r_hill_0);
    let v_hill_0 = sub(v_tr, [v_b_fin[0], v_b_fin[1], v_b_fin[2]]);

    let v_radial   = dot(v_hill_0, r_hat);
    let v_t_cat    = sub(v_hill_0, scale(r_hat, v_radial));  // true transverse velocity
    let z_ref: V3  = [0.0, 0.0, 1.0];

    // Terminator-plane targeting: choose the transverse-kick direction so the
    // resulting orbit normal h_hat = normalize(r × v_t) aligns with sun_hat_arr.
    // Optimal: perp = normalize(sun_hat × r_hat) → h_hat = normalize(sun_hat_perp).
    // Residual inclination error = arcsin(|r_hat · sun_hat|); exactly 0 when approach ⊥ Sun.
    let bennu_helio_arr = v3_from(keplerian_bennu_pos(arr_s));
    let sun_hat_arr     = normalize(scale(bennu_helio_arr, -1.0));
    let sun_cross_r     = cross(sun_hat_arr, r_hat);
    let perp = if norm(sun_cross_r) > 1e-6 {
        normalize(sun_cross_r)
    } else {
        normalize(cross(r_hat, z_ref))   // degenerate: approach exactly along Sun line
    };
    // find_periapsis propagates from a purely-radial baseline + terminator kick.
    // Using v_hill_0 here would mix in the AB execution-error transverse component
    // (~0.1% of 2 km/s ≈ 2 m/s), which the full CAT vector will cancel separately.
    let v_hill_radial = scale(r_hat, v_radial);
    let find_periapsis = |dv_t: f64| -> f64 {
        let v0    = add(v_hill_radial, scale(perp, dv_t));
        let mut r = r_hill_0;
        let mut v = v0;
        let mut t = arr_s;
        let mut min_r = norm(r);
        let mut prev  = min_r;
        for _ in 0..(3 * 24 * 60) {            // max 3-day search, 60 s steps
            let bp0 = { let b = keplerian_bennu_pos(t);       [b[0], b[1], b[2]] };
            let bp1 = { let b = keplerian_bennu_pos(t + 30.0); [b[0], b[1], b[2]] };
            let bp2 = { let b = keplerian_bennu_pos(t + 60.0); [b[0], b[1], b[2]] };
            let (rn, vn) = cat_hill_rk4(r, v, bp0, bp1, bp2, 60.0);
            r = rn; v = vn; t += 60.0;
            let cur = norm(r);
            if cur < min_r { min_r = cur; }
            if cur > prev + 10.0 { break; }    // periapsis clearly passed
            prev = cur;
        }
        min_r
    };

    // Analytical close-approach targeting (CAT)
    //
    // Outside Bennu's Hill sphere (31 km), angular momentum is approximately
    // conserved and periapsis satisfies:
    //
    //   r_p ≈ h / v_approach  =  r · |v_t| / v_approach
    //
    // → required transverse speed:  v_t_needed = FLYBY_PERIAPSIS_M · v_approach / r
    //
    // Full CAT burn = cancel the AB-error transverse + set terminator transverse to v_t_needed.
    // A scalar kick in perp alone would leave the orthogonal transverse (AB error) intact,
    // rotating the orbit plane away from the terminator and producing a wrong periapsis.
    let r_hill_mag  = norm(r_hill_0);
    let v_approach  = norm(v_hill_0).max(0.1);      // ≈ ARRIVAL_APPROACH_V_MPS = 5 m/s
    let v_t_needed  = (FLYBY_PERIAPSIS_M * v_approach / r_hill_mag)
                      .min(v_approach * 0.5);        // cap: transverse can't dominate radial
    // dv_cat: cancel all existing transverse velocity (v_t_cat from AB error) and
    // replace with exactly v_t_needed in the terminator direction.
    let dv_cat     = add(scale(perp, v_t_needed), scale(v_t_cat, -1.0));
    let dv_cat_mag = norm(dv_cat);
    let periapsis_nom = find_periapsis(v_t_needed);

    // Apply to truth (0.5 % execution error on magnitude) and nominal to OD/EKF
    let dv_cat_truth = scale(dv_cat, 1.0 + 0.005 * rng.gauss());
    v_tr = add(v_tr, dv_cat_truth);
    od.add_dv(&Vector3::from(dv_cat));
    ekf.add_dv(&Vector3::from(dv_cat));
    dv_total += dv_cat_mag;

    // Patch the last log row so the CSV reflects the post-AB + post-CAT state
    if let Some(last) = truth_log.last_mut() {
        last.vx = v_tr[0]; last.vy = v_tr[1]; last.vz = v_tr[2];
    }
    if let Some(last) = od_log.last_mut() {
        let ov = v3_from(od.v());
        last.vx = ov[0]; last.vy = ov[1]; last.vz = ov[2];
    }
    if let Some(last) = ekf_log.last_mut() {
        let ev = v3_from(ekf.v());
        last.vx = ev[0]; last.vy = ev[1]; last.vz = ev[2];
    }

    // -- Final report ---------------------------------------------------------
    let truth_miss=norm(sub(r_tr,r_b_fin))/1e3;
    let od_miss=norm(sub(v3_from(od.r()),r_b_fin))/1e3;
    let ekf_miss=norm(sub(v3_from(ekf.r()),r_b_fin))/1e3;

    println!("\n-- Arrival approach burn + close-approach targeting --");
    println!("  v-inf at Bennu     : {:.1} m/s", v_rel_mag);
    println!("  AB delta-V         : {:.1} m/s", dv_ab_mag);
    println!("  CAT delta-V        : {:.4} m/s", dv_cat_mag);
    println!("  Target periapsis   : {:.1} km", FLYBY_PERIAPSIS_M / 1e3);
    println!("  Solved periapsis   : {:.3} km", periapsis_nom / 1e3);
    println!("  Range at handoff   : {:.1} km", d_mag / 1e3);
    {
        let v_post  = add(v_hill_0, dv_cat);
        let h_raw   = cross(r_hill_0, v_post);
        let h_hat   = if norm(h_raw) > 1e-12 { normalize(h_raw) } else { r_hat };
        let align   = dot(h_hat, sun_hat_arr).abs().min(1.0).acos().to_degrees();
        let incl_err = dot(sun_hat_arr, r_hat).abs().min(1.0).asin().to_degrees();
        println!("  Terminator align   : h\u{2225}sun  err={:.2}\u{b0}  \
                  (approach\u{b7}sun={:.3}  max-err={:.2}\u{b0})",
                 align, dot(sun_hat_arr, r_hat), incl_err);
    }

    println!("\n-- Final state --");
    println!("  Truth miss from Bennu    : {:.1} km", truth_miss);
    println!("  Ground OD miss from Bennu: {:.1} km", od_miss);
    println!("  On-board EKF miss        : {:.1} km", ekf_miss);
    println!("  OD estimated C_R         : {:.4}  (truth: {:.2})", od.cr(), C_R_TRUTH);
    println!("  On-board EKF C_R         : {:.4}", ekf.cr());
    println!("  TCMs executed            : {}", tcms.len());
    println!("  Total dV (incl. AB)      : {:.4} m/s", dv_total);

    // -- Miss distance analysis: separate C_R error from execution error -------
    // Propagate the truth state at the last TCM forward to arrival using three
    // different assumptions.  The differences isolate each error source.
    if let Some(last) = tcms.last() {
        let dt_rem = tof_s - last.t_day * 86400.0;
        let v_exec = add(last.v_pre_burn, last.dv_exec);   // truth post-execution velocity
        let v_plan = add(last.v_pre_burn, last.dv_plan);   // hypothetical perfect execution

        let (r_a,_) = propagate_rv(last.r, v_plan, dt_rem, 3600.0, last_tcm_od_cr);
        let (r_b,_) = propagate_rv(last.r, v_exec, dt_rem, 3600.0, last_tcm_od_cr);
        let (r_c,_) = propagate_rv(last.r, v_exec, dt_rem, 3600.0, C_R_TRUTH);

        let miss_a = norm(sub(r_a, r_b_arr)) / 1e3;   // perfect exec + OD C_R (TCM baseline)
        let miss_b = norm(sub(r_b, r_b_arr)) / 1e3;   // real exec   + OD C_R (adds exec error)
        let miss_c = norm(sub(r_c, r_b_arr)) / 1e3;   // real exec   + truth C_R (actual truth miss)

        println!("\n-- Miss distance breakdown (from last TCM, day {:.0}, {:.0} d remaining) --",
                 last.t_day, dt_rem / 86400.0);
        println!("  Perfect exec + OD C_R  ({:.3}): {:>6.1} km  ← TCM targeting baseline",
                 last_tcm_od_cr, miss_a);
        println!("  Real exec   + OD C_R  ({:.3}): {:>6.1} km  ← adds exec error: {:+.1} km",
                 last_tcm_od_cr, miss_b, miss_b - miss_a);
        println!("  Real exec   + truth C_R ({:.2}): {:>6.1} km  ← actual miss  (C_R err: {:+.1} km)",
                 C_R_TRUTH, miss_c, miss_c - miss_b);
        println!("  C_R error magnitude: {:.3}  ({:.1}% of truth)",
                 (last_tcm_od_cr - C_R_TRUTH).abs(),
                 (last_tcm_od_cr - C_R_TRUTH).abs() / C_R_TRUTH * 100.0);
    }

    // -- Save outputs ---------------------------------------------------------
    let save_traj=|path: &str, rows: &Vec<Row>| {
        use std::fmt::Write as W;
        let mut out=String::new();
        writeln!(out,"time_s,x_m,y_m,z_m,vx_ms,vy_ms,vz_ms,cr").unwrap();
        for r in rows {
            writeln!(out,"{:.1},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.8}",
                r.t,r.x,r.y,r.z,r.vx,r.vy,r.vz,r.cr).unwrap();
        }
        std::fs::write(path,&out).unwrap();
        println!("  Saved {} ({} rows)",path,rows.len());
    };
    save_traj("out/cruise_ops/truth.csv",     &truth_log);
    save_traj("out/cruise_ops/ground_od.csv", &od_log);
    save_traj("out/cruise_ops/onboard_ekf.csv",&ekf_log);

    {
        use std::fmt::Write as W;
        let mut out=String::new();
        writeln!(out,"day,x_m,y_m,z_m,dvx_plan,dvy_plan,dvz_plan,\
                      dvx_exec,dvy_exec,dvz_exec,dv_mag_ms,miss_before_km,miss_after_km").unwrap();
        for t in &tcms {
            writeln!(out,"{:.1},{:.4e},{:.4e},{:.4e},{:.6},{:.6},{:.6},\
                          {:.6},{:.6},{:.6},{:.6},{:.2},{:.2}",
                t.t_day,t.r[0],t.r[1],t.r[2],
                t.dv_plan[0],t.dv_plan[1],t.dv_plan[2],
                t.dv_exec[0],t.dv_exec[1],t.dv_exec[2],
                norm(t.dv_exec),t.miss_before,t.miss_after).unwrap();
        }
        std::fs::write("out/cruise_ops/tcm_log.csv",&out).unwrap();
        println!("  Saved out/cruise_ops/tcm_log.csv ({} TCMs)",tcms.len());
    }
    {
        use std::fmt::Write as W;
        let mut out=String::new();
        writeln!(out,"day,miss_km").unwrap();
        for &(d,m) in &miss_log { writeln!(out,"{:.2},{:.2}",d,m).unwrap(); }
        std::fs::write("out/cruise_ops/bplane_history.csv",&out).unwrap();
    }
    {
        use std::fmt::Write as W;
        let mut out=String::new();
        writeln!(out,"time_s,bennu_x,bennu_y,bennu_z,earth_x,earth_y,earth_z").unwrap();
        for k in 0..=600usize {
            let t=tof_s*k as f64/600.0;
            let (rb,_)=bennu_ephem.query(dep_s+t); let (re,_)=earth_rv(dep_s+t);
            writeln!(out,"{:.1},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e}",
                t,rb[0],rb[1],rb[2],re[0],re[1],re[2]).unwrap();
        }
        std::fs::write("out/cruise_ops/body_tracks.csv",&out).unwrap();
        println!("  Saved out/cruise_ops/body_tracks.csv");
    }
    println!("\nTo visualise: python plot/plot_cruise_ops.py");
}

// ── Hill-frame helpers for CAT targeting bisection ──────────────────────────
//
// These mirror the dynamics in proximity_init::hill_rk4 / accel_filter so
// that the CAT bisection and the fast-approach propagator use identical physics.
// The Keplerian bennu position (keplerian_bennu_pos) is the same function used
// by accel_filter inside the proximity simulation.

/// Bennu gravity (point-mass) + linearised solar tidal acceleration [m/s²]
/// in the non-rotating Hill frame, given spacecraft Hill position `r` [m]
/// and Bennu heliocentric position `bp` [m].
fn cat_hill_accel(r: V3, bp: V3) -> V3 {
    const MU_BENNU_CAT: f64 = 4.9;   // m³/s²  (matches lib config::MU_BENNU)

    // Bennu gravity
    let r2 = dot(r, r);
    let r1 = r2.sqrt();
    let a_g = scale(r, -MU_BENNU_CAT / (r2 * r1));

    // Linearised solar tidal: -GM_sun/|r_B|³ · r_sc + 3GM_sun/|r_B|⁵ · (r_B·r_sc) r_B
    let rb2 = dot(bp, bp);
    let rb1 = rb2.sqrt();
    let rb3 = rb2 * rb1;
    let rb5 = rb3 * rb2;
    let d   = dot(bp, r);
    let a_t = add(scale(r, -MU_SUN / rb3),
                  scale(bp, 3.0 * MU_SUN * d / rb5));

    add(a_g, a_t)
}

/// Single RK4 step in the Hill frame using cat_hill_accel.
/// `bp0/1/2` are Bennu heliocentric positions at t, t+dt/2, t+dt respectively.
fn cat_hill_rk4(r: V3, v: V3, bp0: V3, bp1: V3, bp2: V3, dt: f64) -> (V3, V3) {
    let k1r = v;
    let k1v = cat_hill_accel(r, bp0);

    let r2 = add(r, scale(k1r, dt * 0.5));
    let v2 = add(v, scale(k1v, dt * 0.5));
    let k2r = v2;
    let k2v = cat_hill_accel(r2, bp1);

    let r3 = add(r, scale(k2r, dt * 0.5));
    let v3 = add(v, scale(k2v, dt * 0.5));
    let k3r = v3;
    let k3v = cat_hill_accel(r3, bp1);

    let r4 = add(r, scale(k3r, dt));
    let v4 = add(v, scale(k3v, dt));
    let k4r = v4;
    let k4v = cat_hill_accel(r4, bp2);

    let dr = scale(add(k1r, add(scale(k2r, 2.0), add(scale(k3r, 2.0), k4r))), dt / 6.0);
    let dv = scale(add(k1v, add(scale(k2v, 2.0), add(scale(k3v, 2.0), k4v))), dt / 6.0);
    (add(r, dr), add(v, dv))
}
