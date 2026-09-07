//! 6-D linear algebra helpers shared by `periodic_orbits` and `manifolds`.
//!
//! All routines work on the 6-element state vector [x, y, z, vx, vy, vz]
//! and the 6×6 state-transition / monodromy matrices used throughout the CRTBP.

// ─── Vector operations ────────────────────────────────────────────────────────

/// Matrix-vector product  r = M · v  for a 6×6 matrix.
pub fn mat_vec6(m: &[[f64; 6]; 6], v: &[f64; 6]) -> [f64; 6] {
    let mut r = [0.0_f64; 6];
    for i in 0..6 { for j in 0..6 { r[i] += m[i][j] * v[j]; } }
    r
}

/// Euclidean norm of a 6-vector.
pub fn norm6(v: &[f64; 6]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Scale a 6-vector by scalar `s`.
pub fn scale6(v: &[f64; 6], s: f64) -> [f64; 6] {
    [v[0]*s, v[1]*s, v[2]*s, v[3]*s, v[4]*s, v[5]*s]
}

/// Element-wise sum of two 6-vectors.
pub fn add6(a: &[f64; 6], b: &[f64; 6]) -> [f64; 6] {
    [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4], a[5]+b[5]]
}

/// Element-wise difference of two 6-vectors: a − b.
pub fn sub6(a: &[f64; 6], b: &[f64; 6]) -> [f64; 6] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3], a[4]-b[4], a[5]-b[5]]
}

/// Dot product of two 6-vectors.
pub fn dot6(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ─── Eigenstructure ───────────────────────────────────────────────────────────

/// Power iteration to find the dominant eigenvector of a 6×6 matrix.
///
/// Returns `(eigenvalue_magnitude, unit_eigenvector)`.
pub fn dominant_eigenvec6(m: &[[f64; 6]; 6], n_iter: usize) -> (f64, [f64; 6]) {
    let mut v = [1.0_f64, 0.5, 0.0, -0.5, 0.3, 0.0];
    let mut lambda = 1.0;
    for _ in 0..n_iter {
        let mv = mat_vec6(m, &v);
        let n  = norm6(&mv);
        if n < 1e-30 { break; }
        lambda = n;
        v = scale6(&mv, 1.0 / n);
    }
    (lambda, v)
}

// ─── Matrix inverse ───────────────────────────────────────────────────────────

/// 6×6 matrix inverse via Gauss-Jordan elimination with partial pivoting.
///
/// Used to invert the monodromy matrix when finding the stable eigenvector.
pub fn mat_inv6(m: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut a   = *m;
    let mut inv = [[0.0_f64; 6]; 6];
    for i in 0..6 { inv[i][i] = 1.0; }

    for col in 0..6_usize {
        // Partial pivot
        let (mut max_val, mut max_row) = (a[col][col].abs(), col);
        for row in (col+1)..6 {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        a.swap(col, max_row);
        inv.swap(col, max_row);

        let pivot = a[col][col];
        for j in 0..6 { a[col][j] /= pivot; inv[col][j] /= pivot; }

        for row in 0..6_usize {
            if row == col { continue; }
            let f = a[row][col];
            for j in 0..6 {
                a[row][j]   -= f * a[col][j];
                inv[row][j] -= f * inv[col][j];
            }
        }
    }
    inv
}

// ─── Small linear systems ────────────────────────────────────────────────────

/// Solve the 2×2 system  [[a00, a01], [a10, a11]] · [x0, x1] = [b0, b1].
///
/// Returns `(x0, x1)`.  Panics if the system is singular.
pub fn solve_2x2(
    a00: f64, a01: f64,
    a10: f64, a11: f64,
    b0:  f64, b1:  f64,
) -> (f64, f64) {
    let det = a00 * a11 - a01 * a10;
    assert!(det.abs() > 1e-30, "2×2 system is singular (det={det:.2e})");
    ((b0 * a11 - b1 * a01) / det,
     (a00 * b1 - a10 * b0) / det)
}
