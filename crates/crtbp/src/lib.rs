//! CRTBP / BCR4BP mechanics — shared crate for Earth-Moon trajectory design.
//!
//! # Module map
//!
//! | Module             | Responsibility                                                          |
//! |--------------------|-------------------------------------------------------------------------|
//! | `crtbp`            | EOM 2D/3D, Jacobian, Jacobi constant, Lagrange points, frame transform  |
//! | `linalg`           | 6-D vector/matrix helpers (no physics)                                  |
//! | `propagator`       | Dopri5 integrators for CRTBP (2D/3D/+STM) and BCR4BP                   |
//! | `periodic_orbits`  | Differential correctors + family continuation (Lyapunov/Halo/DRO)      |
//! | `manifolds`        | Monodromy matrix, unstable/stable eigenvectors, manifold branch shooting |
//! | `transfers`        | Transfer design — Poincaré sections, WSB capture, LOI/TLI ΔV helpers   |

pub mod crtbp;
pub mod linalg;
pub mod propagator;
pub mod periodic_orbits;
pub mod manifolds;
pub mod transfers;
