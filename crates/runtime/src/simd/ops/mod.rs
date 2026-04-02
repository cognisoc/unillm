//! Fused Operations for Hot Paths
//!
//! These operations are optimized to minimize memory bandwidth by
//! fusing multiple operations into single passes.

mod rmsnorm;
mod rope;
mod swiglu;

pub use rmsnorm::*;
pub use rope::*;
pub use swiglu::*;
