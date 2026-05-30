//! The three-stage vardiff pipeline.
//!
//! Three sequential stages plus a composing adapter that let any
//! (Estimator, Boundary, UpdateRule) triple be automatically a valid
//! [`Vardiff`] implementation. The classic algorithm is one specific
//! composition (see [`classic_composed`]) and is asserted fire-for-fire
//! equivalent to [`crate::vardiff::classic::VardiffState`] by the
//! `vardiff_sim` crate's equivalence-test suite.
//!
//! The recommended production composition (FullRemedy):
//!
//! ```rust,ignore
//! use channels_sv2::vardiff::composed::{
//!     Composed, EwmaEstimator, PoissonCI, PartialRetarget,
//! };
//! use channels_sv2::vardiff::SystemClock;
//! use std::sync::Arc;
//!
//! let v = Composed::new(
//!     EwmaEstimator::new(120),
//!     PoissonCI::default_parametric(),
//!     PartialRetarget::new(0.2),
//!     /* min_hashrate */ 1.0,
//!     Arc::new(SystemClock),
//! );
//! ```
//!
//! See `sim/docs/DESIGN.md` and `sim/docs/FINDINGS.md` for the
//! architectural rationale and the empirical case for `FullRemedy`.
//!
//! [`Vardiff`]: crate::vardiff::Vardiff

pub mod boundary;
pub mod composed;
pub mod decision;
pub mod estimator;
pub mod update;

pub use boundary::{
    AdaptiveCusumBoundary, AsymmetricCusumBoundary, Boundary, CredibleIntervalBoundary,
    CusumBoundary, PoissonCI, StepFunction,
};
pub use composed::{classic_composed, ClassicComposed, Composed};
pub use decision::DecisionRecord;
pub use estimator::Uncertainty;
pub use estimator::{
    BayesianEstimator, CumulativeCounter, Estimator, EstimatorContext, EstimatorSnapshot,
    EwmaEstimator, KalmanEstimator, SlidingWindowEstimator,
};
pub use update::{
    AdaptivePartialRetarget, FullRetargetNoClamp, FullRetargetWithClamp, PartialRetarget,
    UpdateRule,
};
