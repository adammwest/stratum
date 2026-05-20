//! The `Composed` adapter — bundles the three pipeline stages into a
//! single `Vardiff` implementation.
//!
//! `Composed<E, B, U>` carries a blanket `impl Vardiff` so any
//! composition of (Estimator, Boundary, UpdateRule) is automatically a
//! valid production `Vardiff`. The `vardiff_sim` crate adds an
//! `impl Observable for Composed` extension to expose the per-tick
//! decision state for characterization runs.

use std::sync::Arc;

use bitcoin::Target;

use crate::vardiff::{error::VardiffError, Clock, Vardiff};

use super::boundary::{Boundary, StepFunction};
use super::decision::DecisionRecord;
use super::estimator::{CumulativeCounter, Estimator, EstimatorContext};
use super::update::{FullRetargetWithClamp, UpdateRule};

/// A vardiff algorithm composed of three sequential pipeline stages.
///
/// `Composed<E, B, U>` is a drop-in replacement for `VardiffState`
/// for any production code path that holds a `Box<dyn Vardiff>` or
/// accepts an `impl Vardiff`. The three type parameters correspond to:
///
/// - `E`: Estimator — how observations accumulate; produces belief about hashrate.
/// - `B`: Boundary — decides whether the deviation from target is signal or noise.
/// - `U`: UpdateRule — computes the new target when the algorithm fires.
///
/// The deviation statistic (|h_estimate / current_h - 1| × 100) is
/// computed inline by the adapter — it's a fixed normalization step,
/// not a configurable axis.
#[derive(Debug)]
pub struct Composed<E: Estimator, B: Boundary, U: UpdateRule> {
    pub estimator: E,
    pub boundary: B,
    pub update: U,
    pub timestamp_of_last_update: u64,
    pub min_allowed_hashrate: f32,
    pub clock: Arc<dyn Clock>,
    pub last_decision: Option<DecisionRecord>,
}

impl<E, B, U> Composed<E, B, U>
where
    E: Estimator,
    B: Boundary,
    U: UpdateRule,
{
    pub fn new(
        estimator: E,
        boundary: B,
        update: U,
        min_allowed_hashrate: f32,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let timestamp_of_last_update = clock.now_secs();
        Self {
            estimator,
            boundary,
            update,
            timestamp_of_last_update,
            min_allowed_hashrate,
            clock,
            last_decision: None,
        }
    }
}

impl<E, B, U> Vardiff for Composed<E, B, U>
where
    E: Estimator,
    B: Boundary,
    U: UpdateRule,
{
    fn last_update_timestamp(&self) -> u64 {
        self.timestamp_of_last_update
    }
    fn shares_since_last_update(&self) -> u32 {
        self.estimator.shares_count()
    }
    fn set_timestamp_of_last_update(&mut self, ts: u64) {
        self.timestamp_of_last_update = ts;
    }
    fn increment_shares_since_last_update(&mut self) {
        self.estimator.observe(1);
    }
    fn add_shares(&mut self, n: u32) {
        self.estimator.observe(n);
    }
    fn min_allowed_hashrate(&self) -> f32 {
        self.min_allowed_hashrate
    }

    fn reset_counter(&mut self) -> Result<(), VardiffError> {
        self.timestamp_of_last_update = self.clock.now_secs();
        self.estimator.on_fire(0.0, 0.0);
        Ok(())
    }

    fn try_vardiff(
        &mut self,
        hashrate: f32,
        target: &Target,
        shares_per_minute: f32,
    ) -> Result<Option<f32>, VardiffError> {
        self.last_decision = None;

        let now = self.clock.now_secs();
        let dt = now.saturating_sub(self.timestamp_of_last_update);

        if dt <= 15 {
            return Ok(None);
        }

        // Stage 1: Estimator produces its belief.
        let ctx = EstimatorContext {
            current_hashrate: hashrate,
            current_target: target,
            shares_per_minute,
        };
        let snap = self.estimator.snapshot(dt, &ctx);

        // Deviation: fixed normalization (|ratio - 1| × 100).
        let delta = if hashrate > 0.0 {
            ((snap.h_estimate as f64 / hashrate as f64) - 1.0).abs() * 100.0
        } else {
            0.0
        };

        // Stage 2: Boundary decides threshold.
        let threshold = self.boundary.threshold(dt, shares_per_minute, &snap);

        self.last_decision = Some(DecisionRecord {
            delta,
            threshold,
            h_estimate: snap.h_estimate,
            uncertainty: snap.uncertainty,
        });

        if delta < threshold {
            return Ok(None);
        }

        // Stage 3: UpdateRule computes the new target.
        let mut new_hashrate = self
            .update
            .next_hashrate(&snap, hashrate, delta, threshold, shares_per_minute);

        if new_hashrate < self.min_allowed_hashrate {
            new_hashrate = self.min_allowed_hashrate;
        }

        // Notify estimator of the fire.
        self.timestamp_of_last_update = now;
        self.estimator.on_fire(new_hashrate, hashrate);

        Ok(Some(new_hashrate))
    }
}

// ============================================================================
// ClassicComposed — the three-stage representation of VardiffState
// ============================================================================

/// The Classic algorithm expressed as a `Composed` triple. Asserted
/// fire-for-fire equivalent to `VardiffState` by the sim crate's
/// equivalence test suite.
pub type ClassicComposed = Composed<CumulativeCounter, StepFunction, FullRetargetWithClamp>;

/// Constructs the Classic algorithm as a `Composed` triple.
pub fn classic_composed(min_hashrate: f32, clock: Arc<dyn Clock>) -> ClassicComposed {
    Composed::new(
        CumulativeCounter::new(),
        StepFunction::classic_table(),
        FullRetargetWithClamp::classic(),
        min_hashrate,
        clock,
    )
}
