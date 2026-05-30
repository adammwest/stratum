use crate::vardiff::clock::{Clock, SystemClock};
use bitcoin::Target;
use std::sync::Arc;
use tracing::debug;

const DEFAULT_MIN_HASHRATE: f64 = 1.0;

const DEFAULT_INITIAL_LOG_RATIO_VARIANCE: f64 = 1.5;
const DEFAULT_PROCESS_VARIANCE: f64 = 0.004;

const DEFAULT_MAX_LOG_RATIO_STEP: f64 = 9.5;
const DEFAULT_MAX_RELATIVE_HASHRATE_CHANGE: f64 = 300.0;

const DEFAULT_MIN_UPDATE_RELATIVE_CHANGE: f64 = 0.30;

const DEFAULT_LOWER_BOUND_Z: f64 = 0.13;
const DEFAULT_LOWER_BOUND_UNCERTAINTY_OFF: f64 = 0.14;
const DEFAULT_LOWER_BOUND_UNCERTAINTY_FULL: f64 = 0.36;

use super::{error::VardiffError, Vardiff};

/// Represents the dynamic state for an Extended Poisson-Kalman vardiff connection.
///
/// This estimator filters:
///
///     theta = log(true_hashrate / assigned_hashrate)
///
/// using the Poisson observation model:
///
///     observed_shares ~ Poisson(target_shares * exp(theta))
///
/// The controller normally targets the posterior mean. When posterior
/// uncertainty is high, it targets a conservative lower confidence bound of the
/// hashrate estimate to collect more shares. The lower-bound term decays to zero
/// once the filter is confident.
#[derive(Debug)]
pub struct VardiffState {
    /// Count of shares received since the last filter observation.
    pub shares_since_last_update: u32,

    /// Unix timestamp (seconds) of the last filter observation.
    pub timestamp_of_last_update: u64,

    /// The lowest hashrate (H/s) the system will allow; values below this are clamped.
    pub min_allowed_hashrate: f64,

    /// Source of current time for elapsed-time computations.
    pub clock: Arc<dyn Clock>,

    /// Current filtered log hashrate ratio.
    ///
    ///     log_ratio = log(true_hashrate / assigned_hashrate)
    pub log_ratio: f64,

    /// Current variance of `log_ratio`.
    pub log_ratio_variance: f64,

    /// Initial variance of `log_ratio`.
    pub initial_log_ratio_variance: f64,

    /// Process variance added at each observation.
    pub process_variance: f64,

    /// Maximum absolute update to `log_ratio` per observation.
    pub max_log_ratio_step: f64,

    /// Maximum relative hashrate move accepted in one update.
    pub max_relative_hashrate_change: f64,

    /// Minimum relative hashrate change required to return `Some(new_hashrate)`.
    pub min_update_relative_change: f64,

    /// Full lower-bound z value used while uncertainty is high.
    pub lower_bound_z: f64,

    /// Below this uncertainty, lower-bound targeting is disabled.
    pub lower_bound_uncertainty_off: f64,

    /// Above this uncertainty, full lower-bound targeting is enabled.
    pub lower_bound_uncertainty_full: f64,
}

impl std::panic::UnwindSafe for VardiffState {}
impl std::panic::RefUnwindSafe for VardiffState {}

impl VardiffState {
    /// Creates a new `VardiffState` with the default minimum hashrate.
    pub fn new() -> Result<Self, VardiffError> {
        Self::new_with_min(DEFAULT_MIN_HASHRATE as f32)
    }

    /// Creates a new `VardiffState` with a specific minimum hashrate.
    pub fn new_with_min(min_allowed_hashrate: f32) -> Result<Self, VardiffError> {
        Self::new_with_clock(min_allowed_hashrate, Arc::new(SystemClock))
    }

    /// Creates a new `VardiffState` with an injected clock.
    ///
    /// This is used by the simulation framework and tests so time can be
    /// advanced deterministically.
    pub fn new_with_clock(
        min_allowed_hashrate: f32,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, VardiffError> {
        let timestamp_secs = clock.now_secs();

        Ok(VardiffState {
            shares_since_last_update: 0,
            timestamp_of_last_update: timestamp_secs,
            min_allowed_hashrate: (min_allowed_hashrate as f64).max(DEFAULT_MIN_HASHRATE),
            clock,

            log_ratio: 0.0,
            log_ratio_variance: DEFAULT_INITIAL_LOG_RATIO_VARIANCE,
            initial_log_ratio_variance: DEFAULT_INITIAL_LOG_RATIO_VARIANCE,
            process_variance: DEFAULT_PROCESS_VARIANCE,
            max_log_ratio_step: DEFAULT_MAX_LOG_RATIO_STEP,

            max_relative_hashrate_change: DEFAULT_MAX_RELATIVE_HASHRATE_CHANGE,
            min_update_relative_change: DEFAULT_MIN_UPDATE_RELATIVE_CHANGE,

            lower_bound_z: DEFAULT_LOWER_BOUND_Z,
            lower_bound_uncertainty_off: DEFAULT_LOWER_BOUND_UNCERTAINTY_OFF,
            lower_bound_uncertainty_full: DEFAULT_LOWER_BOUND_UNCERTAINTY_FULL,
        })
    }

    /// Sets the count of shares since the last update.
    pub fn set_shares_since_last_update(&mut self, shares_since_last_update: u32) {
        self.shares_since_last_update = shares_since_last_update;
    }

    /// Resets the filter mean to neutral after an external hashrate/difficulty update.
    ///
    /// Variance is intentionally retained. Resetting variance after each accepted
    /// update makes the filter repeatedly overreact to one-window Poisson noise.
    fn reset_log_ratio_filter(&mut self) {
        self.log_ratio = 0.0;
    }

    /// Numerically safe exponential.
    fn safe_exp(x: f64) -> f64 {
        x.clamp(-50.0, 50.0).exp()
    }

    /// Effective lower-bound z value.
    ///
    /// This decays to zero when uncertainty is low, avoiding permanent
    /// conservative bias in steady state.
    fn effective_lower_bound_z(&self) -> f64 {
        let uncertainty = self.log_ratio_variance.max(0.0).sqrt();

        let off = self.lower_bound_uncertainty_off.max(0.0);
        let full = self.lower_bound_uncertainty_full.max(off + 1e-12);

        if uncertainty <= off {
            0.0
        } else if uncertainty >= full {
            self.lower_bound_z
        } else {
            let t = (uncertainty - off) / (full - off);
            self.lower_bound_z * t
        }
    }

    /// Applies one Extended Poisson-Kalman update.
    ///
    /// Model:
    ///
    ///     y ~ Poisson(E * exp(theta))
    ///
    /// where:
    ///
    ///     y     = observed_shares
    ///     E     = target_shares
    ///     theta = log_ratio
    ///
    /// Poisson log-likelihood, ignoring constants:
    ///
    ///     l(theta) = y * theta - E * exp(theta)
    ///
    /// Gradient:
    ///
    ///     dl/dtheta = y - E * exp(theta)
    ///
    /// Negative Hessian / information:
    ///
    ///     -d2l/dtheta2 = E * exp(theta)
    fn update_expkf(&mut self, observed_shares: f64, target_shares: f64) -> f64 {
        let observed_shares = observed_shares.max(0.0);
        let target_shares = target_shares.max(1e-12);

        let prior_theta = self.log_ratio;
        let prior_variance = (self.log_ratio_variance + self.process_variance).max(1e-18);

        let expected_shares = target_shares * Self::safe_exp(prior_theta);

        let score = observed_shares - expected_shares;
        let information = expected_shares.max(1e-12);

        let posterior_variance = 1.0 / (1.0 / prior_variance + information);

        let raw_step = posterior_variance * score;
        let clamped_step = raw_step.clamp(-self.max_log_ratio_step, self.max_log_ratio_step);

        self.log_ratio = (prior_theta + clamped_step).clamp(-50.0, 50.0);
        self.log_ratio_variance = posterior_variance.max(1e-18);

        debug!(
            target: "vardiff",
            "ExPKF update:
            - Observed shares: {:.4}
            - Target shares: {:.4}
            - Prior log-ratio: {:.6}
            - Expected shares: {:.4}
            - Score: {:.4}
            - Information: {:.4}
            - Raw step: {:.6}
            - Clamped step: {:.6}
            - Posterior log-ratio: {:.6}
            - Posterior variance: {:.8}
            - Posterior uncertainty: {:.8}",
            observed_shares,
            target_shares,
            prior_theta,
            expected_shares,
            score,
            information,
            raw_step,
            clamped_step,
            self.log_ratio,
            self.log_ratio_variance,
            self.log_ratio_variance.sqrt(),
        );

        self.log_ratio
    }

    /// Clamps a proposed hashrate update and decides whether it should be emitted.
    fn apply_update_policy(&self, proposed_hashrate: f64, current_hashrate: f64) -> Option<f64> {
        let current_hashrate = current_hashrate
            .max(self.min_allowed_hashrate)
            .max(1.0);

        let proposed_hashrate = if proposed_hashrate.is_finite() {
            proposed_hashrate
                .max(self.min_allowed_hashrate)
                .max(1.0)
        } else {
            current_hashrate
        };

        let lower = (current_hashrate * (1.0 - self.max_relative_hashrate_change))
            .max(self.min_allowed_hashrate)
            .max(1.0);

        let upper = (current_hashrate * (1.0 + self.max_relative_hashrate_change))
            .max(self.min_allowed_hashrate)
            .max(1.0);

        let clamped_hashrate = proposed_hashrate.clamp(lower, upper);
        let relative_change = (clamped_hashrate / current_hashrate - 1.0).abs();

        debug!(
            target: "vardiff",
            "ExPKF update policy:
            - Proposed hashrate: {:.2} H/s
            - Current hashrate: {:.2} H/s
            - Clamped hashrate: {:.2} H/s
            - Relative change: {:.6}
            - Min update relative change: {:.6}
            - Max relative change: {:.6}",
            proposed_hashrate,
            current_hashrate,
            clamped_hashrate,
            relative_change,
            self.min_update_relative_change,
            self.max_relative_hashrate_change,
        );

        if relative_change >= self.min_update_relative_change {
            Some(clamped_hashrate)
        } else {
            None
        }
    }
}

impl Vardiff for VardiffState {
    fn last_update_timestamp(&self) -> u64 {
        self.timestamp_of_last_update
    }

    fn shares_since_last_update(&self) -> u32 {
        self.shares_since_last_update
    }

    fn min_allowed_hashrate(&self) -> f32 {
        self.min_allowed_hashrate as f32
    }

    /// Sets the timestamp of the last update.
    fn set_timestamp_of_last_update(&mut self, timestamp_of_last_update: u64) {
        self.timestamp_of_last_update = timestamp_of_last_update;
    }

    /// Increments the share counter by one.
    fn increment_shares_since_last_update(&mut self) {
        self.shares_since_last_update = self.shares_since_last_update.saturating_add(1);
    }

    /// Adds many shares at once.
    ///
    /// The simulation framework uses this to bulk-add Poisson-sampled shares.
    fn add_shares(&mut self, count: u32) {
        self.shares_since_last_update = self.shares_since_last_update.saturating_add(count);
    }

    /// Resets the share counter and updates the timestamp to now.
    fn reset_counter(&mut self) -> Result<(), VardiffError> {
        let timestamp_secs = self.clock.now_secs();

        self.set_timestamp_of_last_update(timestamp_secs);
        self.set_shares_since_last_update(0);

        Ok(())
    }

    /// Checks channel performance and potentially updates the estimated hashrate.
    ///
    /// This implementation filters the Poisson log-intensity ratio:
    ///
    ///     observed_shares ~ Poisson(target_shares * exp(log_ratio))
    ///
    /// The posterior mean hashrate estimate is:
    ///
    ///     current_hashrate * exp(log_ratio)
    ///
    /// When uncertainty is high, the controller uses a lower-bound estimate:
    ///
    ///     current_hashrate * exp(log_ratio - z_eff * sigma)
    ///
    /// where `z_eff` decays to zero once uncertainty is low.
    fn try_vardiff(
        &mut self,
        hashrate: f32,
        _target: &Target,
        shares_per_minute: f32,
    ) -> Result<Option<f32>, VardiffError> {
        let now = self.clock.now_secs();
        let delta_time = now.saturating_sub(self.timestamp_of_last_update);

        if delta_time <= 15 {
            return Ok(None);
        }

        let current_hashrate = (hashrate as f64)
            .max(self.min_allowed_hashrate)
            .max(1.0);

        let elapsed_minutes = delta_time as f64 / 60.0;
        let target_shares = (shares_per_minute as f64).max(0.0) * elapsed_minutes;
        let observed_shares = self.shares_since_last_update as f64;

        if target_shares <= 0.0 {
            self.reset_counter()?;
            return Ok(None);
        }

        let log_ratio = self.update_expkf(observed_shares, target_shares);
        let estimated_ratio = Self::safe_exp(log_ratio);

        let uncertainty = self.log_ratio_variance.max(0.0).sqrt();
        let effective_lower_bound_z = self.effective_lower_bound_z();

        let lower_bound_log_ratio =
            (log_ratio - effective_lower_bound_z * uncertainty).clamp(-50.0, 50.0);

        let lower_bound_ratio = Self::safe_exp(lower_bound_log_ratio);

        let mean_hashrate = current_hashrate * estimated_ratio;
        let proposed_hashrate = current_hashrate * lower_bound_ratio;

        debug!(
            target: "vardiff",
            "ExPKF vardiff check:
            - Elapsed time: {}s
            - Shares since last update: {}
            - Target shares: {:.4}
            - Current hashrate: {:.2} H/s
            - Log-ratio: {:.6}
            - Estimated ratio: {:.6}
            - Mean hashrate: {:.2} H/s
            - Uncertainty: {:.8}
            - Effective lower-bound z: {:.6}
            - Lower-bound log-ratio: {:.6}
            - Lower-bound ratio: {:.6}
            - Proposed hashrate: {:.2} H/s
            - Current miner target: {:?}",
            delta_time,
            self.shares_since_last_update,
            target_shares,
            current_hashrate,
            log_ratio,
            estimated_ratio,
            mean_hashrate,
            uncertainty,
            effective_lower_bound_z,
            lower_bound_log_ratio,
            lower_bound_ratio,
            proposed_hashrate,
            _target,
        );

        let maybe_new_hashrate = self.apply_update_policy(proposed_hashrate, current_hashrate);

        self.reset_counter()?;

        match maybe_new_hashrate {
            Some(new_hashrate) => {
                self.reset_log_ratio_filter();

                debug!(
                    target: "vardiff",
                    "ExPKF vardiff update accepted:
                    - Previous hashrate: {:.2} H/s
                    - New hashrate: {:.2} H/s
                    - Delta: {:.2}%
                    - Log-ratio variance retained: {:.8}",
                    current_hashrate,
                    new_hashrate,
                    ((new_hashrate - current_hashrate).abs() / current_hashrate) * 100.0,
                    self.log_ratio_variance,
                );

                Ok(Some(new_hashrate as f32))
            }
            None => Ok(None),
        }
    }
}