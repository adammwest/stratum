use crate::vardiff::clock::{Clock, SystemClock};
use bitcoin::Target;
use std::sync::Arc;
use tracing::debug;

const DEFAULT_MIN_HASHRATE: f64 = 1.0;

/// Discounted Gamma-Poisson vardiff parameters.
///
/// State:
///     ratio = true_hashrate / assigned_hashrate
///
/// Observation:
///     observed_shares ~ Poisson(target_shares * ratio)
///
/// Prior/posterior:
///     ratio ~ Gamma(alpha, beta)
///
/// where beta is the rate, not scale.
const DEFAULT_PRIOR_SHARES: f64 = 4.0;
const DEFAULT_INITIAL_RATIO: f64 = 1.0;

/// Exponential forgetting applied before each new observation.
///
/// Lower values react faster but add more jitter.
/// Higher values are steadier but slower after real hashrate changes.
const DEFAULT_DISCOUNT: f64 = 0.99;

/// Conservative control while uncertainty is high.
///
/// Proposed ratio:
///     posterior_mean - z * posterior_std
///
/// Use 0.0 to target posterior mean directly.
const DEFAULT_NORMAL_CONTROL_Z: f64 = 0.19;

/// Control z after a detected statistical break.
///
/// Usually lower than normal z so the controller moves quickly after a break.
const DEFAULT_BREAK_CONTROL_Z: f64 = 0.073;

/// Approximate posterior-predictive break detector.
///
/// Break test:
///     z = (observed - predictive_mean) / sqrt(predictive_variance)
///
/// where:
///     predictive_mean = target_shares * posterior_mean_ratio
///     predictive_variance = predictive_mean
///                         + target_shares^2 * posterior_ratio_variance
const DEFAULT_BREAK_Z: f64 = 3.13;
const DEFAULT_MIN_EXPECTED_SHARES_FOR_BREAK: f64 = 10.0;

/// Posterior strength used when reseeding after a detected break.
const DEFAULT_BREAK_RESET_PRIOR_SHARES: f64 = 1.1;

/// Numerical clamps for the filtered ratio.
const DEFAULT_MIN_RATIO: f64 = 1.0e-6;
const DEFAULT_MAX_RATIO: f64 = 1.0e6;

/// Maximum relative hashrate move accepted in one update.
///
/// 300.0 means up to roughly +30000% in one update.
const DEFAULT_MAX_RELATIVE_HASHRATE_CHANGE: f64 = 300.0;

/// Minimum relative hashrate change required to emit `Some(new_hashrate)`.
const DEFAULT_MIN_UPDATE_RELATIVE_CHANGE: f64 = 0.285;

use super::{error::VardiffError, Vardiff};

/// Represents the dynamic state for a discounted Gamma-Poisson vardiff connection.
///
/// The estimator models:
///
///     observed_shares ~ Poisson(target_shares * ratio)
///
/// where:
///
///     ratio = true_hashrate / assigned_hashrate
///
/// A Gamma posterior is maintained over `ratio`. Under normal conditions the
/// posterior is discounted over time, which lets the estimator adapt without
/// becoming permanently overconfident. A posterior-predictive z-score is used to
/// detect statistical breaks. On break, the posterior is reseeded around the
/// observed ratio so large hashrate jumps are handled quickly without needing a
/// permanently jittery steady-state configuration.
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

    /// Gamma posterior shape for ratio = true_hashrate / assigned_hashrate.
    pub ratio_alpha: f64,

    /// Gamma posterior rate for ratio = true_hashrate / assigned_hashrate.
    pub ratio_beta: f64,

    /// Initial pseudo-share strength.
    pub prior_shares: f64,

    /// Discount factor applied before assimilating each window.
    pub discount: f64,

    /// Normal-mode lower-bound z.
    pub normal_control_z: f64,

    /// Break-mode lower-bound z.
    pub break_control_z: f64,

    /// Absolute z threshold for the approximate posterior-predictive break detector.
    pub break_z: f64,

    /// Minimum predictive expected shares needed before allowing break detection.
    pub min_expected_shares_for_break: f64,

    /// Prior strength used when reseeding the posterior after a break.
    pub break_reset_prior_shares: f64,

    /// Minimum allowed ratio.
    pub min_ratio: f64,

    /// Maximum allowed ratio.
    pub max_ratio: f64,

    /// Maximum relative hashrate move accepted in one update.
    pub max_relative_hashrate_change: f64,

    /// Minimum relative hashrate change required to return `Some(new_hashrate)`.
    pub min_update_relative_change: f64,
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

        let prior_shares = DEFAULT_PRIOR_SHARES.max(1.0e-12);
        let initial_ratio = DEFAULT_INITIAL_RATIO
            .clamp(DEFAULT_MIN_RATIO, DEFAULT_MAX_RATIO);

        Ok(VardiffState {
            shares_since_last_update: 0,
            timestamp_of_last_update: timestamp_secs,
            min_allowed_hashrate: (min_allowed_hashrate as f64).max(DEFAULT_MIN_HASHRATE),
            clock,

            ratio_alpha: prior_shares,
            ratio_beta: prior_shares / initial_ratio,

            prior_shares,
            discount: DEFAULT_DISCOUNT,
            normal_control_z: DEFAULT_NORMAL_CONTROL_Z,
            break_control_z: DEFAULT_BREAK_CONTROL_Z,
            break_z: DEFAULT_BREAK_Z,
            min_expected_shares_for_break: DEFAULT_MIN_EXPECTED_SHARES_FOR_BREAK,
            break_reset_prior_shares: DEFAULT_BREAK_RESET_PRIOR_SHARES,

            min_ratio: DEFAULT_MIN_RATIO,
            max_ratio: DEFAULT_MAX_RATIO,

            max_relative_hashrate_change: DEFAULT_MAX_RELATIVE_HASHRATE_CHANGE,
            min_update_relative_change: DEFAULT_MIN_UPDATE_RELATIVE_CHANGE,
        })
    }

    /// Sets the count of shares since the last update.
    pub fn set_shares_since_last_update(&mut self, shares_since_last_update: u32) {
        self.shares_since_last_update = shares_since_last_update;
    }

    /// Posterior mean of the ratio.
    fn ratio_mean(&self) -> f64 {
        if self.ratio_beta <= 0.0 {
            return 1.0;
        }

        (self.ratio_alpha / self.ratio_beta).clamp(self.min_ratio, self.max_ratio)
    }

    /// Posterior variance of the ratio.
    fn ratio_variance(&self) -> f64 {
        if self.ratio_beta <= 0.0 {
            return self.max_ratio;
        }

        (self.ratio_alpha / self.ratio_beta.powi(2)).max(0.0)
    }

    /// Approximate posterior-predictive z-score for the current count window.
    ///
    /// This uses the law of total variance:
    ///
    ///     Var[Y] = E[Var[Y | ratio]] + Var[E[Y | ratio]]
    ///            = target_shares * mean_ratio
    ///            + target_shares^2 * var_ratio
    fn predictive_z_score(&self, observed_shares: f64, target_shares: f64) -> Option<f64> {
        let target_shares = target_shares.max(0.0);

        if target_shares <= 0.0 {
            return None;
        }

        let mean_ratio = self.ratio_mean();
        let var_ratio = self.ratio_variance();

        let predictive_mean = target_shares * mean_ratio;
        let predictive_variance =
            (predictive_mean + target_shares.powi(2) * var_ratio).max(1.0e-12);

        Some((observed_shares - predictive_mean) / predictive_variance.sqrt())
    }

    /// Returns true when the current count is implausible under the posterior predictive model.
    fn is_break(&self, observed_shares: f64, target_shares: f64) -> bool {
        let mean_ratio = self.ratio_mean();
        let predictive_expected = target_shares * mean_ratio;

        if predictive_expected < self.min_expected_shares_for_break {
            return false;
        }

        let Some(z_score) = self.predictive_z_score(observed_shares, target_shares) else {
            return false;
        };

        z_score.abs() >= self.break_z
    }

    /// Reseeds the Gamma posterior around the observed ratio after a detected break.
    fn reset_posterior_from_observation(&mut self, observed_shares: f64, target_shares: f64) {
        let target_shares = target_shares.max(1.0e-12);
        let observed_ratio = (observed_shares / target_shares)
            .clamp(self.min_ratio, self.max_ratio);

        let strength = self.break_reset_prior_shares.max(1.0e-12);

        self.ratio_alpha = strength;
        self.ratio_beta = strength / observed_ratio;

        self.ratio_alpha += observed_shares.max(0.0);
        self.ratio_beta += target_shares;

        self.clamp_posterior();

        debug!(
            target: "vardiff",
            "Gamma-Poisson break reset:
            - Observed shares: {:.4}
            - Target shares: {:.4}
            - Observed ratio: {:.8}
            - Reset alpha: {:.8}
            - Reset beta: {:.8}
            - Reset posterior mean ratio: {:.8}
            - Reset posterior std ratio: {:.8}",
            observed_shares,
            target_shares,
            observed_ratio,
            self.ratio_alpha,
            self.ratio_beta,
            self.ratio_mean(),
            self.ratio_variance().sqrt(),
        );
    }

    /// Applies one discounted Gamma-Poisson update.
    fn update_gamma_poisson(&mut self, observed_shares: f64, target_shares: f64) {
        let observed_shares = observed_shares.max(0.0);
        let target_shares = target_shares.max(1.0e-12);

        let discount = self.discount.clamp(0.0, 1.0);

        self.ratio_alpha = self.ratio_alpha * discount + observed_shares;
        self.ratio_beta = self.ratio_beta * discount + target_shares;

        self.clamp_posterior();

        debug!(
            target: "vardiff",
            "Gamma-Poisson update:
            - Observed shares: {:.4}
            - Target shares: {:.4}
            - Discount: {:.6}
            - Posterior alpha: {:.8}
            - Posterior beta: {:.8}
            - Posterior mean ratio: {:.8}
            - Posterior std ratio: {:.8}",
            observed_shares,
            target_shares,
            discount,
            self.ratio_alpha,
            self.ratio_beta,
            self.ratio_mean(),
            self.ratio_variance().sqrt(),
        );
    }

    /// Keeps posterior values finite and inside the configured ratio range.
    fn clamp_posterior(&mut self) {
        if !self.ratio_alpha.is_finite() || self.ratio_alpha <= 0.0 {
            self.ratio_alpha = self.prior_shares.max(1.0e-12);
        }

        if !self.ratio_beta.is_finite() || self.ratio_beta <= 0.0 {
            self.ratio_beta = self.ratio_alpha / DEFAULT_INITIAL_RATIO.max(self.min_ratio);
        }

        let mean = self.ratio_alpha / self.ratio_beta;

        if mean < self.min_ratio {
            self.ratio_beta = self.ratio_alpha / self.min_ratio;
        } else if mean > self.max_ratio {
            self.ratio_beta = self.ratio_alpha / self.max_ratio;
        }

        self.ratio_alpha = self.ratio_alpha.max(1.0e-12);
        self.ratio_beta = self.ratio_beta.max(1.0e-12);
    }

    /// Returns the ratio used for control.
    ///
    /// In normal mode, this can target a conservative lower bound. In break mode,
    /// it usually targets the posterior mean to move quickly.
    fn control_ratio(&self, control_z: f64) -> f64 {
        let mean = self.ratio_mean();
        let std = self.ratio_variance().sqrt();

        (mean - control_z.max(0.0) * std).clamp(self.min_ratio, self.max_ratio)
    }

    /// Rebase the posterior after applying a hashrate update.
    ///
    /// If old ratio is:
    ///
    ///     true_hashrate / old_assigned_hashrate
    ///
    /// and the assigned hashrate is multiplied by `applied_ratio`, then the new
    /// ratio is:
    ///
    ///     old_ratio / applied_ratio
    ///
    /// For a Gamma rate parameterization, scaling the random variable by
    /// `1 / applied_ratio` is equivalent to multiplying beta by `applied_ratio`.
    fn rebase_posterior_after_update(&mut self, applied_ratio: f64) {
        let applied_ratio = applied_ratio
            .clamp(self.min_ratio, self.max_ratio)
            .max(1.0e-12);

        self.ratio_beta *= applied_ratio;
        self.clamp_posterior();

        debug!(
            target: "vardiff",
            "Gamma-Poisson posterior rebased:
            - Applied ratio: {:.8}
            - Rebased alpha: {:.8}
            - Rebased beta: {:.8}
            - Rebased mean ratio: {:.8}
            - Rebased std ratio: {:.8}",
            applied_ratio,
            self.ratio_alpha,
            self.ratio_beta,
            self.ratio_mean(),
            self.ratio_variance().sqrt(),
        );
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
            "Gamma-Poisson update policy:
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

    /// Checks channel performance and potentially updates the assigned hashrate.
    ///
    /// This implementation estimates:
    ///
    ///     ratio = true_hashrate / assigned_hashrate
    ///
    /// with a discounted Gamma-Poisson posterior. It uses posterior-predictive
    /// surprise to detect statistical breaks. Normal updates are conservative;
    /// break updates are more direct.
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

        let break_detected = self.is_break(observed_shares, target_shares);
        let predictive_z = self
            .predictive_z_score(observed_shares, target_shares)
            .unwrap_or(0.0);

        if break_detected {
            self.reset_posterior_from_observation(observed_shares, target_shares);
        } else {
            self.update_gamma_poisson(observed_shares, target_shares);
        }

        let control_z = if break_detected {
            self.break_control_z
        } else {
            self.normal_control_z
        };

        let posterior_mean_ratio = self.ratio_mean();
        let posterior_std_ratio = self.ratio_variance().sqrt();
        let control_ratio = self.control_ratio(control_z);
        let proposed_hashrate = current_hashrate * control_ratio;

        debug!(
            target: "vardiff",
            "Gamma-Poisson vardiff check:
            - Elapsed time: {}s
            - Shares since last update: {}
            - Target shares: {:.4}
            - Current hashrate: {:.2} H/s
            - Posterior mean ratio: {:.8}
            - Posterior std ratio: {:.8}
            - Predictive z: {:.6}
            - Break detected: {}
            - Control z: {:.6}
            - Control ratio: {:.8}
            - Proposed hashrate: {:.2} H/s
            - Current miner target: {:?}",
            delta_time,
            self.shares_since_last_update,
            target_shares,
            current_hashrate,
            posterior_mean_ratio,
            posterior_std_ratio,
            predictive_z,
            break_detected,
            control_z,
            control_ratio,
            proposed_hashrate,
            _target,
        );

        let maybe_new_hashrate = self.apply_update_policy(proposed_hashrate, current_hashrate);

        self.reset_counter()?;

        match maybe_new_hashrate {
            Some(new_hashrate) => {
                let applied_ratio = (new_hashrate / current_hashrate)
                    .clamp(self.min_ratio, self.max_ratio);
                self.rebase_posterior_after_update(applied_ratio);

                debug!(
                    target: "vardiff",
                    "Gamma-Poisson vardiff update accepted:
                    - Previous hashrate: {:.2} H/s
                    - New hashrate: {:.2} H/s
                    - Applied ratio: {:.8}
                    - Delta: {:.2}%
                    - Break detected: {}
                    - Rebased posterior mean ratio: {:.8}
                    - Rebased posterior std ratio: {:.8}",
                    current_hashrate,
                    new_hashrate,
                    applied_ratio,
                    ((new_hashrate - current_hashrate).abs() / current_hashrate) * 100.0,
                    break_detected,
                    self.ratio_mean(),
                    self.ratio_variance().sqrt(),
                );

                Ok(Some(new_hashrate as f32))
            }
            None => Ok(None),
        }
    }
}