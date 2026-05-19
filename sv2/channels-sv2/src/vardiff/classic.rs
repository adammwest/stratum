use crate::vardiff::clock::{Clock, SystemClock};
use bitcoin::Target;
use std::sync::Arc;
use tracing::debug;

use super::{error::VardiffError, Vardiff};

const DEFAULT_MIN_HASHRATE: f64 = 1.0;

/// Gamma-Poisson prior for:
///
///     ratio = true_hashrate / assigned_hashrate
///
/// with beta as rate, not scale.
const DEFAULT_PRIOR_SHARES: f64 = 9.809931;
const DEFAULT_INITIAL_RATIO: f64 = 0.801425;

/// Evidence accumulation gate.
const DEFAULT_MIN_TARGET_SHARES_FOR_UPDATE: f64 = 1.066427;
const DEFAULT_MAX_UPDATE_INTERVAL_SECS: u64 = 300;
const DEFAULT_RESET_COUNTER_ON_NO_UPDATE: bool = false;

/// Settled-mode classifier.
///
/// Settled mode means the controller has enough elapsed exposure and the current
/// observation is not surprising under the posterior predictive model.
const DEFAULT_SETTLED_MODE_MIN_AGE_SECS: u64 = 300;
const DEFAULT_SETTLED_MODE_MAX_ABS_PREDICTIVE_Z: f64 = 0.882358;
const DEFAULT_SETTLED_MODE_MIN_TARGET_SHARES: f64 = 8.000000;

/// Recovery means the period immediately after an accepted update. It is treated
/// as changing mode, not settled mode.
const DEFAULT_RECOVERY_MODE_SECS: u64 = 180;

/// Break detection.
const DEFAULT_BREAK_Z: f64 = 3.0;
const DEFAULT_MIN_EXPECTED_SHARES_FOR_BREAK: f64 = 4.0;
const DEFAULT_BREAK_RESET_PRIOR_SHARES: f64 = 2.0;
const DEFAULT_BREAK_RESET_BLEND: f64 = 0.85;
const DEFAULT_BREAK_COOLDOWN_SECS: u64 = 60;

/// Numerical clamps for the filtered ratio.
const DEFAULT_MIN_RATIO: f64 = 1.0e-6;
const DEFAULT_MAX_RATIO: f64 = 1.0e6;

/// Changing-hashrate mode parameters.
///
/// This mode is used during breaks, recovery, and non-settled operation. It is
/// allowed to move faster, and `CHANGING_CONTROL_SCALE` controls how much of the
/// inferred ratio error is applied in one proposal:
///
///     proposed_ratio = 1 + scale * (raw_control_ratio - 1)
const DEFAULT_CHANGING_DISCOUNT: f64 = 0.836136;
const DEFAULT_CHANGING_UPWARD_CONTROL_Z: f64 = 0.926083;
const DEFAULT_CHANGING_DOWNWARD_CONTROL_Z: f64 = 1.600000;
const DEFAULT_CHANGING_CONTROL_RATIO_SMOOTHING: f64 = 0.250000;
const DEFAULT_CHANGING_CONTROL_SCALE: f64 = 1.234353;
const DEFAULT_CHANGING_POISSON_UPDATE_Z: f64 = 1.461906;
const DEFAULT_CHANGING_MIN_UPDATE_RELATIVE_CHANGE: f64 = 0.119178;
const DEFAULT_CHANGING_MAX_UPDATE_RELATIVE_CHANGE: f64 = 1.00;
const DEFAULT_CHANGING_MAX_UPWARD_RELATIVE_CHANGE: f64 = 2.087309;
const DEFAULT_CHANGING_MAX_DOWNWARD_RELATIVE_CHANGE: f64 = 0.495645;
const DEFAULT_CHANGING_POST_UPDATE_COOLDOWN_SECS: u64 = 0;

/// Settled-hashrate mode parameters.
///
/// This mode is used only after enough evidence has accumulated and the current
/// count is not surprising. It can allow smaller accuracy corrections, but with
/// stronger smoothing/cooldown to avoid jitter.
const DEFAULT_SETTLED_DISCOUNT: f64 = 0.971075;
const DEFAULT_SETTLED_UPWARD_CONTROL_Z: f64 = 1.000000;
const DEFAULT_SETTLED_DOWNWARD_CONTROL_Z: f64 = 0.747195;
const DEFAULT_SETTLED_CONTROL_RATIO_SMOOTHING: f64 = 0.950000;
const DEFAULT_SETTLED_CONTROL_SCALE: f64 = 0.600000;
const DEFAULT_SETTLED_POISSON_UPDATE_Z: f64 = 0.450000;
const DEFAULT_SETTLED_MIN_UPDATE_RELATIVE_CHANGE: f64 = 0.320000;
const DEFAULT_SETTLED_MAX_UPDATE_RELATIVE_CHANGE: f64 = 0.20;
const DEFAULT_SETTLED_MAX_UPWARD_RELATIVE_CHANGE: f64 = 0.700000;
const DEFAULT_SETTLED_MAX_DOWNWARD_RELATIVE_CHANGE: f64 = 0.750000;
const DEFAULT_SETTLED_POST_UPDATE_COOLDOWN_SECS: u64 = 0;

/// Optional original-classic safety policies.
const DEFAULT_ENABLE_ZERO_SHARE_POLICY: bool = false;
const DEFAULT_ZERO_SHARE_DECAY_LE_30S: f64 = 1.5;
const DEFAULT_ZERO_SHARE_DECAY_LT_60S: f64 = 2.0;
const DEFAULT_ZERO_SHARE_DECAY_GE_60S: f64 = 3.0;

const DEFAULT_EXTREME_UPWARD_DELTA_THRESHOLD: f64 = 10.0;
const DEFAULT_EXTREME_UPWARD_CAP_LE_30S: f64 = 10.0;
const DEFAULT_EXTREME_UPWARD_CAP_LT_60S: f64 = 5.0;
const DEFAULT_EXTREME_UPWARD_CAP_GE_60S: f64 = 3.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VardiffMode {
    Changing,
    Settled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VardiffModeReason {
    Break,
    Recovery,
    NormalChanging,
    Settled,
}

#[derive(Debug, Clone, Copy)]
struct ModePolicy {
    discount: f64,
    upward_control_z: f64,
    downward_control_z: f64,
    smoothing: f64,
    control_scale: f64,
    poisson_update_z: f64,
    min_update_relative_change: f64,
    max_update_relative_change: f64,
    max_upward_relative_change: f64,
    max_downward_relative_change: f64,
    post_update_cooldown_secs: u64,
}

/// Gamma-Poisson vardiff controller with two fundamental control policies:
///
/// 1. Changing hashrate mode: used during break/recovery/unstable periods.
/// 2. Settled hashrate mode: used when evidence is strong and predictive z is small.
///
/// Pipeline:
///
/// 1. Count shares over elapsed exposure.
/// 2. Compute pre-update predictive z and detect breaks.
/// 3. Classify into Changing or Settled policy.
/// 4. Update Gamma-Poisson posterior using mode-specific discount.
/// 5. Compute raw control ratio from posterior mean/std.
/// 6. Apply changing/settled control scale.
/// 7. Apply mode-specific smoothing.
/// 8. Apply evidence-scaled deadband.
/// 9. Apply asymmetric clamps and cooldown.
/// 10. Rebase posterior after accepted update.
#[derive(Debug)]
pub struct VardiffState {
    pub shares_since_last_update: u32,
    pub timestamp_of_last_update: u64,
    pub timestamp_of_last_accepted_update: u64,
    pub timestamp_of_last_break: u64,
    pub min_allowed_hashrate: f64,
    pub clock: Arc<dyn Clock>,

    pub ratio_alpha: f64,
    pub ratio_beta: f64,
    pub prior_shares: f64,

    pub min_target_shares_for_update: f64,
    pub max_update_interval_secs: u64,
    pub reset_counter_on_no_update: bool,

    pub settled_mode_min_age_secs: u64,
    pub settled_mode_max_abs_predictive_z: f64,
    pub settled_mode_min_target_shares: f64,
    pub recovery_mode_secs: u64,

    pub break_z: f64,
    pub min_expected_shares_for_break: f64,
    pub break_reset_prior_shares: f64,
    pub break_reset_blend: f64,
    pub break_cooldown_secs: u64,

    pub min_ratio: f64,
    pub max_ratio: f64,

    pub changing_discount: f64,
    pub changing_upward_control_z: f64,
    pub changing_downward_control_z: f64,
    pub changing_control_ratio_smoothing: f64,
    pub changing_control_scale: f64,
    pub changing_poisson_update_z: f64,
    pub changing_min_update_relative_change: f64,
    pub changing_max_update_relative_change: f64,
    pub changing_max_upward_relative_change: f64,
    pub changing_max_downward_relative_change: f64,
    pub changing_post_update_cooldown_secs: u64,

    pub settled_discount: f64,
    pub settled_upward_control_z: f64,
    pub settled_downward_control_z: f64,
    pub settled_control_ratio_smoothing: f64,
    pub settled_control_scale: f64,
    pub settled_poisson_update_z: f64,
    pub settled_min_update_relative_change: f64,
    pub settled_max_update_relative_change: f64,
    pub settled_max_upward_relative_change: f64,
    pub settled_max_downward_relative_change: f64,
    pub settled_post_update_cooldown_secs: u64,

    pub last_control_ratio: f64,

    pub enable_zero_share_policy: bool,
    pub zero_share_decay_le_30s: f64,
    pub zero_share_decay_lt_60s: f64,
    pub zero_share_decay_ge_60s: f64,

    pub extreme_upward_delta_threshold: f64,
    pub extreme_upward_cap_le_30s: f64,
    pub extreme_upward_cap_lt_60s: f64,
    pub extreme_upward_cap_ge_60s: f64,
}

impl std::panic::UnwindSafe for VardiffState {}
impl std::panic::RefUnwindSafe for VardiffState {}

impl VardiffState {
    pub fn new() -> Result<Self, VardiffError> {
        Self::new_with_min(DEFAULT_MIN_HASHRATE as f32)
    }

    pub fn new_with_min(min_allowed_hashrate: f32) -> Result<Self, VardiffError> {
        Self::new_with_clock(min_allowed_hashrate, Arc::new(SystemClock))
    }

    pub fn new_with_clock(
        min_allowed_hashrate: f32,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, VardiffError> {
        let timestamp_secs = clock.now_secs();
        let prior_shares = DEFAULT_PRIOR_SHARES.max(1.0e-12);
        let initial_ratio = DEFAULT_INITIAL_RATIO.clamp(DEFAULT_MIN_RATIO, DEFAULT_MAX_RATIO);

        Ok(Self {
            shares_since_last_update: 0,
            timestamp_of_last_update: timestamp_secs,
            timestamp_of_last_accepted_update: 0,
            timestamp_of_last_break: 0,
            min_allowed_hashrate: (min_allowed_hashrate as f64).max(DEFAULT_MIN_HASHRATE),
            clock,

            ratio_alpha: prior_shares,
            ratio_beta: prior_shares / initial_ratio,
            prior_shares,

            min_target_shares_for_update: DEFAULT_MIN_TARGET_SHARES_FOR_UPDATE,
            max_update_interval_secs: DEFAULT_MAX_UPDATE_INTERVAL_SECS,
            reset_counter_on_no_update: DEFAULT_RESET_COUNTER_ON_NO_UPDATE,

            settled_mode_min_age_secs: DEFAULT_SETTLED_MODE_MIN_AGE_SECS,
            settled_mode_max_abs_predictive_z: DEFAULT_SETTLED_MODE_MAX_ABS_PREDICTIVE_Z,
            settled_mode_min_target_shares: DEFAULT_SETTLED_MODE_MIN_TARGET_SHARES,
            recovery_mode_secs: DEFAULT_RECOVERY_MODE_SECS,

            break_z: DEFAULT_BREAK_Z,
            min_expected_shares_for_break: DEFAULT_MIN_EXPECTED_SHARES_FOR_BREAK,
            break_reset_prior_shares: DEFAULT_BREAK_RESET_PRIOR_SHARES,
            break_reset_blend: DEFAULT_BREAK_RESET_BLEND,
            break_cooldown_secs: DEFAULT_BREAK_COOLDOWN_SECS,

            min_ratio: DEFAULT_MIN_RATIO,
            max_ratio: DEFAULT_MAX_RATIO,

            changing_discount: DEFAULT_CHANGING_DISCOUNT,
            changing_upward_control_z: DEFAULT_CHANGING_UPWARD_CONTROL_Z,
            changing_downward_control_z: DEFAULT_CHANGING_DOWNWARD_CONTROL_Z,
            changing_control_ratio_smoothing: DEFAULT_CHANGING_CONTROL_RATIO_SMOOTHING,
            changing_control_scale: DEFAULT_CHANGING_CONTROL_SCALE,
            changing_poisson_update_z: DEFAULT_CHANGING_POISSON_UPDATE_Z,
            changing_min_update_relative_change: DEFAULT_CHANGING_MIN_UPDATE_RELATIVE_CHANGE,
            changing_max_update_relative_change: DEFAULT_CHANGING_MAX_UPDATE_RELATIVE_CHANGE,
            changing_max_upward_relative_change: DEFAULT_CHANGING_MAX_UPWARD_RELATIVE_CHANGE,
            changing_max_downward_relative_change: DEFAULT_CHANGING_MAX_DOWNWARD_RELATIVE_CHANGE,
            changing_post_update_cooldown_secs: DEFAULT_CHANGING_POST_UPDATE_COOLDOWN_SECS,

            settled_discount: DEFAULT_SETTLED_DISCOUNT,
            settled_upward_control_z: DEFAULT_SETTLED_UPWARD_CONTROL_Z,
            settled_downward_control_z: DEFAULT_SETTLED_DOWNWARD_CONTROL_Z,
            settled_control_ratio_smoothing: DEFAULT_SETTLED_CONTROL_RATIO_SMOOTHING,
            settled_control_scale: DEFAULT_SETTLED_CONTROL_SCALE,
            settled_poisson_update_z: DEFAULT_SETTLED_POISSON_UPDATE_Z,
            settled_min_update_relative_change: DEFAULT_SETTLED_MIN_UPDATE_RELATIVE_CHANGE,
            settled_max_update_relative_change: DEFAULT_SETTLED_MAX_UPDATE_RELATIVE_CHANGE,
            settled_max_upward_relative_change: DEFAULT_SETTLED_MAX_UPWARD_RELATIVE_CHANGE,
            settled_max_downward_relative_change: DEFAULT_SETTLED_MAX_DOWNWARD_RELATIVE_CHANGE,
            settled_post_update_cooldown_secs: DEFAULT_SETTLED_POST_UPDATE_COOLDOWN_SECS,

            last_control_ratio: initial_ratio,

            enable_zero_share_policy: DEFAULT_ENABLE_ZERO_SHARE_POLICY,
            zero_share_decay_le_30s: DEFAULT_ZERO_SHARE_DECAY_LE_30S,
            zero_share_decay_lt_60s: DEFAULT_ZERO_SHARE_DECAY_LT_60S,
            zero_share_decay_ge_60s: DEFAULT_ZERO_SHARE_DECAY_GE_60S,

            extreme_upward_delta_threshold: DEFAULT_EXTREME_UPWARD_DELTA_THRESHOLD,
            extreme_upward_cap_le_30s: DEFAULT_EXTREME_UPWARD_CAP_LE_30S,
            extreme_upward_cap_lt_60s: DEFAULT_EXTREME_UPWARD_CAP_LT_60S,
            extreme_upward_cap_ge_60s: DEFAULT_EXTREME_UPWARD_CAP_GE_60S,
        })
    }

    pub fn set_shares_since_last_update(&mut self, shares_since_last_update: u32) {
        self.shares_since_last_update = shares_since_last_update;
    }

    fn ratio_mean(&self) -> f64 {
        if self.ratio_beta <= 0.0 {
            return 1.0;
        }

        (self.ratio_alpha / self.ratio_beta).clamp(self.min_ratio, self.max_ratio)
    }

    fn ratio_variance(&self) -> f64 {
        if self.ratio_beta <= 0.0 {
            return self.max_ratio;
        }

        (self.ratio_alpha / self.ratio_beta.powi(2)).max(0.0)
    }

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

    fn is_break(&self, observed_shares: f64, target_shares: f64, now: u64) -> bool {
        if now.saturating_sub(self.timestamp_of_last_break) < self.break_cooldown_secs {
            return false;
        }

        let predictive_expected = target_shares * self.ratio_mean();
        if predictive_expected < self.min_expected_shares_for_break {
            return false;
        }

        let Some(z_score) = self.predictive_z_score(observed_shares, target_shares) else {
            return false;
        };

        z_score.abs() >= self.break_z
    }

    fn classify_mode(
        &self,
        delta_time: u64,
        target_shares: f64,
        predictive_z: f64,
        break_detected: bool,
        now: u64,
    ) -> (VardiffMode, VardiffModeReason) {
        if break_detected {
            return (VardiffMode::Changing, VardiffModeReason::Break);
        }

        if now.saturating_sub(self.timestamp_of_last_accepted_update) < self.recovery_mode_secs {
            return (VardiffMode::Changing, VardiffModeReason::Recovery);
        }

        let settled = delta_time >= self.settled_mode_min_age_secs
            && target_shares >= self.settled_mode_min_target_shares
            && predictive_z.abs() <= self.settled_mode_max_abs_predictive_z;

        if settled {
            (VardiffMode::Settled, VardiffModeReason::Settled)
        } else {
            (VardiffMode::Changing, VardiffModeReason::NormalChanging)
        }
    }

    fn mode_policy(&self, mode: VardiffMode) -> ModePolicy {
        match mode {
            VardiffMode::Changing => ModePolicy {
                discount: self.changing_discount,
                upward_control_z: self.changing_upward_control_z,
                downward_control_z: self.changing_downward_control_z,
                smoothing: self.changing_control_ratio_smoothing,
                control_scale: self.changing_control_scale,
                poisson_update_z: self.changing_poisson_update_z,
                min_update_relative_change: self.changing_min_update_relative_change,
                max_update_relative_change: self.changing_max_update_relative_change,
                max_upward_relative_change: self.changing_max_upward_relative_change,
                max_downward_relative_change: self.changing_max_downward_relative_change,
                post_update_cooldown_secs: self.changing_post_update_cooldown_secs,
            },
            VardiffMode::Settled => ModePolicy {
                discount: self.settled_discount,
                upward_control_z: self.settled_upward_control_z,
                downward_control_z: self.settled_downward_control_z,
                smoothing: self.settled_control_ratio_smoothing,
                control_scale: self.settled_control_scale,
                poisson_update_z: self.settled_poisson_update_z,
                min_update_relative_change: self.settled_min_update_relative_change,
                max_update_relative_change: self.settled_max_update_relative_change,
                max_upward_relative_change: self.settled_max_upward_relative_change,
                max_downward_relative_change: self.settled_max_downward_relative_change,
                post_update_cooldown_secs: self.settled_post_update_cooldown_secs,
            },
        }
    }

    fn should_wait_for_more_evidence(&self, target_shares: f64, delta_time: u64) -> bool {
        target_shares < self.min_target_shares_for_update.max(0.0)
            && delta_time < self.max_update_interval_secs
    }

    fn reset_posterior_from_observation(&mut self, observed_shares: f64, target_shares: f64) {
        let target_shares = target_shares.max(1.0e-12);
        let observed_ratio = (observed_shares / target_shares).clamp(self.min_ratio, self.max_ratio);
        let old_ratio = self.ratio_mean();
        let blend = self.break_reset_blend.clamp(0.0, 1.0);
        let reset_ratio =
            (blend * observed_ratio + (1.0 - blend) * old_ratio).clamp(self.min_ratio, self.max_ratio);
        let strength = self.break_reset_prior_shares.max(1.0e-12);

        self.ratio_alpha = strength + observed_shares.max(0.0);
        self.ratio_beta = strength / reset_ratio + target_shares;
        self.clamp_posterior();
    }

    fn update_gamma_poisson(&mut self, observed_shares: f64, target_shares: f64, discount: f64) {
        let observed_shares = observed_shares.max(0.0);
        let target_shares = target_shares.max(1.0e-12);
        let discount = discount.clamp(0.0, 1.0);

        self.ratio_alpha = self.ratio_alpha * discount + observed_shares;
        self.ratio_beta = self.ratio_beta * discount + target_shares;
        self.clamp_posterior();
    }

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

    fn raw_control_ratio(&self, policy: ModePolicy) -> f64 {
        let mean = self.ratio_mean();
        let std = self.ratio_variance().sqrt();
        let z = if mean >= 1.0 {
            policy.upward_control_z
        } else {
            policy.downward_control_z
        };

        (mean - z.max(0.0) * std).clamp(self.min_ratio, self.max_ratio)
    }

    fn scale_control_ratio(&self, raw_ratio: f64, policy: ModePolicy) -> f64 {
        let scale = policy.control_scale.max(0.0);
        (1.0 + scale * (raw_ratio - 1.0)).clamp(self.min_ratio, self.max_ratio)
    }

    fn smooth_control_ratio(&mut self, proposed_ratio: f64, policy: ModePolicy) -> f64 {
        let smoothing = policy.smoothing.clamp(0.0, 0.999);
        let last = self.last_control_ratio.clamp(self.min_ratio, self.max_ratio);
        let smoothed =
            (smoothing * last + (1.0 - smoothing) * proposed_ratio).clamp(self.min_ratio, self.max_ratio);

        self.last_control_ratio = smoothed;
        smoothed
    }

    fn evidence_scaled_deadband(&self, target_shares: f64, policy: ModePolicy) -> f64 {
        let exposure = target_shares.max(1.0e-12);
        let poisson_threshold = policy.poisson_update_z.max(0.0) / exposure.sqrt();

        poisson_threshold
            .max(policy.min_update_relative_change.max(0.0))
            .min(policy.max_update_relative_change.max(policy.min_update_relative_change.max(0.0)))
    }

    fn apply_zero_share_policy(
        &self,
        proposed_hashrate: f64,
        current_hashrate: f64,
        observed_shares: f64,
        delta_time: u64,
    ) -> f64 {
        if !self.enable_zero_share_policy || observed_shares > 0.0 {
            return proposed_hashrate;
        }

        let decay = match delta_time {
            dt if dt <= 30 => self.zero_share_decay_le_30s,
            dt if dt < 60 => self.zero_share_decay_lt_60s,
            _ => self.zero_share_decay_ge_60s,
        }
        .max(1.0e-12);

        proposed_hashrate.min(current_hashrate / decay)
    }

    fn apply_extreme_upward_policy(
        &self,
        proposed_hashrate: f64,
        current_hashrate: f64,
        delta_time: u64,
    ) -> f64 {
        let current_hashrate = current_hashrate.max(1.0);
        let signed_relative_change = proposed_hashrate / current_hashrate - 1.0;

        if signed_relative_change <= self.extreme_upward_delta_threshold.max(0.0) {
            return proposed_hashrate;
        }

        let cap_multiplier = match delta_time {
            dt if dt <= 30 => self.extreme_upward_cap_le_30s,
            dt if dt < 60 => self.extreme_upward_cap_lt_60s,
            _ => self.extreme_upward_cap_ge_60s,
        }
        .max(1.0);

        proposed_hashrate.min(current_hashrate * cap_multiplier)
    }

    fn apply_update_policy(
        &self,
        proposed_hashrate: f64,
        current_hashrate: f64,
        observed_shares: f64,
        target_shares: f64,
        delta_time: u64,
        now: u64,
        policy: ModePolicy,
    ) -> Option<f64> {
        if now.saturating_sub(self.timestamp_of_last_accepted_update) < policy.post_update_cooldown_secs {
            return None;
        }

        let current_hashrate = current_hashrate.max(self.min_allowed_hashrate).max(1.0);
        let mut proposed_hashrate = if proposed_hashrate.is_finite() {
            proposed_hashrate.max(self.min_allowed_hashrate).max(1.0)
        } else {
            current_hashrate
        };

        proposed_hashrate =
            self.apply_zero_share_policy(proposed_hashrate, current_hashrate, observed_shares, delta_time);
        proposed_hashrate = self.apply_extreme_upward_policy(proposed_hashrate, current_hashrate, delta_time);

        let lower = (current_hashrate * (1.0 - policy.max_downward_relative_change.clamp(0.0, 1.0)))
            .max(self.min_allowed_hashrate)
            .max(1.0);
        let upper = (current_hashrate * (1.0 + policy.max_upward_relative_change.max(0.0)))
            .max(self.min_allowed_hashrate)
            .max(1.0);

        let clamped_hashrate = proposed_hashrate.clamp(lower, upper);
        let signed_relative_change = clamped_hashrate / current_hashrate - 1.0;
        let relative_change = signed_relative_change.abs();
        let deadband = self.evidence_scaled_deadband(target_shares, policy);

        debug!(
            target: "vardiff",
            "Gamma-Poisson mode update policy:\n            - Proposed hashrate after special policies: {:.2} H/s\n            - Current hashrate: {:.2} H/s\n            - Clamped hashrate: {:.2} H/s\n            - Target shares: {:.4}\n            - Observed shares: {:.4}\n            - Relative change: {:.6}\n            - Evidence-scaled deadband: {:.6}\n            - Cooldown seconds: {}",
            proposed_hashrate,
            current_hashrate,
            clamped_hashrate,
            target_shares,
            observed_shares,
            relative_change,
            deadband,
            policy.post_update_cooldown_secs,
        );

        if relative_change >= deadband {
            Some(clamped_hashrate)
        } else {
            None
        }
    }

    fn rebase_posterior_after_update(&mut self, applied_ratio: f64) {
        let applied_ratio = applied_ratio.clamp(self.min_ratio, self.max_ratio).max(1.0e-12);

        self.ratio_beta *= applied_ratio;
        self.last_control_ratio = (self.last_control_ratio / applied_ratio).clamp(self.min_ratio, self.max_ratio);
        self.clamp_posterior();
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

    fn set_timestamp_of_last_update(&mut self, timestamp_of_last_update: u64) {
        self.timestamp_of_last_update = timestamp_of_last_update;
    }

    fn increment_shares_since_last_update(&mut self) {
        self.shares_since_last_update = self.shares_since_last_update.saturating_add(1);
    }

    fn add_shares(&mut self, count: u32) {
        self.shares_since_last_update = self.shares_since_last_update.saturating_add(count);
    }

    fn reset_counter(&mut self) -> Result<(), VardiffError> {
        let timestamp_secs = self.clock.now_secs();
        self.set_timestamp_of_last_update(timestamp_secs);
        self.set_shares_since_last_update(0);
        Ok(())
    }

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

        let current_hashrate = (hashrate as f64).max(self.min_allowed_hashrate).max(1.0);
        let elapsed_minutes = delta_time as f64 / 60.0;
        let target_shares = (shares_per_minute as f64).max(0.0) * elapsed_minutes;
        let observed_shares = self.shares_since_last_update as f64;

        if target_shares <= 0.0 {
            self.reset_counter()?;
            return Ok(None);
        }

        if self.should_wait_for_more_evidence(target_shares, delta_time) {
            debug!(
                target: "vardiff",
                "Gamma-Poisson vardiff waiting for more evidence:\n                - Elapsed time: {}s\n                - Shares since last update: {}\n                - Target shares: {:.4}\n                - Min target shares for update: {:.4}\n                - Max update interval seconds: {}",
                delta_time,
                self.shares_since_last_update,
                target_shares,
                self.min_target_shares_for_update,
                self.max_update_interval_secs,
            );

            return Ok(None);
        }

        let pre_update_predictive_z = self
            .predictive_z_score(observed_shares, target_shares)
            .unwrap_or(0.0);
        let break_detected = self.is_break(observed_shares, target_shares, now);
        let (mode, mode_reason) = self.classify_mode(
            delta_time,
            target_shares,
            pre_update_predictive_z,
            break_detected,
            now,
        );
        let policy = self.mode_policy(mode);

        if break_detected {
            self.reset_posterior_from_observation(observed_shares, target_shares);
            self.timestamp_of_last_break = now;
        } else {
            self.update_gamma_poisson(observed_shares, target_shares, policy.discount);
        }

        let posterior_mean_ratio = self.ratio_mean();
        let posterior_std_ratio = self.ratio_variance().sqrt();
        let raw_control_ratio = self.raw_control_ratio(policy);
        let scaled_control_ratio = self.scale_control_ratio(raw_control_ratio, policy);
        let control_ratio = self.smooth_control_ratio(scaled_control_ratio, policy);
        let proposed_hashrate = current_hashrate * control_ratio;

        debug!(
            target: "vardiff",
            "Gamma-Poisson vardiff check:\n            - Mode: {:?}\n            - Mode reason: {:?}\n            - Elapsed time: {}s\n            - Shares since last update: {}\n            - Target shares: {:.4}\n            - Current hashrate: {:.2} H/s\n            - Pre-update predictive z: {:.6}\n            - Break detected: {}\n            - Posterior mean ratio: {:.8}\n            - Posterior std ratio: {:.8}\n            - Raw control ratio: {:.8}\n            - Scaled control ratio: {:.8}\n            - Smoothed control ratio: {:.8}\n            - Control scale: {:.6}\n            - Proposed hashrate: {:.2} H/s\n            - Current miner target: {:?}",
            mode,
            mode_reason,
            delta_time,
            self.shares_since_last_update,
            target_shares,
            current_hashrate,
            pre_update_predictive_z,
            break_detected,
            posterior_mean_ratio,
            posterior_std_ratio,
            raw_control_ratio,
            scaled_control_ratio,
            control_ratio,
            policy.control_scale,
            proposed_hashrate,
            _target,
        );

        let maybe_new_hashrate = self.apply_update_policy(
            proposed_hashrate,
            current_hashrate,
            observed_shares,
            target_shares,
            delta_time,
            now,
            policy,
        );

        match maybe_new_hashrate {
            Some(new_hashrate) => {
                self.reset_counter()?;

                let applied_ratio = (new_hashrate / current_hashrate).clamp(self.min_ratio, self.max_ratio);
                self.timestamp_of_last_accepted_update = now;
                self.rebase_posterior_after_update(applied_ratio);

                debug!(
                    target: "vardiff",
                    "Gamma-Poisson vardiff update accepted:\n                    - Mode: {:?}\n                    - Mode reason: {:?}\n                    - Previous hashrate: {:.2} H/s\n                    - New hashrate: {:.2} H/s\n                    - Applied ratio: {:.8}\n                    - Delta: {:.2}%\n                    - Rebased posterior mean ratio: {:.8}\n                    - Rebased posterior std ratio: {:.8}",
                    mode,
                    mode_reason,
                    current_hashrate,
                    new_hashrate,
                    applied_ratio,
                    ((new_hashrate - current_hashrate).abs() / current_hashrate) * 100.0,
                    self.ratio_mean(),
                    self.ratio_variance().sqrt(),
                );

                Ok(Some(new_hashrate as f32))
            }
            None => {
                if self.reset_counter_on_no_update {
                    self.reset_counter()?;
                }

                Ok(None)
            }
        }
    }
}
