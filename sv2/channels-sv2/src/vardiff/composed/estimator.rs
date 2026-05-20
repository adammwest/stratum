//! Estimator — Stage 1 of the three-stage vardiff pipeline.
//!
//! The Estimator carries observation history between ticks and exposes a
//! snapshot of its current belief about the miner's hashrate. It is the
//! piece of the algorithm that lives in statistical estimation theory:
//! design pressure is responsiveness (small effective window, high
//! variance) vs stability (large effective window, low variance but
//! potentially stale data).
//!
//! See `sim/docs/DESIGN.md` for the pipeline architecture.

use crate::target::hash_rate_from_target;
use bitcoin::Target;
use std::collections::VecDeque;
use std::fmt::Debug;

/// Per-tick context an [`Estimator`] needs to construct a snapshot.
///
/// Assembled by the `Composed` adapter from the inputs to `try_vardiff`
/// and passed to [`Estimator::snapshot`]. Carries the *current* target
/// and hashrate so the estimator can compute `H̃` (its own belief about
/// hashrate) from the realized share rate — e.g., via the
/// `hash_rate_from_target` conversion that `VardiffState` uses.
#[derive(Debug, Clone)]
pub struct EstimatorContext<'a> {
    pub current_hashrate: f32,
    pub current_target: &'a Target,
    pub shares_per_minute: f32,
}

/// Uncertainty quantification from the Estimator. Enables the Boundary
/// to adapt its threshold based on how confident the estimate is, rather
/// than inferring confidence from proxy signals (dt_secs, SPM).
#[derive(Debug, Clone, Copy)]
pub struct Uncertainty {
    /// Standard deviation of the ratio estimate (h_estimate / current_h).
    /// Dimensionless. Smaller = more confident.
    pub ratio_std: f64,
    /// Effective independent sample count informing this estimate.
    /// Analogous to 1/(1-discount) for Bayesian, tau/tick for EWMA.
    pub effective_n: f64,
}

/// The Estimator's belief at a tick — the primary handoff struct from
/// Stage 1 to all downstream stages. Carries both a point estimate and
/// optional uncertainty quantification.
#[derive(Debug, Clone)]
pub struct EstimatorSnapshot {
    /// The estimator's belief about the miner's true hashrate (H/s).
    pub h_estimate: f32,
    /// Realized share rate over the window, in shares per minute.
    /// Zero if `dt_secs == 0` or `n_shares == 0`.
    pub realized_share_per_min: f64,
    /// Raw share count observed since the last reset.
    pub n_shares: u32,
    /// Window duration in seconds (`now - timestamp_of_last_update`).
    pub dt_secs: u64,
    /// Optional uncertainty quantification. `None` for estimators that
    /// don't track uncertainty (CumulativeCounter, EWMA, SlidingWindow).
    /// `Some` for estimators that maintain a probabilistic belief
    /// (BayesianEstimator). Downstream stages (especially Boundary) can
    /// use this to auto-calibrate their thresholds.
    pub uncertainty: Option<Uncertainty>,
}

/// Axis 1: how the algorithm accumulates observation history and exposes
/// a snapshot of its current belief about the miner's hashrate.
///
/// **Theory**: statistical estimation (bias, variance, MSE).
///
/// **Design pressure**: responsiveness (small effective window, high
/// variance) vs stability (large effective window, low variance but
/// potentially stale data).
///
/// Implementations: [`CumulativeCounter`] (classic),
/// [`EwmaEstimator`] (EWMA / FullRemedy),
/// [`SlidingWindowEstimator`] (Sliding-Window).
pub trait Estimator: Debug + Send + Sync {
    /// Record `n_shares` new arrivals since the last fire.
    fn observe(&mut self, n_shares: u32);

    /// Notification that the algorithm fired and changed the target.
    /// `new_hashrate` is the target being set; `old_hashrate` is what
    /// it was before the fire.
    ///
    /// Estimators choose how to respond:
    /// - Absolute-rate estimators (CumulativeCounter, EWMA, SlidingWindow):
    ///   full reset — old data is invalid under the new target.
    /// - Ratio-based estimators (Bayesian): rescale posterior to center
    ///   on ratio=1.0 relative to the new target, preserving accumulated
    ///   confidence.
    fn on_fire(&mut self, new_hashrate: f32, old_hashrate: f32);

    /// Compute a snapshot of the estimator's current belief.
    fn snapshot(&self, dt_secs: u64, ctx: &EstimatorContext) -> EstimatorSnapshot;

    /// Return the raw share count, for the `Vardiff::shares_since_last_update`
    /// trait method. Estimators that don't carry an integer count (e.g.,
    /// EWMA) may return a derived value such as `ceil(weighted_sum)`.
    fn shares_count(&self) -> u32;
}

/// The classic estimator: a saturating cumulative share counter, with
/// `H̃` computed at snapshot time via `hash_rate_from_target` (with the
/// linear-scaling fallback used by `VardiffState`).
///
/// This is one of the four trait impls that, when composed together,
/// reproduce `VardiffState` fire-for-fire.
#[derive(Debug, Default)]
pub struct CumulativeCounter {
    shares: u32,
}

impl CumulativeCounter {
    pub fn new() -> Self {
        Self { shares: 0 }
    }
}

impl Estimator for CumulativeCounter {
    fn observe(&mut self, n_shares: u32) {
        self.shares = self.shares.saturating_add(n_shares);
    }

    fn on_fire(&mut self, _new_hashrate: f32, _old_hashrate: f32) {
        self.shares = 0;
    }

    fn shares_count(&self) -> u32 {
        self.shares
    }

    fn snapshot(&self, dt_secs: u64, ctx: &EstimatorContext) -> EstimatorSnapshot {
        // dt_secs > 15 is guarded by the Composed adapter before snapshot()
        // is called, but the divide-by-zero defense is cheap and makes this
        // impl independently testable.
        let realized_share_per_min = if dt_secs == 0 {
            0.0
        } else {
            self.shares as f64 / (dt_secs as f64 / 60.0)
        };

        // Mirror VardiffState::try_vardiff: prefer hash_rate_from_target
        // for exact U256-arithmetic conversion; fall back to linear scaling
        // (mathematically equivalent when target was set for current_hashrate
        // at the configured share rate) on conversion error.
        let h_estimate = match hash_rate_from_target(
            ctx.current_target.to_le_bytes().into(),
            realized_share_per_min,
        ) {
            Ok(h) => h as f32,
            Err(_) => ctx.current_hashrate * realized_share_per_min as f32 / ctx.shares_per_minute,
        };

        EstimatorSnapshot {
            h_estimate,
            realized_share_per_min,
            n_shares: self.shares,
            dt_secs,
            uncertainty: None,
        }
    }
}

/// Exponentially-weighted moving average estimator. Tracks a smoothed
/// per-tick share count: on each `observe(n)` call (one per tick), the
/// internal rate estimate decays toward `n` with continuous-time
/// constant `tau_secs`:
///
/// ```text
/// α = exp(-tick_secs / tau_secs)
/// rate_per_tick ← α × rate_per_tick + (1 − α) × n
/// ```
///
/// ## Interpreting `tau_secs`
///
/// `tau_secs` is the continuous-time exponential time constant: an
/// impulse decays to `1/e ≈ 36.8%` of its initial value after
/// `tau_secs` seconds of elapsed time. The *half-life* (decay to 50%)
/// is `tau_secs × ln(2) ≈ 0.69 × tau_secs`.
///
/// Equivalent per-tick interpretation: at `tick_secs = tau_secs` (the
/// default, `60` and `60`), one tick decays the previous estimate by
/// `1 − e⁻¹ ≈ 63%` and weights the new observation by the same — the
/// EWMA has heavy short-term memory. To get the "lightly smoothed"
/// regime (`α ≈ 0.9`, fade over ~10 ticks) use `tau_secs ≈ 10 ×
/// tick_secs` or more.
///
/// ## Why one-observe-per-tick
///
/// The `Estimator` trait's `observe(n_shares: u32)` doesn't carry the
/// time interval between calls — so this impl assumes each `observe`
/// represents exactly one tick of duration `tick_secs`. The Composed
/// adapter's trial driver calls `add_shares` once per tick, so this
/// assumption holds for all framework-driven scenarios. Custom callers
/// driving observe directly should match the tick cadence to keep the
/// EWMA's effective time constant accurate; otherwise the discrete
/// decay rate diverges from `exp(-real_dt / tau_secs)`.
///
/// `realized_share_per_min` returned from `snapshot` is the rate
/// estimate converted to shares-per-minute: `rate_per_tick × (60 /
/// tick_secs)`.
#[derive(Debug)]
pub struct EwmaEstimator {
    /// Time constant in seconds. Common values: 30, 60, 120, 300.
    pub tau_secs: u64,
    /// Tick interval in seconds. Must match the trial driver's tick
    /// cadence for the EWMA's effective τ to be correct.
    pub tick_secs: u64,
    /// Current rate estimate in shares-per-tick.
    rate_per_tick: f64,
    /// Number of observations since the last reset. The first
    /// observation initializes `rate_per_tick` directly (no decay yet).
    n_observations: u32,
}

impl EwmaEstimator {
    /// Constructs an EWMA with the given time constant. `tick_secs`
    /// defaults to 60 (matching the trial driver's default tick
    /// interval).
    pub fn new(tau_secs: u64) -> Self {
        Self {
            tau_secs,
            tick_secs: 60,
            rate_per_tick: 0.0,
            n_observations: 0,
        }
    }

    /// Decay factor per tick: `exp(-tick_secs / tau_secs)`.
    /// Computed each call rather than cached so changes to `tau_secs`
    /// or `tick_secs` take effect immediately. The cost is negligible.
    fn alpha(&self) -> f64 {
        (-(self.tick_secs as f64) / (self.tau_secs as f64)).exp()
    }
}

impl Estimator for EwmaEstimator {
    fn observe(&mut self, n_shares: u32) {
        let n = n_shares as f64;
        if self.n_observations == 0 {
            self.rate_per_tick = n;
        } else {
            let alpha = self.alpha();
            self.rate_per_tick = alpha * self.rate_per_tick + (1.0 - alpha) * n;
        }
        self.n_observations = self.n_observations.saturating_add(1);
    }

    fn on_fire(&mut self, new_hashrate: f32, old_hashrate: f32) {
        if old_hashrate <= 0.0 || new_hashrate <= 0.0 {
            // Can't compute ratio — full reset
            self.rate_per_tick = 0.0;
            self.n_observations = 0;
            return;
        }

        // The retarget changes the reference point. If the target moved by
        // factor R = new/old, then future shares at the same true hashrate
        // will arrive at rate = old_rate / R (fewer shares at higher difficulty).
        // Scale the EWMA's smoothed rate by this factor so it starts from a
        // reasonable baseline rather than zero.
        //
        // For PartialRetarget (small moves), R ≈ 1.0 and the scaled rate is
        // close to the current smoothed rate — preserving most information.
        // For FullRetarget (jumping to estimate), R can be large and the
        // scaled rate may be inaccurate — but even an approximate starting
        // point is better than zero (which forces the EWMA to re-learn from
        // scratch, causing a lag spike visible as settling delay).
        let ratio = new_hashrate as f64 / old_hashrate as f64;
        if ratio > 0.0 && ratio.is_finite() {
            self.rate_per_tick /= ratio;
            // Keep observation count — the estimate is still informed by
            // prior data, just rescaled. Clearing n_observations would cause
            // the next observe() to skip the exponential blend.
        } else {
            self.rate_per_tick = 0.0;
            self.n_observations = 0;
        }
    }

    fn shares_count(&self) -> u32 {
        self.rate_per_tick.round().max(0.0) as u32
    }

    fn snapshot(&self, dt_secs: u64, ctx: &EstimatorContext) -> EstimatorSnapshot {
        let realized_share_per_min = self.rate_per_tick * (60.0 / self.tick_secs as f64);
        // Same conversion path as CumulativeCounter: prefer
        // hash_rate_from_target, fall back to linear scaling.
        let h_estimate = match hash_rate_from_target(
            ctx.current_target.to_le_bytes().into(),
            realized_share_per_min,
        ) {
            Ok(h) => h as f32,
            Err(_) => ctx.current_hashrate * realized_share_per_min as f32 / ctx.shares_per_minute,
        };
        EstimatorSnapshot {
            h_estimate,
            realized_share_per_min,
            n_shares: self.shares_count(),
            dt_secs,
            uncertainty: None,
        }
    }
}

/// Sliding-window estimator: holds the last `n_ticks` per-tick share
/// observations in a ring buffer and computes `H̃` from the buffer
/// average.
///
/// ## Design simplification
///
/// Like `EwmaEstimator`, this impl assumes one `observe(n_shares)`
/// call per tick of fixed duration `tick_secs`. The Composed adapter's
/// trial driver satisfies this. Custom callers should match the tick
/// cadence to keep the effective window length correct.
///
/// ## Reset behavior
///
/// `reset()` clears the buffer. The Composed adapter calls `reset()`
/// on every fire, so during the cold-start Phase 1 ramp (where the
/// algorithm fires every tick) the buffer holds only a single
/// observation — the estimator behaves identically to
/// `CumulativeCounter` during Phase 1. The sliding-window benefit
/// shows up only post-Phase-1 when the buffer fills with multiple
/// at-truth ticks of data.
///
/// ## When to use
///
/// Pair with [`crate::composed::FullRetargetNoClamp`] for the
/// canonical `SlidingWindow` composition (see `DESIGN.md` §
/// "Algorithm registry"), or with [`crate::composed::PartialRetarget`]
/// for a more damped variant.
/// Always with [`crate::composed::PoissonCI`] boundary so the
/// rate-aware threshold complements the smoothing.
#[derive(Debug)]
pub struct SlidingWindowEstimator {
    /// Maximum number of per-tick observations to retain. The buffer
    /// holds up to this many entries; older entries are dropped on
    /// overflow.
    pub n_ticks: usize,
    /// Tick interval in seconds. Used to convert buffer count → time
    /// when computing realized share rate.
    pub tick_secs: u64,
    /// Ring buffer of per-tick share counts.
    buffer: VecDeque<u32>,
}

impl SlidingWindowEstimator {
    /// Constructs a sliding-window estimator with the given window
    /// length in ticks. Default `tick_secs = 60` matches the trial
    /// driver's tick cadence.
    pub fn new(n_ticks: usize) -> Self {
        Self {
            n_ticks,
            tick_secs: 60,
            buffer: VecDeque::with_capacity(n_ticks),
        }
    }
}

impl Estimator for SlidingWindowEstimator {
    fn observe(&mut self, n_shares: u32) {
        if self.buffer.len() >= self.n_ticks {
            self.buffer.pop_front();
        }
        self.buffer.push_back(n_shares);
    }

    fn on_fire(&mut self, _new_hashrate: f32, _old_hashrate: f32) {
        self.buffer.clear();
    }

    fn shares_count(&self) -> u32 {
        // Saturating sum — at typical n_ticks (≤ 200) and per-tick
        // counts (≤ a few thousand under realistic Poisson) overflow
        // is unreachable. The saturate is defense in depth.
        self.buffer
            .iter()
            .fold(0u32, |acc, &n| acc.saturating_add(n))
    }

    fn snapshot(&self, dt_secs: u64, ctx: &EstimatorContext) -> EstimatorSnapshot {
        let n_actual = self.buffer.len();
        let total_shares: u64 = self.buffer.iter().map(|&x| x as u64).sum();
        // The effective time window covered by the buffer.
        let window_secs = (n_actual as u64) * self.tick_secs;
        let realized_share_per_min = if window_secs > 0 {
            total_shares as f64 / (window_secs as f64 / 60.0)
        } else {
            0.0
        };

        // Same conversion path as CumulativeCounter and EwmaEstimator.
        let h_estimate = match hash_rate_from_target(
            ctx.current_target.to_le_bytes().into(),
            realized_share_per_min,
        ) {
            Ok(h) => h as f32,
            Err(_) => ctx.current_hashrate * realized_share_per_min as f32 / ctx.shares_per_minute,
        };

        EstimatorSnapshot {
            h_estimate,
            realized_share_per_min,
            // Report the buffer total as n_shares — that's what the
            // estimator's snapshot represents.
            n_shares: total_shares.min(u32::MAX as u64) as u32,
            dt_secs,
            uncertainty: None,
        }
    }
}

/// Bayesian Gamma-Poisson estimator with exponential discounting.
///
/// Models the per-tick share rate as a Poisson process with unknown rate λ,
/// and maintains a Gamma posterior over a normalized ratio:
///
/// ```text
/// ratio = true_rate / expected_rate
/// ratio ~ Gamma(alpha, beta)
/// ```
///
/// On each `observe(n_shares)` call (one per tick), the estimator:
/// 1. Discounts the prior: `alpha *= discount`, `beta *= discount`
/// 2. Updates with the observation: `alpha += n_shares`, `beta += 1.0`
///
/// The `+= 1.0` for beta represents "one tick's worth of exposure" — the
/// posterior tracks *shares per tick* normalized by the expected count at
/// snapshot time.
///
/// ## Why Gamma-Poisson?
///
/// This is the conjugate Bayesian model for Poisson data: Gamma prior +
/// Poisson likelihood → Gamma posterior. The update is exact (no approximation)
/// and closed-form (no sampling required). The posterior automatically
/// provides uncertainty quantification (variance shrinks as data accumulates).
///
/// ## Discount factor
///
/// `discount ∈ (0, 1]` controls memory decay per tick:
/// - 0.99: effective memory ~100 ticks (slow, steady)
/// - 0.95: effective memory ~20 ticks (moderate)
/// - 0.90: effective memory ~10 ticks (fast, jittery)
///
/// The effective memory length is approximately `1 / (1 - discount)` ticks.
///
/// ## Relationship to EWMA
///
/// Both have exponential memory. Key difference: EWMA tracks a point estimate
/// of shares-per-tick. Bayesian tracks the full Gamma posterior, which gives
/// both a point estimate (posterior mean) AND an uncertainty measure (posterior
/// variance). This uncertainty can inform downstream decisions (e.g., the
/// boundary can require higher confidence before firing when uncertainty is high).
///
/// ## Break detection / mode-switching
///
/// This estimator does NOT include break detection. That responsibility lives
/// in the Boundary axis. Keeping estimation pure lets us compose the Bayesian
/// estimator with different boundary strategies (PoissonCI, posterior-predictive,
/// etc.) independently.
#[derive(Debug, Clone)]
pub struct BayesianEstimator {
    /// Gamma posterior shape parameter. Roughly "total discounted share count."
    pub alpha: f64,
    /// Gamma posterior rate parameter. Roughly "total discounted tick count."
    pub beta: f64,
    /// Exponential discount factor applied before each tick update. Range (0, 1].
    pub discount: f64,
    /// Initial pseudo-observation strength.
    pub prior_shares: f64,
    /// Tick interval in seconds.
    pub tick_secs: u64,
    /// Total shares observed since last reset (for reporting).
    shares_since_reset: u32,
    /// Number of ticks since last reset.
    n_ticks_since_reset: u32,
}

impl BayesianEstimator {
    /// Constructs a Bayesian estimator.
    ///
    /// `discount` in (0, 1]: exponential forgetting per tick.
    /// `prior_shares`: initial pseudo-count strength. Higher = more
    /// confident initial estimate. Typically 2.0–10.0.
    pub fn new(discount: f64, prior_shares: f64) -> Self {
        let prior_shares = prior_shares.max(0.01);
        Self {
            alpha: prior_shares,
            beta: prior_shares, // ratio_mean = alpha/beta = 1.0 initially
            discount: discount.clamp(0.001, 1.0),
            prior_shares,
            tick_secs: 60,
            shares_since_reset: 0,
            n_ticks_since_reset: 0,
        }
    }

    /// Posterior mean of the rate ratio.
    pub fn ratio_mean(&self) -> f64 {
        if self.beta <= 0.0 {
            return 1.0;
        }
        (self.alpha / self.beta).clamp(1.0e-6, 1.0e6)
    }

    /// Posterior variance of the rate ratio.
    pub fn ratio_variance(&self) -> f64 {
        if self.beta <= 0.0 {
            return 1.0e6;
        }
        (self.alpha / (self.beta * self.beta)).max(0.0)
    }

    /// Posterior standard deviation of the rate ratio.
    pub fn ratio_std(&self) -> f64 {
        self.ratio_variance().sqrt()
    }
}

impl Estimator for BayesianEstimator {
    fn observe(&mut self, n_shares: u32) {
        // Each observe = one tick of data.
        // We accumulate shares and tick count; the actual Bayesian update
        // happens at snapshot time when we have access to SPM (the expected
        // rate). This deferred approach is needed because observe() doesn't
        // receive context about the expected share rate.
        self.shares_since_reset = self.shares_since_reset.saturating_add(n_shares);
        self.n_ticks_since_reset = self.n_ticks_since_reset.saturating_add(1);
    }

    fn on_fire(&mut self, _new_hashrate: f32, _old_hashrate: f32) {
        // After a fire, the new target ≈ old_target × ratio_mean, so the
        // true ratio relative to the new target should be ≈ 1.0.
        //
        // Preserve posterior mass proportional to accumulated data — this
        // is the key difference from full-reset estimators. We shift the
        // center to ratio=1.0 but keep confidence proportional to
        // min(accumulated_info, cap).
        //
        // The cap prevents over-confidence: even if we've seen 100 ticks,
        // after a target change we should be somewhat uncertain about
        // whether the new target is exactly right.
        let info_accumulated = self.alpha.min(self.beta);
        let preserved = info_accumulated.min(self.prior_shares * 4.0).max(self.prior_shares);
        self.alpha = preserved;
        self.beta = preserved; // ratio_mean = 1.0

        self.shares_since_reset = 0;
        self.n_ticks_since_reset = 0;
    }

    fn shares_count(&self) -> u32 {
        self.shares_since_reset
    }

    fn snapshot(&self, dt_secs: u64, ctx: &EstimatorContext) -> EstimatorSnapshot {
        // Compute the posterior after incorporating ALL ticks since last reset.
        // Each tick contributes (observed_shares_that_tick, expected_shares_per_tick)
        // to the Gamma-Poisson update. Since we only have the total, we treat
        // the whole window as one observation:
        //   alpha_new = discount^n * alpha_0 + total_observed
        //   beta_new = discount^n * beta_0 + total_expected
        //
        // This is mathematically equivalent to per-tick updates when the
        // observation is concentrated in one window (which it is, since
        // Composed resets on fire).
        let n_ticks = self.n_ticks_since_reset as f64;
        let total_observed = self.shares_since_reset as f64;

        // Expected shares for the entire window at current difficulty
        let expected_per_tick = (ctx.shares_per_minute as f64) * (self.tick_secs as f64 / 60.0);
        let total_expected = expected_per_tick * n_ticks;

        // Discount factor over the full window
        let discount_factor = self.discount.powf(n_ticks);

        let alpha = discount_factor * self.alpha + total_observed;
        let beta = discount_factor * self.beta + total_expected.max(1.0e-12);

        // Posterior mean ratio
        let ratio_mean = if beta > 0.0 {
            (alpha / beta).clamp(1.0e-6, 1.0e6)
        } else {
            1.0
        };

        // Convert ratio to hashrate estimate
        let h_estimate = (ctx.current_hashrate as f64 * ratio_mean) as f32;

        let realized_share_per_min = if dt_secs > 0 {
            total_observed / (dt_secs as f64 / 60.0)
        } else {
            0.0
        };

        // Posterior standard deviation: sqrt(alpha / beta^2)
        let ratio_std = if beta > 0.0 {
            (alpha / (beta * beta)).sqrt()
        } else {
            1.0
        };
        let effective_n = beta; // beta ≈ total discounted exposure

        EstimatorSnapshot {
            h_estimate,
            realized_share_per_min,
            n_shares: self.shares_since_reset,
            dt_secs,
            uncertainty: Some(Uncertainty {
                ratio_std,
                effective_n,
            }),
        }
    }
}

/// Kalman filter estimator for hashrate tracking.
///
/// Models the hashrate ratio as a scalar state evolving with process noise:
///
/// ```text
/// State model:     ratio(t+1) = ratio(t) + w,   w ~ N(0, Q)
/// Observation:     shares(t) ~ Poisson(SPM * ratio(t) * tick/60)
///                  ≈ N(SPM * ratio * tick/60, SPM * ratio * tick/60)  [Gaussian approx]
/// ```
///
/// The Kalman filter produces both a point estimate (state mean) and
/// uncertainty (state covariance P) at each tick — exactly what the
/// pipeline's `Uncertainty` handoff requires.
///
/// ## Process noise Q
///
/// Q controls how much the filter expects hashrate to change between ticks:
/// - Small Q (e.g., 0.0001): assumes hashrate is nearly constant — smooth
///   estimate, slow to react to real changes
/// - Large Q (e.g., 0.01): assumes hashrate can change rapidly — responsive
///   but noisy
///
/// Analogous to EWMA's τ: `Q ≈ 1/τ²` roughly (larger Q = shorter memory).
///
/// ## Gaussian approximation
///
/// Valid for SPM ≥ 12 (expected shares ≥ 12 per tick). At SPM=6 the Poisson
/// is skewed; the filter still works but uncertainty estimates are less
/// accurate. The approximation `Var[shares] ≈ E[shares]` is used for the
/// measurement noise R.
///
/// ## on_fire behavior
///
/// Rescales the state estimate and covariance by the retarget ratio. Unlike
/// EWMA (which rescales its rate), the Kalman rescales the ratio state
/// itself: `new_ratio = old_ratio / retarget_factor` and P is preserved.
/// This means the filter retains confidence across fires.
#[derive(Debug, Clone)]
pub struct KalmanEstimator {
    /// State estimate: ratio = true_h / target_h.
    ratio_state: f64,
    /// State covariance (uncertainty squared).
    p: f64,
    /// Process noise variance per tick. Controls responsiveness.
    pub q: f64,
    /// Tick interval in seconds.
    pub tick_secs: u64,
    /// Shares accumulated since last fire (for reporting).
    shares_since_fire: u32,
    /// Ticks since last fire.
    n_ticks: u32,
}

impl KalmanEstimator {
    /// Constructs a Kalman estimator with the given process noise.
    ///
    /// `q` is process noise variance per tick. Good values:
    /// - 0.001: moderate responsiveness (similar to EWMA τ=120s)
    /// - 0.005: fast response (similar to EWMA τ=60s)
    /// - 0.0002: very smooth (similar to EWMA τ=300s)
    pub fn new(q: f64) -> Self {
        Self {
            ratio_state: 1.0,
            p: 1.0, // Start uncertain
            q: q.max(1.0e-12),
            tick_secs: 60,
            shares_since_fire: 0,
            n_ticks: 0,
        }
    }
}

impl Estimator for KalmanEstimator {
    fn observe(&mut self, n_shares: u32) {
        self.shares_since_fire = self.shares_since_fire.saturating_add(n_shares);
        self.n_ticks = self.n_ticks.saturating_add(1);
    }

    fn on_fire(&mut self, new_hashrate: f32, old_hashrate: f32) {
        if old_hashrate > 0.0 && new_hashrate > 0.0 {
            let retarget_ratio = new_hashrate as f64 / old_hashrate as f64;
            // After retarget, the new ratio relative to new target should be
            // old_ratio / retarget_ratio (since new_target = old_target * retarget_ratio)
            self.ratio_state /= retarget_ratio;
            // Covariance P is preserved — our confidence about the ratio
            // doesn't change just because the reference point shifted.
        } else {
            self.ratio_state = 1.0;
            self.p = 1.0;
        }
        self.shares_since_fire = 0;
        self.n_ticks = 0;
    }

    fn shares_count(&self) -> u32 {
        self.shares_since_fire
    }

    fn snapshot(&self, dt_secs: u64, ctx: &EstimatorContext) -> EstimatorSnapshot {
        // Run the Kalman predict+update for all accumulated ticks.
        // We do this at snapshot time (like Bayesian) because observe()
        // doesn't have SPM context.
        let mut ratio = self.ratio_state;
        let mut p = self.p;

        if self.n_ticks > 0 && dt_secs > 0 {
            let expected_shares_per_tick =
                (ctx.shares_per_minute as f64) * (self.tick_secs as f64 / 60.0);

            // Per-tick Kalman update (batch: apply n_ticks of predict+update)
            // For efficiency, we do the batch as if all shares arrived uniformly
            let shares_per_tick = self.shares_since_fire as f64 / self.n_ticks as f64;

            for _ in 0..self.n_ticks {
                // Predict step: state unchanged, covariance grows
                p += self.q;

                // Update step (Gaussian-approximated Poisson observation)
                // Measurement: z = shares_this_tick
                // Expected measurement: H = expected_shares_per_tick * ratio
                // Measurement noise: R = expected_shares_per_tick * ratio (Poisson variance)
                let h_matrix = expected_shares_per_tick; // dz/d_ratio
                let predicted_shares = expected_shares_per_tick * ratio;
                let r = predicted_shares.max(1.0); // measurement noise (Poisson variance)

                // Innovation
                let innovation = shares_per_tick - predicted_shares;

                // Innovation covariance: S = H*P*H' + R
                let s = h_matrix * p * h_matrix + r;

                if s > 0.0 {
                    // Kalman gain: K = P*H' / S
                    let k = (p * h_matrix) / s;

                    // State update
                    ratio += k * innovation;

                    // Covariance update: P = (1 - K*H) * P
                    p = (1.0 - k * h_matrix) * p;
                }

                // Clamp ratio to reasonable bounds
                ratio = ratio.clamp(1.0e-6, 1.0e6);
                p = p.max(1.0e-12);
            }
        }

        let h_estimate = (ctx.current_hashrate as f64 * ratio) as f32;
        let ratio_std = p.sqrt();
        let effective_n = if p > 0.0 { 1.0 / p } else { 1.0 };

        let realized_share_per_min = if dt_secs > 0 {
            self.shares_since_fire as f64 / (dt_secs as f64 / 60.0)
        } else {
            0.0
        };

        EstimatorSnapshot {
            h_estimate,
            realized_share_per_min,
            n_shares: self.shares_since_fire,
            dt_secs,
            uncertainty: Some(Uncertainty {
                ratio_std,
                effective_n,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_increments_share_count() {
        let mut e = CumulativeCounter::new();
        e.observe(10);
        e.observe(5);
        assert_eq!(e.shares_count(), 15);
    }

    #[test]
    fn reset_zeroes_share_count() {
        let mut e = CumulativeCounter::new();
        e.observe(10);
        e.on_fire(0.0, 0.0);
        assert_eq!(e.shares_count(), 0);
    }

    #[test]
    fn observe_saturates_at_u32_max() {
        let mut e = CumulativeCounter::new();
        e.observe(u32::MAX);
        e.observe(1);
        assert_eq!(e.shares_count(), u32::MAX);
    }

    #[test]
    fn snapshot_realized_rate_is_shares_per_minute() {
        let mut e = CumulativeCounter::new();
        e.observe(12); // 12 shares over 60s = 12 spm
        let target = Target::MAX;
        let ctx = EstimatorContext {
            current_hashrate: 1.0e15,
            current_target: &target,
            shares_per_minute: 12.0,
        };
        let snap = e.snapshot(60, &ctx);
        assert!((snap.realized_share_per_min - 12.0).abs() < 1e-9);
        assert_eq!(snap.n_shares, 12);
        assert_eq!(snap.dt_secs, 60);
    }

    #[test]
    fn snapshot_zero_dt_yields_zero_rate() {
        let e = CumulativeCounter::new();
        let target = Target::MAX;
        let ctx = EstimatorContext {
            current_hashrate: 1.0e15,
            current_target: &target,
            shares_per_minute: 12.0,
        };
        let snap = e.snapshot(0, &ctx);
        assert_eq!(snap.realized_share_per_min, 0.0);
    }

    // ---- EwmaEstimator ----

    #[test]
    fn ewma_first_observation_sets_rate_directly() {
        let mut e = EwmaEstimator::new(60);
        e.observe(12);
        // First obs initializes rate without decay.
        assert!((e.rate_per_tick - 12.0).abs() < 1e-9);
    }

    #[test]
    fn ewma_converges_to_steady_input() {
        // Feed a constant stream of 12 shares per tick; EWMA should
        // converge to 12 regardless of tau.
        let mut e = EwmaEstimator::new(120);
        for _ in 0..200 {
            e.observe(12);
        }
        assert!((e.rate_per_tick - 12.0).abs() < 1e-6);
    }

    #[test]
    fn ewma_with_long_tau_responds_slowly_to_change() {
        // Long tau → small alpha → strong memory of past values.
        let mut e = EwmaEstimator::new(600);
        for _ in 0..10 {
            e.observe(10);
        }
        // After 10 ticks the rate should be ≈ 10.
        assert!((e.rate_per_tick - 10.0).abs() < 1.0);
        // Now switch to 0; one tick later the rate should still be
        // mostly 10 (long memory).
        e.observe(0);
        // alpha = exp(-60/600) ≈ 0.905. new rate ≈ 0.905 × 10 + 0.095 × 0 = 9.05
        assert!(
            e.rate_per_tick > 8.5,
            "rate fell too fast: {}",
            e.rate_per_tick
        );
        assert!(e.rate_per_tick < 9.5);
    }

    #[test]
    fn ewma_with_short_tau_responds_fast_to_change() {
        // Short tau → large alpha decay → little memory.
        let mut e = EwmaEstimator::new(30);
        for _ in 0..10 {
            e.observe(10);
        }
        // Switch to 0.
        e.observe(0);
        // alpha = exp(-60/30) = exp(-2) ≈ 0.135. new ≈ 0.135 × 10 + 0.865 × 0 = 1.35
        assert!(
            e.rate_per_tick < 2.0,
            "rate fell too slow: {}",
            e.rate_per_tick
        );
    }

    #[test]
    fn ewma_reset_clears_state() {
        let mut e = EwmaEstimator::new(60);
        for _ in 0..5 {
            e.observe(10);
        }
        e.on_fire(0.0, 0.0);
        assert_eq!(e.shares_count(), 0);
        // After reset, first observation again initializes directly.
        e.observe(42);
        assert!((e.rate_per_tick - 42.0).abs() < 1e-9);
    }

    // ---- U256 round-trip precision regression check ----
    //
    // The `hash_rate_to_target` / `hash_rate_from_target` round-trip
    // should be approximately linear: `recovered ≈ hashrate ×
    // realized_spm / configured_spm`. A pre-fix bug at
    // `target.rs:184` truncated the scale factor too aggressively
    // (×100 instead of ×100_000), producing up to 49% inflation at
    // realized_spm ∈ (30, 60). See FINDINGS.md §3 for the analysis.
    //
    // The fix (`× 100_000` scaling on both sides of the U256
    // arithmetic) brings round-trip precision below 1% for all
    // realistic inputs. This test pins that precision so a regression
    // (someone reverting to `× 100`) gets caught immediately.

    #[test]
    fn hash_rate_round_trip_is_precise_after_u256_fix() {
        use crate::target::{hash_rate_from_target, hash_rate_to_target};

        // (hashrate, configured_spm, realized_spm) — points that hit
        // the worst pre-fix inflation. Each is from the SPM=30
        // cold-start trace at the corresponding tick. Post-fix all
        // should round-trip within 1% of the linear prediction.
        let cases: &[(f64, f64, f64)] = &[
            // Pre-fix inflation: 1.49×. Post-fix: ~1×.
            (7.29e12, 30.0, 4031.0),
            // Pre-fix inflation: 1.09×. Post-fix: ~1×.
            (2.19e13, 30.0, 1375.0),
            // Pre-fix inflation: 1.08×. Post-fix: ~1×.
            (6.56e13, 30.0, 464.0),
            // Pre-fix inflation: 1.01×. Post-fix: ~1×.
            (1.97e14, 30.0, 169.0),
            // Always-precise cases (realized matches configured).
            (1.0e15, 30.0, 30.0),
            (1.0e15, 12.0, 60.0),
            // Additional coverage at production-typical share rates.
            (1.0e15, 6.0, 6.0),
            (1.0e15, 120.0, 120.0),
        ];

        for &(hashrate, configured, realized) in cases {
            let target = hash_rate_to_target(hashrate, configured)
                .expect("target conversion should succeed");
            let recovered = hash_rate_from_target(target.to_le_bytes().into(), realized)
                .expect("hashrate recovery should succeed");
            let linear_prediction = hashrate * realized / configured;
            let inflation = recovered / linear_prediction;
            // 2% tolerance after the U256 fix. Includes the small
            // target+1 ≈ target and 2^256 − target ≈ 2^256
            // approximations (both relative error < 1e-13) plus the
            // residual truncation in `60/realized × 100_000 as u128`.
            assert!(
                (inflation - 1.0).abs() < 0.02,
                "round-trip inflation at hashrate={}, cfg={}, realized={}: \
                 expected ~1.0, got {} (recovered={}, linear_prediction={}). \
                 This likely means the `× 100_000` U256 scale factor in \
                 target.rs was reverted to `× 100` — see FINDINGS.md §3.",
                hashrate,
                configured,
                realized,
                inflation,
                recovered,
                linear_prediction,
            );
        }
    }

    // ---- SlidingWindowEstimator ----

    #[test]
    fn sliding_window_buffer_grows_then_evicts_oldest() {
        let mut e = SlidingWindowEstimator::new(3);
        e.observe(10);
        e.observe(20);
        e.observe(30);
        // Buffer is full; next observe evicts the oldest.
        e.observe(40);
        assert_eq!(e.shares_count(), 20 + 30 + 40);
        e.observe(50);
        assert_eq!(e.shares_count(), 30 + 40 + 50);
    }

    #[test]
    fn sliding_window_reset_clears_buffer() {
        let mut e = SlidingWindowEstimator::new(5);
        for _ in 0..5 {
            e.observe(10);
        }
        assert_eq!(e.shares_count(), 50);
        e.on_fire(0.0, 0.0);
        assert_eq!(e.shares_count(), 0);
        // Buffer is empty; first observe after reset starts fresh.
        e.observe(7);
        assert_eq!(e.shares_count(), 7);
    }

    #[test]
    fn sliding_window_short_buffer_uses_actual_length() {
        // Window n_ticks=10 but only 3 observations. realized rate
        // should be (sum / 3 ticks), not (sum / 10 ticks).
        let mut e = SlidingWindowEstimator::new(10);
        e.observe(30);
        e.observe(30);
        e.observe(30);
        let target = Target::MAX;
        let ctx = EstimatorContext {
            current_hashrate: 1.0e15,
            current_target: &target,
            shares_per_minute: 30.0,
        };
        let snap = e.snapshot(180, &ctx);
        // 3 ticks × 60s = 180s window. 90 shares total. realized = 90/3 = 30 spm.
        assert!((snap.realized_share_per_min - 30.0).abs() < 1e-6);
    }

    #[test]
    fn sliding_window_full_buffer_averages_correctly() {
        // Constant 30 shares per tick across 10 ticks. realized_per_min
        // = 30. Exact.
        let mut e = SlidingWindowEstimator::new(10);
        for _ in 0..10 {
            e.observe(30);
        }
        let target = Target::MAX;
        let ctx = EstimatorContext {
            current_hashrate: 1.0e15,
            current_target: &target,
            shares_per_minute: 30.0,
        };
        let snap = e.snapshot(600, &ctx);
        assert!((snap.realized_share_per_min - 30.0).abs() < 1e-6);
    }

    #[test]
    fn sliding_window_oldest_drops_out() {
        // Fill with 60-per-tick (high rate), then add ticks at 0
        // (truth has dropped). Once the buffer fully turns over, the
        // estimate should reflect the new rate.
        let mut e = SlidingWindowEstimator::new(5);
        for _ in 0..5 {
            e.observe(60);
        }
        // Buffer = [60, 60, 60, 60, 60]; total = 300; realized = 60 spm.
        // Now feed 5 ticks of 0:
        for _ in 0..5 {
            e.observe(0);
        }
        // Buffer = [0, 0, 0, 0, 0]; realized = 0 spm.
        assert_eq!(e.shares_count(), 0);
        let target = Target::MAX;
        let ctx = EstimatorContext {
            current_hashrate: 1.0e15,
            current_target: &target,
            shares_per_minute: 60.0,
        };
        let snap = e.snapshot(300, &ctx);
        assert_eq!(snap.realized_share_per_min, 0.0);
    }

    #[test]
    fn sliding_window_empty_buffer_yields_zero_rate() {
        let e = SlidingWindowEstimator::new(10);
        let target = Target::MAX;
        let ctx = EstimatorContext {
            current_hashrate: 1.0e15,
            current_target: &target,
            shares_per_minute: 30.0,
        };
        let snap = e.snapshot(60, &ctx);
        assert_eq!(snap.realized_share_per_min, 0.0);
    }

    #[test]
    fn ewma_snapshot_returns_rate_in_shares_per_minute() {
        let mut e = EwmaEstimator::new(60);
        // Feed 12 shares per tick for many ticks.
        for _ in 0..50 {
            e.observe(12);
        }
        // rate_per_tick ≈ 12, tick_secs = 60.
        // shares_per_min = 12 * (60/60) = 12.
        let target = Target::MAX;
        let ctx = EstimatorContext {
            current_hashrate: 1.0e15,
            current_target: &target,
            shares_per_minute: 12.0,
        };
        let snap = e.snapshot(60, &ctx);
        assert!((snap.realized_share_per_min - 12.0).abs() < 1e-3);
    }
}
