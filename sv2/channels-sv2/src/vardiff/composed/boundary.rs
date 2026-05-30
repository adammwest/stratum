//! Boundary — Stage 2 of the three-stage vardiff pipeline.
//!
//! The Boundary computes the threshold θ that the deviation δ must
//! exceed for the algorithm to fire. It is the piece of the algorithm
//! that lives in decision theory: design pressure is Type I vs Type II
//! error rates, plus *rate-awareness* — the noise floor of δ depends on
//! share rate, so the threshold should too.
//!
//! The Boundary receives the full `EstimatorSnapshot` including optional
//! uncertainty, enabling uncertainty-aware implementations that adapt
//! their threshold based on estimator confidence.

use std::fmt::Debug;

use super::estimator::EstimatorSnapshot;

/// The decision threshold θ that the deviation δ must exceed to fire.
/// Returns `f64` in percentage points (e.g., `60.0` means δ must be at
/// least 60%).
///
/// Implementations: [`StepFunction`] (classic), [`PoissonCI`]
/// (rate-aware), [`CredibleIntervalBoundary`] (uncertainty-aware),
/// [`CusumBoundary`] (sequential-testing).
pub trait Boundary: Debug + Send + Sync {
    fn threshold(&self, dt_secs: u64, shares_per_minute: f32, snap: &EstimatorSnapshot) -> f64;
}

/// A piecewise-constant threshold over `dt_secs`. Share-rate-blind.
///
/// Constructed via [`StepFunction::classic_table`] for byte-for-byte
/// equivalence with `VardiffState::try_vardiff`'s threshold cascade:
///
/// ```text
/// dt <  60s:  θ = 100%   (only very large δ fires; the >=100% short-circuit)
/// dt <  120s: θ =  60%
/// dt <  180s: θ =  50%
/// dt <  240s: θ =  45%
/// dt <  300s: θ =  30%
/// dt ≥  300s: θ =  15%
/// ```
///
/// To explore other step functions (e.g., flatter at high `dt_secs`),
/// construct `StepFunction { table: custom_table }` directly. The table
/// MUST be sorted ascending by `dt_threshold` and MUST have a final entry
/// with `dt_threshold == u64::MAX` so the function is defined for all
/// inputs.
#[derive(Debug, Clone)]
pub struct StepFunction {
    /// Sorted ascending by `dt_threshold`. For `dt_secs < dt_threshold`
    /// the function returns `value`. The last entry must have
    /// `dt_threshold == u64::MAX`.
    pub table: Vec<(u64, f64)>,
}

impl StepFunction {
    /// The classic threshold ladder from `VardiffState::try_vardiff`.
    pub fn classic_table() -> Self {
        Self {
            table: vec![
                (60, 100.0),
                (120, 60.0),
                (180, 50.0),
                (240, 45.0),
                (300, 30.0),
                (u64::MAX, 15.0),
            ],
        }
    }
}

impl Boundary for StepFunction {
    fn threshold(&self, dt_secs: u64, _shares_per_minute: f32, _snap: &EstimatorSnapshot) -> f64 {
        for &(threshold_dt, value) in &self.table {
            if dt_secs < threshold_dt {
                return value;
            }
        }
        // Unreachable when the table includes the required u64::MAX entry,
        // but defensive fallback rather than panic.
        self.table.last().map(|(_, v)| *v).unwrap_or(100.0)
    }
}

/// Parametric boundary: threshold derived from the Poisson confidence
/// interval on the realized share count under the null hypothesis (no
/// genuine change in miner hashrate).
///
/// Formula (in *percentage points*, matching the deviation
/// convention):
///
/// ```text
/// λ̄ = (SPM / 60) × Δt              (expected share count under H₀)
/// θ_fraction = (z·√λ̄ + 0.5) / λ̄ + margin
/// θ_pct = θ_fraction × 100
/// ```
///
/// The `z` coefficient gives the desired Type I error rate (e.g.,
/// `z = 2.576` for ~1% per-tick false-fire rate under H₀). The
/// `margin` term adds a flat slack above the Poisson floor — without
/// it the algorithm fires too readily when the statistic happens to
/// sit just above the boundary on small-variance trials. The default
/// `(z = 2.576, margin = 0.05)` is the parameterization used by the
/// `Parametric` and `FullRemedy` algorithms in the registry.
///
/// This boundary is the only stage where `Parametric` differs from
/// `ClassicComposed`; Estimator and UpdateRule are unchanged.
#[derive(Debug, Clone, Copy)]
pub struct PoissonCI {
    /// Two-sided normal quantile. `2.576` ≈ 99% CI.
    pub z: f64,
    /// Additive margin in fractional form (e.g., `0.05` for +5%).
    pub margin: f64,
}

impl PoissonCI {
    /// The default Parametric parameters: `z = 2.576` (99% CI),
    /// `margin = 0.05` (+5%).
    pub fn default_parametric() -> Self {
        Self {
            z: 2.576,
            margin: 0.05,
        }
    }

    /// Construct with arbitrary `z` and `margin`. Use this to explore
    /// the Type I error frontier: e.g. `with_z(3.0, 0.05)` for a 99.7%
    /// CI ("3σ"), or `with_z(3.891, 0.05)` for 99.99%.
    ///
    /// **Why explore beyond the default?** The default `z = 2.576` gives
    /// a ~1% per-tick false-fire rate under H₀, which is the right
    /// trade-off when each tick is independent. But at very low share
    /// rates the per-tick Poisson tail is heavy enough that even
    /// 1-in-100 outliers cascade through the algorithm. See
    /// `sim/docs/FINDINGS.md` § "Parametric SPM=6 cascade" for the
    /// concrete failure mode that motivates exploring stricter z.
    pub fn with_z(z: f64, margin: f64) -> Self {
        Self { z, margin }
    }

    /// 99.7% CI ("3σ") preset: `z = 3.0`, default margin.
    /// Roughly 0.3% per-tick false-fire rate under H₀.
    pub fn strict_3sigma() -> Self {
        Self {
            z: 3.0,
            margin: 0.05,
        }
    }
}

impl Boundary for PoissonCI {
    fn threshold(&self, dt_secs: u64, shares_per_minute: f32, _snap: &EstimatorSnapshot) -> f64 {
        // Expected share count under H₀ over the window.
        let lambda_bar = (shares_per_minute as f64 / 60.0) * dt_secs as f64;
        if lambda_bar <= 0.0 {
            // Pathological — fall back to a very strict threshold so
            // the algorithm only fires on overwhelming evidence.
            return 100.0;
        }
        let bound_fraction = (self.z * lambda_bar.sqrt() + 0.5) / lambda_bar + self.margin;
        // The deviation δ is in percentage points;
        // convert the fractional bound to match.
        bound_fraction * 100.0
    }
}

/// Credible-interval boundary: fires when the estimator's posterior
/// credible interval excludes ratio = 1.0, meaning the algorithm is
/// confident that the miner's hashrate differs from the current target.
///
/// This boundary uses the estimator's reported `uncertainty.ratio_std`
/// directly, rather than inferring noise from dt_secs or SPM. This
/// makes it self-calibrating: a confident estimate (many ticks of data,
/// small ratio_std) triggers on small deviations; an uncertain estimate
/// (fresh after fire, large ratio_std) requires large deviations.
///
/// ## Formula
///
/// The deviation δ = |ratio - 1| × 100 (percentage points).
/// The boundary computes:
///
/// ```text
/// θ = z × ratio_std × 100
/// ```
///
/// Fire iff δ ≥ θ, i.e., the deviation is at least z standard deviations
/// from ratio=1.0. With z=1.96, this corresponds to a 95% credible
/// interval excluding 1.0.
///
/// ## Fallback
///
/// When `snap.uncertainty` is `None` (estimator doesn't report confidence),
/// falls back to PoissonCI behavior. This ensures the boundary works with
/// all estimators, not just Bayesian.
///
/// ## Parameters
///
/// - `z`: number of standard deviations required (1.96 = 95% CI, 2.576 = 99%)
/// - `fallback`: PoissonCI used when uncertainty is unavailable
#[derive(Debug, Clone, Copy)]
pub struct CredibleIntervalBoundary {
    /// Z-score for the credible interval (e.g., 1.96 for 95% CI).
    pub z: f64,
    /// Fallback boundary for estimators that don't report uncertainty.
    pub fallback: PoissonCI,
}

impl CredibleIntervalBoundary {
    /// 95% credible interval boundary.
    pub fn new_95() -> Self {
        Self {
            z: 1.96,
            fallback: PoissonCI::default_parametric(),
        }
    }

    /// Custom z-score credible interval boundary.
    pub fn with_z(z: f64) -> Self {
        Self {
            z,
            fallback: PoissonCI::default_parametric(),
        }
    }
}

impl Boundary for CredibleIntervalBoundary {
    fn threshold(&self, dt_secs: u64, shares_per_minute: f32, snap: &EstimatorSnapshot) -> f64 {
        match &snap.uncertainty {
            Some(u) if u.ratio_std > 0.0 && u.effective_n > 0.0 => {
                // Fire when |ratio - 1.0| > z × ratio_std
                // In percentage points (matching deviation convention):
                self.z * u.ratio_std * 100.0
            }
            _ => {
                // No uncertainty available — delegate to PoissonCI
                self.fallback.threshold(dt_secs, shares_per_minute, snap)
            }
        }
    }
}

/// Sequential-testing boundary inspired by CUSUM (Cumulative Sum).
///
/// Unlike PoissonCI (which treats each tick independently), this boundary
/// accounts for the fact that evidence accumulates over ticks. A genuine
/// hashrate change produces *persistent* deviations across all subsequent
/// ticks, not just one spike. The threshold shrinks faster with dt_secs
/// than PoissonCI, making it more sensitive to sustained changes.
///
/// ## Formula
///
/// ```text
/// n_ticks = dt_secs / tick_secs
/// θ = (sensitivity / n_ticks + floor) × 100
/// ```
///
/// Where:
/// - `sensitivity`: how many ticks of sustained deviation needed to fire
///   (analogous to CUSUM's h/k ratio). Lower = more sensitive.
/// - `floor`: minimum threshold as a fraction (prevents firing on tiny
///   deviations even with many ticks). Acts as the "slack" k in CUSUM.
///
/// The key difference from PoissonCI: PoissonCI threshold decreases as
/// 1/√n (Poisson CI width). CusumBoundary threshold decreases as 1/n
/// (linear evidence accumulation). At n=1 tick they're similar; at n=10
/// ticks CUSUM is much tighter.
///
/// ## When uncertainty is available
///
/// If the estimator reports uncertainty, the floor is scaled by ratio_std
/// to avoid firing when the estimate itself is uncertain.
#[derive(Debug, Clone, Copy)]
pub struct CusumBoundary {
    /// Ticks of sustained deviation needed to fire at maximum sensitivity.
    /// Lower = fires faster. Typical range: 2.0–5.0.
    pub sensitivity: f64,
    /// Minimum fractional threshold regardless of accumulated evidence.
    /// Prevents firing on tiny deviations. Typical: 0.03–0.10 (3–10%).
    pub floor: f64,
    /// Tick interval for converting dt_secs to tick count.
    pub tick_secs: u64,
}

impl CusumBoundary {
    pub fn new(sensitivity: f64, floor: f64) -> Self {
        Self {
            sensitivity,
            floor,
            tick_secs: 60,
        }
    }
}

/// Rate-adaptive CUSUM: sensitivity scales with share rate so the
/// boundary is appropriately conservative at low SPM (where Poisson
/// noise is high) and aggressive at high SPM (where noise is low).
///
/// ```text
/// effective_sensitivity = base_sensitivity × √(SPM / reference_spm)
/// ```
///
/// At reference_spm, effective_sensitivity = base_sensitivity. At lower
/// SPM it's smaller (tighter → more conservative, less jitter). At higher
/// SPM it's larger (looser → but the EWMA estimate is better anyway).
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveCusumBoundary {
    pub base_sensitivity: f64,
    pub reference_spm: f64,
    pub floor: f64,
    pub tick_secs: u64,
}

impl AdaptiveCusumBoundary {
    pub fn new(base_sensitivity: f64, floor: f64) -> Self {
        Self {
            base_sensitivity,
            reference_spm: 30.0,
            floor,
            tick_secs: 60,
        }
    }
}

impl Boundary for AdaptiveCusumBoundary {
    fn threshold(&self, dt_secs: u64, shares_per_minute: f32, snap: &EstimatorSnapshot) -> f64 {
        let n_ticks = (dt_secs as f64 / self.tick_secs as f64).max(1.0);

        // Rate-adaptive sensitivity: more conservative at low SPM
        let spm_factor = ((shares_per_minute as f64) / self.reference_spm).sqrt();
        let sensitivity = self.base_sensitivity * spm_factor;

        let effective_floor = match &snap.uncertainty {
            Some(u) if u.ratio_std > 0.0 => self.floor + u.ratio_std * 0.5,
            _ => self.floor,
        };

        let threshold_fraction = (sensitivity / n_ticks) + effective_floor;
        threshold_fraction * 100.0
    }
}

/// Asymmetric rate-adaptive CUSUM: uses different thresholds for
/// tightening (miner over-performing) vs easing (miner under-performing).
///
/// **Rationale:** Making difficulty harder costs ~1 rejected share per fire
/// (in-flight work becomes invalid). Making difficulty easier costs nothing
/// (old harder work is still valid). Therefore the boundary should:
/// - Fire **quickly** to ease difficulty (miner slowing → detect fast, free action)
/// - Fire **cautiously** to tighten difficulty (miner speeding → avoid rejecting in-flight shares)
///
/// Direction is determined from the snapshot: if `realized_share_per_min > shares_per_minute`,
/// the miner is over-performing and firing would tighten (costly direction).
///
/// The `tighten_multiplier` (>1.0) makes the threshold higher when tightening,
/// requiring more evidence before making difficulty harder.
#[derive(Debug, Clone, Copy)]
pub struct AsymmetricCusumBoundary {
    /// Base sensitivity (same as AdaptiveCusumBoundary).
    pub base_sensitivity: f64,
    /// Reference SPM for rate scaling.
    pub reference_spm: f64,
    /// Minimum threshold floor.
    pub floor: f64,
    /// Multiplier applied to threshold when firing would TIGHTEN difficulty.
    /// Values > 1.0 make tightening more conservative. Typical: 1.5–3.0.
    pub tighten_multiplier: f64,
    /// Tick interval.
    pub tick_secs: u64,
}

impl AsymmetricCusumBoundary {
    /// Constructs with default reference_spm=30 and tick_secs=60.
    /// `tighten_multiplier` controls the asymmetry: 1.0 = symmetric,
    /// 2.0 = requires 2× more evidence to tighten than to ease.
    pub fn new(base_sensitivity: f64, floor: f64, tighten_multiplier: f64) -> Self {
        Self {
            base_sensitivity,
            reference_spm: 30.0,
            floor,
            tighten_multiplier: tighten_multiplier.max(1.0),
            tick_secs: 60,
        }
    }
}

impl Boundary for AsymmetricCusumBoundary {
    fn threshold(&self, dt_secs: u64, shares_per_minute: f32, snap: &EstimatorSnapshot) -> f64 {
        let n_ticks = (dt_secs as f64 / self.tick_secs as f64).max(1.0);

        let spm_factor = ((shares_per_minute as f64) / self.reference_spm).sqrt();
        let sensitivity = self.base_sensitivity * spm_factor;

        let effective_floor = match &snap.uncertainty {
            Some(u) if u.ratio_std > 0.0 => self.floor + u.ratio_std * 0.5,
            _ => self.floor,
        };

        let base_threshold = (sensitivity / n_ticks) + effective_floor;

        // Determine direction: is the miner over-performing (tighten) or under-performing (ease)?
        // realized_spm > configured_spm means miner is faster → fire would tighten → costly
        let would_tighten = snap.realized_share_per_min > shares_per_minute as f64;

        let threshold_fraction = if would_tighten {
            base_threshold * self.tighten_multiplier
        } else {
            base_threshold
        };

        threshold_fraction * 100.0
    }
}

impl Boundary for CusumBoundary {
    fn threshold(&self, dt_secs: u64, _shares_per_minute: f32, snap: &EstimatorSnapshot) -> f64 {
        let n_ticks = (dt_secs as f64 / self.tick_secs as f64).max(1.0);

        // Base floor — scaled by uncertainty if available
        let effective_floor = match &snap.uncertainty {
            Some(u) if u.ratio_std > 0.0 => self.floor + u.ratio_std * 0.5,
            _ => self.floor,
        };

        // Threshold decreases as 1/n_ticks (sequential evidence accumulation)
        let threshold_fraction = (self.sensitivity / n_ticks) + effective_floor;

        // Convert to percentage points (matching deviation convention)
        threshold_fraction * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_snap() -> EstimatorSnapshot {
        EstimatorSnapshot {
            h_estimate: 1.0e15,
            realized_share_per_min: 12.0,
            n_shares: 12,
            dt_secs: 60,
            uncertainty: None,
        }
    }

    #[test]
    fn classic_table_matches_vardiff_state_cascade() {
        let b = StepFunction::classic_table();
        // Below the first boundary: 100% (only very large δ fires).
        assert_eq!(b.threshold(0, 12.0, &dummy_snap()), 100.0);
        assert_eq!(b.threshold(15, 12.0, &dummy_snap()), 100.0);
        assert_eq!(b.threshold(59, 12.0, &dummy_snap()), 100.0);
        // Each subsequent rung.
        assert_eq!(b.threshold(60, 12.0, &dummy_snap()), 60.0);
        assert_eq!(b.threshold(119, 12.0, &dummy_snap()), 60.0);
        assert_eq!(b.threshold(120, 12.0, &dummy_snap()), 50.0);
        assert_eq!(b.threshold(179, 12.0, &dummy_snap()), 50.0);
        assert_eq!(b.threshold(180, 12.0, &dummy_snap()), 45.0);
        assert_eq!(b.threshold(239, 12.0, &dummy_snap()), 45.0);
        assert_eq!(b.threshold(240, 12.0, &dummy_snap()), 30.0);
        assert_eq!(b.threshold(299, 12.0, &dummy_snap()), 30.0);
        // Floor: 15% past dt = 300s.
        assert_eq!(b.threshold(300, 12.0, &dummy_snap()), 15.0);
        assert_eq!(b.threshold(1_800, 12.0, &dummy_snap()), 15.0);
        assert_eq!(b.threshold(u64::MAX - 1, 12.0, &dummy_snap()), 15.0);
    }

    #[test]
    fn classic_threshold_is_share_rate_blind() {
        // The classic ladder ignores share rate — the same dt produces the
        // same threshold regardless of SPM. This is the property that
        // motivates the Parametric boundary (which IS rate-aware).
        let b = StepFunction::classic_table();
        for &spm in &[6.0f32, 12.0, 30.0, 60.0, 120.0] {
            assert_eq!(b.threshold(120, spm, &dummy_snap()), 50.0);
            assert_eq!(b.threshold(300, spm, &dummy_snap()), 15.0);
        }
    }

    // ---- PoissonCI ----

    #[test]
    fn poisson_ci_matches_reference_values() {
        // At dt=1200s, hand-computed reference values:
        //   SPM=12  → θ ≈ 0.218
        //   SPM=60  → θ ≈ 0.125
        //   SPM=120 → θ ≈ 0.103
        // (using z=2.576, margin=0.05; θ_fraction = (z·√λ̄ + 0.5)/λ̄ + margin)
        // Our boundary returns these × 100 (percentage points).
        let b = PoissonCI::default_parametric();
        let t12 = b.threshold(1200, 12.0, &dummy_snap());
        let t60 = b.threshold(1200, 60.0, &dummy_snap());
        let t120 = b.threshold(1200, 120.0, &dummy_snap());
        assert!((t12 - 21.8).abs() < 0.1, "SPM=12 got {}", t12);
        assert!((t60 - 12.5).abs() < 0.1, "SPM=60 got {}", t60);
        assert!((t120 - 10.3).abs() < 0.1, "SPM=120 got {}", t120);
    }

    #[test]
    fn poisson_ci_is_rate_aware() {
        // The defining property — unlike StepFunction. As SPM
        // increases the threshold strictly decreases (the noise floor
        // shrinks, so the algorithm can detect smaller real changes).
        let b = PoissonCI::default_parametric();
        let t6 = b.threshold(600, 6.0, &dummy_snap());
        let t12 = b.threshold(600, 12.0, &dummy_snap());
        let t60 = b.threshold(600, 60.0, &dummy_snap());
        let t120 = b.threshold(600, 120.0, &dummy_snap());
        assert!(t6 > t12, "{} not > {}", t6, t12);
        assert!(t12 > t60);
        assert!(t60 > t120);
    }

    #[test]
    fn poisson_ci_returns_strict_threshold_on_degenerate_inputs() {
        let b = PoissonCI::default_parametric();
        assert_eq!(b.threshold(0, 12.0, &dummy_snap()), 100.0); // dt = 0 → λ̄ = 0
        assert_eq!(b.threshold(60, 0.0, &dummy_snap()), 100.0); // SPM = 0 → λ̄ = 0
    }

    #[test]
    fn poisson_ci_strict_3sigma_returns_higher_threshold_than_default() {
        // The strict variant (z=3.0) sits above the default (z=2.576)
        // at every λ̄ — that's the whole point: trade a higher
        // missed-detection rate for a tighter false-fire rate. Holds
        // across share rates.
        let default = PoissonCI::default_parametric();
        let strict = PoissonCI::strict_3sigma();
        for &spm in &[6.0f32, 12.0, 30.0, 60.0, 120.0] {
            for &dt in &[60u64, 300, 600, 1200] {
                let d = default.threshold(dt, spm, &dummy_snap());
                let s = strict.threshold(dt, spm, &dummy_snap());
                assert!(
                    s > d,
                    "strict ({}) should be > default ({}) at dt={}, spm={}",
                    s,
                    d,
                    dt,
                    spm,
                );
            }
        }
    }

    #[test]
    fn poisson_ci_with_z_matches_strict_3sigma_preset() {
        let a = PoissonCI::strict_3sigma();
        let b = PoissonCI::with_z(3.0, 0.05);
        assert_eq!(a.z, b.z);
        assert_eq!(a.margin, b.margin);
        for &(dt, spm) in &[(60u64, 12.0f32), (600, 60.0), (1800, 120.0)] {
            assert_eq!(
                a.threshold(dt, spm, &dummy_snap()),
                b.threshold(dt, spm, &dummy_snap())
            );
        }
    }
}
