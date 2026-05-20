use crate::vardiff::clock::{Clock, SystemClock};
use bitcoin::Target;
use statrs::function::gamma::ln_gamma;
use std::sync::Arc;
use tracing::debug;

use super::{error::VardiffError, Vardiff};

const DEFAULT_MIN_HASHRATE: f32 = 1.0;
// BEST BAYES PARAMS: min_interval_secs=300 expected_break_interval_secs=3600.000 prior_strength_minutes=0.01000 max_run_lengths=25 update_threshold_pct=25.349 changepoint_threshold=9.115953
// METRICS: SP50:0.092005 | SP99:0.265230 | CR:96.10% | CP50:307.2s | CP99:805.2s | RR:0.232000 | RP50:164.6s | RP99:197.1s | JP50:0.000000 | JM:0.005139 | JP99:0.062098
const BAYES_MIN_INTERVAL_SECS: u64 = 300;
const BAYES_EXPECTED_BREAK_INTERVAL_SECS: f64 = 3600.0;
const BAYES_PRIOR_STRENGTH_MINUTES: f64 = 0.01;
const BAYES_MAX_RUN_LENGTHS: usize = 25;
const BAYES_UPDATE_THRESHOLD_PCT: f32 = 25.349;
const BAYES_CHANGEPOINT_THRESHOLD: f64 = 9.115953;

#[derive(Debug, Clone, Copy)]
pub struct BayesParams {
    pub min_interval_secs: u64,
    pub expected_break_interval_secs: f64,
    pub prior_strength_minutes: f64,
    pub max_run_lengths: usize,
    pub update_threshold_pct: f32,
    pub changepoint_threshold: f64,
}

impl Default for BayesParams {
    fn default() -> Self {
        Self {
            min_interval_secs: BAYES_MIN_INTERVAL_SECS,
            expected_break_interval_secs: BAYES_EXPECTED_BREAK_INTERVAL_SECS,
            prior_strength_minutes: BAYES_PRIOR_STRENGTH_MINUTES,
            max_run_lengths: BAYES_MAX_RUN_LENGTHS,
            update_threshold_pct: BAYES_UPDATE_THRESHOLD_PCT,
            changepoint_threshold: BAYES_CHANGEPOINT_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PoissonGammaBocpd {
    prior_strength_minutes: f64,
    expected_break_interval_secs: f64,
    max_run_lengths: usize,
    log_run_probs: Vec<f64>,
    shapes: Vec<f64>,
    rates: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct BocpdObservation {
    pub posterior_share_rate_per_min: f64,
    pub changepoint_probability: f64,
    pub reset_run_probability: f64,
    pub active_run_lengths: usize,
}

impl PoissonGammaBocpd {
    pub fn new(prior_mean_share_rate_per_min: f64, params: &BayesParams) -> Self {
        let prior_mean_share_rate_per_min = prior_mean_share_rate_per_min.max(f64::EPSILON);
        let prior_shape = prior_mean_share_rate_per_min * params.prior_strength_minutes;
        let prior_rate = params.prior_strength_minutes;

        Self {
            prior_strength_minutes: params.prior_strength_minutes,
            expected_break_interval_secs: params.expected_break_interval_secs,
            max_run_lengths: params.max_run_lengths,
            log_run_probs: vec![0.0],
            shapes: vec![prior_shape],
            rates: vec![prior_rate],
        }
    }

    pub fn observe(
        &mut self,
        shares: u32,
        exposure_minutes: f64,
        prior_mean_share_rate_per_min: f64,
        elapsed_secs: u64,
    ) -> BocpdObservation {
        let exposure_minutes = exposure_minutes.max(f64::EPSILON);
        let prior_mean_share_rate_per_min = prior_mean_share_rate_per_min.max(f64::EPSILON);
        let prior_shape = prior_mean_share_rate_per_min * self.prior_strength_minutes;
        let prior_rate = self.prior_strength_minutes;

        let hazard = self.hazard_for_elapsed_secs(elapsed_secs);
        let log_hazard = hazard.ln();
        let log_growth = (1.0 - hazard).ln();

        let n = self.log_run_probs.len();
        let mut next_log_run_probs = vec![f64::NEG_INFINITY; n + 1];
        let mut changepoint_terms = Vec::with_capacity(n);

        for i in 0..n {
            let log_pred = poisson_gamma_logpred(
                shares,
                exposure_minutes,
                self.shapes[i],
                self.rates[i],
            );

            changepoint_terms.push(self.log_run_probs[i] + log_pred + log_hazard);
            next_log_run_probs[i + 1] = self.log_run_probs[i] + log_pred + log_growth;
        }

        next_log_run_probs[0] = logsumexp(&changepoint_terms);

        let log_norm = logsumexp(&next_log_run_probs);
        for p in &mut next_log_run_probs {
            *p -= log_norm;
        }

        let mut next_shapes = Vec::with_capacity(n + 1);
        let mut next_rates = Vec::with_capacity(n + 1);

        next_shapes.push(prior_shape);
        next_rates.push(prior_rate);

        for i in 0..n {
            next_shapes.push(self.shapes[i] + shares as f64);
            next_rates.push(self.rates[i] + exposure_minutes);
        }

        self.log_run_probs = next_log_run_probs;
        self.shapes = next_shapes;
        self.rates = next_rates;
        self.truncate();

        let reset_run_probability = self.log_run_probs[0].exp();
        BocpdObservation {
            posterior_share_rate_per_min: self.posterior_rate_mean(),
            changepoint_probability: reset_run_probability,
            reset_run_probability,
            active_run_lengths: self.log_run_probs.len(),
        }
    }

    fn hazard_for_elapsed_secs(&self, elapsed_secs: u64) -> f64 {
        let elapsed_secs = elapsed_secs as f64;
        let hazard = 1.0 - (-elapsed_secs / self.expected_break_interval_secs).exp();
        hazard.clamp(1.0e-9, 1.0 - 1.0e-9)
    }

    fn posterior_rate_mean(&self) -> f64 {
        let mut mean = 0.0;

        for i in 0..self.log_run_probs.len() {
            mean += self.log_run_probs[i].exp() * self.shapes[i] / self.rates[i];
        }

        mean
    }

    fn truncate(&mut self) {
        if self.log_run_probs.len() <= self.max_run_lengths {
            return;
        }

        self.log_run_probs.truncate(self.max_run_lengths);
        self.shapes.truncate(self.max_run_lengths);
        self.rates.truncate(self.max_run_lengths);

        let log_norm = logsumexp(&self.log_run_probs);
        for p in &mut self.log_run_probs {
            *p -= log_norm;
        }
    }
}

#[derive(Debug)]
pub struct VardiffState {
    pub shares_since_last_update: u32,
    pub timestamp_of_last_update: u64,
    pub min_allowed_hashrate: f32,
    pub clock: Arc<dyn Clock>,
    pub params: BayesParams,
    pub bocpd: PoissonGammaBocpd,
}

impl VardiffState {
    pub fn new() -> Result<Self, VardiffError> {
        Self::new_with_min(DEFAULT_MIN_HASHRATE)
    }

    pub fn new_with_min(min_allowed_hashrate: f32) -> Result<Self, VardiffError> {
        Self::new_with_clock(min_allowed_hashrate, Arc::new(SystemClock))
    }

    pub fn new_with_clock(
        min_allowed_hashrate: f32,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, VardiffError> {
        Self::new_with_clock_and_params(min_allowed_hashrate, clock, BayesParams::default())
    }

    pub fn new_with_clock_and_params(
        min_allowed_hashrate: f32,
        clock: Arc<dyn Clock>,
        params: BayesParams,
    ) -> Result<Self, VardiffError> {
        let timestamp_secs = clock.now_secs();

        Ok(Self {
            shares_since_last_update: 0,
            timestamp_of_last_update: timestamp_secs,
            min_allowed_hashrate,
            clock,
            params,
            bocpd: PoissonGammaBocpd::new(1.0, &params),
        })
    }

    pub fn set_shares_since_last_update(&mut self, shares_since_last_update: u32) {
        self.shares_since_last_update = shares_since_last_update;
    }

    fn try_vardiff_bocpd(
        &mut self,
        hashrate: f32,
        shares_per_minute: f32,
    ) -> Result<Option<f32>, VardiffError> {
        let now = self.clock.now_secs();
        let elapsed_secs = now.saturating_sub(self.timestamp_of_last_update);

        let target_share_rate = shares_per_minute.max(f32::EPSILON);
        let min_elapsed_secs =
            (self.params.min_interval_secs as f32 * 6.0 / target_share_rate).ceil() as u64;

        if elapsed_secs <= min_elapsed_secs {
            return Ok(None);
        }

        let exposure_minutes = elapsed_secs as f64 / 60.0;
        

        let observation = self.bocpd.observe(
            self.shares_since_last_update,
            exposure_minutes,
            target_share_rate as f64,
            elapsed_secs,
        );

        let posterior_share_rate = observation.posterior_share_rate_per_min as f32;
        let mut new_hashrate = hashrate * posterior_share_rate / target_share_rate;

        if new_hashrate < self.min_allowed_hashrate {
            new_hashrate = self.min_allowed_hashrate;
        }

        let delta_pct = if hashrate > 0.0 {
            ((new_hashrate - hashrate).abs() / hashrate) * 100.0
        } else {
            100.0
        };

        debug!(
            target: "vardiff",
            "BOCPD vardiff: elapsed={}s shares={} posterior_spm={:.4} target_spm={:.4} cp_prob={:.4} active_runs={} new_hashrate={:.2} delta_pct={:.2}",
            elapsed_secs,
            self.shares_since_last_update,
            observation.posterior_share_rate_per_min,
            shares_per_minute,
            observation.changepoint_probability,
            observation.active_run_lengths,
            new_hashrate,
            delta_pct,
        );

        self.reset_counter()?;

        let confidence = (observation.changepoint_probability / self.params.changepoint_threshold)
            .clamp(0.0, 1.0) as f32;

        let confidence_boost = 1.0 + confidence;

        let effective_update_threshold_pct =
            (self.params.update_threshold_pct / confidence_boost)
                .max(1.0);

        if delta_pct < effective_update_threshold_pct {
            return Ok(None);
        }

        Ok(Some(new_hashrate))
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
        self.min_allowed_hashrate
    }

    fn set_timestamp_of_last_update(&mut self, timestamp_of_last_update: u64) {
        self.timestamp_of_last_update = timestamp_of_last_update;
    }

    fn increment_shares_since_last_update(&mut self) {
        self.shares_since_last_update += 1;
    }

    fn add_shares(&mut self, n: u32) {
        self.shares_since_last_update = self.shares_since_last_update.saturating_add(n);
    }

    fn reset_counter(&mut self) -> Result<(), VardiffError> {
        self.timestamp_of_last_update = self.clock.now_secs();
        self.shares_since_last_update = 0;
        Ok(())
    }

    fn try_vardiff(
        &mut self,
        hashrate: f32,
        _target: &Target,
        shares_per_minute: f32,
    ) -> Result<Option<f32>, VardiffError> {
        self.try_vardiff_bocpd(hashrate, shares_per_minute)
    }
}

fn poisson_gamma_logpred(shares: u32, exposure_minutes: f64, shape: f64, rate: f64) -> f64 {
    let x = shares as f64;
    let exposure_minutes = exposure_minutes.max(f64::EPSILON);
    let rate_plus_exposure = rate + exposure_minutes;

    ln_gamma(shape + x) - ln_gamma(shape) - ln_gamma(x + 1.0)
        + shape * (rate / rate_plus_exposure).ln()
        + x * (exposure_minutes / rate_plus_exposure).ln()
}

fn logsumexp(values: &[f64]) -> f64 {
    let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !max_value.is_finite() {
        return max_value;
    }

    let sum: f64 = values.iter().map(|v| (*v - max_value).exp()).sum();
    max_value + sum.ln()
}
