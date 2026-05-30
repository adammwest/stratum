use std::collections::BTreeMap;
use std::env;
use std::sync::Arc;
use std::time::Instant;

use channels_sv2::vardiff::classic::{
    VardiffUpdateCurve,
};
use channels_sv2::vardiff::vanilla::VardiffState as VanillaVardiffState;
use channels_sv2::vardiff::MockClock;
use channels_sv2::VardiffState;
use vardiff_sim::baseline::{
    default_cells, Cell, CellResult, Scenario, DEFAULT_BASELINE_SEED, DEFAULT_TRIAL_COUNT,
    MIN_SETTLED_WINDOW_SECS, QUIET_WINDOW_SECS, REACT_WINDOW_SECS, SETTLE_BUFFER_SECS,
    STEP_EVENT_AT_SECS,
};
use vardiff_sim::metrics::{
    convergence_time_distribution, jitter_distribution, reaction_time_distribution,
    settled_accuracy_distribution,
};
use vardiff_sim::run_trial;

#[derive(Debug, Clone, Copy)]
struct Range {
    min: f64,
    max: f64,
}

impl Range {
    fn from_env(prefix: &str, default_min: f64, default_max: f64) -> Self {
        let min = env_or(&format!("{}_MIN", prefix), default_min);
        let max = env_or(&format!("{}_MAX", prefix), default_max);
        Self {
            min: min.min(max),
            max: min.max(max),
        }
    }

    fn width(self) -> f64 {
        (self.max - self.min).max(1.0e-12)
    }

    fn clamp(self, value: f64) -> f64 {
        value.clamp(self.min, self.max)
    }

    fn sample(self, rng: &mut SplitMix64) -> f64 {
        self.min + rng.next_f64() * self.width()
    }
}

#[derive(Debug, Clone, Copy)]
struct SearchBounds {
    start_pct: Range,
    floor_pct: Range,
    half_life_secs: Range,
    power: Range,
}

#[derive(Debug, Clone, Copy)]
struct TuneParams {
    start_pct: f32,
    floor_pct: f32,
    half_life_secs: f32,
    power: f32,
}

impl TuneParams {
    fn from_curve(curve: VardiffUpdateCurve) -> Self {
        Self {
            start_pct: curve.start_pct,
            floor_pct: curve.floor_pct,
            half_life_secs: curve.half_life_secs,
            power: curve.power,
        }
    }

    fn to_curve(self) -> VardiffUpdateCurve {
        VardiffUpdateCurve {
            start_pct: self.start_pct,
            floor_pct: self.floor_pct,
            half_life_secs: self.half_life_secs,
            power: self.power,
        }
    }
}

impl SearchBounds {
    fn random_candidate(self, rng: &mut SplitMix64) -> TuneParams {
        TuneParams {
            start_pct: self.start_pct.sample(rng) as f32,
            floor_pct: self.floor_pct.sample(rng) as f32,
            half_life_secs: self.half_life_secs.sample(rng) as f32,
            power: self.power.sample(rng) as f32,
        }
    }

    fn perturb(self, current: TuneParams, step: f64, rng: &mut SplitMix64) -> TuneParams {
        TuneParams {
            start_pct: self
                .start_pct
                .clamp(
                    current.start_pct as f64 + rng.normalish() * step * self.start_pct.width(),
                ) as f32,
            floor_pct: self
                .floor_pct
                .clamp(
                    current.floor_pct as f64 + rng.normalish() * step * self.floor_pct.width(),
                ) as f32,
            half_life_secs: self
                .half_life_secs
                .clamp(
                    current.half_life_secs as f64
                        + rng.normalish() * step * self.half_life_secs.width(),
                ) as f32,
            power: self
                .power
                .clamp(
                    current.power as f64 + rng.normalish() * step * self.power.width(),
                ) as f32,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn normalish(&mut self) -> f64 {
        let mut sum = 0.0;
        for _ in 0..6 {
            sum += self.next_f64();
        }
        (sum - 3.0) / 1.224744871391589
    }
}

fn main() {
    let trial_count = env_or("VARDIFF_SCAN_TRIALS", DEFAULT_TRIAL_COUNT / 10);
    let base_seed = env_or_seed("VARDIFF_SCAN_SEED", DEFAULT_BASELINE_SEED);
    let cells = default_cells();

    let max_evals = env_or("VARDIFF_SCAN_ITERS", 5000usize).max(1);
    let restart_count = env_or("VARDIFF_RESTARTS", 6usize).max(1);
    let patience = env_or("VARDIFF_PATIENCE", 60usize).max(1);
    let optimizer_seed = env_or_seed("VARDIFF_OPT_SEED", base_seed ^ 0x9e3779b97f4a7c15);

    let d = TuneParams::from_curve(VardiffUpdateCurve::default());
    let bounds = SearchBounds {
        start_pct: Range::from_env(
            "VARDIFF_SCAN_START_PCT",
            80.0,
            160.0,
        ),
        floor_pct: Range::from_env(
            "VARDIFF_SCAN_FLOOR_PCT",
            8.0,
            25.0,
        ),
        half_life_secs: Range::from_env(
            "VARDIFF_SCAN_HALF_LIFE_SECS",
            30.0,
            420.0,
        ),
        power: Range::from_env(
            "VARDIFF_SCAN_POWER",
            0.5,
            4.0,
        ),
};

    eprintln!(
        "Random-restart hill climbing for up to {} evaluations ({} cells x {} trials each, {} restarts)",
        max_evals,
        cells.len(),
        trial_count,
        restart_count
    );

    let vanilla_results = run_scan_vanilla(&cells, trial_count, base_seed);
    let vanilla_summary = summarize(&vanilla_results);
    eprintln!(
        "Vanilla baseline: SP50:{:.6} | SP99:{:.6} | CR:{:.2}% | CP50:{:.1}s | CP99:{:.1}s | RR:{:.6} | RP50:{:.1}s | RP99:{:.1}s | JP50:{:.6} | JM:{:.6} | JP99:{:.6}",
        vanilla_summary.settled_p50.mean,
        vanilla_summary.settled_p99.mean,
        vanilla_summary.conv_rate.mean * 100.0,
        vanilla_summary.conv_p50.mean,
        vanilla_summary.conv_p99.mean,
        vanilla_summary.react_rate.mean,
        vanilla_summary.react_p50.mean,
        vanilla_summary.react_p99.mean,
        vanilla_summary.jitter_p50.mean,
        vanilla_summary.jitter_mean.mean,
        vanilla_summary.jitter_p99.mean,
    );

    let started = Instant::now();
    let mut rng = SplitMix64::new(optimizer_seed);
    let mut evals = 0usize;
    let mut best: Option<(f64, TuneParams, MetricsSummary, f64, f64)> = None;
    let evals_per_restart = (max_evals / restart_count).max(1);

    for restart in 0..restart_count {
        if evals >= max_evals {
            break;
        }

        let mut current_params = if restart == 0 { d } else { bounds.random_candidate(&mut rng) };
        let current_results =
            run_scan_classic_with_params(&cells, trial_count, base_seed, current_params);
        let mut current_summary = summarize(&current_results);
        let mut current_breakdown = score_per_cell(&current_results, &vanilla_results);
        let mut current_reg_score = current_breakdown.reg;
        let mut current_beat_score = current_breakdown.beat;
        let mut current_score = current_reg_score - current_beat_score / (1.0 + current_reg_score);
        evals += 1;

        if best
            .as_ref()
            .map(|(best_score, _, _, _, _)| current_score < *best_score)
            .unwrap_or(true)
        {
            best = Some((
                current_score,
                current_params,
                current_summary,
                current_reg_score,
                current_beat_score,
            ));
            log_best(
                evals,
                max_evals,
                current_score,
                current_params,
                current_summary,
                current_reg_score,
                current_beat_score,
                current_breakdown,
                restart + 1,
                None,
            );
        }

        let mut step = env_or("VARDIFF_INITIAL_STEP", 1.0f64);
        let min_step = env_or("VARDIFF_MIN_STEP", 0.0001f64);
        let max_step = env_or("VARDIFF_MAX_STEP", 5.0f64);
        let mut misses = 0usize;
        let restart_eval_limit = evals.saturating_add(evals_per_restart).min(max_evals);

        while evals < restart_eval_limit {
            let proposal = bounds.perturb(current_params, step, &mut rng);
            let proposal_results =
                run_scan_classic_with_params(&cells, trial_count, base_seed, proposal);
            let proposal_summary = summarize(&proposal_results);
            let proposal_breakdown = score_per_cell(&proposal_results, &vanilla_results);
            let proposal_reg_score = proposal_breakdown.reg;
            let proposal_beat_score = proposal_breakdown.beat;
            let proposal_score = proposal_reg_score - proposal_beat_score / (1.0 + proposal_reg_score);
            evals += 1;

            if proposal_score < current_score {
                current_params = proposal;
                current_summary = proposal_summary;
                current_reg_score = proposal_reg_score;
                current_beat_score = proposal_beat_score;
                current_breakdown = proposal_breakdown;
                current_score = proposal_score;
                misses = 0;
                step = (step * 1.08).min(max_step);

                if best
                    .as_ref()
                    .map(|(best_score, _, _, _, _)| current_score < *best_score)
                    .unwrap_or(true)
                {
                    best = Some((
                        current_score,
                        current_params,
                        current_summary,
                        current_reg_score,
                        current_beat_score,
                    ));
                    log_best(
                        evals,
                        max_evals,
                        current_score,
                        current_params,
                        current_summary,
                        current_reg_score,
                        current_beat_score,
                        current_breakdown,
                        restart + 1,
                        Some(step),
                    );
                }
            } else {
                misses += 1;
                if misses >= patience {
                    step *= 0.5;
                    misses = 0;
                    if step < min_step {
                        break;
                    }
                }
            }
        }
    }

    let elapsed = started.elapsed().as_secs_f64();
    eprintln!("Search complete in {:.2}s after {} evaluations", elapsed, evals);

    if let Some((score, params, s, reg, beat)) = best {
        println!("BEST SCORE: {:.6}", score);
        println!("BEST REG_SCORE: {:.6}", reg);
        println!("BEST BEAT_SCORE: {:.6}", beat);
        println!(
            "BEST CURVE PARAMS: start_pct={:.3} floor_pct={:.3} half_life_secs={:.3} power={:.5}",
            params.start_pct,
            params.floor_pct,
            params.half_life_secs,
            params.power,
        );
        println!(
            "METRICS: SP50:{:.6} | SP99:{:.6} | CR:{:.2}% | CP50:{:.1}s | CP99:{:.1}s | RR:{:.6} | RP50:{:.1}s | RP99:{:.1}s | JP50:{:.6} | JM:{:.6} | JP99:{:.6}",
            s.settled_p50.mean,
            s.settled_p99.mean,
            s.conv_rate.mean * 100.0,
            s.conv_p50.mean,
            s.conv_p99.mean,
            s.react_rate.mean,
            s.react_p50.mean,
            s.react_p99.mean,
            s.jitter_p50.mean,
            s.jitter_mean.mean,
            s.jitter_p99.mean,
        );
    }
}

fn log_best(
    evals: usize,
    max_evals: usize,
    score: f64,
    p: TuneParams,
    s: MetricsSummary,
    reg_score: f64,
    beat_score: f64,
    breakdown: ScoreBreakdown,
    restart: usize,
    step: Option<f64>,
) {
    if let Some(step) = step {
        eprintln!(
            "[{}/{}] NEW BEST restart={} score={:.6} reg={:.6} beat={:.6} start={:.2} floor={:.2} half_life={:.2}s power={:.4} step={:.5} | cold(r={:.3},b={:.3}) stable(r={:.3},b={:.3}) stepR(r={:.3},b={:.3}) stepQ(r={:.3},b={:.3}) | SP50:{:.6} | SP99:{:.6} | CR:{:.2}% | CP50:{:.1}s | CP99:{:.1}s | RR:{:.6} | RP50:{:.1}s | RP99:{:.1}s | JP50:{:.6} | JM:{:.6} | JP99:{:.6}",
            evals,
            max_evals,
            restart,
            score,
            reg_score,
            beat_score,
            p.start_pct,
            p.floor_pct,
            p.half_life_secs,
            p.power,
            step,
            breakdown.cold_reg,
            breakdown.cold_beat,
            breakdown.stable_reg,
            breakdown.stable_beat,
            breakdown.step_reaction_reg,
            breakdown.step_reaction_beat,
            breakdown.step_quality_reg,
            breakdown.step_quality_beat,
            s.settled_p50.mean,
            s.settled_p99.mean,
            s.conv_rate.mean * 100.0,
            s.conv_p50.mean,
            s.conv_p99.mean,
            s.react_rate.mean,
            s.react_p50.mean,
            s.react_p99.mean,
            s.jitter_p50.mean,
            s.jitter_mean.mean,
            s.jitter_p99.mean,
        );
    } else {
        eprintln!(
            "[{}/{}] NEW BEST restart={} score={:.6} reg={:.6} beat={:.6} start={:.2} floor={:.2} half_life={:.2}s power={:.4} | cold(r={:.3},b={:.3}) stable(r={:.3},b={:.3}) stepR(r={:.3},b={:.3}) stepQ(r={:.3},b={:.3}) | SP50:{:.6} | SP99:{:.6} | CR:{:.2}% | CP50:{:.1}s | CP99:{:.1}s | RR:{:.6} | RP50:{:.1}s | RP99:{:.1}s | JP50:{:.6} | JM:{:.6} | JP99:{:.6}",
            evals,
            max_evals,
            restart,
            score,
            reg_score,
            beat_score,
            p.start_pct,
            p.floor_pct,
            p.half_life_secs,
            p.power,
            breakdown.cold_reg,
            breakdown.cold_beat,
            breakdown.stable_reg,
            breakdown.stable_beat,
            breakdown.step_reaction_reg,
            breakdown.step_reaction_beat,
            breakdown.step_quality_reg,
            breakdown.step_quality_beat,
            s.settled_p50.mean,
            s.settled_p99.mean,
            s.conv_rate.mean * 100.0,
            s.conv_p50.mean,
            s.conv_p99.mean,
            s.react_rate.mean,
            s.react_p50.mean,
            s.react_p99.mean,
            s.jitter_p50.mean,
            s.jitter_mean.mean,
            s.jitter_p99.mean,
        );
    }
}

fn run_scan_classic_with_params(
    cells: &[Cell],
    trial_count: usize,
    base_seed: u64,
    params: TuneParams,
) -> Vec<CellResult> {
    cells
        .iter()
        .enumerate()
        .map(|(idx, cell)| run_cell_with_classic_params(cell, trial_count, base_seed, idx as u64, params))
        .collect()
}

fn run_scan_vanilla(cells: &[Cell], trial_count: usize, base_seed: u64) -> Vec<CellResult> {
    cells
        .iter()
        .enumerate()
        .map(|(idx, cell)| run_cell_with_vanilla(cell, trial_count, base_seed, idx as u64))
        .collect()
}

fn run_cell_with_classic_params(
    cell: &Cell,
    trial_count: usize,
    base_seed: u64,
    cell_index: u64,
    params: TuneParams,
) -> CellResult {
    let (config, schedule) = cell.scenario.build(cell.shares_per_minute);
    let mut trials = Vec::with_capacity(trial_count);

    for trial_index in 0..trial_count {
        let seed = base_seed
            .wrapping_add(cell_index.wrapping_shl(20))
            .wrapping_add(trial_index as u64);
        let clock = Arc::new(MockClock::new(0));
        let vardiff = VardiffState::new_with_clock_and_update_curve(1.0, clock.clone(), params.to_curve())
            .expect("VardiffState construction should never fail");
        let trial = run_trial(vardiff, clock, config.clone(), &schedule, seed);
        trials.push(trial);
    }

    cell_result_from_trials(cell, &trials)
}

fn run_cell_with_vanilla(
    cell: &Cell,
    trial_count: usize,
    base_seed: u64,
    cell_index: u64,
) -> CellResult {
    let (config, schedule) = cell.scenario.build(cell.shares_per_minute);
    let mut trials = Vec::with_capacity(trial_count);

    for trial_index in 0..trial_count {
        let seed = base_seed
            .wrapping_add(cell_index.wrapping_shl(20))
            .wrapping_add(trial_index as u64);
        let clock = Arc::new(MockClock::new(0));
        let vardiff = VanillaVardiffState::new_with_clock(1.0, clock.clone())
            .expect("Vanilla VardiffState construction should never fail");
        let trial = run_trial(vardiff, clock, config.clone(), &schedule, seed);
        trials.push(trial);
    }

    cell_result_from_trials(cell, &trials)
}

fn cell_result_from_trials(cell: &Cell, trials: &[vardiff_sim::trial::Trial]) -> CellResult {
    let (convergence_rate, conv_dist) = convergence_time_distribution(trials, QUIET_WINDOW_SECS);
    let accuracy = settled_accuracy_distribution(trials);
    let jitter = jitter_distribution(
        trials,
        QUIET_WINDOW_SECS,
        SETTLE_BUFFER_SECS,
        MIN_SETTLED_WINDOW_SECS,
    );

    let (reaction_rate_opt, reaction_dist) = match cell.scenario {
        Scenario::Step { .. } => {
            let (rate, dist) = reaction_time_distribution(trials, STEP_EVENT_AT_SECS, REACT_WINDOW_SECS);
            (Some(rate), Some(dist))
        }
        _ => (None, None),
    };

    CellResult {
        shares_per_minute: cell.shares_per_minute,
        scenario_key: cell.scenario.key(),
        convergence_rate,
        convergence_p10_secs: conv_dist.p10(),
        convergence_p50_secs: conv_dist.p50(),
        convergence_p90_secs: conv_dist.p90(),
        convergence_p95_secs: conv_dist.p95(),
        convergence_p99_secs: conv_dist.p99(),
        settled_accuracy_p10: accuracy.p10(),
        settled_accuracy_p50: accuracy.p50(),
        settled_accuracy_p90: accuracy.p90(),
        settled_accuracy_p95: accuracy.p95(),
        settled_accuracy_p99: accuracy.p99(),
        jitter_p50_per_min: jitter.p50(),
        jitter_p90_per_min: jitter.p90(),
        jitter_p95_per_min: jitter.p95(),
        jitter_p99_per_min: jitter.p99(),
        jitter_mean_per_min: jitter.mean(),
        reaction_rate: reaction_rate_opt,
        reaction_p10_secs: reaction_dist.as_ref().and_then(|d| d.p10()),
        reaction_p50_secs: reaction_dist.as_ref().and_then(|d| d.p50()),
        reaction_p90_secs: reaction_dist.as_ref().and_then(|d| d.p90()),
        reaction_p99_secs: reaction_dist.as_ref().and_then(|d| d.p99()),
    }
}

#[derive(Debug, Clone, Copy)]
struct CellStats {
    mean: f64,
}

#[derive(Debug, Clone, Copy)]
struct MetricsSummary {
    settled_p50: CellStats,
    settled_p99: CellStats,
    conv_rate: CellStats,
    conv_p50: CellStats,
    conv_p99: CellStats,
    react_rate: CellStats,
    react_p50: CellStats,
    react_p99: CellStats,
    jitter_p50: CellStats,
    jitter_mean: CellStats,
    jitter_p99: CellStats,
}

fn stats_of<I>(iter: I) -> CellStats
where
    I: Iterator<Item = f64>,
{
    let values: Vec<f64> = iter.filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        return CellStats { mean: 0.0 };
    }
    CellStats {
        mean: values.iter().sum::<f64>() / values.len() as f64,
    }
}

fn summarize(results: &[CellResult]) -> MetricsSummary {
    MetricsSummary {
        settled_p50: stats_of(results.iter().filter_map(|r| r.settled_accuracy_p50)),
        settled_p99: stats_of(results.iter().filter_map(|r| r.settled_accuracy_p99)),
        conv_rate: stats_of(results.iter().map(|r| r.convergence_rate)),
        conv_p50: stats_of(results.iter().filter_map(|r| r.convergence_p50_secs)),
        conv_p99: stats_of(results.iter().filter_map(|r| r.convergence_p99_secs)),
        react_rate: stats_of(results.iter().filter_map(|r| r.reaction_rate)),
        react_p50: stats_of(results.iter().filter_map(|r| r.reaction_p50_secs)),
        react_p99: stats_of(results.iter().filter_map(|r| r.reaction_p99_secs)),
        jitter_p50: stats_of(results.iter().filter_map(|r| r.jitter_p50_per_min)),
        jitter_mean: stats_of(results.iter().filter_map(|r| r.jitter_mean_per_min)),
        jitter_p99: stats_of(results.iter().filter_map(|r| r.jitter_p99_per_min)),
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct GroupAcc {
    reg_sum: f64,
    beat_sum: f64,
    term_count: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ScoreBreakdown {
    reg: f64,
    beat: f64,
    cold_reg: f64,
    cold_beat: f64,
    stable_reg: f64,
    stable_beat: f64,
    step_reaction_reg: f64,
    step_reaction_beat: f64,
    step_quality_reg: f64,
    step_quality_beat: f64,
}

const SCALE_RATE: f64 = 0.05;
const SCALE_TIME: f64 = 60.0;
const SCALE_ACC_P50: f64 = 0.05;
const SCALE_ACC_P99: f64 = 0.10;
const SCALE_JMEAN: f64 = 0.01;
const SCALE_JP99: f64 = 0.05;
const MAX_METRIC_BEAT: f64 = 25.0;

fn lower_regression(delta: f64, tolerance: f64, scale: f64) -> f64 {
    let excess = delta - tolerance;
    if excess <= 0.0 {
        0.0
    } else {
        (excess / scale.max(1e-9)).powi(2)
    }
}

fn lower_beat(delta: f64, tolerance: f64, scale: f64) -> f64 {
    let gain = -delta - tolerance;
    if gain <= 0.0 {
        0.0
    } else {
        ((gain / scale.max(1e-9)).powi(2)).min(MAX_METRIC_BEAT)
    }
}

fn higher_regression(delta: f64, tolerance: f64, scale: f64) -> f64 {
    let shortfall = -delta - tolerance;
    if shortfall <= 0.0 {
        0.0
    } else {
        (shortfall / scale.max(1e-9)).powi(2)
    }
}

fn higher_beat(delta: f64, tolerance: f64, scale: f64) -> f64 {
    let gain = delta - tolerance;
    if gain <= 0.0 {
        0.0
    } else {
        ((gain / scale.max(1e-9)).powi(2)).min(MAX_METRIC_BEAT)
    }
}

fn add_lower(group: &mut GroupAcc, weight: f64, delta: Option<f64>, tolerance: f64, scale: f64, allow_beat: bool) {
    if let Some(d) = delta {
        group.reg_sum += weight * lower_regression(d, tolerance, scale);
        if allow_beat {
            group.beat_sum += weight * lower_beat(d, tolerance, scale);
        }
        group.term_count += 1.0;
    }
}

fn add_higher(group: &mut GroupAcc, weight: f64, delta: Option<f64>, tolerance: f64, scale: f64, allow_beat: bool) {
    if let Some(d) = delta {
        group.reg_sum += weight * higher_regression(d, tolerance, scale);
        if allow_beat {
            group.beat_sum += weight * higher_beat(d, tolerance, scale);
        }
        group.term_count += 1.0;
    }
}

fn group_norm(g: GroupAcc) -> (f64, f64) {
    let denom = g.term_count.max(1.0);
    (g.reg_sum / denom, g.beat_sum / denom)
}

fn score_per_cell(current: &[CellResult], vanilla: &[CellResult]) -> ScoreBreakdown {
    let vmap: BTreeMap<String, &CellResult> = vanilla
        .iter()
        .map(|r| (format!("spm_{}.{}", r.shares_per_minute as u32, r.scenario_key), r))
        .collect();
    let cmap: BTreeMap<String, &CellResult> = current
        .iter()
        .map(|r| (format!("spm_{}.{}", r.shares_per_minute as u32, r.scenario_key), r))
        .collect();

    let mut cold = GroupAcc::default();
    let mut stable = GroupAcc::default();
    let mut step_reaction = GroupAcc::default();
    let mut step_quality = GroupAcc::default();

    for (k, v) in &vmap {
        let Some(c) = cmap.get(k) else { continue };
        let is_cold = v.scenario_key.starts_with("cold_start");
        let is_stable = v.scenario_key.starts_with("stable");
        let is_step = v.scenario_key.starts_with("step_");

        let dd_cr = Some(c.convergence_rate - v.convergence_rate);
        let dd_cp50 = pair_delta(c.convergence_p50_secs, v.convergence_p50_secs);
        let dd_cp99 = pair_delta(c.convergence_p99_secs, v.convergence_p99_secs);
        let dd_sa50 = pair_delta(c.settled_accuracy_p50, v.settled_accuracy_p50);
        let dd_sa99 = pair_delta(c.settled_accuracy_p99, v.settled_accuracy_p99);
        let dd_jm = pair_delta(c.jitter_mean_per_min, v.jitter_mean_per_min);
        let dd_j99 = pair_delta(c.jitter_p99_per_min, v.jitter_p99_per_min);
        let dd_rr = pair_delta(c.reaction_rate, v.reaction_rate);
        let dd_rp50 = pair_delta(c.reaction_p50_secs, v.reaction_p50_secs);
        let dd_rp99 = pair_delta(c.reaction_p99_secs, v.reaction_p99_secs);

        if is_cold {
            add_higher(&mut cold, 8.0, dd_cr, 0.005, SCALE_RATE, true);
            add_lower(&mut cold, 6.0, dd_cp50, 30.0, SCALE_TIME, true);
            add_lower(&mut cold, 8.0, dd_cp99, 60.0, SCALE_TIME, true);

            // Regression-only cold-start quality terms.
            add_lower(&mut cold, 1.0, dd_sa50, 0.08, SCALE_ACC_P50, false);
            add_lower(&mut cold, 2.0, dd_sa99, 0.10, SCALE_ACC_P99, false);
            add_lower(&mut cold, 1.0, dd_jm, 0.005, SCALE_JMEAN, false);
            add_lower(&mut cold, 1.0, dd_j99, 0.02, SCALE_JP99, false);
        } else if is_stable {
            add_lower(&mut stable, 3.0, dd_sa50, 0.02, SCALE_ACC_P50, true);
            add_lower(&mut stable, 4.0, dd_sa99, 0.05, SCALE_ACC_P99, true);
            add_lower(&mut stable, 4.0, dd_jm, 0.001, SCALE_JMEAN, true);
            add_lower(&mut stable, 5.0, dd_j99, 0.01, SCALE_JP99, true);
        } else if is_step {
            add_higher(&mut step_reaction, 10.0, dd_rr, 0.03, SCALE_RATE, true);
            add_lower(&mut step_reaction, 6.0, dd_rp50, 30.0, SCALE_TIME, true);
            add_lower(&mut step_reaction, 8.0, dd_rp99, 60.0, SCALE_TIME, true);

            add_lower(&mut step_quality, 2.0, dd_sa50, 0.05, SCALE_ACC_P50, true);
            add_lower(&mut step_quality, 2.0, dd_sa99, 0.10, SCALE_ACC_P99, true);
            add_lower(&mut step_quality, 1.0, dd_jm, 0.002, SCALE_JMEAN, true);
            add_lower(&mut step_quality, 1.0, dd_j99, 0.02, SCALE_JP99, true);
        }
    }

    let (cold_reg, cold_beat) = group_norm(cold);
    let (stable_reg, stable_beat) = group_norm(stable);
    let (step_reaction_reg, step_reaction_beat) = group_norm(step_reaction);
    let (step_quality_reg, step_quality_beat) = group_norm(step_quality);

    let reg = 1.0 * cold_reg
        + 1.5 * stable_reg
        + 2.5 * step_reaction_reg
        + 1.5 * step_quality_reg;

    let beat = 1.0 * cold_beat
        + 1.0 * stable_beat
        + 2.0 * step_reaction_beat
        + 0.75 * step_quality_beat;

    ScoreBreakdown {
        reg,
        beat,
        cold_reg,
        cold_beat,
        stable_reg,
        stable_beat,
        step_reaction_reg,
        step_reaction_beat,
        step_quality_reg,
        step_quality_beat,
    }
}

fn pair_delta(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x - y),
        _ => None,
    }
}

fn env_or<T: std::str::FromStr>(var: &str, default: T) -> T {
    env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_or_seed(var: &str, default: u64) -> u64 {
    if let Ok(s) = env::var(var) {
        if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            return u64::from_str_radix(hex, 16).unwrap_or(default);
        }
        return s.parse().unwrap_or(default);
    }
    default
}
