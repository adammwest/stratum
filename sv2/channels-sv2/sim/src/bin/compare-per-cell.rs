use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::env;
use std::fs;

#[derive(Debug, Clone, Default)]
struct CellMetrics {
    scenario: String,
    convergence_rate: Option<f64>,
    convergence_p50_secs: Option<f64>,
    convergence_p99_secs: Option<f64>,
    settled_accuracy_p50: Option<f64>,
    settled_accuracy_p99: Option<f64>,
    jitter_mean_per_min: Option<f64>,
    jitter_p99_per_min: Option<f64>,
    reaction_rate: Option<f64>,
    reaction_p50_secs: Option<f64>,
    reaction_p99_secs: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellKind {
    ColdStart,
    Stable,
    Step,
    Other,
}

#[derive(Debug, Clone)]
struct CellDelta {
    cell: String,
    kind: CellKind,
    d_cr: Option<f64>,
    d_cp50: Option<f64>,
    d_cp99: Option<f64>,
    d_sa_p50: Option<f64>,
    d_sa_p99: Option<f64>,
    d_jmean: Option<f64>,
    d_jp99: Option<f64>,
    d_rr: Option<f64>,
    d_rp50: Option<f64>,
    d_rp99: Option<f64>,
    score: f64,
    beat_score: f64,
    warns: Vec<String>,
    fails: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct MetricZ {
    cr: f64,
    cp50: f64,
    cp99: f64,
    sa_p50: f64,
    sa_p99: f64,
    jmean: f64,
    jp99: f64,
    rr: f64,
    rp50: f64,
    rp99: f64,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let vanilla_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "vardiff_baseline.toml".to_string());
    let bocpd_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "vardiff_best.toml".to_string());

    let vanilla = parse_cells(&vanilla_path).expect("failed to parse vanilla TOML");
    let bocpd = parse_cells(&bocpd_path).expect("failed to parse BOCPD TOML");

    let mut all: Vec<CellDelta> = Vec::new();
    let mut hard_fail = false;

    for (cell, v) in &vanilla {
        let Some(b) = bocpd.get(cell) else { continue };
        let kind = cell_kind(&v.scenario);

        let mut d = CellDelta {
            cell: cell.clone(),
            kind,
            d_cr: delta(b.convergence_rate, v.convergence_rate),
            d_cp50: delta(b.convergence_p50_secs, v.convergence_p50_secs),
            d_cp99: delta(b.convergence_p99_secs, v.convergence_p99_secs),
            d_sa_p50: delta(b.settled_accuracy_p50, v.settled_accuracy_p50),
            d_sa_p99: delta(b.settled_accuracy_p99, v.settled_accuracy_p99),
            d_jmean: delta(b.jitter_mean_per_min, v.jitter_mean_per_min),
            d_jp99: delta(b.jitter_p99_per_min, v.jitter_p99_per_min),
            d_rr: delta(b.reaction_rate, v.reaction_rate),
            d_rp50: delta(b.reaction_p50_secs, v.reaction_p50_secs),
            d_rp99: delta(b.reaction_p99_secs, v.reaction_p99_secs),
            score: 0.0,
            beat_score: 0.0,
            warns: Vec::new(),
            fails: Vec::new(),
        };

        apply_guards(&mut d);
        if !d.fails.is_empty() {
            hard_fail = true;
        }
        all.push(d);
    }

    let z = metric_z(&all);
    for d in &mut all {
        let (reg, beat) = cell_score_z(d, z);
        d.score = reg;
        d.beat_score = beat;
    }

    all.sort_by(|a, b| a.cell.cmp(&b.cell));

    print_table("Full Per-Cell Diff", &all);

    let regressions: Vec<CellDelta> = all
        .iter()
        .filter(|c| has_any_regression(c))
        .cloned()
        .collect();
    print_table("Regression Table (Worse Cells)", &regressions);

    let majors: Vec<CellDelta> = all
        .iter()
        .filter(|c| !c.fails.is_empty())
        .cloned()
        .collect();
    print_table("Major Regression Table", &majors);

    print_worst_lists(&all);

    let total_score: f64 = all.iter().map(|c| c.score).sum();
    let near_zero = total_score <= 1e-9;
    let accepted = !hard_fail && near_zero;

    println!("\nFinal Verdict: {}", if accepted { "PASS" } else { "REJECT" });
    println!("Total Per-Cell Regression Score: {:.6}", total_score);
    let total_beat_score: f64 = all.iter().map(|c| c.beat_score).sum();
    println!("Total Per-Cell Beat Score: {:.6}", total_beat_score);
    println!("Hard-Guard Failures: {}", if hard_fail { "YES" } else { "NO" });
    println!(
        "Criteria: no hard-guard fails, near-zero per-cell regression score, no hidden low-SPM ±50% sacrifice."
    );
}

fn delta(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x - y),
        _ => None,
    }
}

fn cell_kind(scenario: &str) -> CellKind {
    if scenario.starts_with("cold_start") {
        CellKind::ColdStart
    } else if scenario.starts_with("stable") {
        CellKind::Stable
    } else if scenario.starts_with("step_") {
        CellKind::Step
    } else {
        CellKind::Other
    }
}

fn parse_cells(path: &str) -> Result<BTreeMap<String, CellMetrics>, String> {
    let txt = fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut section: Option<String> = None;
    let mut cells: BTreeMap<String, CellMetrics> = BTreeMap::new();

    for raw in txt.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(sec) = line.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
            section = Some(sec.to_string());
            continue;
        }
        let Some(sec) = &section else { continue };
        if !sec.starts_with("cell.") {
            continue;
        }
        let cell = sec.trim_start_matches("cell.").to_string();
        let Some((k, vraw)) = line.split_once('=') else { continue };
        let k = k.trim();
        let vraw = vraw.trim();
        let entry = cells.entry(cell).or_default();

        match k {
            "scenario" => {
                entry.scenario = parse_string(vraw).unwrap_or_default();
            }
            "convergence_rate" => entry.convergence_rate = parse_num(vraw),
            "convergence_p50_secs" => entry.convergence_p50_secs = parse_num(vraw),
            "convergence_p99_secs" => entry.convergence_p99_secs = parse_num(vraw),
            "settled_accuracy_p50" => entry.settled_accuracy_p50 = parse_num(vraw),
            "settled_accuracy_p99" => entry.settled_accuracy_p99 = parse_num(vraw),
            "jitter_mean_per_min" => entry.jitter_mean_per_min = parse_num(vraw),
            "jitter_p99_per_min" => entry.jitter_p99_per_min = parse_num(vraw),
            "reaction_rate" => entry.reaction_rate = parse_num(vraw),
            "reaction_p50_secs" => entry.reaction_p50_secs = parse_num(vraw),
            "reaction_p99_secs" => entry.reaction_p99_secs = parse_num(vraw),
            _ => {}
        }
    }

    Ok(cells)
}

fn parse_num(s: &str) -> Option<f64> {
    s.parse::<f64>().ok()
}

fn parse_string(s: &str) -> Option<String> {
    s.strip_prefix('"')
        .and_then(|x| x.strip_suffix('"'))
        .map(|x| x.to_string())
}

fn apply_guards(d: &mut CellDelta) {
    match d.kind {
        CellKind::ColdStart => {
            fail_if(d, d.d_cr, |x| x < -0.01, "cold: dCR < -0.01");
            fail_if(d, d.d_cp50, |x| x > 60.0, "cold: dCP50s > +60");
            fail_if(d, d.d_cp99, |x| x > 120.0, "cold: dCP99s > +120");
            warn_if(d, d.d_sa_p50, |x| x > 0.03, "cold: WARN dSA_p50 > +0.03");
            fail_if(d, d.d_sa_p50, |x| x > 0.08, "cold: dSA_p50 > +0.08");
        }
        CellKind::Stable => {
            fail_if(d, d.d_sa_p50, |x| x > 0.02, "stable: dSA_p50 > +0.02");
            fail_if(d, d.d_sa_p99, |x| x > 0.05, "stable: dSA_p99 > +0.05");
            fail_if(d, d.d_jmean, |x| x > 0.001, "stable: dJmean > +0.001");
            fail_if(d, d.d_jp99, |x| x > 0.01, "stable: dJP99 > +0.01");
            warn_if(
                d,
                d.d_jmean,
                |x| x > 0.0,
                "stable: jitter increase is suspicious",
            );
        }
        CellKind::Step => {
            fail_if(d, d.d_rr, |x| x < -0.05, "step: dRR < -0.05");
            fail_if(d, d.d_rp50, |x| x > 60.0, "step: dRP50s > +60");
            fail_if(d, d.d_rp99, |x| x > 120.0, "step: dRP99s > +120");
            fail_if(d, d.d_sa_p50, |x| x > 0.05, "step: dSA_p50 > +0.05");
            fail_if(d, d.d_sa_p99, |x| x > 0.10, "step: dSA_p99 > +0.10");
            fail_if(d, d.d_jmean, |x| x > 0.002, "step: dJmean > +0.002");
            fail_if(d, d.d_jp99, |x| x > 0.02, "step: dJP99 > +0.02");

            if is_pm_50_step(&d.cell) {
                fail_if(d, d.d_rr, |x| x < -0.03, "hard: ±50% step dRR < -0.03");
                if is_low_spm(&d.cell) {
                    fail_if(
                        d,
                        d.d_sa_p50,
                        |x| x > 0.10,
                        "hard: low-SPM ±50% step dSA_p50 > +0.10",
                    );
                }
            }
        }
        CellKind::Other => {}
    }
}

fn is_pm_50_step(cell: &str) -> bool {
    cell.contains("step_minus_50") || cell.contains("step_plus_50")
}

fn is_low_spm(cell: &str) -> bool {
    cell.starts_with("spm_6.") || cell.starts_with("spm_12.")
}

fn fail_if<F: Fn(f64) -> bool>(d: &mut CellDelta, v: Option<f64>, pred: F, msg: &str) {
    if let Some(x) = v {
        if pred(x) {
            d.fails.push(msg.to_string());
        }
    }
}

fn warn_if<F: Fn(f64) -> bool>(d: &mut CellDelta, v: Option<f64>, pred: F, msg: &str) {
    if let Some(x) = v {
        if pred(x) {
            d.warns.push(msg.to_string());
        }
    }
}

fn lower_regression(delta: f64, tolerance: f64, z: f64) -> f64 {
    let excess = delta - tolerance;
    if excess <= 0.0 {
        return 0.0;
    }
    let denom = z.max(1e-9);
    (excess / denom).powi(2)
}

fn higher_regression(delta: f64, tolerance: f64, z: f64) -> f64 {
    let shortfall = -delta - tolerance;
    if shortfall <= 0.0 {
        return 0.0;
    }
    let denom = z.max(1e-9);
    (shortfall / denom).powi(2)
}

fn lower_beat(delta: f64, tolerance: f64, z: f64) -> f64 {
    let gain = -delta - tolerance;
    if gain <= 0.0 {
        return 0.0;
    }
    let denom = z.max(1e-9);
    (gain / denom).powi(2)
}

fn higher_beat(delta: f64, tolerance: f64, z: f64) -> f64 {
    let gain = delta - tolerance;
    if gain <= 0.0 {
        return 0.0;
    }
    let denom = z.max(1e-9);
    (gain / denom).powi(2)
}

fn cell_score_z(d: &CellDelta, z: MetricZ) -> (f64, f64) {
    let mut reg = 0.0;
    let mut beat = 0.0;

    match d.kind {
        CellKind::ColdStart => {
            add_h_pair(&mut reg, &mut beat, 3.0, d.d_cr, 0.01, z.cr);
            add_l_pair(&mut reg, &mut beat, 2.0, d.d_cp50, 60.0, z.cp50);
            add_l_pair(&mut reg, &mut beat, 3.0, d.d_cp99, 120.0, z.cp99);
            add_l_pair(&mut reg, &mut beat, 1.0, d.d_sa_p50, 0.08, z.sa_p50);
            add_l_pair(&mut reg, &mut beat, 2.0, d.d_sa_p99, 0.0, z.sa_p99);
            add_l_pair(&mut reg, &mut beat, 1.0, d.d_jmean, 0.0, z.jmean);
            add_l_pair(&mut reg, &mut beat, 1.0, d.d_jp99, 0.0, z.jp99);
        }
        CellKind::Stable => {
            add_l_pair(&mut reg, &mut beat, 3.0, d.d_sa_p50, 0.02, z.sa_p50);
            add_l_pair(&mut reg, &mut beat, 4.0, d.d_sa_p99, 0.05, z.sa_p99);
            add_l_pair(&mut reg, &mut beat, 4.0, d.d_jmean, 0.001, z.jmean);
            add_l_pair(&mut reg, &mut beat, 5.0, d.d_jp99, 0.01, z.jp99);
        }
        CellKind::Step => {
            add_h_pair(&mut reg, &mut beat, 5.0, d.d_rr, 0.05, z.rr);
            add_l_pair(&mut reg, &mut beat, 3.0, d.d_rp50, 60.0, z.rp50);
            add_l_pair(&mut reg, &mut beat, 4.0, d.d_rp99, 120.0, z.rp99);
            add_l_pair(&mut reg, &mut beat, 4.0, d.d_sa_p50, 0.05, z.sa_p50);
            add_l_pair(&mut reg, &mut beat, 5.0, d.d_sa_p99, 0.10, z.sa_p99);
            add_l_pair(&mut reg, &mut beat, 2.0, d.d_jmean, 0.002, z.jmean);
            add_l_pair(&mut reg, &mut beat, 3.0, d.d_jp99, 0.02, z.jp99);
        }
        CellKind::Other => {}
    }

    (reg, beat)
}

fn add_l_pair(reg: &mut f64, beat: &mut f64, w: f64, delta: Option<f64>, tol: f64, z: f64) {
    if let Some(d) = delta {
        *reg += w * lower_regression(d, tol, z);
        *beat += w * lower_beat(d, tol, z);
    }
}

fn add_h_pair(reg: &mut f64, beat: &mut f64, w: f64, delta: Option<f64>, tol: f64, z: f64) {
    if let Some(d) = delta {
        *reg += w * higher_regression(d, tol, z);
        *beat += w * higher_beat(d, tol, z);
    }
}

fn has_any_regression(c: &CellDelta) -> bool {
    c.score > 0.0 || !c.fails.is_empty() || !c.warns.is_empty()
}

fn print_table(title: &str, rows: &[CellDelta]) {
    println!("\n{}", title);
    println!("cell | dCR | dCP50s | dCP99s | dSA_p50 | dSA_p99 | dJmean | dJP99 | dRR | dRP50s | dRP99s | reg_score | beat_score | flags");
    for r in rows {
        let mut flags = Vec::new();
        if !r.warns.is_empty() {
            flags.push(format!("WARN:{}", r.warns.join(";")));
        }
        if !r.fails.is_empty() {
            flags.push(format!("FAIL:{}", r.fails.join(";")));
        }

        println!(
            "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.6} | {:.6} | {}",
            r.cell,
            fmt_opt(r.d_cr, 3),
            fmt_opt(r.d_cp50, 0),
            fmt_opt(r.d_cp99, 0),
            fmt_opt(r.d_sa_p50, 4),
            fmt_opt(r.d_sa_p99, 4),
            fmt_opt(r.d_jmean, 4),
            fmt_opt(r.d_jp99, 4),
            fmt_opt(r.d_rr, 3),
            fmt_opt(r.d_rp50, 0),
            fmt_opt(r.d_rp99, 0),
            r.score,
            r.beat_score,
            if flags.is_empty() { "-".to_string() } else { flags.join(" | ") }
        );
    }
}

fn metric_z(rows: &[CellDelta]) -> MetricZ {
    MetricZ {
        cr: z_std(rows.iter().filter_map(|r| r.d_cr)),
        cp50: z_std(rows.iter().filter_map(|r| r.d_cp50)),
        cp99: z_std(rows.iter().filter_map(|r| r.d_cp99)),
        sa_p50: z_std(rows.iter().filter_map(|r| r.d_sa_p50)),
        sa_p99: z_std(rows.iter().filter_map(|r| r.d_sa_p99)),
        jmean: z_std(rows.iter().filter_map(|r| r.d_jmean)),
        jp99: z_std(rows.iter().filter_map(|r| r.d_jp99)),
        rr: z_std(rows.iter().filter_map(|r| r.d_rr)),
        rp50: z_std(rows.iter().filter_map(|r| r.d_rp50)),
        rp99: z_std(rows.iter().filter_map(|r| r.d_rp99)),
    }
}

fn z_std<I: Iterator<Item = f64>>(it: I) -> f64 {
    let vals: Vec<f64> = it.collect();
    if vals.is_empty() {
        return 1.0;
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / vals.len() as f64;
    var.sqrt().max(1e-6)
}

fn fmt_opt(v: Option<f64>, p: usize) -> String {
    match v {
        Some(x) => format!("{:+.*}", p, x),
        None => "-".to_string(),
    }
}

fn print_worst_lists(rows: &[CellDelta]) {
    fn topk<F: Fn(&CellDelta) -> Option<f64>>(rows: &[CellDelta], f: F) -> Vec<(String, f64)> {
        let mut v: Vec<(String, f64)> = rows
            .iter()
            .filter_map(|r| f(r).map(|x| (r.cell.clone(), x)))
            .filter(|(_, x)| *x > 0.0)
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        v.truncate(5);
        v
    }

    let sa = topk(rows, |r| r.d_sa_p50.or(r.d_sa_p99));
    let rr = topk(rows, |r| r.d_rr.map(|x| -x));
    let rp = topk(rows, |r| {
        let a = r.d_rp50.unwrap_or(0.0).max(0.0);
        let b = r.d_rp99.unwrap_or(0.0).max(0.0);
        Some(a.max(b))
    });
    let jit = topk(rows, |r| {
        let a = r.d_jmean.unwrap_or(0.0).max(0.0);
        let b = r.d_jp99.unwrap_or(0.0).max(0.0);
        Some(a.max(b))
    });

    print_top("Worst Settled Accuracy Regressions", &sa);
    print_top("Worst Reaction Rate Regressions", &rr);
    print_top("Worst Reaction p50/p99 Regressions", &rp);
    print_top("Worst Jitter Regressions", &jit);
}

fn print_top(title: &str, rows: &[(String, f64)]) {
    println!("\n{}", title);
    if rows.is_empty() {
        println!("- none");
        return;
    }
    for (cell, v) in rows {
        println!("- {}: {:.6}", cell, v);
    }
}
