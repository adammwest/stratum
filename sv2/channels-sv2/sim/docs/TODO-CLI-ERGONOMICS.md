# Follow-up: CLI ergonomics overhaul

The current workflow for running vardiff investigations requires
editing binary source code, recompiling, and reading raw markdown.
This should be replaced with a config-driven CLI.

## Proposed interface

```bash
# Compare algorithms (sorted by fitness, winners highlighted)
cargo run --release --bin vardiff-compare -- \
    --algorithms full_remedy,ada_cusum_050,ada_cusum_020 \
    --spm 6,8,10,12,15,20,30 \
    --trials 1000 \
    --format table  # or: json, csv, markdown

# Sweep a parameter
cargo run --release --bin vardiff-sweep -- \
    --base ada_cusum \
    --vary eta=0.2,0.3,0.4,0.5 \
    --spm 6,12,30 \
    --trials 500

# Regenerate baselines for all registered algorithms
cargo run --release --bin vardiff-baseline -- --all

# Single-trial trace
cargo run --release --bin vardiff-trace -- \
    ada_cusum_050 cold_start --spm 12 --seed 0xCAFE
```

## Algorithm registry (name → factory)

Algorithms should be addressable by short name rather than requiring
the user to know the full factory function signature:

```
full_remedy         → EwmaEstimator(120) + PoissonCI + PartialRetarget(0.2)
ada_cusum_020       → EwmaEstimator(120) + AdaptiveCusum(1.5, 0.05) + PartialRetarget(0.2)
ada_cusum_050       → EwmaEstimator(120) + AdaptiveCusum(1.5, 0.05) + PartialRetarget(0.5)
classic             → CumulativeCounter + StepFunction + FullRetargetWithClamp
parametric          → CumulativeCounter + PoissonCI + FullRetargetWithClamp
vardiff_state       → VardiffState (production reference)
```

Custom compositions via inline spec:
```bash
--algorithm "ewma(60)+cusum(2.0,0.05)+partial(0.3)"
```

## Share rate presets

```bash
--spm operational   # 6,8,10,12,15,20,30 (default)
--spm low           # 2,3,4,6
--spm full          # 2,3,4,6,8,10,12,15,20,25,30
--spm 6,12          # explicit
```

## Output improvements

- Sort algorithms by operational_fitness (best first)
- Bold per-metric winners at each SPM
- Show rank delta vs baseline (e.g., "+3.2pp" vs FullRemedy)
- Summary line: "Algorithm X dominates at SPM Y-Z, loses at SPM W"
- Optional: terminal color output for quick scanning

## Baseline management

```bash
# Show which baselines are stale (different from current code)
vardiff-baseline --check

# Regenerate only stale baselines
vardiff-baseline --update

# Regenerate all
vardiff-baseline --all
```

## Implementation priority

1. Algorithm name registry (lookup table: name → AlgorithmSpec)
2. CLI argument parsing for compare/sweep/trace
3. Output formatting (sort + bold + rank)
4. Share rate presets
5. Baseline management commands
