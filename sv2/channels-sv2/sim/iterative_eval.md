# Algorithm comparison (1000 trials/cell, SPM=6–30)

Algorithms sorted by mean operational fitness (best first).
**Bold** = best value at that SPM for that metric.

## Overall ranking

| Rank | Algorithm | Mean Fitness |
| --- | --- | --- |
| 1 | EWMA-AdaCUSUM-tau120-s15-f5-eta50 ★ | 0.676 |
| 2 | EWMA-AdaCUSUM-tau120-s15-f5-eta40 | 0.634 |
| 3 | EWMA-AdaCUSUM-tau120-s15-f5-eta30 | 0.585 |
| 4 | EWMA-AdaCUSUM-tau120-s15-f5-eta20 | 0.583 |
| 5 | FullRemedy | 0.565 |

## Reaction rate at -10% step (higher = better)

| SPM | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta40 | EWMA-AdaCUSUM-tau120-s15-f5-eta30 | EWMA-AdaCUSUM-tau120-s15-f5-eta20 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | **85.9%** | 84.3% | **85.6%** | 84.9% | 32.9% |
| 8 | **76.1%** | **75.6%** | **75.6%** | 74.8% | 30.0% |
| 10 | **67.5%** | **67.8%** | **68.0%** | 67.2% | 33.0% |
| 12 | 61.9% | **62.7%** | 61.9% | 61.2% | 34.1% |
| 15 | **55.2%** | 54.6% | 54.2% | 54.1% | 34.2% |
| 20 | **49.6%** | **49.3%** | 47.8% | 46.6% | 35.9% |
| 30 | **48.7%** | 48.0% | 47.0% | 46.6% | 37.0% |

## Reaction rate at -50% step (higher = better)

| SPM | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta40 | EWMA-AdaCUSUM-tau120-s15-f5-eta30 | EWMA-AdaCUSUM-tau120-s15-f5-eta20 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | **98.7%** | **99.0%** | **99.2%** | **99.5%** | 87.1% |
| 8 | **99.1%** | **99.2%** | **99.5%** | **99.5%** | 92.3% |
| 10 | **99.4%** | **99.7%** | **99.9%** | **99.9%** | 95.4% |
| 12 | **99.2%** | **99.4%** | **99.2%** | **99.4%** | 97.5% |
| 15 | **99.9%** | **99.8%** | **99.9%** | **99.8%** | **99.6%** |
| 20 | **99.3%** | **99.8%** | **99.9%** | **99.9%** | **99.9%** |
| 30 | **99.2%** | **99.3%** | **99.5%** | **99.5%** | **100.0%** |

## Cold-start convergence time p50 (lower = better)

| SPM | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta40 | EWMA-AdaCUSUM-tau120-s15-f5-eta30 | EWMA-AdaCUSUM-tau120-s15-f5-eta20 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | **8m** | 9m | 10m | 14m | 9m |
| 8 | **6m** | 8m | 10m | 14m | 9m |
| 10 | **6m** | 8m | 10m | 14m | 10m |
| 12 | **6m** | 7m | 10m | 14m | 10m |
| 15 | **5m** | 7m | 10m | 14m | 11m |
| 20 | **5m** | 7m | 9m | 14m | 11m |
| 30 | **1m** | 5m | 8m | 13m | 11m |

## Cold-start convergence rate (higher = better)

| SPM | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta40 | EWMA-AdaCUSUM-tau120-s15-f5-eta30 | EWMA-AdaCUSUM-tau120-s15-f5-eta20 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | 93.7% | 91.5% | 91.8% | 86.4% | **99.9%** |
| 8 | 98.6% | 98.5% | 98.4% | 93.8% | **100.0%** |
| 10 | **100.0%** | **99.8%** | **99.6%** | 96.3% | **100.0%** |
| 12 | **100.0%** | **100.0%** | **99.9%** | 96.8% | **99.9%** |
| 15 | **100.0%** | **99.9%** | **99.9%** | 97.4% | **100.0%** |
| 20 | **100.0%** | **100.0%** | **100.0%** | 98.7% | **100.0%** |
| 30 | **100.0%** | **100.0%** | **100.0%** | **99.8%** | **100.0%** |

## Stable-load jitter (fires/min) (lower = better)

| SPM | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta40 | EWMA-AdaCUSUM-tau120-s15-f5-eta30 | EWMA-AdaCUSUM-tau120-s15-f5-eta20 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | 0.175 | 0.171 | 0.171 | 0.169 | **0.032** |
| 8 | 0.143 | 0.141 | 0.139 | 0.137 | **0.030** |
| 10 | 0.125 | 0.123 | 0.119 | 0.116 | **0.027** |
| 12 | 0.107 | 0.104 | 0.102 | 0.099 | **0.026** |
| 15 | 0.088 | 0.086 | 0.082 | 0.079 | **0.023** |
| 20 | 0.068 | 0.062 | 0.058 | 0.057 | **0.020** |
| 30 | 0.039 | 0.033 | 0.027 | 0.022 | **0.014** |

## Ramp target overshoot p99 (lower = better)

| SPM | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta40 | EWMA-AdaCUSUM-tau120-s15-f5-eta30 | EWMA-AdaCUSUM-tau120-s15-f5-eta20 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | 45.6% | 38.2% | 31.2% | 18.9% | **5.5%** |
| 8 | 35.6% | 31.1% | 22.4% | 13.3% | **2.3%** |
| 10 | 30.5% | 26.5% | 18.1% | 9.5% | **3.3%** |
| 12 | 26.4% | 21.5% | 15.6% | 7.7% | **3.2%** |
| 15 | 24.1% | 17.6% | 12.6% | 3.4% | **2.5%** |
| 20 | 18.0% | 13.0% | 6.7% | **0.0%** | **1.3%** |
| 30 | 13.0% | 8.1% | **1.7%** | **0.0%** | **0.0%** |

## Settled accuracy p50 (lower = better)

| SPM | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta40 | EWMA-AdaCUSUM-tau120-s15-f5-eta30 | EWMA-AdaCUSUM-tau120-s15-f5-eta20 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | 9.0% | 8.2% | 7.2% | **5.9%** | 6.6% |
| 8 | 7.7% | 7.2% | 6.0% | **5.2%** | 5.7% |
| 10 | 7.7% | 6.4% | 5.2% | **4.2%** | 5.3% |
| 12 | 6.7% | 5.9% | 4.9% | **3.8%** | 4.8% |
| 15 | 5.8% | 4.8% | 4.0% | **3.3%** | 4.3% |
| 20 | 4.7% | 4.2% | 3.4% | **2.6%** | 3.9% |
| 30 | 6.6% | 5.5% | 4.3% | **3.0%** | 3.3% |

