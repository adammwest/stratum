# Algorithm comparison (1000 trials/cell, SPM=6–30)

Algorithms sorted by mean operational fitness (best first).
**Bold** = best value at that SPM for that metric.

## Overall ranking

| Rank | Algorithm | Mean Fitness |
| --- | --- | --- |
| 1 | EWMA-AsymCUSUM-tau120-s15-f5-t30-eta50 ★ | 0.751 |
| 2 | EWMA-AsymCUSUM-tau120-s15-f5-t20-eta50 | 0.727 |
| 3 | EWMA-AsymCUSUM-tau120-s15-f5-t15-eta50 | 0.707 |
| 4 | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | 0.676 |
| 5 | FullRemedy | 0.565 |

## Reaction rate at -10% step (higher = better)

| SPM | EWMA-AsymCUSUM-tau120-s15-f5-t30-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t20-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t15-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | 64.7% | 74.5% | 77.8% | **85.9%** | 32.9% |
| 8 | 56.3% | 68.3% | 70.5% | **76.1%** | 30.0% |
| 10 | 47.6% | 56.8% | 62.3% | **67.5%** | 33.0% |
| 12 | 44.3% | 49.3% | 56.1% | **61.9%** | 34.1% |
| 15 | 42.8% | 46.9% | 49.3% | **55.2%** | 34.2% |
| 20 | 47.4% | 47.7% | 47.6% | **49.6%** | 35.9% |
| 30 | **49.9%** | **49.9%** | **50.0%** | 48.7% | 37.0% |

## Reaction rate at -50% step (higher = better)

| SPM | EWMA-AsymCUSUM-tau120-s15-f5-t30-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t20-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t15-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | **98.4%** | **98.5%** | **98.7%** | **98.7%** | 87.1% |
| 8 | **98.7%** | **98.8%** | **99.1%** | **99.1%** | 92.3% |
| 10 | **99.0%** | **99.1%** | **99.4%** | **99.4%** | 95.4% |
| 12 | **99.0%** | **99.0%** | **99.0%** | **99.2%** | 97.5% |
| 15 | **99.8%** | **99.8%** | **99.8%** | **99.9%** | **99.6%** |
| 20 | **99.4%** | **99.4%** | **99.4%** | **99.3%** | **99.9%** |
| 30 | **99.3%** | **99.3%** | **99.3%** | **99.2%** | **100.0%** |

## Cold-start convergence time p50 (lower = better)

| SPM | EWMA-AsymCUSUM-tau120-s15-f5-t30-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t20-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t15-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | **4m** | 5m | 7m | 8m | 9m |
| 8 | 5m | **4m** | 6m | 6m | 9m |
| 10 | 5m | **4m** | 5m | 6m | 10m |
| 12 | **4m** | **4m** | **4m** | 6m | 10m |
| 15 | **1m** | 4m | 4m | 5m | 11m |
| 20 | **1m** | 4m | 3m | 5m | 11m |
| 30 | **1m** | **1m** | **1m** | **1m** | 11m |

## Cold-start convergence rate (higher = better)

| SPM | EWMA-AsymCUSUM-tau120-s15-f5-t30-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t20-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t15-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | **100.0%** | **99.4%** | 98.3% | 93.7% | **99.9%** |
| 8 | **100.0%** | **100.0%** | **99.9%** | 98.6% | **100.0%** |
| 10 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| 12 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **99.9%** |
| 15 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| 20 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| 30 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

## Stable-load jitter (fires/min) (lower = better)

| SPM | EWMA-AsymCUSUM-tau120-s15-f5-t30-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t20-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t15-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | 0.108 | 0.132 | 0.150 | 0.175 | **0.032** |
| 8 | 0.084 | 0.107 | 0.122 | 0.143 | **0.030** |
| 10 | 0.063 | 0.088 | 0.103 | 0.125 | **0.027** |
| 12 | 0.045 | 0.071 | 0.087 | 0.107 | **0.026** |
| 15 | 0.026 | 0.054 | 0.071 | 0.088 | **0.023** |
| 20 | **0.008** | 0.029 | 0.049 | 0.068 | 0.020 |
| 30 | **0.001** | 0.003 | 0.014 | 0.039 | 0.014 |

## Ramp target overshoot p99 (lower = better)

| SPM | EWMA-AsymCUSUM-tau120-s15-f5-t30-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t20-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t15-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | 36.1% | 40.9% | 43.4% | 45.6% | **5.5%** |
| 8 | 26.5% | 31.3% | 33.4% | 35.6% | **2.3%** |
| 10 | 18.4% | 25.2% | 26.7% | 30.5% | **3.3%** |
| 12 | 16.6% | 20.9% | 24.3% | 26.4% | **3.2%** |
| 15 | 14.1% | 15.4% | 19.5% | 24.1% | **2.5%** |
| 20 | 5.8% | 13.3% | 14.6% | 18.0% | **1.3%** |
| 30 | **0.0%** | 13.7% | 12.9% | 13.0% | **0.0%** |

## Settled accuracy p50 (lower = better)

| SPM | EWMA-AsymCUSUM-tau120-s15-f5-t30-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t20-eta50 | EWMA-AsymCUSUM-tau120-s15-f5-t15-eta50 | EWMA-AdaCUSUM-tau120-s15-f5-eta50 | FullRemedy |
| --- | --- | --- | --- | --- | --- |
| 6 | 10.5% | 9.6% | 9.2% | 9.0% | **6.6%** |
| 8 | 9.1% | 8.2% | 8.1% | 7.7% | **5.7%** |
| 10 | 9.4% | 7.0% | 6.7% | 7.7% | **5.3%** |
| 12 | 8.9% | 6.9% | 6.3% | 6.7% | **4.8%** |
| 15 | 8.4% | 7.2% | 5.9% | 5.8% | **4.3%** |
| 20 | 7.7% | 7.4% | 6.4% | 4.7% | **3.9%** |
| 30 | 6.7% | 6.8% | 7.1% | 6.6% | **3.3%** |

