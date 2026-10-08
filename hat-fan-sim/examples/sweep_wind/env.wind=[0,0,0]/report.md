# V1 23-inch ring-fan hat env.wind=[0,0,0]: airflow to the face

**PARTLY - air reaches the face but it is weak**

![summary](summary.png)

| metric | value |
|---|---|
| mean air speed over face | 0.46 m/s (faint) |
| max air speed over face | 2.17 m/s |
| face area feeling > 0.2 m/s | 78% |
| face area cooled > 0.5 m/s | 36% |
| fan exit speed / open area | 2.5 m/s / 478 cm² |
| fan volume flow | 120 L/s (253 CFM) |

## Probes

| point | mean speed m/s | turbulence rms m/s | feel |
|---|---|---|---|
| forehead | 0.63 | 0.01 | cooling breeze |
| left eye | 0.64 | 0.01 | cooling breeze |
| right eye | 0.63 | 0.01 | cooling breeze |
| nose tip | 2.18 | 0.02 | strong breeze |
| left cheek | 0.68 | 0.02 | cooling breeze |
| right cheek | 0.52 | 0.02 | cooling breeze |
| mouth | 0.12 | 0.02 | not felt |
| chin | 0.14 | 0.02 | not felt |

## Solver

163,863 cells at 12 mm, 2176 steps (0.8 s simulated, averaged over the last 0.40 s), runtime 26 s.
