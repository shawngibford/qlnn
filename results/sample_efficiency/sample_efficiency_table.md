# Sample-efficiency sweep (Claim 3 / Step 6)

Test MAE on the locked h=3 evaluation, fraction of training
windows truncated chronologically from the start. mean ± 95% CI.

| Stack | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| Classical H=4 | 0.2788 ± 0.0239 | 0.2546 ± 0.0288 | 0.2564 ± 0.0265 | 0.2594 ± 0.0206 |
| QLNN | — | — | — | — |

Window counts (training only):

| Stack | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| Classical H=4 | 47 | 118 | 236 | 472 |
| QLNN | 47 | — | — | — |

### Claim 3 verdict

Target X = classical H=4 test MAE at 100% data = **0.2594**.
- 10%: classical=0.2788 | QLNN=— | QLNN reaches target: NO | classical reaches target: NO
- 25%: classical=0.2546 | QLNN=— | QLNN reaches target: NO | classical reaches target: YES
- 50%: classical=0.2564 | QLNN=— | QLNN reaches target: NO | classical reaches target: YES
- 100%: classical=0.2594 | QLNN=— | QLNN reaches target: NO | classical reaches target: YES
