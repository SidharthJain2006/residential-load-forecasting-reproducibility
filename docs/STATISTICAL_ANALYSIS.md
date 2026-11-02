# Statistical analysis and inference limits

## UCI
The primary comparison is a hierarchical moving-block bootstrap with 5,000 replicates and seven-day block length, with three-day and fourteen-day sensitivity analyses. Candidate origins: 215; accepted complete test windows: 192; rejected: 23. Trainable model seed variability is resampled; TimesFM is a single zero-shot run. Primary seven-day model differences against Vanilla Transformer are in `Results/UCI/primary_7day_block_bootstrap.csv`. Do not label these outputs as DM or MCS tests.

## MORED V3
The unit is original raw recording session ID (4 clusters: 6–9). Errors within overlapping 24-hour forecast windows are aggregated within each session before pairwise comparisons. Directional differences across all four sessions are reported along with cluster-bootstrap intervals. Exact two-sided sign p-values equal 0.125 and Holm-adjusted values equal 1.000. With only four independent clusters, this is underpowered; bootstrap intervals are exploratory and do not establish conventional significance.

## London
The five-seed BES ON/OFF difference is descriptive. No paired inferential test or equivalence claim is made.
