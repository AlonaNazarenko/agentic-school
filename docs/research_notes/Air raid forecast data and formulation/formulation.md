# Problem formulation and modelling techniques for rare spatio-temporal event onsets (beyond a pooled GBM)

Verification note: all findings below come from web-search result summaries (13 searches). No full-text page was fetched, so **every claim is snippet-level** unless marked otherwise. Treat the numbers as provisional until checked against the primary PDFs.

## Hurdle / two-stage models (P(any national event) x P(region | event))

### Takeaway
Two-part hurdle models are a standard, well-motivated way to handle zero-inflated spatial event data, and ML (boosted-tree) hurdle versions exist for crashes and crime. I found no study that measures the gain of a "national wave x regional share" decomposition over a pooled classifier for onset forecasting.

### Cited Findings
- A 2023 review describes the hurdle model as a sensible two-part model for spatial zero-inflated data: one process produces the zeros, and positives follow a zero-truncated distribution. [snippet] — [arXiv 2304.02476](https://arxiv.org/pdf/2304.02476)
- A 2024 Applied Sciences paper proposes an ML hurdle framework (CatBoost, two stages) for zero-inflated crash-frequency data, as an alternative to Poisson/NB. [snippet; gain size not seen] — [DOAJ / Applied Sciences 2024](https://doaj.org/article/dd81da29e6de4ac98fb00e2f3c52d7a4)
- An earth-embedding urban study uses a two-part hurdle (classifier for occurrence, then regressor for magnitude) with XGBoost/LightGBM for zero-inflated indicators such as crime rates. [snippet] — [arXiv 2604.03456](https://arxiv.org/pdf/2604.03456)
- Political-violence work in Africa combines a negative-binomial hurdle with model-based (statistical) boosting and spatial structure. [snippet] — [TUM mediaTUM](https://mediatum.ub.tum.de/node?id=1632629)
- Spatio-temporal hurdle with correlated random effects (ED visits: any use x count given use). [snippet] — [SMMR 2014](https://journals.sagepub.com/doi/full/10.1177/0962280214527079)
- A KIT thesis on the 2023/24 VIEWS challenge has a hurdle-model section; results not visible in the snippet. — [KIT](https://publikationen.bibliothek.kit.edu/1000190424/174932541)

### Inferences
- For air-raid alerts, the natural hurdle is: stage 1 = P(a wave starts nationally in the next h min), stage 2 = P(oblast o alerts | wave), trained only on wave-positive windows. Stage 2 then gets a much higher base rate, which may help the oblast-allocation pattern. The product is the oblast probability. This is an untested hypothesis. Compare it against the pooled model on the same per-oblast PR-AUC.
- A cheaper variant is to feed the stage-1 national probability as a feature to the pooled model (stacking).

### Gaps
- No quantitative evidence found on hurdle vs single-model gains for binary onset classification (as opposed to counts).

## Discrete-time hazard / survival GBMs vs fixed-horizon classifiers

### Takeaway
Discrete-time survival is a binary classifier on person-period rows, so GBMs apply directly. Reported gains over other formulations depend on the dataset. The main documented benefit is that the target matches the warning objective, not guaranteed higher accuracy.

### Cited Findings
- Discrete-time survival models predict the hazard per interval on a person-period dataset, and any binary classifier can estimate it. Some discrete-time ML models beat Cox. Random survival forests were sometimes outperformed. Relative ranking varied by dataset. [snippet] — [PMC9316420 (BMC Med Res Methodol 2022)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9316420/)
- Survival studies usually report C-index, which makes comparison with classifiers hard. The literature gives limited guidance on converting survival outputs to classification metrics. [snippet] — [arXiv 2603.11598](https://arxiv.org/html/2603.11598v1)
- Vineyard-disease early warning compared "daily status" and "event onset within 3-7 days" formulations with LR and XGBoost. The onset formulation's value is "explicit alignment between the supervised target and the early-warning objective", not guaranteed numerical superiority. [snippet] — [arXiv 2605.04548](https://arxiv.org/pdf/2605.04548)

### Inferences
- A discrete hazard model at 5-min resolution (target: onset in the next 5-min bin, conditional on no active alert) produces 15/30/60-min probabilities as 1 - prod(1 - h_t) over future bins. That gives monotone-consistent horizons for free. It needs future features, though, so in practice it reduces to forecasting with horizon-indexed rows (see next section).
- Onset targets must exclude bins where an alert is already active (the at-risk set). Check that the current labels do this.

### Gaps
- XGBoost AFT, LightGBM survival and pycox were not checked in docs this round. No head-to-head hazard-GBM vs fixed-horizon classifier result was found for event onsets.

## Multi-horizon / multi-task learning

### Takeaway
There is indirect evidence only. In forecasting, the direct multi-step strategy (one model across horizons or one per horizon, not recursive) explains much of the apparent deep-model gains. I found no source on a horizon-as-feature GBM with monotonic consistency for binary onset.

### Cited Findings
- Zeng et al.: DLinear, a simple linear model, beat Transformer long-term forecasters on 9 benchmarks "in most cases by a large margin". They attribute the Transformers' gains to the non-autoregressive direct multi-step strategy, not to attention. [snippet] — [arXiv 2205.13504](https://arxiv.org/pdf/2205.13504)

### Inferences
- Stacking horizons as rows with h in {15, 30, 60} as a feature triples the data and shares strength. LightGBM `monotone_constraints` on h can enforce P(15) <= P(30) <= P(60). This is a design suggestion, not an evidenced gain.

### Gaps
- No empirical source found on monotone multi-horizon event classifiers.

## Learning-to-rank (LambdaRank) and focal loss

### Takeaway
Learning-to-rank tuned to a top-k objective won the 2017 NIJ crime-forecasting challenge on the PAI metric. Focal loss for GBMs has weak evidence: one thesis found plain class weighting did better.

### Cited Findings
- Mohler et al. 2020 (CrimeRank): hotspot methods are judged by area-normalized Recall@k but trained by likelihood. A boosting learning-to-rank method trained for top-k capture was the top-performing solution (PAI metric) in the 2017 NIJ challenge. [snippet] — [Crime Science 2020](https://link.springer.com/article/10.1186/s40163-020-00112-x)
- LightGBM supports LambdaRank (LambdaMART) with query groups and NDCG@k. [snippet] — [SynapseML LightGBM docs](https://microsoft.github.io/SynapseML/docs/1.0.3/Explore%20Algorithms/LightGBM/Quickstart%20-%20Classification,%20Ranking,%20and%20Regression/)
- Focal loss in GBMs needs a custom objective (gradient and hessian). [snippet; tutorial-grade source] — [apxml course](https://apxml.com/courses/mastering-gradient-boosting-algorithms/chapter-7-boosting-advanced-topics-customization/handling-imbalanced-data-boosting)
- A thesis on HTRU2 and credit default found that weighted LightGBM gave the best performance, beating the focal-loss variant. [snippet; thesis] — [UGM repository](https://etd.repository.ugm.ac.id/penelitian/detail/240861)
- A focal-aware cost-sensitive boosted tree was proposed for imbalanced credit scoring (Expert Systems with Applications). Gains not seen in the snippet. — [datalearner listing](https://www.datalearner.com/academic/journal-papers/0957-4174/volumes-and-issues/412/paper-detail/19276)

### Inferences
- For "which oblasts to alert" at time t, group = timestamp (24 oblasts). LambdaRank then optimizes cross-oblast ordering, which is exactly what a wave-driven problem needs once a wave is predicted. It does not give calibrated probabilities, so pair it with a classifier or the hurdle stage 1.
- PR-AUC is threshold-free, so focal loss or weights mainly change ranking only through tree structure. Expect small effects.

### Gaps
- No direct LambdaRank vs binary GBM comparison for spatial event onsets.

## Per-cluster vs pooled vs per-region; hierarchical models

### Takeaway
Theory and evidence favour global (pooled) models, even for heterogeneous series. Local or partitioned models are a complexity-control tool, and which works best depends on the data.

### Cited Findings
- Montero-Manso and Hyndman: for any dataset there exists a global model at least as good as a set of local models. Global models work surprisingly well even on heterogeneous groups. They suggest raising complexity via longer lags, nonlinearity and **data partitioning** (clustering). [snippet] — [arXiv 2008.00444](https://arxiv.org/pdf/2008.00444)
- In hierarchical forecasting, local vs global performance depends on the information available, the horizon, intermittency and seasonality. [snippet] — [arXiv 2411.06394](https://arxiv.org/pdf/2411.06394)

### Inferences
- Keep the pooled model, add an oblast ID / cluster ID as a categorical, and test cluster-specific models (frontline / central / western) as an ensemble member rather than a replacement.

### Gaps
- No mixed-effects GBM evidence (e.g. GPBoost) gathered.

## Ensembling, stacking, conformal

### Takeaway
GBM-based ensembles won M5, and simple ensembles beat the average model in the ViEWS conflict competition. For calibrated binary risk, Venn-Abers is the conformal-family tool.

### Cited Findings
- M5 (2020, Walmart): winners of both tracks were GBDT (LightGBM)-based. Neural nets (DeepAR, N-BEATS) placed 2nd and 3rd in Accuracy. Top Kagglers blended or stacked LightGBM. [snippet; sources partly disagree on 2nd/3rd place methods] — [Amazon Science "forecasting with trees"](https://cdn.amazon.science/17/31/0d99831d4add92e711a56abc75f8/forecasting-with-trees.pdf); [m5-methods GitHub](https://github.com/umitkaanusta/m5-methods)
- ViEWS 2020/21 competition (15 teams): a simple ensemble of the entries was significantly better than the average individual model. [snippet] — [PRIO blog](https://www.prio.org/comments/585)
- VIEWS 2023/24: tree-based ensembles produced probabilistic forecasts on 13,110 PRIO-GRID cells. [snippet] — [arXiv 2512.06210](https://arxiv.org/pdf/2512.06210)
- Venn-Abers combines Venn predictors with isotonic regression to give calibrated binary probabilities as an interval [p0, p1], with finite-sample guarantees. [snippet] — [arXiv 2502.05676](https://arxiv.org/html/2502.05676v3); [PMLR Gustafsson 2026](https://proceedings.mlr.press/v329/gustafsson26a.html)

### Inferences
- An ensemble of the pooled GBM, a hurdle model, a LambdaRank model and a hazard model is a cheap test. Average the ranks or stack with a logistic regression on a later time fold.
- Calibration matters for thresholding alerts. Conformal guarantees assume exchangeability, which is broken under war-regime drift, so use rolling recalibration.

### Gaps
- The magnitude of the ViEWS ensemble gain was not seen.

## Sequence models vs GBM with engineered features

### Takeaway
Trees remain state-of-the-art on medium-sized tabular data. Simple models often match Transformers in time-series forecasting. Deep sequence models are not a likely quick win over the GBM.

### Cited Findings
- Grinsztajn et al. (NeurIPS 2022), 45 datasets: tree models (XGBoost, RF) remain state-of-the-art on medium-sized data (~10K samples). Neural networks struggle with uninformative features, non-rotation-invariant data and irregular functions. [snippet] — [arXiv 2207.08815](https://arxiv.org/abs/2207.08815)
- DLinear beats Transformer forecasters, and Transformer performance often fails to improve with longer inputs. [snippet] — [arXiv 2205.13504](https://arxiv.org/pdf/2205.13504)

### Inferences
- The Grinsztajn evidence covers medium data. The air-raid dataset (24 oblasts x ~100k+ 5-min steps) is larger, so a neural point process or TCN could be revisited later, but as an ensemble member.

### Gaps
- No direct benchmark found for neural Hawkes / TPP vs GBM on onset forecasting.

## Lessons from operational early warning

### Takeaway
Warning quality and warning threshold must be judged separately. Raising lead time or POD costs false alarms. Excess false alarms carry real costs (casualties, alert fatigue). Deployed models often underperform their reported validation.

### Cited Findings
- Brooks and Correia 2018: signal detection theory separates the system's skill from its threshold. NWS tornado lead time was stable over 1986-2011 while FAR slowly fell. In 2012, shorter warnings and a focus on fewer false alarms lowered lead time, POD and FAR together. Single metrics viewed alone mislead. [snippet] — [NOAA repository](https://repository.library.noaa.gov/view/noaa/25664)
- A 1-SD increase in tornado FAR raises expected fatalities by 12-29% and injuries by 14-32%. The effects of FAR and lead time are about equal in size. [snippet; via Mercatus summary] — [Mercatus](https://www.mercatus.org/students/research/journal-articles/false-alarms-tornado-warnings-and-tornado-casualties)
- The first warning for a storm in an outbreak is much less likely to be a false alarm. [student REU paper, not peer-reviewed] — [OU REU 2022](https://reu.caps.ou.edu/reu22/finalpapers/Bohlmann-finalpaper.html)
- Epic Sepsis Model external validation (Wong et al., JAMA Intern Med 2021; 38,455 hospitalizations): AUC 0.63 vs higher developer-reported values. Sensitivity 33%, PPV 12%. It alerted on 18% of patients and caught only 7% of sepsis cases clinicians missed, causing alert fatigue. [snippet via secondary coverage] — [PMC8218233](https://pmc.ncbi.nlm.nih.gov/articles/PMC8218233); [MedCity News](https://medcitynews.com/2021/06/popular-sepsis-prediction-model-works-substantially-worse-than-claimed-researchers-find)
- Daily wildfire occurrence (Morocco, temporal split 2019-22 train / 2023-24 test): XGBoost PR-AUC 0.739, ROC-AUC 0.813. Fire Weather Index features added value. [snippet] — [EECSS 2026 proceedings](https://www.avestia.com/EECSS2026_Proceedings/files/paper/MVML/MVML_123.pdf)

### Inferences
- Report precision/recall at fixed alert budgets (e.g. top-k oblasts per 5-min step) and lead-time gained beyond the official alert. Do not report PR-AUC alone.
- The "first warning in an outbreak" finding matches the wave structure: a model of wave onset (national) and of in-wave spread are different problems, which supports the hurdle decomposition.
- Validate on a strictly later period. The ESM case shows reported performance can degrade badly out of sample.

### Gaps
- ShakeAlert lead-time and threshold documentation not retrieved.
