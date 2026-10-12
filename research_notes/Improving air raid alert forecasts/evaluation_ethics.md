# Training/evaluating rare-event forecasters under drift + legal/ethical aspects (air-raid alert forecaster)

## Q1. Class imbalance for gradient boosting (scale_pos_weight / is_unbalance / undersampling / focal loss)

### Takeaway
LightGBM's own docs warn that is_unbalance / scale_pos_weight may improve ranking metrics but give poor probability estimates. Undersampling also biases posteriors without changing their ranking. If the forecaster publishes probabilities, train unweighted, or re-weight and then recalibrate on a later time block.

### Cited Findings
- LightGBM `is_unbalance` (default false) and `scale_pos_weight` (default 1.0) apply only to binary/multiclassova. The docs say that enabling them "should increase the overall performance metric of your model" but "will also result in poor estimates of the individual class probabilities". — [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- Undersampling changes the class priors and biases the posterior probabilities. The ranking order is unaffected, but accuracy and calibration are significantly affected. Dal Pozzolo et al. give an analytical correction p = β·pₛ / (β·pₛ − pₛ + 1), where β is the majority-class sampling rate. — [IEEE Xplore, CIDM 2015](https://ieeexplore.ieee.org/document/7376606); [authors' slides](https://www.slideshare.net/slideshow/calibrating-probability-with-undersampling-for-unbalanced-classification/56254737)
- Whether undersampling helps depends on the sample size, classifier variance, degree of imbalance and posterior value ("When Is Undersampling Effective…", ECML-PKDD 2015). — [mlanthology](https://mlanthology.org/ecmlpkdd/2015/pozzolo2015ecmlpkdd-undersampling)
- The analytical correction is not valid for balanced random forests, because of bootstrapping. Recent work compares Platt and isotonic recalibration after undersampling. — [arXiv 2412.16209](https://arxiv.org/pdf/2412.16209); [arXiv 2410.18144](https://arxiv.org/pdf/2410.18144)

### Inferences
- Since ROC/PR-AUC depend only on ranking, re-weighting may change them little. Its main effect is on Brier score and log loss. Run an ablation with scale_pos_weight ∈ {1, sqrt(neg/pos), neg/pos} plus post-hoc calibration.

### Gaps
- I did not verify a primary source for focal loss with GBDT (Lin et al. 2017 is a detection paper; LightGBM has no built-in focal objective). Treat it as optional or experimental.

## Q2. Metrics for rare events

### Takeaway
Use PR-AUC (average precision) as the main ranking metric, and report the base rate alongside it. Pair it with proper scoring rules (log loss, Brier with decomposition or Brier skill vs. a climatology baseline) and reliability diagrams. Calibrate with sigmoid when data are scarce and isotonic only when there are roughly 1000 or more samples.

### Cited Findings
- Saito & Rehmsmeier (2015): "the visual interpretability of ROC plots in the context of imbalanced datasets can be deceptive". PRC plots "can provide the viewer with an accurate prediction of future classification performance". Only PRC changes with the positive:negative ratio. — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)
- scikit-learn says Brier score and log loss are strictly proper but mix calibration (reliability), resolution and uncertainty. A lower Brier score "does not necessarily mean a better calibrated model" unless you compute the decomposition. — [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
- Sigmoid (Platt) calibration is "most effective for small sample sizes". Isotonic regression "is more prone to overfitting, especially on small datasets". Isotonic performs as well or better "when there is enough data (greater than ~1000 samples)", citing Niculescu-Mizil & Caruana (ICML 2005). — [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html)

### Inferences
- For an oblast with a 0.3% base rate, one test month has very few positives. Isotonic calibration per oblast is likely to overfit, so prefer a pooled calibrator, or sigmoid per oblast.
- Report both pooled metrics (dominated by high-base-rate oblasts) and the macro average across oblasts. PR-AUC is not comparable across oblasts with different base rates, so normalize it by base rate (e.g. AP/prevalence) or use Brier skill score.

### Gaps
- I did not fetch a primary source for the Murphy (1973) Brier decomposition. It is standard, but cite it after checking.

## Q3. Time-series CV, leakage, significance tests

### Takeaway
Use rolling-origin (walk-forward) evaluation with several test folds, not a single latest month. Add a gap (embargo) between train and test that is at least as long as the feature lookback and label horizon. Compare models with Diebold-Mariano (HLN-corrected) on per-period loss differences.

### Cited Findings
- fpp3: in time-series CV, each training set uses only observations before the test set, which is "evaluation on a rolling forecasting origin". Accuracy is averaged over the test sets. — [Hyndman & Athanasopoulos, fpp3 §5.10](https://otexts.com/fpp3/tscv.html)
- Bergmeir & Benítez (2012, Information Sciences, doi 10.1016/j.ins.2011.12.028) recommend blocked CV as the standard for time-series predictor evaluation. They found it gave more robust model selection than a last-block holdout. — [Monash research portal](https://research.monash.edu/en/publications/on-the-use-of-cross-validation-for-time-series-predictor-evaluati/)
- Diebold-Mariano (1995, JBES 13:253–263) tests for zero expected loss differential using a HAC variance. Harvey, Leybourne & Newbold (1997, IJF 13(2):281–291) add a small-sample correction, which is implemented in R `forecast::dm.test` and in statsmodels via `harvey_adj`. — [R dm.test docs](https://search.r-project.org/CRAN/refmans/forecast/html/dm.test.html); [statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.diebold_mariano_test.html)
- The standard DM test can be undersized and lose power when the mean loss differential varies over time (Harvey, Leybourne & Zu). This matters under drift. — [Nottingham repository](https://nottingham-repository.worktribe.com/output/40565499/testing-for-equal-average-forecast-accuracy-in-possibly-unstable-environments)

### Inferences
- Use the per-hour (or per-day) log-loss or Brier difference as the DM loss series. For PR-AUC differences, use a moving or circular block bootstrap over days, so that within-day autocorrelation is kept.

### Gaps
- I did not fetch primary sources for López de Prado's purging/embargo (Advances in Financial Machine Learning, 2018, ch. 7) or for block bootstrap (Künsch 1989; Politis & Romano 1994). The concepts are well known, but the citations are unverified here.

## Q4. Concept drift

### Takeaway
The standard reference is Gama et al. 2014. Its strategies are drift detection, windowing (sliding windows) and forgetting/weighting of old data. For changes in attack tactics, compare training-window lengths and time-decay sample weights in the walk-forward evaluation, and monitor metrics month by month.

### Cited Findings
- Gama, Žliobaitė, Bifet, Pechenizkiy & Bouchachia, "A Survey on Concept Drift Adaptation", ACM Computing Surveys 46(4):44, 2014, doi 10.1145/2523813. It covers adaptive learning, a taxonomy of strategies, and how to evaluate adaptive methods. — [TU/e portal](https://research.tue.nl/en/publications/a-survey-on-concept-drift-adaptation/); [Bournemouth eprint](https://eprints.bournemouth.ac.uk/22491/)
- An example of drift in the reporting data itself: in Aug 2026, Ukraine's Air Force restricted its morning reports on ballistic missiles (numbers launched or shot down). It said it would keep sending real-time threat alerts via Telegram, Viber and WhatsApp. — [Ukranews](https://ukranews.com/en/news/1168459-ihnat-confirms-that-ukrainian-air-forces-change-format-of-reports-on-air-attacks-by-occupation); [LIGA](https://news.liga.net/en/politics/news/no-missiles-for-patriot-air-force-explains-why-it-no-longer-reports-the-number-of-russian-ballistic-missiles)

### Inferences
- Features built from official reports or Telegram channels can change without warning when reporting formats change. Treat source-format changes as covariate drift and monitor feature distributions (e.g. PSI).

### Gaps
- I found no peer-reviewed study of drift in air-raid alert patterns specifically.

## Q5. Operational metrics for early-warning systems

### Takeaway
The earthquake early-warning (EEW) literature scores events as matched, missed, false and late alerts, and measures warning time (lead time) to an event. It avoids metrics that depend on true negatives. Tolerating more false alerts reduces missed ones.

### Cited Findings
- The ShakeAlert testing platform classifies alerts as matched, missed or false, with an upper limit on time to first alert. — [SCEC](https://central.scec.org/node/5061)
- ShakeAlert 2.0 historic false/missed rates were 8%/16% for M5+ and 10%/36.7% for M3.5+. — [Caltech repository, SRL 2020](https://authors.library.caltech.edu/records/dz4m3-02r27)
- An on-site EEW study in Taiwan reported a 2.5% false-alarm and 14.1% missed-alarm probability. — [NTU](https://scholars.lib.ntu.edu.tw/handle/123456789/493313)
- The Transformer EEW paper treats too-late warnings as misses and uses precision, recall, F1 and PR-AUC, avoiding true negatives. — [arXiv 2009.06316](https://arxiv.org/pdf/2009.06316)
- Minson et al.: missed alerts are more common than false ones, and accepting more false alerts gives the best overall performance. — [USGS](https://pubs.usgs.gov/publication/70215102) (the journal and DOI in the metadata are inconsistent, so verify them)

### Inferences
- Report event-level metrics: the share of alert starts preceded by a forecast above threshold within an X-minute window (the hit rate), the median lead time, and false alarms per oblast per week, at fixed operating thresholds.

### Gaps
- I did not verify weather-warning sources (e.g. NWS POD/FAR/CSI definitions).

## Q6. Legal/ethical aspects

### Takeaway
Ukraine criminalizes unauthorized dissemination of information on movements of weapons and of the Armed Forces under martial law (Criminal Code Art. 114-2, added in 2022). The Air Force warned in 2023 that real-time posting of missile/drone tracks, which can reveal air-defense locations, could lead to criminal cases. Telegram's API terms prohibit using Telegram data to train AI/ML models. That is the most direct compliance risk for a forecaster trained on scraped channels.

### Cited Findings
- Art. 114-2 was introduced by Law No. 2160-IX, in force 27 Mar 2022. It covers unauthorized dissemination, under martial law, of information on the direction and movement of weapons and ammunition into Ukraine and on the movement or deployment of the Armed Forces of Ukraine. — [DonNU analysis](https://r.donnu.edu.ua/handle/123456789/2595?locale=en); law text at [zakon.rada.gov.ua/laws/show/2160-20](https://zakon.rada.gov.ua/laws/show/2160-20) (returned 403 to my fetcher, so verify the exact wording manually). A 2024 study reviewed 112 guilty verdicts from 2022–2023. — [VNU PDF](https://evnuir.vnu.edu.ua/bitstream/123456789/25712/1/2788-6018.2024.05.115.pdf)
- On 22 Mar 2023, Air Force spokesman Yurii Ihnat said "criminal cases may be opened because some Telegram channels provide real-time information", calling it "a serious crime against Ukraine – informing the enemy". On 24 Mar he clarified there was no ban on established channels. His concern was radar screenshots and flight paths showing where targets disappeared, which reveal air-defense positions. — [Kyiv Post](https://kyivpost.com/post/14817); [Ukrainska Pravda](https://pravda.com.ua/eng/news/2023/03/22/7394545)
- Telegram API ToS §1.5: developers are "prohibited from using, accessing or aggregating data obtained from the Telegram platform to train" AI/ML models. Under §4, violators are notified and API access is discontinued if not fixed within 10 days. — [Telegram API Terms](https://core.telegram.org/api/terms)
- The general Telegram ToS covers spam and violence, but I found no explicit scraping clause there. — [telegram.org/tos](https://telegram.org/tos)

### Inferences
- A model that forecasts alert *starts* from public alert history has lower risk than one that republishes trajectory or interception data. Still: (a) don't publish features or explanations that reveal where threats disappear or where air defense is effective; (b) publish with a delay or in aggregate; (c) never present the forecast as a substitute for official alerts. These are design recommendations, not legal advice.
- Using Telegram API data (e.g. via Telethon) for model training appears to conflict with the API ToS §1.5. Consider official alert sources or licensed APIs instead, and get legal review.

### Gaps
- I could not fetch the official Rada text of Art. 114-2 (403), so the exact penalties are unverified. Secondary reporting mentions up to 12 years.
- I found no official Ukrainian guidance specifically on publishing *probabilistic forecasts* of alerts.
