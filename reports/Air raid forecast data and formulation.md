# Reframe the target before adding data

The best remaining gains for per-oblast alert-start forecasting are more likely to come from **changing what the model is asked to predict** than from more alert-history features. The finding that exponentially decayed Hawkes-proxy counts added nothing to LightGBM v2 (mean per-oblast PR-AUC 0.222±0.020 at 30 min over six monthly folds, Mar–Sep 2026) suggests the model already uses most of the self-excitation signal in the alert record. That rear oblasts gain most from spatial features points to the remaining structure: a wave starts nationally, then spreads along a few fixed corridors at known drone speeds. Three changes follow from this:

- **Reframe the target as a hurdle:** P(national wave) × P(oblast joins | wave).
- **Rebuild spatial inputs from raion-level alert data.** Measure them as fractions of neighbouring raions alerted and as corridor-ordered lags, not oblast-level flags. The raion data is free and MIT-licensed.
- **Treat the 15 December 2025 switch to raion-level alerting as a structural break.** It changes what an "oblast alert start" means in your training data.

Only one new historical source is backfillable and leading in the right sense: raion and hromada alert timestamps from the Vadimkin repository. The Kaggle attack dataset is useful only as a lagged regime covariate. Threat-typed and direction data must be collected going forward, because no archive exists. Much of the methods evidence is snippet-level, as flagged below, so every proposal here is a hypothesis to test inside your existing rolling folds.

*Verification note:* **[snippet]** marks claims seen only in search-result summaries, not on the full primary page.

## Raion-level alert history is the only strong backfillable addition

The **Vadimkin/ukrainian-air-raid-sirens-dataset** is MIT-licensed and updated daily. It holds official and volunteer (eTryvoga) data from 2022 onward ([repo](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)).

The official file `official_data_en.csv` has these columns: `oblast, raion, hromada, level, started_at, finished_at, source`.

- **Timestamps are UTC**, for example `2022-03-15 16:10:34+00:00`.
- Official rows start on 15 March 2022 ([raw CSV](https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv)).
- **The newest official row seen was dated 7 September 2026.** The volunteer file (`region, started_at, finished_at, naive`) ran to 12 October 2026 ([raw CSV](https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_en.csv)).
- Volunteer rows with no recorded end are flagged `naive`, and their end is set to start + 30 minutes ([datasets README](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset/blob/main/datasets/README.md)).

Before relying on the official file for the September fold, check whether it lags or is simply unsorted.

**Why it matters:** the `level` column lets you count, at time t, how many raions in each neighbouring oblast are already under alert. That is a finer and earlier propagation signal than "neighbour oblast alerting". These alerts are public as they happen, so the features are leakage-safe if built only from rows with `started_at` < t.

**Caveats:**

- **There is no threat-type field**, so the dataset cannot separate drone waves from ballistic events.
- **Volunteer `naive` end times are invented.** Use them only for start events, never for duration or "alert currently active" features.

The **Kaggle "Massive Missile Attacks on Ukraine"** dataset (piterfm) is still maintained: version 214 was updated on 10 October 2026. It is licensed **CC BY-NC-SA 4.0**, which rules out commercial use ([Kaggle API](https://www.kaggle.com/api/v1/datasets/list?search=massive+missile+attacks)).

- **What a row is:** one attack wave per weapon model. Columns include `time_start`, `time_end`, `model`, `launch_place`, `target`, `launched`, `destroyed`, `not_reach_goal`, `border_crossing`, `affected_region` and `source`.
- **How it is compiled:** by hand from Air Force and General Staff posts ([Kaggle metadata](https://www.kaggle.com/api/v1/datasets/metadata/piterfm/massive-missile-attacks-on-ukraine)).
- **The leakage trap:** `time_start` is the time the attack happened, not the time the row became knowable. There is no publication-time column. You have to impose a lag, for example making a row available only from 09:00 Kyiv time the next day.
- **A reporting break inside your test window.** The maintainer notes that the Air Force stopped reporting the number of some kinds of missiles launched and intercepted from **10 August 2026** (same source). ISIS-online says separate Shahed counts stopped being published consistently after about 5 May 2026 ([ISIS Aug 2026](https://isis-online.org/uploads/isis-reports/documents/August_2026-Updated-Analysis-of-Russian-Shahed-type-UAVs-Deployment-Against-Ukraine_Final.pdf)).

Count features built from Kaggle therefore change meaning in at least two of your six folds. Use them only as coarse daily-intensity covariates, and run an ablation that drops them after May.

**The other sources fail on backfill or timing:**

| Source | Why it fails |
|---|---|
| alerts.in.ua `alert_type` | Types are `air_raid`, `artillery_shelling`, `urban_fights`, `chemical`, `nuclear`. History goes back only one month at 2 requests/min, and the terms bar critical-infrastructure use ([devs.alerts.in.ua](https://devs.alerts.in.ua/)). Coarse even when live. |
| ukrainealarm API | Needs a key; its Swagger page returned 403 ([swagger](https://api.ukrainealarm.com/swagger/index.html)). Unverified. |
| UCDP Candidate | Monthly, with roughly a one-month lag ([UCDP](https://ucdp.uu.se/downloads/)). |
| ISIS-online | Monthly PDFs ([index](https://isis-online.org/isis-reports/monthly-analysis-of-russian-shahed-136-deployment-against-ukraine)). |
| MotherDuck SkyFall | A visualisation of Air Force daily reports for Sep 2022–Jun 2026 with no downloadable table found ([MotherDuck](https://motherduck.com/dive-gallery/dives/skyfall-1350-nights-of-drones-missiles-over-ukraine)). |
| NOTAMs | FAA and ICAO iSTARS APIs exist, but Russian archive depth is unconfirmed ([ICAO](https://icao.int/safety/iStars/Pages/Get-NOTAM-Data.aspx)). |

No open archive of strategic-aviation takeoffs, Air Force direction posts or outage schedules was found. The leading signals that matter most ("group of UAVs heading toward X") exist only as Telegram text. The earlier report already flagged that route as legally constrained.

## Attack geometry turns spatial features into corridor lags

The physics is simple enough to encode directly.

**Drone speeds:**

- Propeller Shaheds (Geran-2) fly at about **180–185 km/h** by Ukrainian intelligence estimates ([Euromaidan Press](https://euromaidanpress.com/2025/06/11/russia-jet-powered-geran-3-drone-hits-kyiv/)), and up to about 250 km/h in Texty's figures.
- Jet Geran-3s fly at about **500–600 km/h** at roughly 4 km altitude ([Texty](https://texty.org.ua/articles/118344/endless-cycle-of-shaheds-how-jet-powered-drones-bring-kyiv-to-a-standstill/)).
- Propeller-drone routes are mostly long straight segments. Jet drones change course several times on approach to Kyiv (same source).

**Launch sites form a fixed set:**

- Oryol/Tsymbulova, about 175 km from the border and the main site.
- Navlya in Bryansk.
- Kursk-Khalino.
- Shatalovo in Smolensk, about 46 km from Belarus, in use since early 2025.
- Primorsko-Akhtarsk.
- Millerovo.
- Crimea and Donetsk airport.

Sources: [Militarnyi](https://militarnyi.com/en/news/russia-expands-shahed-drone-sites/), [Molfar](https://www.molfar.institute/en/orlando-milan-shanghai-where-do-russians-launch-shahed-drones-from-molfar-intelligence-institute-analyzes-secret-documents-from-alabuga/), [ISIS](https://isis-online.org/isis-reports/a-comprehensive-analytical-review-of-russian-shahed-type-uavs-deployment-against-ukraine-in-2025). These are OSINT sources, not official ones. In 2024 drones also transited Belarusian airspace, one flying more than 260 km to Stolin ([Online.ua](https://news.online.ua/en/17-shaheeds-flew-into-belarus-overnight-during-russias-attack-on-ukraine-889472/)). Shatalovo plus the Belarus transits imply a northern approach to Volyn, Rivne and Zhytomyr.

**What this means for features (my inference):** at 180–250 km/h a drone covers 100 km in about 24–33 minutes. Rear oblasts sit 1.5 to 6+ hours downstream of the border. That is exactly why they gain from spatial features, and why the 60-minute horizon should benefit most from **corridor-ordered lag features**: the share of raions alerted in the upstream oblast 15, 30, 60, 90 and 120 minutes ago, plus distance-weighted sums over the launch-side frontier. The lag structure itself is not published anywhere: no open per-oblast propagation study or drone-track dataset was found. You will have to estimate it from cross-correlations of alert starts within training folds.

**Weapon types set the forecastable ceiling:**

- **Kinzhal reaches Kyiv in about 3.2 minutes** ([Telegraf](https://news.telegraf.com.ua/ukraina/2025-06-03/5910970-dlya-spaseniya-est-schitannye-minuty-za-skolko-kinzhal-doletaet-do-ukrainskikh-gorodov-karta)).
- **Iskander-M from Bryansk takes about 4 minutes** (a media calculation) ([Telegraf](https://news.telegraf.com.ua/ukraina/2024-12-27/5891976-za-yakiy-chas-mozhe-doletiti-do-kieva-ta-dnipra-iskander-m-yakim-rosiya-vdarila-po-zaporizhzhyu)).
- **Caspian-launched Kh-101 cruise missiles give about an hour or more.** One example: launched around 17:30, first detected at 19:56 ([Militarnyi](https://militarnyi.com/en/news/russia-attacks-ukraine-with-kh-101-cruise-missiles-during-daytime/); [Ukranews](https://ukranews.com/en/news/1097100-there-is-not-enough-time-for-detection-air-forces-explain-features-of-russian-missile-attack-on-kyiv)).
- **Every MiG-31K takeoff triggers a nationwide alert** ([ITC](https://itc.ua/news/kazhdyj-vzlet-myg-31k-neset-opasnost-puska-kynzhala-sposobnogo-porazhat-lyubuyu-tochku-v-ukrayne-v-techenye-neskolkyh-mynut/)).

Near-simultaneous starts across most oblasts form a separate event class with essentially no lead time. Flag them using only past data (for example, ≥N oblasts starting within 2 minutes), then score PR-AUC with and without them. Part of the 0.222 ceiling is probably irreducible ballistic or MiG noise. Quantify that share before chasing more features.

**Timing has shifted:**

- **Rhythm since mid-2025:** a steady ~180 drones a day, with **mass attacks every 5–8 days**. Attacks of more than 400 drones came 4–5 times a month from summer 2025 ([ISIS](https://isis-online.org/isis-reports/a-comprehensive-analytical-review-of-russian-shahed-type-uavs-deployment-against-ukraine-in-2025)).
- **Waves now run for days.** An April 2026 attack lasted about 32 hours and used 983 drones **[snippet]** ([ISIS May 2026](https://isis-online.org/uploads/isis-reports/documents/May-2026-Updated-Analysis-of-Russian-Shahed-type-UAVs-Deployment-Against-Ukraine_Final.pdf)).
- **Daytime alerts are growing.** From mid-August 2026, jet drones caused near-continuous alerts in Kyiv, with an evening-commute peak at 17:00–19:00. Kyiv also uses a "yellow" alert level for jet-drone threats ([Texty](https://texty.org.ua/articles/118344/endless-cycle-of-shaheds-how-jet-powered-drones-bring-kyiv-to-a-standstill/)).

A fixed hour-of-day prior learned on 2025 data is therefore stale for Aug–Sep 2026. Let hour × recent-month interactions in (for example, a rolling 28-day per-oblast hour-of-day start rate) rather than a static hour feature.

## December 2025 rewrote the label

On **15 December 2025** the Cabinet extended raion-level alerting nationwide, except in Donetsk and Luhansk oblasts. It had been piloted in 13 regions first. The handoff from the Air Force to DSNS now takes 8–15 seconds ([Cabinet of Ministers](https://www.kmu.gov.ua/news/yuliia-svyrydenko-zaprovadzhuiemo-novyi-pidkhid-do-oholoshennia-povitrianykh-tryvoh-poraionne-opovishchennia); [Militarnyi](https://militarnyi.com/uk/news/kabmin-ogolosyv-novu-systemu-spovishhennya-pro-povitryani-tryvogy-po-vsij-ukrayini/)). The government reported large effects:

- Some Dnipropetrovsk hromadas gained 100+ alert-free days.
- Sumy Oblast's total alert time fell by 50 days.
- Some Kharkiv hromadas lost more than a month of alert time.

Kharkiv had split alerts by raion even earlier ([Suspilne](https://suspilne.media/kharkiv/1086637-u-harkivskij-oblasti-diut-okremi-trivogi-dla-rajoniv-ci-mozlivo-ce-dla-gromad/)). The exact resolution text and the dates of the pilot regions were not found.

**The modelling consequence is direct.** All six test folds (Mar–Sep 2026) fall after the break. If training includes pre-December data, the model learns from a label that was produced differently:

- Before the switch, an oblast alert usually meant the whole oblast was under alert.
- After it, an "oblast start" depends on how you aggregate raions. "Any raion starts" produces more, shorter, partial events.

Four concrete fixes:

1. Rebuild the target consistently from raion rows, as "first raion start after the oblast was fully clear".
2. Add a regime indicator, with per-oblast pilot dates where they can be recovered from the data itself (the first date with raion-level rows for that oblast).
3. Test training only on post-switch data versus all data with stronger recency weights.
4. Consider a raion-level model aggregated up, using 1 − ∏(1 − p_raion) as an extra feature or ensemble member.

Recent ISIS figures put decoys (Gerbera, Parody) at about 40% of Q4 2025 launches ([ISIS](https://isis-online.org/isis-reports/a-comprehensive-analytical-review-of-russian-shahed-type-uavs-deployment-against-ukraine-in-2025)). Raion-level alerting means decoy tracks now trigger narrower alerts. That helps keep the raion-share signal informative.

## Formulation changes beat new architectures

The evidence for the methods below is mostly snippet-level, and none of it comes from alert data. The strongest general result is that **pooled, global models are hard to beat**:

- Montero-Manso and Hyndman show that for any set of local models there exists a global model at least as good. They recommend adding complexity through longer lags, nonlinearity and **data partitioning** rather than fitting per-series models **[snippet]** ([arXiv 2008.00444](https://arxiv.org/pdf/2008.00444)).
- Trees remain state of the art on medium-sized tabular data (Grinsztajn et al., 45 datasets) **[snippet]** ([arXiv 2207.08815](https://arxiv.org/abs/2207.08815)).
- DLinear beat Transformer forecasters on 9 benchmarks. The authors attribute the Transformers' apparent gains to direct multi-step output, not attention **[snippet]** ([arXiv 2205.13504](https://arxiv.org/pdf/2205.13504)).
- M5 winners in both tracks were LightGBM-based **[snippet]** ([Amazon Science](https://cdn.amazon.science/17/31/0d99831d4add92e711a56abc75f8/forecasting-with-trees.pdf)).

So cluster-specific models (frontline, central, western) belong in an ensemble, not as replacements. Sequence models (TCN, neural TPP) are low priority.

**The hurdle decomposition fits the wave structure best.** Hurdle models are the standard treatment for zero-inflated spatial data **[snippet]** ([arXiv 2304.02476](https://arxiv.org/pdf/2304.02476)), and boosted two-stage versions exist for crash data **[snippet]** ([Applied Sciences 2024](https://doaj.org/article/dd81da29e6de4ac98fb00e2f3c52d7a4)). Tornado-warning research finds the first warning in an outbreak much less likely to be a false alarm. That source is a non-peer-reviewed student paper ([OU REU](https://reu.caps.ou.edu/reu22/finalpapers/Bohlmann-finalpaper.html)), but it supports treating wave onset and in-wave spread as separate problems.

How the stages would work:

- **Stage 1:** P(a national wave starts within h). Its base rate is far higher than any single oblast's, and it can use Kaggle regime features.
- **Stage 2:** P(oblast o starts | wave active or imminent), trained only on wave windows. Corridor lags dominate here.

No study measuring the gain of this decomposition for binary onsets was found. The cheapest test is to add the stage-1 probability as a stacked feature.

**Ranking matches the per-timestamp structure.** CrimeRank, a boosted learning-to-rank model trained for top-k capture, won the 2017 NIJ crime-forecasting challenge on the PAI metric **[snippet]** ([Crime Science 2020](https://link.springer.com/article/10.1186/s40163-020-00112-x)). LightGBM supports LambdaRank with query groups ([SynapseML docs](https://microsoft.github.io/SynapseML/docs/1.0.3/Explore%20Algorithms/LightGBM/Quickstart%20-%20Classification,%20Ranking,%20and%20Regression/)). Grouping by timestamp across the 24 oblasts directly optimises "which oblasts are next". Ranking gives no calibrated probabilities, so it must feed a calibrator or ensemble.

**Focal loss is unlikely to help.** In one thesis, a weighted LightGBM beat a focal-loss variant **[snippet]** ([UGM](https://etd.repository.ugm.ac.id/penelitian/detail/240861)).

**Discrete hazard and multi-horizon models mainly buy consistency.** Any classifier on person-period rows estimates a discrete-time hazard **[snippet]** ([PMC9316420](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9316420/)). Onset-targeted formulations are valued for aligning the target with the warning goal, not for guaranteed accuracy gains **[snippet]** ([arXiv 2605.04548](https://arxiv.org/pdf/2605.04548)).

Two practical steps:

- Stack the 15/30/60-minute rows with horizon as a feature and a LightGBM monotone constraint on it. This shares strength across horizons and forces P15 ≤ P30 ≤ P60. It is my design suggestion, not an evidenced gain.
- Check that labels exclude bins where the oblast is already under alert (the at-risk set). Including them inflates both positives and difficulty.

**Ensembles reliably beat the average member.** In the ViEWS competition, a simple ensemble of 15 teams' models beat the average individual model **[snippet]** ([PRIO](https://www.prio.org/comments/585)). Venn-Abers gives calibrated binary probability intervals **[snippet]** ([arXiv 2502.05676](https://arxiv.org/html/2502.05676v3)), but its guarantees assume exchangeability, which wartime drift breaks. Use it only with rolling recalibration on top of the calibration test already run.

**Operational lessons constrain how skill should be reported:**

- Signal-detection analysis of NWS tornado warnings shows that skill and threshold must be judged separately. A 2012 policy shift lowered lead time, probability of detection and false-alarm rate together **[snippet]** ([NOAA](https://repository.library.noaa.gov/view/noaa/25664)).
- A 1-SD rise in tornado false-alarm rate raises expected fatalities by 12–29% **[snippet]** ([Mercatus](https://www.mercatus.org/students/research/journal-articles/false-alarms-tornado-warnings-and-tornado-casualties)).
- The Epic Sepsis Model fell to an external AUC of 0.63 with 12% PPV, well below what its developer reported **[snippet]** ([PMC8218233](https://pmc.ncbi.nlm.nih.gov/articles/PMC8218233)).

Report precision at a fixed alert budget (top-k oblasts per step) and lead time gained over the official alert, alongside PR-AUC.

## Prioritized experiments

Rolling-origin CV, decay features and the calibration test are done. All experiments below use that same rolling CV. Impact estimates are my judgement.

| # | Experiment | Expected impact | Effort | Leakage notes |
|---|---|---|---|---|
| 1 | **Rebuild the label from raion rows** ("first raion start after the oblast was fully clear"), add a regime dummy with per-oblast switch dates, and compare post-Dec-2025-only training against all data | High: fixes label inconsistency across the training span | Low–medium | Derive switch dates from the first raion-level rows; check the official file's tail (last seen 7 Sep 2026) |
| 2 | **Raion-share spatial features:** fraction of raions alerted in each neighbour or upstream oblast now and 15/30/60/90/120 min ago, plus national raion share | High for rear oblasts and the 60-min horizon | Low | Only `started_at` < t; ignore `naive` end times |
| 3 | **Corridor lags:** estimate per-pair lead-lag from cross-correlation of starts within each training fold; add the top-k upstream oblasts' lagged starts and distance to the nearest launch site | Medium–high | Medium | Estimate lags on training data only, never on the test month |
| 4 | **Simultaneous-start (MiG or ballistic) class:** flag ≥N oblasts starting within 2 min, then score with and without these events to measure the irreducible share | Diagnostic; sets a realistic ceiling | Low | The flag uses past starts only; the evaluation split is post hoc |
| 5 | **Hurdle:** stage-1 national-wave model, first as a stacked feature and then as a full product model | Medium | Medium | Wave definition from past data at t; stage-1 out-of-fold predictions for stacking |
| 6 | **Stacked multi-horizon model** with a monotone constraint on horizon; verify the at-risk-set labels | Low–medium for PR-AUC, high for consistency | Low | None beyond the existing setup |
| 7 | **LambdaRank** grouped by timestamp, rank-averaged with v2 | Medium on top-k precision | Low–medium | Group only contemporaneous rows |
| 8 | **Kaggle regime covariates:** days since >400-drone attack, rolling 1/3/7-day launches by model, launch-place direction shares | Medium for 60 min and quiet-period onsets | Medium (text parsing) | Available only from next-day 09:00 Kyiv; counts break after May and Aug 2026; NC licence |
| 9 | **Rolling hour-of-day prior:** 28-day per-oblast start rate by hour | Low–medium (daytime jet-drone shift) | Low | Trailing window only |
| 10 | **Cluster-specific models** (front, centre, west) and a raion-aggregated model as ensemble members; weights learned on a later fold | Low–medium | Medium | Weights fit out of time |
| 11 | **Prospective logging** of alerts.in.ua types and raion status for future folds | Long-term | Medium | No backfill exists |
| 12 | **Neural TPP or TCN** | Low | High | Ensemble member only |

## Conclusion

The flat result from the Hawkes proxy is informative. It says the remaining error is not about how soon an oblast re-alerts. It is about *whether a wave is coming* and *which way it travels*, and those are national-regime and geometric questions. The raion-level data and the December 2025 policy change are both a liability and an opportunity: they make the old oblast label partly inconsistent, but they also expose a finer, earlier spatial signal that the current model never sees. Fix the label first. Otherwise every later ablation is measured against a moving target.
