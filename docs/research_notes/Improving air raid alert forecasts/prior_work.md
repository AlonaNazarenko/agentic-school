# Prior work on forecasting / analysing Ukrainian air raid alerts and Russian aerial attacks (2022–2026)

Research date: 2026-10-12. About 13 tool calls. Coverage is partial: Kaggle and GitHub pages were mostly not machine-readable, and those gaps are marked.

## 1. Academic papers modelling or forecasting alerts and attacks

### Takeaway
I found only one directly comparable ML paper: Pavlyshenko & Pavlyshenko (arXiv 2411.14625, Nov 2024). It uses per-region Random Forests on the Vadimkin/Klymenko data with 5- and 15-min horizons. Its target is "alert active in next 5 min", not alert starts, and the text reports no numeric metrics. The project's start-event PR-AUC benchmark therefore has no direct published comparator.

### Cited Findings
- **Paper:** "Predictive Analytics of Air Alerts in the Russian-Ukrainian War", Demian & Bohdan Pavlyshenko, arXiv:2411.14625, submitted 21 Nov 2024 (cs.LG) — [arXiv](https://arxiv.org/html/2411.14625v1)
  - **Data and period:** Vadym Klymenko's air-raid datasets (GitHub), 1-minute grid, 25 Mar 2022 – 6 Nov 2024 — [arXiv](https://arxiv.org/html/2411.14625v1)
  - **Target:** binary "alert in region within next 5 min". An ongoing alert counts as 1, so the target is alert state, not onset. A 15-min horizon appears in Appendix A.1 — [arXiv](https://arxiv.org/html/2411.14625v1)
  - **Features:** lagged binary alert states of all oblasts, current alert duration per region, hour/day-of-week/month, and `ndays` (trend) — [arXiv](https://arxiv.org/html/2411.14625v1)
  - **Model:** one RandomForestClassifier per region (n_estimators=500), fitted for Lviv, Vinnytsia, Kyiv oblast and Kharkiv oblasts only — [arXiv](https://arxiv.org/html/2411.14625v1)
  - **Validation:** time split at 2024-07-01. Metrics are accuracy, confusion matrix and ROC-AUC, shown only in figures. The text gives no numbers, and some figure labels appear duplicated or mislabelled — [arXiv](https://arxiv.org/html/2411.14625v1)
  - **Conclusions:** neighbouring-region features are the most important, seasonality matters, and patterns drift over time (`ndays` is important) — [arXiv](https://arxiv.org/html/2411.14625v1)
  - **Telegram forecasts:** the authors also ran a Telegram channel predicting alert starts with NLP plus heuristics over messages from monitoring Telegram channels — [arXiv (via search summary)](https://arxiv.org/html/2411.14625v1)
- **PNAS 2023:** Van Dijcke, Wright & Polyak, "Public response to government alerts saves lives during Russian invasion of Ukraine". It used the timing of more than 3,000 alerts and pings from 17M devices. Sheltering response decayed over time (alert fatigue), and the authors estimate 35–45% of civilian casualties were averted. This studies behaviour, not forecasting — [UChicago PDF](https://harris.uchicago.edu/files/pnas.2220160120.pdf); [UMich news](https://news.umich.edu/researchers-find-lifesaving-impact-of-early-warning-systems-in-ukraine)
- **National Defence University of Ukraine:** a Ukrainian article uses alert statistics (probability, duration and timing) to plan the teaching process and mentions neural networks. It is statistical and applied, not an ML benchmark — [znp-vo.nuou.org.ua](https://znp-vo.nuou.org.ua/article/download/363678/350511/848660)
- **CSIS Futures Lab "Russian Firepower Strike Tracker" (podcast, 9 May 2025):**
  - strike volume is reportedly not statistically tied to frontline activity
  - mixed salvos (cruise, ballistic and drones) get through more often
  - planned work adds weather data and lasso variable selection

  — [CSIS](https://www.csis.org/analysis/truth-matter-analyzing-missile-attacks-ukraine)
- **CSIS missile analysis:** uses the Petro Ivaniuk Kaggle dataset, cross-checked against Ukrainian Air Force posts. It counts 11,466 missiles launched from Sep 2022 to Sep 2024 — [CSIS (via search snippet)](https://www.csis.org/analysis/truth-matter-analyzing-missile-attacks-ukraine)

### Inferences
- Pavlyshenko's state target is much easier than ours: persistence of ongoing alerts inflates AUC. Their results are not comparable to start-only PR-AUC.
- Their feature findings (neighbour lags, seasonality, drift) match what a LightGBM baseline likely already uses. Adding external launch signals would be the novel contribution.

### Gaps
- I found no paper reporting PR-AUC or Brier scores for alert-onset prediction.
- I did not search Google Scholar or SSRN exhaustively. Other conference papers (Ukrainian journals, IEEE) may exist.
- I could not verify a LinkedIn claim (Yevhenii Opria) of a production LightGBM forecasting 24h-ahead alert probability for 25 regions: [LinkedIn](https://ua.linkedin.com/in/yevhenii-opria)

## 2. GitHub, Kaggle and API projects

### Takeaway
The main alert-history source is Vadimkin/ukrainian-air-raid-sirens-dataset (MIT, updated daily, official and volunteer alerts). For live data, alerts.in.ua offers oblast/raion/hromada status and threat-type tags, but its documented history only goes back one month. I found no public forecasting repo or Kaggle notebook beyond the Pavlyshenko work.

### Cited Findings
- **Vadimkin/ukrainian-air-raid-sirens-dataset:**
  - air-raid alerts by region, two types: official, and unofficial collected by volunteers from the eTryvoga channel
  - "Datasets updated daily", MIT license
  - regenerated by `process.py` from Telegram (needs API id and hash)

  — [GitHub](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
- **Klymenko data in journalism:** El País built its one-year siren analysis on data collated by developer Vadym Klymenko from eTryvoga alerts — [IIB Awards](https://www.informationisbeautifulawards.com/showcase/6047-after-a-year-of-war-air-raid-sirens-have-sounded-more-than-15-000-times-in-ukraine)
- **alerts.in.ua API (devs.alerts.in.ua):**
  - **Endpoints:** active alerts JSON; IoT status strings per oblast and per UID (A = full, P = partial, N = none); history per UID (`month_ago` only)
  - **Location types:** oblast, raion, city, hromada
  - **Alert types:** air raid, artillery, urban fights, chemical, nuclear
  - **Threat tags:** tactical aviation, strategic aviation, MiG-31K, ballistic, cruise, drones, guided bombs, etc.
  - **Rate limits:** about 8–12 req/min, and 2 req/min for history
  - **Caveat:** the service says it is not for critical use

  — [alerts.in.ua docs](https://devs.alerts.in.ua/)
- **Other tools (none predictive):**
  - ALERTua/air_raid_threat_reporter filters Telegram threat channels with Ollama LLM classification — [GitHub](https://github.com/ALERTua/air_raid_threat_reporter)
  - Ukraine Siren Alerts (UASA) harvests alerts from municipal sites and social media — [Wikipedia](https://en.wikipedia.org/wiki/Ukraine_Siren_Alerts)
  - air.alarms.in.ua has collected alerts since 24 Feb 2022 — [Kyiv Post](https://www.kyivpost.com/amp/post/18257)

### Inferences
- The alerts.in.ua threat tags (e.g. MiG-31K takeoff, strategic aviation) are near-real-time precursors. They would need to be logged prospectively, since history is limited to one month.
- Volunteer eTryvoga records may lead official alerts and could serve as a feature or a label-noise check.

### Gaps
- I could not read `datasets/README.md` for field definitions or raion coverage.
- I found no Kaggle notebook predicting alerts. Kaggle pages did not render.
- I did not reach the ukrainealarm.com API docs.

## 3. Missile and drone attack datasets

### Takeaway
The best feature source is the Petro Ivaniuk (piterfm) Kaggle dataset "Massive Missile Attacks on Ukraine", parsed from Air Force daily reports and used by CSIS. Its schema could not be read directly in this session. The DiXi Group / Energy Map file and the Airwars Shahed map are narrower alternatives.

### Cited Findings
- **Kaggle "Massive Missile Attacks on Ukraine"** (kaggle.com/datasets/piterfm/massive-missile-attacks-on-ukraine). The page exists, but the content did not render — [Kaggle](https://www.kaggle.com/datasets/piterfm/massive-missile-attacks-on-ukraine)
  - CSIS attributes its strike data to Petro Ivaniuk on Kaggle and says it matches the Ukrainian Air Force's official numbers — [CSIS](https://www.csis.org/analysis/truth-matter-analyzing-missile-attacks-ukraine)
- **MotherDuck "Skyfall: 1350 nights"** (Volodymyr Pivoshenko): an interactive dive over every recorded attack, Sep 2022 – Jun 2026, from Air Force daily reports — [MotherDuck](https://motherduck.com/dive-gallery/dives/skyfall-1350-nights-of-drones-missiles-over-ukraine)
- **DiXi Group / Energy Map:** massive strikes on critical infrastructure, 11 Sep 2022 – 26 Aug 2024. It gives weapon type and subtype with counts launched and shot down, from official Armed Forces reports, updated irregularly — [Energy Map](https://map.ua-energy.org/en/resources/12f3148d-841a-478d-b9ed-72bf0764b286)
- **Airwars Shahed map:** tracked every Shahed launch from 13 Sep 2022 to 30 Aug 2023 with OSINT, including shoot-down locations and possible launch sites — [Airwars](https://airwars.org/shahed-map/)
- **CSIS drone counts (via UNN):** more than 44,000 Shahed-type drones in the first 10 months of 2025, +303% versus all of 2024, about 64% destroyed — [UNN](https://unn.ua/en/news/number-of-russian-kamikaze-drone-launches-against-ukraine-quadrupled-this-year-media)

### Inferences
- Air Force reports are daily or overnight aggregates. They suit features like "yesterday's or overnight launch counts" and regime indicators, not 5-min timing.
- Using them live risks leakage: a night's report is published the next morning, so features must be lagged to publication time.

### Gaps
- I did not verify the Kaggle column list. From memory it includes time_start/time_end, model, launch_place, target, launched and destroyed, but this is UNVERIFIED.
- I did not reach the ACLED Ukraine data in this session.
- I did not locate a primary CSIS Shahed tracker page.

## 4. Analyses of alert patterns

### Takeaway
Pattern analyses found were descriptive journalism with counts. The key mechanics for raion-level declaration and the threat-type taxonomy come from the alerts.in.ua docs.

### Cited Findings
- **2022:** 14,870 sirens across Ukraine — [Ukrinform](https://www.ukrinform.net/rubric-society/3644868-14870-air-raid-sirens-rang-across-ukraine-in-2022.html)
- **2023:** Kyiv Post / air.alarms.in.ua, "40 sirens every day", with 6,807 national alerts in 2023 at the time of reporting — [Kyiv Post](https://www.kyivpost.com/amp/post/18257)
- **Partial alerts:** alerts can be partial (P) when only some raions or hromadas are under alert. Some cities, such as Zaporizhzhia and Kharkiv, declare separately — [alerts.in.ua docs](https://devs.alerts.in.ua/)
- **Alert fatigue:** the alert-fatigue effect points to user behaviour, not alert frequency, as the outcome of interest — [PNAS PDF](https://harris.uchicago.edu/files/pnas.2220160120.pdf)

### Inferences
- Oblast-level "start" labels depend on how partial raion alerts are aggregated. The label definition should be documented when comparing with others.

### Gaps
- I did not reach texty.org.ua analyses or official documentation of the 2023 switch to raion-level declaration.

## 5. Student and hackathon projects

### Takeaway
I found no verified KSE or university hackathon project forecasting alerts.

### Cited Findings
- None verified. The only Ukrainian academic item found is the NUOU education-planning article — [NUOU](https://znp-vo.nuou.org.ua/article/download/363678/350511/848660)

### Gaps
- Searches for KSE, Ukrainian university and hackathon projects were limited to one Ukrainian-language query. Follow up via GitHub topic search and the KSE repositories.
