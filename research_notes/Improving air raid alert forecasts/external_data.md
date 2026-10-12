# External data sources that precede air raid alerts in Ukraine (forecasting features, leakage)

Research date: 2026-10-12. About 13 tool calls. Search results skewed to 2023-2025, so 2026 coverage is thin.

## 1. Ukrainian Air Force official Telegram channel

### Takeaway
The Air Force posts two kinds of messages: (a) real-time threat notices (aircraft takeoffs, missile and drone directions), which come before or at the same time as alerts and leak if used carelessly, and (b) morning summaries of launched and downed drones and missiles, which are published after the night is over. The summaries are only safe as lagged (t-1 or earlier) daily features.

### Cited Findings
- Daily summaries on Telegram give drones and missiles launched, shot down, and "lost" (electronic warfare). Example: 28 of 74 shot down and 46 "lost". — [MarketScreener/Reuters](https://ca.marketscreener.com/quote/currency/US-DOLLAR-RUSSIAN-ROUBLE--2370597/news/Ukraine-says-it-destroys-28-Russian-drones-overnight-48546658/)
- Example summary for 25 Aug 2025: 76 of 104 Shahed-type and decoy drones intercepted. — [NV](https://english.nv.ua/nation/ukraine-downs-76-russian-drones-50539700.html)
- Record night: 728 drones and 13 missiles launched, 718 drones and 7 missiles destroyed (date not shown in the snippet; this was July 2025). — [Yahoo/Reuters](https://www.yahoo.com/news/russia-launches-record-728-drones-055628002.html)
- Summaries name the drone types (Shahed, Gerbera, "other") and the number of strike locations. — [Yahoo/UP](https://www.yahoo.com/news/articles/ukrainian-air-force-downs-35-062300748.html)
- Real-time posts announce Tu-95MS takeoffs and estimate when bombers will reach launch lines (for example, "about 05:00-05:30"). — [Ukrainska Pravda, 8 Jan 2024](https://www.pravda.com.ua/eng/news/2024/01/8/7436250/)
- MiG-31K takeoff causes a nationwide alert immediately, so the post and the alert happen at the same time. — [UP, 6 Nov 2023](https://www.pravda.com.ua/eng/news/2023/11/06/7427447/)
- Summaries are later revised and include decoys (Gerbera, Italmas). — [ISIS-online June 2026 report](https://isis-online.org/uploads/isis-reports/documents/June-2026-Updated-Analysis-of-Russian-Shahed-type-UAVs-Deployment-Against-Ukraine-up-to-July-6.pdf)

### Inferences
- Leakage: a morning summary describes the night that just ended. Join it to day d+1 or later, never to night d. Use the Telegram message `date`, not the event date. Track `edit_date`, because posts get edited.
- Real-time threat posts are useful for nowcasting over the next 0-6 hours (bombers), not for day-ahead forecasts. MiG-31K posts give zero lead time.

### Gaps
- The handle t.me/kpszsu was not confirmed by a primary-page fetch. It is widely cited as the Air Force channel ("Povitriani Syly ZSU"), but verify it on the official mil.gov.ua / Air Force website.
- No systematic measurement of the lag between a threat post and an alert was found.

## 2. Monitoring Telegram channels and NLP research

### Takeaway
Volunteer "monitoring" channels add route and timing detail, but they are informal, sometimes inaccurate, and have been pressured to limit real-time detail. One academic paper (Pavlyshenko and Pavlyshenko, 2024) uses Telegram text plus NLP heuristics to forecast alert starts.

### Cited Findings
- arXiv:2411.14625 (21 Nov 2024, D. Pavlyshenko and B. Pavlyshenko) builds a Telegram channel that predicts alert starts from messages in other Telegram channels using NLP and heuristics. It finds that alerts in neighbouring regions and seasonality features (hour, day of week, month) are strong predictors. The abstract reports no accuracy or lead-time metrics. — [arXiv](https://arxiv.org/abs/2411.14625)
- Monitoring channels gave launch-line estimates (Saratov or Caspian zone around 00:00-01:00) and missile arrival windows (about 07:40-08:25). — [Censor.net](https://censor.net/en/news/3501616/strategic_bombers_take_off_in_russia); [Censor.net](https://censor.net/en/news/3520072/aktyvnist_rosiyiskoyi_aviatsiyi)
- AFP describes spotter channels as giving "less certainty, shorter warning" than a weather app. — [Asharq Al-Awsat/AFP](https://english.aawsat.com/node/5307591)
- Air Force spokesman Ihnat warned real-time tracking channels about possible criminal liability, and some channels made their updates vaguer. He later said there was no ban. — [Kyiv Post](https://kyivpost.com/post/14817)
- The SBU has sent more than 1,500 channels for blocking (propaganda and fakes, not spotters). — [IMI](https://imi.org.ua/en/news/sbu-named-telegram-channels-coordinated-by-russia-s-special-services-i44524)
- Qualitative study of how Telegram air-raid channels produce OSINT knowledge (AoIR 2024, Bystrytska). — [SPIR](https://spir.aoir.org/ojs/index.php/spir/article/download/13912/11770/89742)

### Inferences
- Channel behaviour changes over time (policy pressure), so text features are non-stationary.
- Many monitoring posts are reactions to alerts already in progress. Use only messages timestamped strictly before the forecast cutoff, and exclude posts that repeat the alert itself.
- Access: Telethon/MTProto with a user api_id and api_hash. History is available per channel. Deleted messages are lost, and edits overwrite the text (only `edit_date` is kept). Archive messages in real time if exact-as-of text is needed.

### Gaps
- No published quantitative accuracy or lead-time evaluation of monitoring channels was found.
- No Ukrainian-language NER work specific to this domain was found in this pass.

## 3. Weather

### Takeaway
Officials name weather as one factor in choosing attack dates, but no quantitative study linking wind, cloud, or fog to Shahed volume was found. Open-Meteo gives free hourly historical and forecast weather.

### Cited Findings
- A Ukrainian military intelligence spokesperson said attack-date choice depends on missile availability, weather conditions, holidays, and other factors. — [ThePage.ua](https://thepage.ua/news/raketnye-obstrely-ukrainy-kogda-zhdat-novuyu-ataku-raket-i-shahed-rf)
- Fog made interception harder on 12 Dec 2023, according to the monitoring channel "Nikolayevskyi Vanyok". This was not an Air Force statement. — [Euromaidan Press](https://euromaidanpress.com/2023/12/12/ukraine-downs-9-15-explosive-drones-two-missiles-during-russias-latest-air-attack/)
- Open-Meteo archive endpoint: `https://archive-api.open-meteo.com/v1/archive`. Hourly variables include cloud cover (total, low, mid, high), wind at 10 m and 100 m plus gusts, and precipitation, rain, and snow. ERA5 covers 1940 to present with a 5-day delay. ECMWF IFS covers 2017 onward, updated every 6 hours. Non-commercial use is free; commercial use needs an API key. — [Open-Meteo docs](https://open-meteo.com/en/docs/historical-weather-api)

### Inferences
- Leakage: reanalysis (ERA5) is "observed" weather that was not known at forecast time. For a realistic setup, use archived forecasts issued before the cutoff (Open-Meteo has a separate Historical Forecast API, not verified here). Otherwise label it an oracle feature.
- Consider weather along launch corridors (Bryansk, Kursk, Krasnodar/Primorsko-Akhtarsk, Crimea) as well as over targets.

### Gaps
- No peer-reviewed or official quantitative weather-attack relationship was found.

## 4. Moon phase

### Takeaway
No evidence was found linking moon phase or illumination to attack timing. It is cheap to compute (for example with astral or ephem) and has no leakage, so test it empirically.

### Cited Findings
- Searches found only general statements about night attacks and pauses. One example: the first night without Shaheds in 48 days, in October 2024. — [Espreso](https://global.espreso.tv/russia-ukraine-war-russia-halts-nighttime-shahed-drone-attacks-on-ukraine-for-first-time-in-48-days)

### Gaps
- No studies or official statements were found.

## 5. Strategic aviation, MiG-31K, Black Sea ships

### Takeaway
Tu-95MS/Tu-160 takeoffs (Engels, Olenya) are announced publicly by the Air Force and monitoring channels, usually about 3-9 hours before the expected launch. They are strong same-night precursors, but some flights turn back or only simulate launches. MiG-31K takeoffs trigger alerts immediately, so they are part of the target, not a feature.

### Cited Findings
- Tu-95MS takeoff from Olenya with an estimated launch window of 05:00-05:30. — [UP, 8 Jan 2024](https://www.pravda.com.ua/eng/news/2024/01/8/7436250/)
- Conditional Air Force estimate: "if combat mission, missiles around 04:00-05:00". — [UP, 17 Sep 2023](https://pravda.com.ua/eng/news/2023/09/17/7420149)
- Examples of simulated launches and of Tu-22s returning to base. — [Censor.net](https://censor.net/en/news/3520072/aktyvnist_rosiyiskoyi_aviatsiyi); [Censor.net](https://censor.net/en/news/3501616/strategic_bombers_take_off_in_russia)
- MiG-31K takeoff leads to a nationwide alert. Durations: about 1.5 h (6 Nov 2023), about 2 h 40 min (14 Nov 2023), three alerts on 4 Aug 2025. — [UP](https://www.pravda.com.ua/eng/news/2023/11/06/7427447/); [UP](https://pravda.com.ua/eng/news/2023/11/14/7428722); [UP](https://www.pravda.com.ua/eng/news/2025/08/4/7524681)

### Gaps
- Black Sea Kalibr carriers: not covered in this pass. Navy spokespeople report the number of carriers at sea, but no source with timing was verified.
- No structured dataset of takeoff events was found. It would have to be built from Air Force Telegram posts.

## 6. FIRMS, ISW, ACLED, CSIS/ISIS Shahed data

### Takeaway
All of these are daily-level and lagged. Treat them as t-1 or slower features and time them by publication time, not event time.

### Cited Findings
- FIRMS: global NRT data are available within about 3 hours of the satellite pass (ultra-real-time about 30 min at some stations). The API needs a MAP_KEY. The limit is 5,000 transactions per 10 minutes, according to a FIRMS expert on the Earthdata forum (not in official docs). The API has area/country CSV endpoints and a data_availability endpoint for NRT vs standard-processing data. — [Earthdata FIRMS](https://www.earthdata.nasa.gov/learn/find-data/near-real-time/firms?page=1); [Earthdata forum](https://forum.earthdata.nasa.gov/viewtopic.php?p=23828); [FIRMS API](https://firms.modaps.eosdis.nasa.gov/api/)
- ACLED Ukraine Conflict Monitor: updated weekly on Wednesdays. Some users get faster access. — [ACLED](https://acleddata.com/use-access/when-are-acled-data-updated)
- ACLED reporting lag (Africa study, 2026): more than half of events are reported within 2 weeks, and the lag varies by event type. — [arXiv 2603.25964](https://arxiv.org/abs/2603.25964v1)
- ISW: daily "Russian Offensive Campaign Assessment" is dated to a day. No official publication time or data cutoff was found. ISW has flagged signs of upcoming massive strikes. — [UkraNews](https://ukranews.com/en/news/1109287-kremlin-prepares-for-massive-strike-on-ukraine-isw-points-to-signs)
- Institute for Science and International Security (isis-online.org, not CSIS) publishes monthly Shahed analyses. June 2026: average about 260 per day (May 29-Jun 11) vs about 151 per day (Jun 12-30). Peaks: 656 drones on Jun 2 and 611 on Jun 15. The reports describe 24-32 h multi-wave cycles running into daylight (April 2026). They rely on Air Force reports, which can be revised. — [ISIS June 2026](https://isis-online.org/uploads/isis-reports/documents/June-2026-Updated-Analysis-of-Russian-Shahed-type-UAVs-Deployment-Against-Ukraine-up-to-July-6.pdf)

### Inferences
- FIRMS hotspots could come from either strikes or ordinary fires, and they mostly reflect damage after an attack. They are better as a lagged target or confirmation signal than as a precursor.
- ACLED's weekly cadence and revisions make it unsuitable for intraday forecasting. If used at all, use it as a weekly lagged intensity measure and keep snapshot versions.
- ISW reports are usually published in the evening US Eastern time (this needs verification). Use them only for days after the publication timestamp.

### Gaps
- The ISW publication timestamp was not verified.
- No "CSIS Shahed dataset" was found. CSIS Futures Lab has a "Russian Firepower Strike Tracker", but its download access is unverified.

## 7. Cycles of mass attacks

### Takeaway
Russia stockpiles missiles between salvos. In 2025 the average gap between major salvos was about 8 days, but it varied from 2 to 38 days depending on production and diplomacy. "Days since last mass strike" is a leakage-free and plausibly useful feature.

### Cited Findings
- CSIS (Jensen and Atalan, 9 Sep 2025): 2025 average gap of 8.2 days, 38 days in August 2025 during negotiations, as short as 2 days at times. Average salvo size: about 100 weapons in 2022 vs about 370 in 2025. About 6,300 UAVs and missiles in July 2025 vs about 3,300 in August. — [CSIS](https://www.csis.org/analysis/russias-massed-strikes-strategy-coercion-salvo)
- ISW (via UP/UkraNews): Russia accumulates missiles and fires them roughly every 2 weeks in strikes of more than 40 missiles (Sep 2025). — [UkraNews](https://ukranews.com/en/news/1109287-kremlin-prepares-for-massive-strike-on-ukraine-isw-points-to-signs)
- Air Force: the stockpile supports roughly weekly mass strikes (Ihnat, about Nov 2025). — [NV](https://english.nv.ua/russian-war/russia-can-launch-massive-strikes-weekly-despite-factory-problems-air-force-says-50564585.html)
- Defence Intelligence in 2023: the interval between massive attacks grew to 3-4 weeks, and Russia spent a month stockpiling for the 9 Mar 2023 strike. — [UkraNews](https://ukranews.com/en/amp/news/921697-invaders-not-abandon-missile-terror-but-interval-between-massive-attacks-increased-to-3-4-weeks); [UP](https://pravda.com.ua/eng/news/2023/03/10/7392832)

### Inferences
- Use hazard-style features (days since the last strike with more than N missiles, rolling launch counts). Compute them only from data available before the cutoff, using t-1 Air Force summaries.
- The regime changes over time (2023 vs 2025 vs 2026), so validate with time-based splits.

### Gaps
- No public, downloadable salvo-interval dataset was verified.
