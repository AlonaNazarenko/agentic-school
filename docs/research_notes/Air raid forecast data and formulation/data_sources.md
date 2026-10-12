# Backfillable Data Sources for Forecasting Air-Raid Alert Starts per Oblast (2025–2026)

Verification date: 2026-10-12. "Verified" means I read it directly from the source (API/raw file), not from a search snippet.

## 1. Kaggle "Massive Missile Attacks on Ukraine" (piterfm / Petro Ivaniuk)

### Takeaway
This dataset is still active (version 214, last updated 2026-10-10). Each row is one attack wave per weapon model, with `time_start`/`time_end`, `launch_place`, `target`, `launched`/`destroyed`, and a hit-region list. It was compiled after the fact from Air Force posts, so it is useful for lagged features (for example, previous nights' intensity and launch areas). Its timestamps are not usable as real-time leading signals. Usefulness: MEDIUM as lagged/regime features; LOW as a leading signal.

### Cited Findings
- Kaggle API metadata (verified): owner Petro Ivaniuk (piterfm), licence **CC BY-NC-SA 4.0** (non-commercial), current version 214, lastUpdated 2026-10-10T15:30Z, 22,251 downloads — [Kaggle API list](https://www.kaggle.com/api/v1/datasets/list?search=massive+missile+attacks); [dataset page](https://www.kaggle.com/datasets/piterfm/massive-missile-attacks-on-ukraine)
- Files (verified): `missile_attacks_daily.csv` (902,798 bytes) and `missiles_and_uavs.csv` (5,396 bytes) — [Kaggle files API](https://www.kaggle.com/api/v1/datasets/list/piterfm/massive-missile-attacks-on-ukraine)
- `missile_attacks_daily.csv` columns (from the dataset description, verified): `time_start`, `time_end`, `model`, `launch_place`, `target`, `target_main`, `status_data` (e.g. `hidden` = counts not published; if only destroyed is published, launched is filled with the destroyed value as a minimum), `carrier`, `launched`, `destroyed`, `not_reach_goal` (since July 2024), `border_crossing`, `still_attacking`, `num_hit_location`, `num_fall_fragment_location`, `affected_region`, `destroyed_details`, `launched_details`, `launch_place_details`, `source` — [Kaggle metadata API](https://www.kaggle.com/api/v1/datasets/metadata/piterfm/massive-missile-attacks-on-ukraine)
- `missiles_and_uavs.csv` columns: `model, category, national_origin, type, launch_platform, name, name_NATO, in_service, designer, manufacturer, guidance_system, unit_cost` — same source
- The description gives a lookup table from `launch_place` to coordinates (e.g. Navlya/Bryansk 52.8281,34.4989; Khalino/Kursk; Millerovo/Rostov; Primorsko-Akhtarsk/Krasnodar; Shatalovo/Smolensk) — same source
- Source: compiled manually from Air Force Command (kpszsu) and General Staff Facebook/Telegram reports, sometimes corrected with the Monitor/Monitorwar Telegram channels. Coverage starts October 2022 (massive strikes), plus some single UAVs — same source
- Data warning: "The Ukrainian Air Force has stopped reporting the number of some kind of missiles launched and intercepted since August 10, 2026" — same source. ISIS-online likewise notes that the Air Force stopped consistently publishing separate Shahed/Geran counts after about 5 May 2026, and that its daily reports have become less detailed — [ISIS Aug 2026 report](https://isis-online.org/uploads/isis-reports/documents/August_2026-Updated-Analysis-of-Russian-Shahed-type-UAVs-Deployment-Against-Ukraine_Final.pdf)
- There is no GitHub mirror under the piterfm account. Its public repos are unrelated and date from 2017–2021 — [GitHub API piterfm repos](https://api.github.com/users/piterfm/repos)

### Inferences
- Timestamp semantics: `time_start`/`time_end` are the event times the Air Force states for the attack. The Air Force summary is typically posted the next morning, and Kaggle updates happen every few days. To use the data without leakage, a row should only become available after its publication time (a conservative rule is time_end plus several hours, or the next day 09:00 Kyiv). There is no publication-time column, so this lag has to be imposed rather than observed.
- The dataset is useful for building features such as rolling 1/3/7-day launch counts by model, days since the last massive attack, and the share of drones vs. missiles. `launch_place` can serve as a direction prior for oblasts.
- `target`/`affected_region` are free text and need parsing to map them to oblasts.

### Gaps
- I could not download the CSV without Kaggle authentication, so I could not confirm exact value formats, 2026 row counts, or the share of null `launch_place`.

## 2. Vadimkin/ukrainian-air-raid-sirens-dataset

### Takeaway
This repo has official and volunteer (eTryvoga) alert start/end times at oblast, raion and hromada level, updated daily through 2026-10-12, under an MIT licence. It has **no threat-type or reason field**. It is a strong source for the target variable, but it adds no new leading information.

### Cited Findings
- Licence MIT, updated daily, coverage since 24 Feb 2022, official and volunteer (eTryvoga) sources — [repo](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
- Files: `official_data_en.csv`, `official_data_uk.csv`, `volunteer_data_en.csv`, `volunteer_data_uk.csv`, plus geojson. Volunteer rows with a missing end time are marked naive, with end = start + 30 min — [datasets/README](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset/blob/main/datasets/README.md)
- Header of `official_data_en.csv` (verified by curl): `oblast,raion,hromada,level,started_at,finished_at,source`. Timestamps are UTC (e.g. `2022-03-15 16:10:34+00:00`). Official data starts 2022-03-15, and the last row I saw was 2026-09-07 (raion level) — [raw CSV](https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv)
- Header of `volunteer_data_en.csv` (verified): `region,started_at,finished_at,naive`. It runs from 2022-02-25 to 2026-10-12 — [raw CSV](https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_en.csv)

### Inferences
- `level` (oblast/raion/hromada) lets you derive features such as "how many raions in a neighbouring oblast are already alerted". These are legitimate leading spatial-propagation features, because alerts in neighbouring areas are public at event time.
- The official file's tail ended on 2026-09-07 at the time I checked, while the volunteer file was current. The official file may lag, or the rows may not be sorted; check this before use.

### Gaps
- Neither file gives a threat type. eTryvoga's own Telegram posts include threat text, but no structured archive was found.

## 3. Alert archives with a threat type (alerts.in.ua, ukrainealarm)

### Takeaway
alerts.in.ua exposes `alert_type`, but its history endpoint only goes back one month (`month_ago`), so there is no backfill. A threat-typed history would have to be collected going forward by polling. The ukrainealarm API docs returned 403, so I could not verify them.

### Cited Findings
- alerts.in.ua `alert_type` values: `air_raid`, `artillery_shelling`, `urban_fights`, `chemical`, `nuclear`. History endpoint: `/v1/regions/{uid}/alerts/month_ago.json`, limited to 2 requests/min. Overall rate limit is about 8–12 requests/min per IP. A token is required, and the terms say it must not be used for critical infrastructure — [devs.alerts.in.ua](https://devs.alerts.in.ua/)
- api.ukrainealarm.com needs an API key requested via a form, and its Swagger is at /swagger/index.html — [PaulAnnekov/ukrainealarm README](https://raw.githubusercontent.com/PaulAnnekov/ukrainealarm/main/README.md); [PyPI ua-alarm](https://pypi.org/project/ua-alarm/1.3.9/). The Swagger and root pages returned HTTP 403 to my fetch — [swagger](https://api.ukrainealarm.com/swagger/index.html)

### Inferences
- alerts.in.ua's alert types describe the category of alert (air raid, shelling, etc.), not the weapon (drone, missile, ballistic). So even live, they would add little for air-raid starts.

### Gaps
- I could not confirm ukrainealarm's `alertType` enum or whether it has a history endpoint (403).
- I found no public archive of weapon-specific threat types (e.g. "ballistic threat", "Shahed toward X"). Those messages exist only as text in Air Force/monitor Telegram channels and would need scraping with message timestamps (which are publication times, so they are leakage-safe).

## 4. Shahed/launch-site/direction data, NOTAMs, strategic aviation

### Takeaway
The Institute for Science and International Security (ISIS-online) publishes monthly aggregates as PDFs, not event data. No open, structured, time-stamped dataset was found for drone entry directions, Russian NOTAMs, or Tu-95/Tu-160 movements. The realistic route to these signals is scraping Air Force and monitoring Telegram channels.

### Cited Findings
- ISIS-online (Institute for Science and International Security) publishes monthly PDF analyses of Shahed deployment based on Air Force data. 2025 totals: 54,538 Shahed-type UAVs, of which about 32,200 were strike UAVs. May 2026 was a record at 8,161 — [2025 review](https://isis-online.org/isis-reports/a-comprehensive-analytical-review-of-russian-shahed-type-uavs-deployment-against-ukraine-in-2025); [May 2026 PDF](https://isis-online.org/uploads/isis-reports/documents/May-2026-Updated-Analysis-of-Russian-Shahed-type-UAVs-Deployment-Against-Ukraine_Final.pdf)
- A "Shahed Tracker" is cited as a source in Re: Russia charts, but no downloadable dataset was confirmed — [Datawrapper chart](https://datawrapper.dwcdn.net/XGEdA/1)
- Airwars mapped Shahed launches only for 13 Sep 2022 – 30 Aug 2023, which is outside 2025–26 — [Airwars](https://airwars.org/?p=92726)
- NOTAMs: the FAA NOTAM API serves active NOTAMs. ICAO iSTARS has a "Stored NOTAMs" API sourced from FAA DINS every 3 hours. Russian coverage and archive depth are unconfirmed — [ICAO iSTARS](https://icao.int/safety/iStars/Pages/Get-NOTAM-Data.aspx); [NBAA on FAA archive](https://nbaa.org/aircraft-operations/airspace/navigator-improved-notam-search-coming-this-fall/)

### Inferences
- If a historical Russian NOTAM archive can be obtained, NOTAMs are publication-timed and so leakage-safe. They could be a leading indicator for missile launches from the Caspian/Black Sea, but this is unproven.

### Gaps
- I found no open dataset tracking Russian strategic aviation (Tu-95MS/Tu-22M3 take-offs). Only monitor channel posts exist.
- I found no structured archive of Air Force direction posts ("group of UAVs moving toward...").

## 5. Electricity outage schedules (Ukrenergo, DTEK)

### Takeaway
No public historical archive of outage schedules was found in this search. Outages are also mostly a consequence of strikes, not a leading signal. Usefulness: LOW.

### Cited Findings
- None verified. A GitHub repository search for outage-schedule archives returned no relevant results ([GitHub search API](https://api.github.com/search/repositories?q=ukraine+power+outage+schedule+archive)).

### Gaps
- I did not verify whether Ukrenergo or DTEK publish historical schedules. Community Telegram bots probably have logs, but this is not confirmed.

## 6. Other strike datasets (UCDP Candidate, Airwars, MotherDuck Skyfall, DiXi)

### Takeaway
UCDP Candidate is a free, global, monthly event dataset with roughly a one-month lag. That makes it too slow for any feature except long-run regime context. MotherDuck "Skyfall" is a visualisation built on Air Force daily reports (probably overlapping with piterfm), and I found no downloadable dataset for it. I did not verify DiXi's energy map.

### Cited Findings
- UCDP Candidate: monthly releases with a lag of no more than about one month. Data is extracted on the 20th of each month. It is global after v20.0.10, free to download as CSV, and also available through the UCDP API. 2026 releases (26.0.x) are listed — [UCDP downloads](https://ucdp.uu.se/downloads/); [codebook 1.5](https://ucdp.uu.se/downloads/candidateged/ucdp-candidate-codebook1.5.pdf)
- MotherDuck "SkyFall: 1,350 nights of drones & missiles over Ukraine" (Volodymyr Pivoshenko, 11 Jun 2026) covers Sep 2022 – Jun 2026 and is built from the official UA Air Force daily reports — [MotherDuck dive](https://motherduck.com/dive-gallery/dives/skyfall-1350-nights-of-drones-missiles-over-ukraine)

### Inferences
- UCDP and ACLED have event dates at daily resolution and publication lags of days to weeks. Fitting this to 15–60-minute horizons is only possible as lagged daily or weekly intensity features.

### Gaps
- DiXi energy map: not researched or verified.
- Skyfall's underlying table schema and access are unknown.

## Overall usefulness ranking (author judgment)
1. Vadimkin sirens (raion/hromada level, neighbour propagation): HIGH, leakage-safe at event time.
2. piterfm Kaggle (lagged attack intensity and launch places): MEDIUM, needs an imposed publication lag.
3. Telegram scraping (Air Force/monitor channels; threat type and direction, publication-timed): potentially HIGH, but needs to be built. Not a ready dataset.
4. alerts.in.ua alert_type: LOW (one month of history, coarse types).
5. UCDP/ACLED/ISIS/Skyfall: LOW (daily or monthly, lagged).
6. NOTAMs, strategic aviation, outage schedules: no accessible historical data confirmed.
