# How Russian aerial attacks on Ukraine unfold in time and space (2025–2026): domain knowledge for per-oblast alert forecasting

Scope note: these are historical and aggregate facts only. Search was limited to about 13 tool calls. Primary Air Force Telegram posts were not fetched. The ISIS June 2026 PDF could not be parsed (no PDF tools), so its contents are not used beyond the search snippet.

## Q1. Shahed speed, range, launch sites/routes, time to reach Kyiv and the west

### Takeaway
Propeller Shaheds (Geran-2) fly at roughly 180–250 km/h. Jet versions (Geran-3) fly at roughly 500–600 km/h at about 4 km altitude. Launches come from a fixed set of sites in Russia and occupied territory: Oryol/Tsymbulova, Navlya (Bryansk), Kursk-Khalino, Shatalovo (Smolensk, near Belarus), Primorsko-Akhtarsk, Millerovo, Crimea (Cape Chauda) and Donetsk airport. So the entry direction (north, northeast, east, south) is structurally predictable, and lag times scale with distance divided by speed.

### Cited Findings
- Regular Shahed is about 250 km/h. The jet Shahed is about 600 km/h and flies at about 4,000 m. Regular Shahed routes are mostly long straight segments; jet drones approach Kyiv from several directions and change course several times (Texty, 24 Sep 2026) — [Texty](https://texty.org.ua/articles/118344/endless-cycle-of-shaheds-how-jet-powered-drones-bring-kyiv-to-a-standstill/)
- Ukrainian military intelligence puts Shahed-136 speed at up to about 180–185 km/h. Jet Geran-3 is put at about 500–600 km/h, with the Air Force spokesperson saying "over 500 km/h" (June 2025) — [Euromaidan Press/Defense Express](https://euromaidanpress.com/2025/06/11/russia-jet-powered-geran-3-drone-hits-kyiv/); [Kyiv Post](https://www.kyivpost.com/amp/post/57347)
- Jet drones were first used at relatively large scale against Kyiv in July 2025 (up to 8). From mid-August 2026 they caused near-continuous alerts in Kyiv at night, morning, day and evening, with concentrations at 17:00–19:00 (evening commute) — [Texty](https://texty.org.ua/articles/118344/endless-cycle-of-shaheds-how-jet-powered-drones-bring-kyiv-to-a-standstill/)
- Launch sites:
  - Oryol/Tsymbulova: about 175 km from the border, the main site, with at least 12 launchers.
  - Navlya (Bryansk): about 70 km from the border, vehicle launches only.
  - Shatalovo (Smolensk): about 46 km from the Belarus border, in use since early 2025. As of January 2026 it had 4 fixed pads and 4 launch vehicles, with at least 22 drones per salvo.
  - Primorsko-Akhtarsk: 8 racks, up to 25 drones per wave, struck in August 2025.
  - Kursk-Khalino.
  - Millerovo (Rostov): new storage site.
  - Sources: [Molfar](https://www.molfar.institute/en/orlando-milan-shanghai-where-do-russians-launch-shahed-drones-from-molfar-intelligence-institute-analyzes-secret-documents-from-alabuga/); [Militarnyi](https://militarnyi.com/en/news/russia-expands-shahed-drone-sites/); [Espreso](https://global.espreso.tv/russia-ukraine-war-ukrainian-strike-levels-russian-drone-hub-on-black-sea-coast-satellite-photos-reveal); [Komersant](https://komersant.ua/en/rosiia-rozghornula-p-iat-maydanchykiv-dlia-zapusku-shakhediv-poblyzu-kordonu-z-bilorussiu/) (these are OSINT, not official)
- Launches also come from Crimea and Donetsk. Ukraine struck the Donetsk airport launch site on 5 Nov and 30 Dec 2025. Oblasts near the border or front (Sumy, Chernihiv, Kharkiv, Odesa) have shorter flight times. Some UAVs strayed into Moldova, Romania, Poland and Belarus (ISIS, 22 Jan 2026) — [ISIS](https://isis-online.org/isis-reports/a-comprehensive-analytical-review-of-russian-shahed-type-uavs-deployment-against-ukraine-in-2025)
- Shaheds transited Belarusian airspace in 2024; one flew more than 260 km over Belarus to Stolin. The likely launch area was Bryansk — [Online.ua](https://news.online.ua/en/17-shaheeds-flew-into-belarus-overnight-during-russias-attack-on-ukraine-889472/); [Babel](https://babel.ua/en/news/109499-5-shahed-drones-flew-to-belarus-at-night-there-have-been-more-such-incidents)

### Inferences
- At 180–250 km/h, a drone takes about 24–33 minutes per 100 km.
  - Bryansk/Kursk to Kyiv (about 300–450 km) is roughly 1.5–2.5 hours of flight.
  - Western oblasts (Lviv/Volyn) are 700–1,000+ km away by route, so 4–6+ hours.
  - Jet drones are about 3x faster.
  - Feature idea: per-oblast lags of upstream oblast alerts along corridors (for example Sumy/Chernihiv, then Kyiv, then Zhytomyr, then Rivne/Volyn; and Odesa/Mykolaiv from the south). Also oblast distance to the nearest launch site.
- Shatalovo plus the Belarus transit implies a northern approach to Volyn, Rivne and Zhytomyr along the Belarus border.

### Gaps
- No source measured actual launch-to-arrival times for Kyiv or western oblasts. No open dataset of drone tracks was found; Texty has route maps for Kyiv only. Hvardiiske (Crimea) was not confirmed as a launch site.

## Q2. How air raid alerts are declared; 2025–2026 policy changes

### Takeaway
Since 15 December 2025, alerts are issued by raion (district) nationwide, except in Donetsk and Luhansk oblasts. They are run jointly by DSNS (State Emergency Service), the Air Force and the oblast military administrations, after a pilot in 13 regions. Signal transfer from the Air Force to DSNS dropped from minutes to 8–15 seconds. Every MiG-31K takeoff triggers a nationwide alert, because it carries the Kinzhal.

### Cited Findings
- PM Svyrydenko (15 Dec 2025) announced raion-level alerting. It was piloted in 13 regions and extended to all oblasts except Donetsk and Luhansk. Reported effects:
  - Some Dnipropetrovsk hromadas gained 100+ alert-free days.
  - Sumy Oblast alert time fell by 50 days.
  - Some Kharkiv hromadas lost more than a month of alert time.
  - Air Force to DSNS signal transfer now takes 8–15 seconds.
  - Source: [Cabinet of Ministers](https://www.kmu.gov.ua/news/yuliia-svyrydenko-zaprovadzhuiemo-novyi-pidkhid-do-oholoshennia-povitrianykh-tryvoh-poraionne-opovishchennia); [Militarnyi](https://militarnyi.com/uk/news/kabmin-ogolosyv-novu-systemu-spovishhennya-pro-povitryani-tryvogy-po-vsij-ukrayini/)
- Kharkiv OVA had earlier split alerts by raion and did not rule out splitting further to the hromada level — [Suspilne](https://suspilne.media/kharkiv/1086637-u-harkivskij-oblasti-diut-okremi-trivogi-dla-rajoniv-ci-mozlivo-ce-dla-gromad/)
- A nationwide alert is declared on MiG-31K takeoff, because each takeoff is a potential Kinzhal launch (Ihnat) — [ITC](https://itc.ua/news/kazhdyj-vzlet-myg-31k-neset-opasnost-puska-kynzhala-sposobnogo-porazhat-lyubuyu-tochku-v-ukrayne-v-techenye-neskolkyh-mynut/); [Vozpopuli](https://www.vozpopuli.com/internacional/activan-alerta-aerea-toda-ucrania-posible-lanzamiento-misil-hipersonico-ruso.html)
- Kyiv uses a "yellow" alert level for jet-drone threats. A deadly strike on 23 Sep 2026 came during a yellow alert — [Texty](https://texty.org.ua/articles/118344/endless-cycle-of-shaheds-how-jet-powered-drones-bring-kyiv-to-a-standstill/)

### Inferences
- There is a structural break around December 2025 (and earlier in pilot oblasts). Oblast-level series built as "any raion alerting" will look different afterwards: shorter alerts and more partial alerts. Add a regime dummy, or model at raion level.
- Nationwide simultaneous alert starts (all oblasts within a minute) are a signature of MiG-31K or strategic-aviation events. Treat them as a separate event class.

### Gaps
- The exact Cabinet resolution text, the alert criteria and thresholds, and the list and dates of the 13 pilot regions were not found. No official statement on the "yellow" alert criteria was retrieved.

## Q3. Structure and timing of combined strikes; missile flight times

### Takeaway
Typical mass strikes start with drones launched in the evening or late night. Cruise and ballistic missiles follow, often in the early morning, and official tallies come out around 07:30–08:30. Warning time differs sharply by weapon:
- Kinzhal: about 3 minutes to Kyiv.
- Iskander-M: about 4 minutes from Bryansk.
- Kh-101 from the Caspian: about 2–2.5 hours from launch to entering Ukrainian airspace.

From mid-2025 Russia shifted from daily launches to mass attacks every 5–8 days on top of a steady ~180 drones per day.

### Cited Findings
- Monthly launch totals in 2025 (ISIS, 22 Jan 2026):
  - July: 6,297 (about 203 per day, the peak)
  - August: about 133 per day
  - September: about 188 per day
  - October: 5,298
  - November: 5,447
  - December: 5,131
- Other findings from the same ISIS report:
  - January–May 2025 saw daily launches. From June 2025 the pattern became a steady ~180 per day plus cumulative mass attacks every 5–8 days.
  - From summer 2025, attacks with more than 400 drones happened 4–5 times a month.
  - Attack waves often last several hours.
  - Source: [ISIS](https://isis-online.org/isis-reports/a-comprehensive-analytical-review-of-russian-shahed-type-uavs-deployment-against-ukraine-in-2025)
- Decoys (Gerbera, Parody) made up about 40% of launches in Q4 2025; strike drones were about 60% (58/62/60% in Oct/Nov/Dec). The strike share falls in mass attacks and reaches about 80% in small targeted ones — [ISIS](https://isis-online.org/isis-reports/a-comprehensive-analytical-review-of-russian-shahed-type-uavs-deployment-against-ukraine-in-2025)
- In April 2026 a single attack lasted about 32 hours and used 983 Shahed-type drones, 20 Kh-101, 23 Iskander-M and 5 Iskander-K. Official totals may undercount launches by 5–10% as daytime attacks grow (ISIS 2026 reports, from a search snippet; the PDF was not read) — [ISIS May 2026](https://isis-online.org/uploads/isis-reports/documents/May-2026-Updated-Analysis-of-Russian-Shahed-type-UAVs-Deployment-Against-Ukraine_Final.pdf)
- Record overnight attacks: 472 drones (June 2025, Reuters via Deccan Herald) and 524 drones plus 22 missiles (May 2026, Reuters via Arab News). Tallies are issued at about 07:30–08:30 — [Deccan Herald](https://www.deccanherald.com/amp/story/world%2Frussia-launches-472-drones-at-ukraine-overnight-air-force-says-3566722); [Arab News](https://www.arabnews.pk/node/2643986/world)
- The Air Force described drone groups moving toward several regions, followed by cruise and ballistic missiles — [Hromadske](https://hromadske.ua/en/amp/war/265624-russia-launches-ballistic-missiles-and-hundreds-of-drones-at-ukraine-overnight)
- Kh-101 timing examples (older, 2023–2024):
  - Caspian launch at about midnight, entering Ukrainian airspace at about 02:20 (June 2023).
  - Launch at about 17:30, first detection at 19:56 (April 2024).
  - Sources: [Ukrainska Pravda](https://www.pravda.com.ua/eng/news/2023/06/24/7408284/); [Militarnyi](https://militarnyi.com/en/news/russia-attacks-ukraine-with-kh-101-cruise-missiles-during-daytime/)
- Ihnat said cruise missiles from the Caspian or Black Sea give about an hour to react, while Iskanders give only minutes. Iskander-M takes about 238 s at 500 km range, so about 4 minutes to Kyiv from Bryansk (a media calculation) — [Ukranews](https://ukranews.com/en/news/1097100-there-is-not-enough-time-for-detection-air-forces-explain-features-of-russian-missile-attack-on-kyiv); [Telegraf](https://news.telegraf.com.ua/ukraina/2024-12-27/5891976-za-yakiy-chas-mozhe-doletiti-do-kieva-ta-dnipra-iskander-m-yakim-rosiya-vdarila-po-zaporizhzhyu)
- Kinzhal takes about 3.2 minutes to Kyiv (UA War Infographics, via Telegraf, June 2025) — [Telegraf](https://news.telegraf.com.ua/ukraina/2025-06-03/5910970-dlya-spaseniya-est-schitannye-minuty-za-skolko-kinzhal-doletaet-do-ukrainskikh-gorodov-karta)

### Inferences
- Useful features:
  - Hour of day, with an evening/night peak for drone alert starts and an early-morning peak for missile-driven alerts.
  - Days since the last mass attack, given the 5–8 day cycle.
  - Rolling national drone count.
  - A nationwide-alert flag.
  - Upstream oblast alerts lagged 0.5–3 hours.
- Ballistic alerts have near-zero lead time, so they are essentially unforecastable from upstream propagation. Separate them out or treat them as noise.
- Daytime and jet-drone activity rose in 2026, which weakens a purely nocturnal prior. Time-varying seasonality is needed.

### Gaps
- No systematic day-of-week analysis was found; ISIS explicitly gives none. No quantitative seasonal analysis was found beyond monthly totals. There is no measured drone-to-missile interval for 2025–2026. No CSIS/RUSI timing analysis was retrieved.

## Q4. Open analyses of alert spatial propagation

### Takeaway
There is little open spatial-propagation analysis. Texty.org.ua has Kyiv-focused route and alert-cycle infographics. ISIS publishes monthly aggregates. OSINT groups (Molfar, Belaruski Hajun) map launch sites and Belarus transits.

### Cited Findings
- Texty jet-drone route maps over Kyiv and an infographic of the full-day alert cycle on 28 Aug 2026 — [Texty](https://texty.org.ua/articles/118344/endless-cycle-of-shaheds-how-jet-powered-drones-bring-kyiv-to-a-standstill/)
- ISIS monthly Shahed reports, the basis for national intensity features — [ISIS index](https://isis-online.org/isis-reports/monthly-analysis-of-russian-shahed-136-deployment-against-ukraine)

### Inferences
- Propagation lags will likely need to be learned empirically from the alert data itself, for example cross-correlation between oblast alert starts.

### Gaps
- No published per-oblast alert propagation study was found.
