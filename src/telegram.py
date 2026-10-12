"""Telegram monitoring channels: scraping, message parsing and leakage-safe grid features.

Leakage rules:
- a message is used only from the end of the 5-min bin it was published in (bin t → decision time t + STEP);
- the text we scrape is the *current* text; messages edited later than EDIT_TOLERANCE after
  publication are dropped, because their text may contain information from after publication.
"""
import asyncio
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

STEP = pd.Timedelta("5min")
EDIT_TOLERANCE = pd.Timedelta("5min")

# ---------------------------------------------------------------- dictionaries (Ukrainian, lowercased)
OBLAST_PATTERNS = {
    "Vinnytska oblast":        r"вінниц\w*|вінничин\w*",
    "Volynska oblast":         r"волин\w*|луцьк\w*|ковел\w*",
    "Dnipropetrovska oblast":  r"дніпропетровщин\w*|дніпропетровськ\w*|\bдніпр[оаіу]\b|кривий ріг|кривого рогу|кривому розі|кривор\w*|нікопол\w*|павлоград\w*|кам'янськ\w*|марганц\w*",
    "Donetska oblast":         r"донеччин\w*|донецьк\w*|краматорськ\w*|слов'янськ\w*|покровськ\w*|костянтинівк\w*|дружківк\w*",
    "Zhytomyrska oblast":      r"житомир\w*|бердичів\w*|коростен\w*",
    "Zakarpatska oblast":      r"закарпат\w*|ужгород\w*|мукачев\w*",
    "Zaporizka oblast":        r"запоріж\w*",
    "Ivano-Frankivska oblast": r"франківськ\w*|прикарпатт\w*",
    "Kyivska oblast":          r"київщин\w*|київськ\w* обл\w*|бровар\w*|борисп\w*|біл\w* церкв\w*|вишгород\w*|фастів\w*|ірпін\w*|обухів\w*",
    "Kyiv City":               r"\bки[їє]в(?!щин|ськ)\w*",
    "Kirovohradska oblast":    r"кіровоградщин\w*|кропивницьк\w*",
    "Lvivska oblast":          r"львів\w*",
    "Mykolaivska oblast":      r"миколаїв\w*|вознесенськ\w*",
    "Odeska oblast":           r"одес\w*|ізмаїл\w*|чорноморськ\w*|білгород-дністровськ\w*",
    "Poltavska oblast":        r"полтав\w*|кременчук\w*|миргород\w*|лубн\w*",
    "Rivnenska oblast":        r"\bрівн(?:е|ого|ому|енщин\w*|енськ\w*)\b",
    "Sumska oblast":           r"\bсум(?:и|ах|ам|ами|щин\w*|ськ\w*)\b|шостк\w*|конотоп\w*|охтирк\w*",
    "Ternopilska oblast":      r"тернопіл\w*|тернопільщин\w*",
    "Kharkivska oblast":       r"харків\w*|куп'янськ\w*|ізюм\w*|чугуїв\w*",
    "Khersonska oblast":       r"херсон\w*",
    "Khmelnytska oblast":      r"хмельниц\w*|хмельниччин\w*|старокостянтинів\w*",
    "Cherkaska oblast":        r"черкас\w*|уман\w*",
    "Chernivetska oblast":     r"чернівц\w*|буковин\w*",
    "Chernihivska oblast":     r"чернігів\w*|черніг\w*|ніжин\w*|прилук\w*",
}

WEAPON_PATTERNS = {
    "kab":       r"\bкаб\w*|авіабомб\w*|керован\w* авіа\w* бомб\w*|\bфаб\b|\bумпк\b",
    "shahed":    r"шахед\w*|shahed|\bбпла\b|ударн\w* (?:бпла|дрон\w*)|мопед\w*|герань",
    "missile":   r"ракет\w*|крилат\w*|х-101|х-555|х-59|х-69|калібр\w*|швидкісн\w* ціл\w*",
    "ballistic": r"балістик\w*|балістичн\w*|іскандер\w*|кн-23|с-300|с-400",
    "aviation":  r"ту-95|ту-160|ту-22|міг-31|тактичн\w* авіаці\w*|су-34|су-35",
}
DIRECTION_PATTERN = r"курсом на|у напрямку|в напрямку|напрямок|рухаються на|летять на|в бік|у бік"

_OBL = {o: re.compile(p) for o, p in OBLAST_PATTERNS.items()}
_WPN = {w: re.compile(p) for w, p in WEAPON_PATTERNS.items()}
_DIR = re.compile(DIRECTION_PATTERN)


def parse_message(text: str) -> dict:
    """{'oblasts': [...], 'weapons': [...], 'direction': bool}"""
    t = (text or "").lower().replace("’", "'").replace("ʼ", "'")
    return {"oblasts": [o for o, r in _OBL.items() if r.search(t)],
            "weapons": [w for w, r in _WPN.items() if r.search(t)],
            "direction": bool(_DIR.search(t))}


# ---------------------------------------------------------------- scraping
def _load_env(path: Path = Path(".env")):
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


SAVE_EVERY = 1000          # checkpoint to disk every N new posts
WAIT_BETWEEN_REQUESTS = 1  # seconds between 100-message history requests (gentle on rate limits)
MAX_AUTO_FLOOD_WAIT = 300  # Telegram "slow down" waits up to 5 min are slept through; longer → save and stop


def _save(old: pd.DataFrame, rows: list, out: Path) -> pd.DataFrame:
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["channel", "id"])
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)                                     # atomic: an interrupted save never corrupts the file
    return df


async def _scrape(channels, since, out: Path, session: str):
    import time
    from telethon import TelegramClient
    from telethon.errors import FloodWaitError
    _load_env()
    api_id, api_hash = int(os.environ["TG_API_ID"]), os.environ["TG_API_HASH"]
    old = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    rows, total = [], len(old)
    async with TelegramClient(session, api_id, api_hash, flood_sleep_threshold=MAX_AUTO_FLOOD_WAIT) as client:
        try:
            for ch in channels:
                min_id = int(old.loc[old.channel == ch, "id"].max()) if len(old) and (old.channel == ch).any() else 0
                print(f"{ch}: {'resuming after id ' + str(min_id) if min_id else 'starting from ' + str(since.date())}")
                n, t0 = 0, time.time()
                async for m in client.iter_messages(ch, reverse=True, offset_date=since, min_id=min_id,
                                                    wait_time=WAIT_BETWEEN_REQUESTS):
                    if not m.message:
                        continue
                    rows.append({"channel": ch, "id": m.id, "date": m.date, "edit_date": m.edit_date,
                                 "fwd": m.fwd_from is not None, "text": m.message})
                    n += 1
                    if n % 100 == 0:
                        print(f"\r  {ch}: {n} new posts, up to {m.date:%Y-%m-%d %H:%M} UTC "
                              f"({n / max(time.time() - t0, 1):.0f}/s)", end="", flush=True)
                    if len(rows) >= SAVE_EVERY:
                        old, rows = _save(old, rows, out), []
                        print(f"  → saved ({len(old)} total)", flush=True)
                print(f"\n{ch}: done, {n} new posts")
        except FloodWaitError as e:
            print(f"\nTelegram asked to wait {e.seconds}s — saving and stopping. Re-run later to continue.")
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nInterrupted — saving what was downloaded. Re-run to continue.")
        finally:
            old = _save(old, rows, out) if rows else old
            print(f"saved {len(old)} posts in total ({len(old) - total} new this run) → {out}")
    return old


def scrape(channels, since, out="data/telegram_raw.parquet", session="data/tg_session"):
    """Download all text messages since `since`. Saves every SAVE_EVERY posts and on Ctrl+C;
    re-running resumes after the last saved post id."""
    try:
        return asyncio.run(_scrape(channels, pd.Timestamp(since).to_pydatetime(), Path(out), session))
    except KeyboardInterrupt:
        return pd.read_parquet(out) if Path(out).exists() else pd.DataFrame()


# ---------------------------------------------------------------- cleaning + grid features
def clean(raw: pd.DataFrame, tz="Europe/Kyiv") -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(tz)
    ed = pd.to_datetime(df["edit_date"], utc=True).dt.tz_convert(tz)
    late_edit = ed.notna() & (ed - df["date"] > EDIT_TOLERANCE)
    df = df[~late_edit & ~df["fwd"].astype(bool)]           # forwarded posts: original time unknown
    parsed = df["text"].map(parse_message)
    df["oblasts"] = parsed.map(lambda p: p["oblasts"])
    df["weapons"] = parsed.map(lambda p: p["weapons"])
    df["direction"] = parsed.map(lambda p: p["direction"])
    return df.reset_index(drop=True)


TG_WINDOWS = {"15m": "15min", "60m": "60min"}


def tg_features(msgs: pd.DataFrame, grid: pd.DatetimeIndex, oblasts, times) -> pd.DataFrame:
    """Long frame (time, oblast) aligned with features.build_features.

    Per oblast: messages mentioning it (any / per weapon) in the last 15 / 60 min.
    Global: messages per weapon type in the last 15 / 60 min.
    """
    b = ((msgs["date"] - grid[0]) // STEP).astype(int).values      # bin of publication
    ok = (b >= 0) & (b < len(grid))
    msgs, b = msgs[ok], b[ok]
    O, T = len(oblasts), len(grid)
    col = {o: i for i, o in enumerate(oblasts)}
    weapons = list(WEAPON_PATTERNS)

    per_obl = {k: np.zeros((T, O), np.float32) for k in ["any"] + weapons}
    glob = {k: np.zeros((T, 1), np.float32) for k in ["all"] + weapons}
    for bi, obls, wps in zip(b, msgs["oblasts"], msgs["weapons"]):
        glob["all"][bi, 0] += 1
        for w_ in wps:
            glob[w_][bi, 0] += 1
        for o in obls:
            if o in col:
                per_obl["any"][bi, col[o]] += 1
                for w_ in wps:
                    per_obl[w_][bi, col[o]] += 1

    def roll(x, n):
        cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
        lo = np.maximum(np.arange(len(x)) + 1 - n, 0)
        return cs[np.arange(len(x)) + 1] - cs[lo]

    pos = grid.get_indexer(pd.DatetimeIndex(times))
    out = {}
    for wn, wd in TG_WINDOWS.items():
        n = int(pd.Timedelta(wd) / STEP)
        for k, a in per_obl.items():
            out[f"tg_obl_{k}_{wn}"] = roll(a, n)[pos].ravel()
        for k, a in glob.items():
            out[f"tg_all_{k}_{wn}"] = np.repeat(roll(a, n)[pos], O, axis=1).ravel()
    idx = pd.MultiIndex.from_arrays([np.repeat(grid[pos], O), np.tile(list(oblasts), len(pos))],
                                    names=["time", "oblast"])
    return pd.DataFrame(out, index=idx)


# ================================================================ v2: gazetteer, location vs target, counts, upstream, relative
_ADJ_SUFFIX = re.compile(r"(?:ськ|цьк|зьк)(?:ий|а)$")


def _stems_from_name(name: str) -> list:
    """'Уманський район' → ['уманськ', 'уман'];  'м. Миргород та Миргородська …' → ['миргород', …]."""
    name = name.lower().replace("’", "'").replace("ʼ", "'")
    out = []
    m = re.match(r"м\.\s*([\w'-]+)", name)
    if m:
        out.append(m.group(1))
        name = name.split(" та ", 1)[-1]
    word = name.split()[0]
    if word.endswith(("ий", "а")):
        out.append(word[:-2] if word.endswith("ий") else word[:-1])   # 'уманськ' – matches 'Уманському районі'
        city = _ADJ_SUFFIX.sub("", word)                                # 'уман'    – matches 'Умань', 'Умані'
        if city != word:
            out.append(city)
    return out


def build_gazetteer(uk_csv: Path, en_csv: Path, min_len: int = 4) -> dict:
    """{stem: oblast_en} from raion and hromada names of the official dataset (Ukrainian file, row-aligned with
    the English one). Stems shared by several oblasts or shorter than `min_len` are dropped."""
    uk = pd.read_csv(uk_csv, usecols=["oblast", "raion", "hromada"])
    en = pd.read_csv(en_csv, usecols=["oblast"])
    uk["oblast_en"] = en["oblast"].values
    pairs = {}
    for col in ["raion", "hromada"]:
        for name, obl in uk[[col, "oblast_en"]].dropna().drop_duplicates().itertuples(index=False):
            for s in _stems_from_name(name):
                if len(s) >= min_len:
                    pairs.setdefault(s, set()).add(obl)
    return {s: next(iter(o)) for s, o in pairs.items() if len(o) == 1}


_COUNT = re.compile(r"(\d+)\s*(?:х\s*)?(?:ударн\w*\s*)?(?:бпла|шахед\w*|ракет\w*|каб\w*|ціл\w*|дрон\w*)")
_GROUP = re.compile(r"груп\w*")


class Parser:
    """Message parser with an optional gazetteer; separates where a threat IS from where it is HEADING."""

    def __init__(self, gazetteer: dict | None = None):
        self.obl = dict(_OBL)
        self.gaz = None
        if gazetteer:
            alt = "|".join(sorted(map(re.escape, gazetteer), key=len, reverse=True))
            self.gaz = (re.compile(rf"\b({alt})\w*"), gazetteer)

    def _oblasts(self, t: str) -> list:
        found = [o for o, r in self.obl.items() if r.search(t)]
        if self.gaz:
            rx, g = self.gaz
            for m in rx.finditer(t):
                o = g[m.group(1)]
                if o not in found:
                    found.append(o)
        return found

    def parse(self, text: str) -> dict:
        t = (text or "").lower().replace("’", "'").replace("ʼ", "'")
        d = _DIR.search(t)
        loc_txt, tgt_txt = (t[: d.start()], t[d.start():]) if d else (t, "")
        loc, tgt = self._oblasts(loc_txt), self._oblasts(tgt_txt)
        n = sum(int(x) for x in _COUNT.findall(t) if int(x) < 200)
        return {"oblasts": list(dict.fromkeys(loc + tgt)), "loc": loc, "tgt": tgt,
                "weapons": [w for w, r in _WPN.items() if r.search(t)], "direction": bool(d),
                "count": n if n else (2 if _GROUP.search(t) else 1)}


def clean_v2(raw: pd.DataFrame, parser: Parser, tz="Europe/Kyiv") -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(tz)
    ed = pd.to_datetime(df["edit_date"], utc=True).dt.tz_convert(tz)
    df = df[~(ed.notna() & (ed - df["date"] > EDIT_TOLERANCE)) & ~df["fwd"].astype(bool)]
    p = df["text"].map(parser.parse)
    for k in ["oblasts", "loc", "tgt", "weapons", "direction", "count"]:
        df[k] = p.map(lambda x: x[k])
    return df.reset_index(drop=True)


TG2_WINDOWS = {"15m": "15min", "60m": "60min", "180m": "180min"}


def tg_features_v2(msgs: pd.DataFrame, grid: pd.DatetimeIndex, oblasts, times, adj: np.ndarray,
                   lift: np.ndarray) -> pd.DataFrame:
    """Per (time, oblast), all trailing windows of posts published in bins <= t:
      tg2_loc / tg2_tgt         – oblast mentioned as current location / as destination
      tg2_cnt                   – number of objects (drones/missiles) in posts mentioning the oblast
      tg2_<weapon>              – mentions per threat type
      tg2_adj_loc / tg2_adj_tgt – same for border neighbours;  tg2_up_loc – lift-weighted upstream oblasts
      tg2_nat_<weapon>, tg2_nat_cnt – national totals
      tg2_rel_*                 – counts relative to the channel's posting rate over the previous 7 days
    """
    b = ((msgs["date"] - grid[0]) // STEP).astype(int).values
    ok = (b >= 0) & (b < len(grid))
    msgs, b = msgs[ok], b[ok]
    T, O = len(grid), len(oblasts)
    col = {o: i for i, o in enumerate(oblasts)}
    weapons = list(WEAPON_PATTERNS)
    per = {k: np.zeros((T, O), np.float32) for k in ["loc", "tgt", "cnt"] + weapons}
    nat = {k: np.zeros((T, 1), np.float32) for k in ["all", "cnt"] + weapons}
    for bi, loc, tgt, wps, cnt in zip(b, msgs["loc"], msgs["tgt"], msgs["weapons"], msgs["count"]):
        nat["all"][bi, 0] += 1
        nat["cnt"][bi, 0] += cnt
        for w_ in wps:
            nat[w_][bi, 0] += 1
        for o in loc:
            if o in col:
                per["loc"][bi, col[o]] += 1
        for o in tgt:
            if o in col:
                per["tgt"][bi, col[o]] += 1
        for o in set(loc) | set(tgt):
            if o in col:
                per["cnt"][bi, col[o]] += cnt
                for w_ in wps:
                    per[w_][bi, col[o]] += 1

    def roll(x, n):
        cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
        lo = np.maximum(np.arange(len(x)) + 1 - n, 0)
        return cs[np.arange(len(x)) + 1] - cs[lo]

    pos = grid.get_indexer(pd.DatetimeIndex(times))
    deg = np.maximum(adj.sum(axis=0, keepdims=True), 1)
    lw = lift / np.maximum(lift.sum(axis=0, keepdims=True), 1e-9)
    rate7 = roll(nat["all"], int(pd.Timedelta("7D") / STEP)) / (7 * 24)          # posts per hour, last 7 days
    out = {}
    for wn, wd in TG2_WINDOWS.items():
        n = int(pd.Timedelta(wd) / STEP)
        R = {k: roll(a, n) for k, a in per.items()}
        out[f"tg2_loc_{wn}"], out[f"tg2_tgt_{wn}"], out[f"tg2_cnt_{wn}"] = R["loc"], R["tgt"], R["cnt"]
        out[f"tg2_adj_loc_{wn}"] = R["loc"] @ adj / deg
        out[f"tg2_adj_tgt_{wn}"] = R["tgt"] @ adj / deg
        out[f"tg2_up_loc_{wn}"] = R["loc"] @ lw
        if wn != "180m":
            for w_ in weapons:
                out[f"tg2_{w_}_{wn}"] = R[w_]
        Nall = roll(nat["all"], n)
        out[f"tg2_nat_all_{wn}"] = np.repeat(Nall, O, axis=1)
        out[f"tg2_nat_cnt_{wn}"] = np.repeat(roll(nat["cnt"], n), O, axis=1)
        for w_ in weapons:
            out[f"tg2_nat_{w_}_{wn}"] = np.repeat(roll(nat[w_], n), O, axis=1)
        hours = n * STEP / pd.Timedelta("1h")
        rel = 1.0 / np.maximum(rate7 * hours, 1e-3)
        out[f"tg2_rel_nat_{wn}"] = np.repeat(Nall * rel, O, axis=1)
        out[f"tg2_rel_loc_{wn}"] = R["loc"] * rel
        out[f"tg2_rel_tgt_{wn}"] = R["tgt"] * rel
    cols = {k: v[pos].ravel().astype(np.float32) for k, v in out.items()}
    idx = pd.MultiIndex.from_arrays([np.repeat(grid[pos], O), np.tile(list(oblasts), len(pos))],
                                    names=["time", "oblast"])
    return pd.DataFrame(cols, index=idx)
