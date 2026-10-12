"""Download, clean and grid the official air raid alert history (same logic as notebook 01)."""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = ("https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
       "main/datasets/official_data_en.csv")
TZ = "Europe/Kyiv"
STEP = pd.Timedelta("5min")
MAX_DURATION_MIN = 3 * 24 * 60


def download(path: Path, force: bool = False) -> Path:
    if force or not path.exists():
        r = requests.get(URL, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["started_at", "finished_at"]:
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(TZ)
    df = df.dropna(subset=["started_at", "finished_at"])
    df = df[df["finished_at"] > df["started_at"]]
    dur = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60
    return df[dur < MAX_DURATION_MIN]


def _merge(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("started_at")
    s, e = g["started_at"].values, g["finished_at"].values
    run_end = np.maximum.accumulate(e)
    grp = np.cumsum(np.r_[True, s[1:] > run_end[:-1]])
    return pd.DataFrame({"start": pd.Series(s).groupby(grp).min().values,
                         "end": pd.Series(run_end).groupby(grp).max().values})


def episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Oblast on alert = any oblast/raion/hromada alert active in it; overlapping alerts merged."""
    ep = (df.groupby("oblast")[["started_at", "finished_at"]]
            .apply(_merge).reset_index(level=0).reset_index(drop=True))
    for c in ["start", "end"]:
        ep[c] = pd.to_datetime(ep[c], utc=True).dt.tz_convert(TZ)
    return ep


def to_grid(ep: pd.DataFrame, step: pd.Timedelta = STEP):
    """Return (state, starts): time × oblast int8 frames on a regular grid.
    Bin t covers [t, t+step); its information is known at the end of the bin."""
    t0, t1 = ep["start"].min().floor(step), ep["end"].max().ceil(step)
    grid = pd.date_range(t0, t1, freq=step)
    state, starts = {}, {}
    for o, e in ep.groupby("oblast"):
        i0 = ((e["start"] - t0) // step).astype(int).values
        i1 = np.ceil((e["end"] - t0) / step).astype(int).values
        d = np.zeros(len(grid) + 1, dtype=np.int32)
        np.add.at(d, i0, 1); np.add.at(d, i1, -1)
        state[o] = (np.cumsum(d)[:-1] > 0).astype(np.int8)
        s = np.zeros(len(grid), dtype=np.int8); s[i0] = 1
        starts[o] = s
    return pd.DataFrame(state, index=grid), pd.DataFrame(starts, index=grid)


def sub_grid(df: pd.DataFrame, grid: pd.DatetimeIndex, oblasts, step: pd.Timedelta = STEP):
    """Raion-level detail on the same grid.
    Returns (sub_active, obl_level_on):
      sub_active   – number of raion/hromada alerts active in the bin, per oblast
      obl_level_on – 1 if a whole-oblast alert is active in the bin."""
    t0 = grid[0]
    df = df[(df["finished_at"] > t0) & (df["started_at"] < grid[-1] + step)]
    sub, full = {}, {}
    for o in oblasts:
        out = []
        for lvl_mask in [df["level"] != "oblast", df["level"] == "oblast"]:
            e = df[(df["oblast"] == o) & lvl_mask]
            i0 = np.clip(((e["started_at"] - t0) // step).astype(int).values, 0, len(grid))
            i1 = np.clip(np.ceil((e["finished_at"] - t0) / step).astype(int).values, 0, len(grid))
            d = np.zeros(len(grid) + 1, dtype=np.int32)
            np.add.at(d, i0, 1); np.add.at(d, i1, -1)
            out.append(np.cumsum(d)[:-1])
        sub[o], full[o] = out[0].astype(np.float32), (out[1] > 0).astype(np.int8)
    return pd.DataFrame(sub, index=grid), pd.DataFrame(full, index=grid)


def raion_share_grid(df: pd.DataFrame, grid: pd.DatetimeIndex, oblasts, step: pd.Timedelta = STEP) -> pd.DataFrame:
    """Share of an oblast's raions under alert in each bin (1.0 while a whole-oblast alert is active).
    Overlapping alerts of the same raion are merged first, so each raion counts once."""
    t0 = grid[0]
    df = df[(df["finished_at"] > t0) & (df["started_at"] < grid[-1] + step)]
    out = {}
    for o in oblasts:
        d = np.zeros(len(grid) + 1, dtype=np.int32)
        r = df[(df["oblast"] == o) & df["raion"].notna()]
        n_raions = max(r["raion"].nunique(), 1)
        for _, g in r.groupby("raion"):
            e = _merge(g)
            s = pd.to_datetime(e["start"], utc=True); f = pd.to_datetime(e["end"], utc=True)
            i0 = np.clip(((s - t0) // step).astype(int).values, 0, len(grid))
            i1 = np.clip(np.ceil((f - t0) / step).astype(int).values, 0, len(grid))
            np.add.at(d, i0, 1); np.add.at(d, i1, -1)
        share = np.minimum(np.cumsum(d)[:-1] / n_raions, 1.0)
        full = df[(df["oblast"] == o) & (df["level"] == "oblast")]
        if len(full):
            f_ = np.zeros(len(grid) + 1, dtype=np.int32)
            i0 = np.clip(((full["started_at"] - t0) // step).astype(int).values, 0, len(grid))
            i1 = np.clip(np.ceil((full["finished_at"] - t0) / step).astype(int).values, 0, len(grid))
            np.add.at(f_, i0, 1); np.add.at(f_, i1, -1)
            share = np.where(np.cumsum(f_)[:-1] > 0, 1.0, share)
        out[o] = share.astype(np.float32)
    return pd.DataFrame(out, index=grid)


def regime_switch_date(df: pd.DataFrame, threshold: float = 0.10, window: str = "30D") -> pd.Timestamp:
    """First day on which whole-oblast alerts fall below `threshold` of all alerts (30-day rolling),
    i.e. when alerting moved to raion level."""
    d = df.set_index("started_at").sort_index()
    share = (d["level"] == "oblast").astype(float).resample("D").mean().rolling(window, min_periods=10).mean()
    return share[share < threshold].index[0]
