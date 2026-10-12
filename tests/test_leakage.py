"""Features at bin t must not change when all data after bin t is removed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, build_targets, compute_lift, HORIZONS

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def data():
    state = pd.read_parquet(DATA / "state_5min.parquet")
    starts = pd.read_parquet(DATA / "starts_5min.parquet")
    split = {k: pd.Timestamp(v) for k, v in json.loads((DATA / "split.json").read_text()).items()}
    lo = split["val_start"] - pd.Timedelta(days=400)
    return state.loc[lo:], starts.loc[lo:], split


def test_features_ignore_future(data):
    state, starts, split = data
    lift = compute_lift(starts, split["val_start"])
    rng = np.random.default_rng(0)
    cand = state.loc[split["val_start"]:].index[:-20]
    for t in rng.choice(cand, size=20, replace=False):
        t = pd.Timestamp(t)
        full = build_features(state, starts, lift, [t])
        trunc = build_features(state.loc[:t], starts.loc[:t], lift, [t])
        pd.testing.assert_frame_equal(full, trunc)


def test_lift_uses_only_train(data):
    state, starts, split = data
    a = compute_lift(starts, split["val_start"])
    b = compute_lift(starts.loc[: split["val_start"] - pd.Timedelta("5min")], split["val_start"])
    pd.testing.assert_frame_equal(a, b)


def test_target_is_strictly_future(data):
    state, starts, _ = data
    t = starts.index[1000]
    y = build_targets(starts, [t])
    # zero out everything after t+k: target must become 0; zero out up to t: target unchanged
    for h, k in HORIZONS.items():
        s2 = starts.copy(); s2.iloc[: 1001] = 0
        assert (build_targets(s2, [t])[f"y_{h}"].values == y[f"y_{h}"].values).all()


def test_features_v2_ignore_future(data):
    from src.data import load_clean, sub_grid
    from src.features import build_features_v2
    state, starts, split = data
    state, starts = state.loc[split["val_start"] - pd.Timedelta(days=5):], starts.loc[split["val_start"] - pd.Timedelta(days=5):]
    df = load_clean(DATA / "official_data_en.csv")
    sa, fl = sub_grid(df, state.index, list(state.columns))
    lift = compute_lift(pd.read_parquet(DATA / "starts_5min.parquet"), split["val_start"])
    rng = np.random.default_rng(1)
    for t in rng.choice(state.loc[split["val_start"]:].index[:-20], size=10, replace=False):
        t = pd.Timestamp(t)
        full = build_features_v2(state, starts, lift, [t], sa, fl)
        trunc = build_features_v2(state.loc[:t], starts.loc[:t], lift, [t], sa.loc[:t], fl.loc[:t])
        pd.testing.assert_frame_equal(full, trunc)


def test_features_v3_ignore_future(data):
    from src.data import load_clean, sub_grid
    from src.features import build_features_v3
    state, starts, split = data
    state, starts = state.loc[split["val_start"] - pd.Timedelta(days=5):], starts.loc[split["val_start"] - pd.Timedelta(days=5):]
    df = load_clean(DATA / "official_data_en.csv")
    sa, fl = sub_grid(df, state.index, list(state.columns))
    lift = compute_lift(pd.read_parquet(DATA / "starts_5min.parquet"), split["val_start"])
    rng = np.random.default_rng(2)
    for t in rng.choice(state.loc[split["val_start"]:].index[:-20], size=10, replace=False):
        t = pd.Timestamp(t)
        full = build_features_v3(state, starts, lift, [t], sa, fl)
        trunc = build_features_v3(state.loc[:t], starts.loc[:t], lift, [t], sa.loc[:t], fl.loc[:t])
        pd.testing.assert_frame_equal(full, trunc, rtol=1e-5)


def test_features_v4_ignore_future(data):
    from src.data import load_clean, sub_grid, raion_share_grid
    from src.features import build_features_v4
    state, starts, split = data
    state, starts = state.loc[split["val_start"] - pd.Timedelta(days=5):], starts.loc[split["val_start"] - pd.Timedelta(days=5):]
    df = load_clean(DATA / "official_data_en.csv")
    sa, fl = sub_grid(df, state.index, list(state.columns))
    rs = raion_share_grid(df, state.index, list(state.columns))
    lift = compute_lift(pd.read_parquet(DATA / "starts_5min.parquet"), split["val_start"])
    reg = pd.Timestamp("2025-11-01", tz="Europe/Kyiv")
    rng = np.random.default_rng(3)
    for t in rng.choice(state.loc[split["val_start"]:].index[:-20], size=10, replace=False):
        t = pd.Timestamp(t)
        full = build_features_v4(state, starts, lift, [t], sa, fl, rs, reg)
        trunc = build_features_v4(state.loc[:t], starts.loc[:t], lift, [t], sa.loc[:t], fl.loc[:t], rs.loc[:t], reg)
        pd.testing.assert_frame_equal(full, trunc)
