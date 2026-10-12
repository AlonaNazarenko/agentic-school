"""Leakage-safe feature builder.

Convention: a row for grid bin t is a forecast made at decision time tau = t + STEP
(the end of bin t). Features use only bins <= t; the target uses bins t+1 ... t+k.
"""
import numpy as np
import pandas as pd

STEP = pd.Timedelta("5min")
HORIZONS = {"15min": 3, "30min": 6, "60min": 12}
CAP_STEPS = int(pd.Timedelta("7D") / STEP)


def w(d: str) -> int:
    return int(pd.Timedelta(d) / STEP)


# ---------------------------------------------------------------- target
def future_starts(S: pd.DataFrame, k: int) -> pd.DataFrame:
    """1 if any start in bins t+1 … t+k, NaN where the window runs past the data."""
    cs = np.vstack([np.zeros((1, S.shape[1])), np.cumsum(S.values, axis=0)])
    n = len(S)
    y = np.full(S.shape, np.nan)
    y[: n - k] = cs[k + 1: n + 1] - cs[1: n - k + 1]
    out = np.where(np.isnan(y), np.nan, (y > 0).astype(float))
    return pd.DataFrame(out, index=S.index, columns=S.columns)


# ---------------------------------------------------------------- graph from train only
def compute_lift(starts: pd.DataFrame, end, days: int = 365, k: int = HORIZONS["30min"]) -> pd.DataFrame:
    """lift[A, B] = P(B starts within k steps | A starts now) / P(B starts within k steps).
    Uses only bins strictly before `end` (targets never cross `end`)."""
    S = starts.loc[end - pd.Timedelta(days=days): end - STEP]
    F = future_starts(S, k)
    ok = F.notna().all(axis=1)
    S, F = S[ok].astype(float), F[ok]
    cond = (S.T.values @ F.values) / np.maximum(S.sum().values[:, None], 1)
    lift = cond / np.maximum(F.mean().values[None, :], 1e-9)
    np.fill_diagonal(lift, 0.0)
    return pd.DataFrame(lift, index=S.columns, columns=S.columns)


# ---------------------------------------------------------------- helpers
def _steps_since(event: np.ndarray) -> np.ndarray:
    """Steps since the last True (0 if True now), capped at CAP_STEPS."""
    idx = np.arange(len(event), dtype=float)[:, None]
    last = pd.DataFrame(np.where(event, idx, np.nan)).ffill().values
    return np.minimum(np.nan_to_num(idx - last, nan=CAP_STEPS), CAP_STEPS)


def _roll_sum(x: np.ndarray, n: int) -> np.ndarray:
    cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    lo = np.maximum(np.arange(len(x)) + 1 - n, 0)
    return cs[np.arange(len(x)) + 1] - cs[lo]


OWN = ["on_now", "since_start", "since_on", "starts_1h", "starts_6h", "starts_24h", "share_on_24h"]
CAL = ["hour_sin", "hour_cos", "dow", "night"]
SPATIAL = ["n_on", "d_n_on_15m", "d_n_on_30m", "d_n_on_60m", "n_starts_15m", "n_starts_30m", "n_starts_60m",
           "lift_starts_15m", "lift_starts_30m", "lift_starts_60m", "lift_on"]


def build_features(state: pd.DataFrame, starts: pd.DataFrame, lift: pd.DataFrame,
                   times, top_k: int = 3) -> pd.DataFrame:
    """Long frame indexed by (time, oblast). `times` are grid bins t (decision time t + STEP).
    All operations are trailing, so the row for t depends only on bins <= t."""
    oblasts = list(state.columns)
    st = state.values.astype(np.float32)
    sr = starts.values.astype(np.float32)
    pos = state.index.get_indexer(pd.DatetimeIndex(times))
    if (pos < 0).any():
        raise KeyError("times must be grid bins")

    wide = {
        "on_now": st,
        "since_start": _steps_since(sr > 0),
        "since_on": _steps_since(st > 0),
        "starts_1h": _roll_sum(sr, w("1h")),
        "starts_6h": _roll_sum(sr, w("6h")),
        "starts_24h": _roll_sum(sr, w("24h")),
        "share_on_24h": _roll_sum(st, w("24h")) / w("24h"),
    }
    # global "attack size / wave"
    n_on = st.sum(axis=1, keepdims=True)
    glob = {"n_on": n_on}
    for m in ["15m", "30m", "60m"]:
        n = w(m.replace("m", "min"))
        prev = np.vstack([np.full((n, 1), np.nan), n_on[:-n]])
        glob[f"d_n_on_{m}"] = n_on - prev
        glob[f"n_starts_{m}"] = _roll_sum(sr, n).sum(axis=1, keepdims=True)
    # lift-weighted neighbour activity (graph learned on train)
    W = lift.loc[oblasts, oblasts].values.astype(np.float32)
    for m in ["15m", "30m", "60m"]:
        wide[f"lift_starts_{m}"] = _roll_sum(sr, w(m.replace("m", "min"))) @ W
    wide["lift_on"] = st @ W / np.maximum(W.sum(axis=0, keepdims=True), 1e-9)
    # top-k individual "early-warning" neighbours per target oblast
    since_start = wide["since_start"]
    for i in range(top_k):
        src = [lift[o].drop(o).nlargest(top_k).index[i] for o in oblasts]
        j = [oblasts.index(s) for s in src]
        wide[f"nbr{i+1}_on"] = st[:, j]
        wide[f"nbr{i+1}_since_start"] = since_start[:, j]

    T, O = len(pos), len(oblasts)
    cols = {name: a[pos].reshape(T, -1) for name, a in wide.items()}
    cols.update({name: np.repeat(a[pos], O, axis=1) for name, a in glob.items()})
    out = pd.DataFrame({k: v.ravel() for k, v in cols.items()})

    tau = state.index[pos] + STEP                       # decision time
    hour = tau.hour + tau.minute / 60
    out["hour_sin"] = np.repeat(np.sin(2 * np.pi * hour / 24), O)
    out["hour_cos"] = np.repeat(np.cos(2 * np.pi * hour / 24), O)
    out["dow"] = np.repeat(tau.dayofweek, O)
    out["night"] = np.repeat(((hour >= 22) | (hour < 6)).astype(int), O)
    out.index = pd.MultiIndex.from_arrays(
        [np.repeat(state.index[pos], O), np.tile(oblasts, T)], names=["time", "oblast"])
    return out


# ---------------------------------------------------------------- v2: wave direction, adjacency, raion detail
REGIONS = {
    "east":   ["Kharkivska oblast", "Donetska oblast", "Luhanska oblast", "Sumska oblast",
               "Dnipropetrovska oblast", "Zaporizka oblast", "Poltavska oblast"],
    "south":  ["Khersonska oblast", "Mykolaivska oblast", "Odeska oblast"],
    "north":  ["Chernihivska oblast", "Kyivska oblast", "Kyiv City", "Zhytomyrska oblast"],
    "centre": ["Cherkaska oblast", "Kirovohradska oblast", "Vinnytska oblast"],
    "west":   ["Lvivska oblast", "Volynska oblast", "Rivnenska oblast", "Ternopilska oblast", "Khmelnytska oblast",
               "Ivano-Frankivska oblast", "Zakarpatska oblast", "Chernivetska oblast"],
}
_ADJ = {
    "Vinnytska oblast": ["Zhytomyrska oblast", "Kyivska oblast", "Cherkaska oblast", "Kirovohradska oblast",
                         "Odeska oblast", "Khmelnytska oblast", "Chernivetska oblast"],
    "Volynska oblast": ["Rivnenska oblast", "Lvivska oblast"],
    "Dnipropetrovska oblast": ["Poltavska oblast", "Kharkivska oblast", "Donetska oblast", "Zaporizka oblast",
                               "Khersonska oblast", "Mykolaivska oblast", "Kirovohradska oblast"],
    "Donetska oblast": ["Kharkivska oblast", "Zaporizka oblast", "Luhanska oblast"],
    "Zhytomyrska oblast": ["Kyivska oblast", "Khmelnytska oblast", "Rivnenska oblast"],
    "Zakarpatska oblast": ["Lvivska oblast", "Ivano-Frankivska oblast"],
    "Zaporizka oblast": ["Khersonska oblast"],
    "Ivano-Frankivska oblast": ["Lvivska oblast", "Chernivetska oblast", "Ternopilska oblast"],
    "Kyivska oblast": ["Kyiv City", "Cherkaska oblast", "Poltavska oblast", "Chernihivska oblast"],
    "Kirovohradska oblast": ["Cherkaska oblast", "Poltavska oblast", "Mykolaivska oblast", "Odeska oblast"],
    "Lvivska oblast": ["Rivnenska oblast", "Ternopilska oblast"],
    "Mykolaivska oblast": ["Odeska oblast", "Khersonska oblast"],
    "Poltavska oblast": ["Cherkaska oblast", "Kharkivska oblast", "Sumska oblast", "Chernihivska oblast"],
    "Rivnenska oblast": ["Ternopilska oblast", "Khmelnytska oblast"],
    "Sumska oblast": ["Chernihivska oblast", "Kharkivska oblast"],
    "Ternopilska oblast": ["Khmelnytska oblast", "Chernivetska oblast"],
    "Kharkivska oblast": ["Luhanska oblast"],
    "Khmelnytska oblast": ["Chernivetska oblast"],
}


def adjacency(oblasts) -> np.ndarray:
    """Symmetric 0/1 border adjacency matrix for the given oblast order."""
    ix = {o: i for i, o in enumerate(oblasts)}
    A = np.zeros((len(oblasts), len(oblasts)), np.float32)
    for a, nbrs in _ADJ.items():
        for b in nbrs:
            if a in ix and b in ix:
                A[ix[a], ix[b]] = A[ix[b], ix[a]] = 1
    return A


def region_of(o: str) -> str:
    return next(r for r, os_ in REGIONS.items() if o in os_)


WAVE = ([f"reg_{r}_{s}" for r in REGIONS for s in ["on", "d15", "d30"]]
        + ["own_reg_on", "own_reg_d15", "n_starts_5m", "burst_max_30m"])
ADJ_F = ["adj_on_share", "adj_starts_15m", "adj_starts_30m", "adj_since_start"]
SUB = ["sub_active", "sub_active_d15", "obl_level_on", "adj_sub_active", "lift_sub_active"]


def _lagdiff(x: np.ndarray, n: int) -> np.ndarray:
    return x - np.vstack([np.full((n, x.shape[1]), np.nan), x[:-n]])


def build_features_v2(state, starts, lift, times, sub_active, obl_level_on, top_k: int = 3) -> pd.DataFrame:
    """v1 features + wave direction, border adjacency and raion-level detail (all trailing)."""
    base = build_features(state, starts, lift, times, top_k)
    oblasts = list(state.columns)
    O = len(oblasts)
    st = state.values.astype(np.float32)
    sr = starts.values.astype(np.float32)
    sa = sub_active[oblasts].reindex(state.index).fillna(0).values.astype(np.float32)
    fl = obl_level_on[oblasts].reindex(state.index).fillna(0).values.astype(np.float32)
    pos = state.index.get_indexer(pd.DatetimeIndex(times))

    wide, glob = {}, {}
    reg_cols = {r: [i for i, o in enumerate(oblasts) if o in os_] for r, os_ in REGIONS.items()}
    reg_on = {r: st[:, ix].mean(axis=1, keepdims=True) if ix else np.zeros((len(st), 1), np.float32)
              for r, ix in reg_cols.items()}
    for r, a in reg_on.items():
        glob[f"reg_{r}_on"] = a
        glob[f"reg_{r}_d15"] = _lagdiff(a, w("15min"))
        glob[f"reg_{r}_d30"] = _lagdiff(a, w("30min"))
    own_r = [region_of(o) for o in oblasts]
    wide["own_reg_on"] = np.hstack([reg_on[r] for r in own_r])
    wide["own_reg_d15"] = np.hstack([glob[f"reg_{r}_d15"] for r in own_r])
    n5 = sr.sum(axis=1, keepdims=True)
    glob["n_starts_5m"] = n5
    glob["burst_max_30m"] = pd.DataFrame(n5).rolling(w("30min"), min_periods=1).max().values

    A = adjacency(oblasts)
    deg = np.maximum(A.sum(axis=0, keepdims=True), 1)
    wide["adj_on_share"] = st @ A / deg
    wide["adj_starts_15m"] = _roll_sum(sr, w("15min")) @ A
    wide["adj_starts_30m"] = _roll_sum(sr, w("30min")) @ A
    since = _steps_since(sr > 0)
    wide["adj_since_start"] = np.stack([since[:, A[:, j] > 0].min(axis=1) if A[:, j].any()
                                        else np.full(len(st), CAP_STEPS) for j in range(O)], axis=1)

    W = lift.loc[oblasts, oblasts].values.astype(np.float32)
    wide["sub_active"] = sa
    wide["sub_active_d15"] = _lagdiff(sa, w("15min"))
    wide["obl_level_on"] = fl
    wide["adj_sub_active"] = sa @ A
    wide["lift_sub_active"] = sa @ W / np.maximum(W.sum(axis=0, keepdims=True), 1e-9)

    cols = {k: a[pos].ravel() for k, a in wide.items()}
    cols.update({k: np.repeat(a[pos], O, axis=1).ravel() for k, a in glob.items()})
    return base.join(pd.DataFrame(cols, index=base.index))


def nbr_cols(top_k: int = 3):
    return [f"nbr{i+1}_{s}" for i in range(top_k) for s in ["on", "since_start"]]


def build_targets(starts: pd.DataFrame, times) -> pd.DataFrame:
    """Long frame of targets y_15min, y_30min, y_60min aligned with build_features."""
    out = {}
    for h, k in HORIZONS.items():
        out[f"y_{h}"] = future_starts(starts, k).loc[pd.DatetimeIndex(times)].stack(future_stack=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- v3: Hawkes-proxy decayed start counts
DECAY_HALF_LIVES = {"15m": "15min", "1h": "1h", "6h": "6h", "24h": "24h"}
DECAY_SCOPES = ["own", "adj", "reg", "nat"]
DECAY = [f"dec_{s}_{h}" for s in DECAY_SCOPES for h in DECAY_HALF_LIVES]


def _decay(x: np.ndarray, half_life_steps: float) -> np.ndarray:
    """e[t] = a * e[t-1] + x[t]  (exponential kernel, causal)."""
    from scipy.signal import lfilter
    a = 0.5 ** (1.0 / half_life_steps)
    return lfilter([1.0], [1.0, -a], x, axis=0)


def build_features_v3(state, starts, lift, times, sub_active, obl_level_on, top_k: int = 3) -> pd.DataFrame:
    """v2 + exponentially decayed counts of past alert starts at several time scales,
    for the oblast itself, its border neighbours, its macro-region and the whole country.
    These are the building blocks of a Hawkes intensity, without fitting one."""
    X = build_features_v2(state, starts, lift, times, sub_active, obl_level_on, top_k)
    oblasts = list(state.columns)
    sr = starts.values.astype(np.float64)
    A = adjacency(oblasts)
    R = np.array([[1.0 if region_of(a) == region_of(b) else 0.0 for b in oblasts] for a in oblasts])
    np.fill_diagonal(R, 0)
    pos = state.index.get_indexer(pd.DatetimeIndex(times))
    cols = {}
    for hn, hl in DECAY_HALF_LIVES.items():
        e = _decay(sr, pd.Timedelta(hl) / STEP)
        cols[f"dec_own_{hn}"] = e
        cols[f"dec_adj_{hn}"] = e @ A
        cols[f"dec_reg_{hn}"] = e @ R
        cols[f"dec_nat_{hn}"] = np.repeat(e.sum(axis=1, keepdims=True), len(oblasts), axis=1)
    D = pd.DataFrame({k: v[pos].ravel().astype(np.float32) for k, v in cols.items()}, index=X.index)
    return X.join(D)


# ---------------------------------------------------------------- v4: raion-share spatial features + regime flag
RS_LAGS = {"15": "15min", "30": "30min", "60": "60min", "90": "90min", "120": "120min"}
RAION = (["rs_own", "rs_own_lag15", "rs_adj", "rs_lift", "rs_nat", "rs_nat_d30"]
         + [f"rs_adj_lag{k}" for k in RS_LAGS] + [f"rs_lift_lag{k}" for k in ["30", "60", "120"]])
REGIME = ["regime_raion"]


def _lag(x: np.ndarray, n: int) -> np.ndarray:
    return np.vstack([np.full((n, x.shape[1]), np.nan, dtype=np.float32), x[:-n]])


def build_features_v4(state, starts, lift, times, sub_active, obl_level_on, raion_share,
                      regime_date, top_k: int = 3) -> pd.DataFrame:
    """v2 + raion-share of the oblast, its border neighbours (now and lagged 15–120 min),
    lift-weighted upstream oblasts and the whole country, plus a raion-alerting regime flag."""
    X = build_features_v2(state, starts, lift, times, sub_active, obl_level_on, top_k)
    oblasts = list(state.columns)
    rs = raion_share[oblasts].reindex(state.index).fillna(0).values.astype(np.float32)
    A = adjacency(oblasts); deg = np.maximum(A.sum(axis=0, keepdims=True), 1)
    W = lift.loc[oblasts, oblasts].values.astype(np.float32); wsum = np.maximum(W.sum(axis=0, keepdims=True), 1e-9)
    pos = state.index.get_indexer(pd.DatetimeIndex(times))
    adj, lw = rs @ A / deg, rs @ W / wsum
    nat = rs.mean(axis=1, keepdims=True)
    cols = {"rs_own": rs, "rs_own_lag15": _lag(rs, w("15min")), "rs_adj": adj, "rs_lift": lw,
            "rs_nat": np.repeat(nat, len(oblasts), axis=1),
            "rs_nat_d30": np.repeat(nat - _lag(nat, w("30min")), len(oblasts), axis=1)}
    for k, d in RS_LAGS.items():
        cols[f"rs_adj_lag{k}"] = _lag(adj, w(d))
    for k in ["30", "60", "120"]:
        cols[f"rs_lift_lag{k}"] = _lag(lw, w(RS_LAGS[k]))
    D = pd.DataFrame({k: v[pos].ravel() for k, v in cols.items()}, index=X.index)
    tau = X.index.get_level_values("time") + STEP
    D["regime_raion"] = (tau >= regime_date).astype(np.float32)
    return X.join(D)
