"""Build dashboard/index.html: a Ukraine map of model-predicted probabilities of an air raid alert starting.

Replays the out-of-sample (walk-forward test) predictions — a historical backtest, not a live feed.

Usage:
    .venv/bin/python dashboard/make_dashboard.py                       # v4, latest test fold (notebook 06)
    .venv/bin/python dashboard/make_dashboard.py --preds data/preds_08_telegram_v2.parquet --model "v4 + Telegram v2"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

UK_TO_EN = {
    "Вінницька область": "Vinnytska oblast", "Волинська область": "Volynska oblast",
    "Дніпропетровська область": "Dnipropetrovska oblast", "Донецька область": "Donetska oblast",
    "Житомирська область": "Zhytomyrska oblast", "Закарпатська область": "Zakarpatska oblast",
    "Запорізька область": "Zaporizka oblast", "Івано-Франківська область": "Ivano-Frankivska oblast",
    "Київська область": "Kyivska oblast", "Кіровоградська область": "Kirovohradska oblast",
    "Луганська область": "Luhanska oblast", "Львівська область": "Lvivska oblast",
    "Миколаївська область": "Mykolaivska oblast", "Одеська область": "Odeska oblast",
    "Полтавська область": "Poltavska oblast", "Рівненська область": "Rivnenska oblast",
    "Сумська область": "Sumska oblast", "Тернопільська область": "Ternopilska oblast",
    "Харківська область": "Kharkivska oblast", "Херсонська область": "Khersonska oblast",
    "Хмельницька область": "Khmelnytska oblast", "Черкаська область": "Cherkaska oblast",
    "Чернівецька область": "Chernivetska oblast", "Чернігівська область": "Chernihivska oblast",
    "Автономна Республіка Крим": "Crimea",
}
LABEL = {
    "Vinnytska oblast": "Vinnytsia", "Volynska oblast": "Volyn", "Dnipropetrovska oblast": "Dnipropetrovsk",
    "Donetska oblast": "Donetsk", "Zhytomyrska oblast": "Zhytomyr", "Zakarpatska oblast": "Zakarpattia",
    "Zaporizka oblast": "Zaporizhzhia", "Ivano-Frankivska oblast": "Ivano-Frankivsk", "Kyivska oblast": "Kyiv oblast",
    "Kirovohradska oblast": "Kirovohrad", "Luhanska oblast": "Luhansk", "Lvivska oblast": "Lviv",
    "Mykolaivska oblast": "Mykolaiv", "Odeska oblast": "Odesa", "Poltavska oblast": "Poltava",
    "Rivnenska oblast": "Rivne", "Sumska oblast": "Sumy", "Ternopilska oblast": "Ternopil",
    "Kharkivska oblast": "Kharkiv", "Khersonska oblast": "Kherson", "Khmelnytska oblast": "Khmelnytskyi",
    "Cherkaska oblast": "Cherkasy", "Chernivetska oblast": "Chernivtsi", "Chernihivska oblast": "Chernihiv",
    "Kyiv City": "Kyiv City", "Crimea": "Crimea",
}
KYIV_CITY_LONLAT = (30.5234, 50.4501)
W, H = 1000, 680


def project(lon, lat, lon0=31.2, lat0=48.4):
    k = np.cos(np.radians(lat0))
    return (np.asarray(lon) - lon0) * k, -(np.asarray(lat) - lat0)


def simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    keep = [pts[0]]
    for p in pts[1:-1]:
        if np.hypot(*(p - keep[-1])) >= tol:
            keep.append(p)
    keep.append(pts[-1])
    return np.array(keep)


def build_paths(geo_path: Path):
    g = json.loads(geo_path.read_text())
    rings = []
    for f in g["features"]:
        name = UK_TO_EN.get(f["properties"].get("name:uk") or f["properties"].get("name"))
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for ring in poly:
                a = np.array(ring)
                rings.append((name, np.column_stack(project(a[:, 0], a[:, 1]))))
    allp = np.vstack([r for _, r in rings])
    lo, hi = allp.min(0), allp.max(0)
    pad = 12
    s = min((W - 2 * pad) / (hi - lo)[0], (H - 2 * pad) / (hi - lo)[1])
    off = np.array([pad, pad]) + (np.array([W, H]) - 2 * pad - (hi - lo) * s) / 2
    paths = {}
    for name, r in rings:
        p = simplify((r - lo) * s + off, tol=0.9)
        d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in p) + "Z"
        paths[name] = paths.get(name, "") + d
    kx, ky = project(*KYIV_CITY_LONLAT)
    kyiv = ((np.array([kx, ky]) - lo) * s + off).round(1).tolist()
    return paths, kyiv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(DATA / "preds_06_experiments.parquet"))
    ap.add_argument("--model", default="v4 (+raion share)")
    ap.add_argument("--fold", type=int, default=None, help="default: latest fold")
    ap.add_argument("--every", default="15min", help="time step shown in the replay")
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "index.html"))
    a = ap.parse_args()

    P = pd.read_parquet(a.preds)
    fold = a.fold or int(P["fold"].max())
    P = P[P["fold"] == fold]
    horizons = ["15min", "30min", "60min"]
    state = pd.read_parquet(DATA / "state_5min.parquet")
    oblasts = sorted(P.index.get_level_values("oblast").unique())

    times = P.index.get_level_values("time").unique().sort_values()
    times = times[times.floor(a.every) == times]
    prob, actual = {}, {}
    for h in horizons:
        ph = P[P["horizon"] == h]
        prob[h] = ph[a.model].unstack("oblast").reindex(index=times, columns=oblasts)
        actual[h] = ph["y"].unstack("oblast").reindex(index=times, columns=oblasts)
    on_now = state.reindex(times)[oblasts]

    # base rate per oblast & horizon in this test fold (for the "× usual" readout)
    base = {h: P[P["horizon"] == h].groupby(level="oblast")["y"].mean().reindex(oblasts).fillna(0).round(4).tolist()
            for h in horizons}

    paths, kyiv = build_paths(DATA / "ua_oblasts.geojson")
    payload = {
        "model": a.model, "fold": fold,
        "times": [t.isoformat() for t in (times + pd.Timedelta("5min"))],   # decision time = end of bin
        "oblasts": oblasts, "labels": [LABEL.get(o, o) for o in oblasts],
        "horizons": horizons,
        # probabilities in per-mille (0–1000) to keep the file small
        "p": {h: np.nan_to_num(prob[h].values * 1000).round().astype(int).tolist() for h in horizons},
        "y": {h: np.nan_to_num(actual[h].values, nan=-1).astype(int).tolist() for h in horizons},
        "on": on_now.fillna(0).astype(int).values.tolist(),
        "base": base,
        "paths": paths, "kyiv": kyiv, "W": W, "H": H,
    }
    html = (ROOT / "dashboard" / "template.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    Path(a.out).write_text(html)
    print(f"wrote {a.out}: {len(times)} time steps, fold {fold}, model '{a.model}', "
          f"{Path(a.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
