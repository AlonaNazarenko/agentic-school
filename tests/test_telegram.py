import pandas as pd

from src.telegram import parse_message, tg_features, clean


def test_parse_kab_direction():
    p = parse_message("Пуски КАБ курсом на Харківщину та Сумщину")
    assert set(p["oblasts"]) == {"Kharkivska oblast", "Sumska oblast"}
    assert p["weapons"] == ["kab"] and p["direction"]


def test_parse_kyiv_city_vs_oblast():
    assert parse_message("Шахеди на Київщині, курсом на Бровари")["oblasts"] == ["Kyivska oblast"]
    assert parse_message("Балістика на Київ!")["oblasts"] == ["Kyiv City"]
    assert set(parse_message("Ракета через Київщину на Київ")["oblasts"]) == {"Kyivska oblast", "Kyiv City"}


def test_parse_no_false_rivne_sumy():
    assert parse_message("рівномірно сумарно")["oblasts"] == []


def test_message_used_only_after_its_bin():
    grid = pd.date_range("2026-01-01 10:00", periods=12, freq="5min", tz="Europe/Kyiv")
    raw = pd.DataFrame({"channel": ["x"], "id": [1], "date": [pd.Timestamp("2026-01-01 10:12", tz="Europe/Kyiv")],
                        "edit_date": [None], "fwd": [False], "text": ["КАБ на Харків"]})
    f = tg_features(clean(raw), grid, ["Kharkivska oblast"], grid)
    s = f["tg_obl_kab_15m"].droplevel("oblast")
    # published 10:12 → bin 10:10 → known at decision time 10:15 (row t=10:10); nothing before
    assert s.loc[:"2026-01-01 10:05"].sum() == 0 and s.loc["2026-01-01 10:10"] == 1


def test_late_edits_dropped():
    raw = pd.DataFrame({"channel": ["x"], "id": [1], "date": [pd.Timestamp("2026-01-01 10:00", tz="UTC")],
                        "edit_date": [pd.Timestamp("2026-01-01 11:00", tz="UTC")], "fwd": [False], "text": ["КАБ"]})
    assert len(clean(raw)) == 0


def test_parse_air_force_style():
    p = parse_message("Група ударних БпЛА на півночі Сумщини, курсом на Полтавщину")
    assert set(p["oblasts"]) == {"Sumska oblast", "Poltavska oblast"} and "shahed" in p["weapons"] and p["direction"]
    assert "missile" in parse_message("Швидкісна ціль на Київ!")["weapons"]


def test_parser_v2_gazetteer_and_direction():
    from src.telegram import Parser
    gaz = {"брусилів": "Zhytomyrska oblast", "макарів": "Kyivska oblast", "вознесенськ": "Mykolaivska oblast"}
    p = Parser(gaz)
    r = p.parse("🚀Групи ракет на Брусилів (Житомирщина) та Макарів (Київщина)")
    assert set(r["oblasts"]) == {"Zhytomyrska oblast", "Kyivska oblast"}
    r = p.parse("5 БпЛА у Вознесенському районі, курсом на Кропивницький")
    assert r["loc"] == ["Mykolaivska oblast"] and r["tgt"] == ["Kirovohradska oblast"] and r["count"] == 5


def test_tg_v2_ignores_future_posts():
    import numpy as np
    from src.telegram import Parser, clean_v2, tg_features_v2
    grid = pd.date_range("2026-01-01", periods=200, freq="5min", tz="Europe/Kyiv")
    obl = ["Kharkivska oblast", "Sumska oblast"]
    raw = pd.DataFrame({"channel": "x", "id": range(4),
                        "date": [grid[10], grid[50], grid[120], grid[150]] ,
                        "edit_date": [None] * 4, "fwd": [False] * 4,
                        "text": ["КАБ на Харків", "БпЛА на Сумщині курсом на Харківщину", "КАБ на Суми", "ракета"]})
    m = clean_v2(raw, Parser())
    A = np.array([[0, 1], [1, 0]], np.float32)
    t = grid[100]
    full = tg_features_v2(m, grid, obl, [t], A, A)
    trunc = tg_features_v2(m[m.date < t + pd.Timedelta("5min")], grid, obl, [t], A, A)
    pd.testing.assert_frame_equal(full, trunc)
