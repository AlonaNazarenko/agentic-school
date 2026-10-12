"""Usage:  .venv/bin/python scripts/scrape_telegram.py --since 2025-06-01 [extra channels ...]

Default channels: DEFAULT_CHANNELS (official Ukrainian Air Force). Extra channels are added to them.

Needs TG_API_ID / TG_API_HASH in .env (get them at https://my.telegram.org → API development tools).
First run asks for your phone number and login code; the session is saved to data/tg_session.session.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                      # .env, data/ and the session file live in the project root
sys.path.insert(0, str(ROOT))

from src.telegram import scrape

DEFAULT_CHANNELS = ["kpszsu"]   # Повітряні сили ЗСУ (official)

p = argparse.ArgumentParser()
p.add_argument("channels", nargs="*", default=[], help="extra public channel usernames (without @)")
p.add_argument("--since", default="2025-06-01")
p.add_argument("--out", default="data/telegram_raw.parquet")
a = p.parse_args()
channels = list(dict.fromkeys(DEFAULT_CHANNELS + a.channels))
df = scrape(channels, a.since, a.out)
print(df.groupby("channel").agg(n=("id", "size"), first=("date", "min"), last=("date", "max")))
