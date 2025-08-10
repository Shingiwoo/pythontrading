import os
import csv
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "trades.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "trades.csv")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            side TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            qty REAL,
            fee REAL,
            pnl REAL,
            pnl_pct REAL,
            reason TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbol_ts ON trades(symbol, timestamp)")
    conn.commit()
    conn.close()


_init_db()


def log_trade(data: Dict[str, any]):
    # CSV
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

    # SQLite
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO trades (timestamp, symbol, side, entry, sl, tp, qty, fee, pnl, pnl_pct, reason)
        VALUES (:timestamp, :symbol, :side, :entry, :sl, :tp, :qty, :fee, :pnl, :pnl_pct, :reason)
        """,
        data,
    )
    conn.commit()
    conn.close()


def get_trades(symbol: Optional[str] = None) -> List[Dict[str, any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if symbol:
        cur.execute("SELECT * FROM trades WHERE symbol=? ORDER BY timestamp", (symbol,))
    else:
        cur.execute("SELECT * FROM trades ORDER BY timestamp")
    rows = cur.fetchall()
    conn.close()
    cols = [d[0] for d in cur.description] if rows else []
    return [dict(zip(cols, row)) for row in rows]


def daily_summary(date: Optional[str] = None) -> Dict[str, any]:
    if date is None:
        date = datetime.utcnow().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT symbol, SUM(pnl) FROM trades WHERE date(timestamp)=? GROUP BY symbol",
        (date,),
    )
    rows = cur.fetchall()
    conn.close()
    return {"date": date, "pnl": {r[0]: r[1] for r in rows}}
