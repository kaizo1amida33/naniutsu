import os
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta, timezone

# --- 日付決定（CSV不要） ---
# NEXT_DATE_OVERRIDE があればそれを Dmax とみなす（例: 2025-09-26）
ovr = os.environ.get("NEXT_DATE_OVERRIDE", "").strip()
if ovr:
    dmax = pd.to_datetime(ovr)
else:
    # CSVが無い場合は「JSTの今日-1日」を Dmax とみなす（= 出力日は今日）
    JST = timezone(timedelta(hours=9))
    today_jst = datetime.now(JST).date()
    dmax = pd.to_datetime(str(today_jst - timedelta(days=1)))

target_date = (dmax + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

# --- ダミーHTMLを読む ---
with open("dummy.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

rows = []
table = soup.find("table", {"id": "machines"})
for tr in table.find_all("tr")[1:]:
    tds = [td.get_text(strip=True) for td in tr.find_all("td")]
    if len(tds) == 3:
        series, num, score = tds
        rows.append({
            "date": target_date,
            "series": series,
            "num": int(num),
            "score": float(score)
        })

df = pd.DataFrame(rows, columns=["date","series","num","score"])
Path("outputs").mkdir(exist_ok=True)
df.to_csv("outputs/predict_777.csv", index=False)
print(f"[OK] predict_777.csv generated for {target_date}: {len(df)} rows")
