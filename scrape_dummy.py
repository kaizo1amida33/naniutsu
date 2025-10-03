import pandas as pd
from bs4 import BeautifulSoup
import glob
from pathlib import Path

# === 1) Dmax検出 ===
def detect_max_date():
    files = glob.glob("*_test_ge_*_v14.csv")
    if not files:
        raise RuntimeError("テストCSVが見つかりません")
    max_dt = None
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["date"])
        except Exception:
            continue
        if "date" not in df:
            continue
        d = pd.to_datetime(df["date"], errors="coerce")
        if d.notna().any():
            m = d.max().normalize()
            if (max_dt is None) or (m > max_dt):
                max_dt = m
    if max_dt is None:
        raise RuntimeError("有効なdateが見つかりません")
    return max_dt

dmax = detect_max_date()
target_date = (dmax + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

# === 2) ダミーHTMLをスクレイピング ===
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

# === 3) 出力 ===
Path("outputs").mkdir(exist_ok=True)
df.to_csv("outputs/predict_777.csv", index=False)
print(f"predict_777.csv generated for {target_date}: {len(df)} rows")
