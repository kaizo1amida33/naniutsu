# -*- coding: utf-8 -*-
"""
scraping_auto_csv_data_v3_fixed.py
- v3系の挙動を再現（ログ文言も近づける）
- 出力: *_plazahakata_all_days.csv を更新（追記・重複除去）
       {series}_plazahakata_all_days_v14.csv (<=2025-08-16)
       {series}_test_ge_2025-08-17_v14.csv   (>=2025-08-17)
"""

import os, re, time, sys, datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://min-repo.com"

KISHU_CONFIGS = [
    {
        "name": "北斗の拳",
        "series": "hokuto",
        "start_url": "https://min-repo.com/2633508/?kishu=L%E3%82%B9%E3%83%9E%E3%82%B9%E3%83%AD%E5%8C%97%E6%96%97%E3%81%AE%E6%8B%B3",
        "csv_file": "hokuto_plazahakata_all_days.csv",
        "last_url_file": "last_url_hokuto.txt",
    },
    {
        "name": "モンキーターン",
        "series": "monkey",
        "start_url": "https://min-repo.com/2633508/?kishu=%E3%82%B9%E3%83%9E%E3%82%B9%E3%83%AD%E3%83%A2%E3%83%B3%E3%82%AD%E3%83%BC%E3%82%BF%E3%83%BC%E3%83%B3V",
        "csv_file": "monkey_plazahakata_all_days.csv",
        "last_url_file": "last_url_monkey.txt",
    },
    {
        "name": "東京喰種",
        "series": "ghoul",
        "start_url": "https://min-repo.com/2633508/?kishu=L%E6%9D%B1%E4%BA%AC%E5%96%B0%E7%A8%AE",
        "csv_file": "ghoul_plazahakata_all_days.csv",
        "last_url_file": "last_url_ghoul.txt",
    },
    {
        "name": "マイジャグラーV",
        "series": "myjugglerV",
        "start_url": "https://min-repo.com/2633508/?kishu=%E3%83%9E%E3%82%A4%E3%82%B8%E3%83%A3%E3%82%B0%E3%83%A9%E3%83%BCV",
        "csv_file": "myjugglerV_plazahakata_all_days.csv",
        "last_url_file": "last_url_myjugglerV.txt",
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

TRAIN_CUTOFF = "2025-08-16"
TEST_FROM    = "2025-08-17"
ORDER6 = ["date","num","samai","g_num","avg","series"]

def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[LOG {ts}] {msg}", flush=True)

def normalize_url(u: str) -> str:
    if not u: return u
    s = u.strip()
    if s.startswith("hhttps://"): s = "https://" + s[len("hhttps://"):]
    if s.startswith("ttps://"):   s = "https://" + s[len("ttps://"):]
    if s.startswith("//"):        s = "https:" + s
    if not s.startswith("http"):  s = "https://" + s.lstrip("/")
    return s

def date_to_ymd(txt: str) -> str:
    txt = txt.strip()
    # 1) YYYY-MM-DD / YYYY/M/D
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", txt)
    if m:
        y, mo, d = map(int, m.groups())
        return f"{y:04d}-{mo:02d}-{d:02d}"
    # 2) M/D → 今年補完
    m = re.search(r"(\d{1,2})/(\d{1,2})", txt)
    if m:
        y = datetime.datetime.now().year
        mo, d = map(int, m.groups())
        return f"{y:04d}-{mo:02d}-{d:02d}"
    # 3) 日本語 M月D日（曜日入りも許容）
    m = re.search(r"(\d{1,2})月(\d{1,2})日", txt)
    if m:
        y = datetime.datetime.now().year
        mo, d = map(int, m.groups())
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return ""

def get_date(soup) -> str:
    # v3 互換：とれたら YYYY-MM-DD を返す
    # h1
    h1 = soup.find("h1")
    if h1:
        ymd = date_to_ymd(h1.get_text(" ", strip=True))
        if ymd: return ymd
    # よくある候補
    for sel in ["h2", ".day_title", ".date", ".head_ttl", ".page_title", ".ttl", "title"]:
        tag = soup.select_one(sel)
        if tag:
            ymd = date_to_ymd(tag.get_text(" ", strip=True))
            if ymd: return ymd
    # 全文フォールバック
    ymd = date_to_ymd(soup.get_text(" ", strip=True))
    return ymd

def _parse_avg(text: str) -> int:
    text = (text or "").strip()
    m = re.search(r'1\s*/\s*(\d+)', text) or re.search(r'1\s*[\\/／]\s*(\d+)', text)
    return int(m.group(1)) if m else 0

def get_slot_data_from_slump_list(soup, date_ymd: str) -> List[Dict]:
    data = []
    for li in soup.select("ul.slump_list > li"):
        a = li.find("a")
        num = a.get_text(strip=True) if a else ""
        if not num.isdigit(): continue
        table = li.find("table")
        if not table: continue
        trs = table.find_all("tr")
        if not trs: continue
        tr = trs[1] if len(trs) >= 2 else trs[0]
        tds = tr.find_all("td")
        if len(tds) < 2: continue
        samai = tds[0].get_text(strip=True).replace(",", "")
        g_num = tds[1].get_text(strip=True).replace(",", "")
        avg = _parse_avg(tds[2].get_text(strip=True)) if len(tds) >= 3 else 0
        data.append({"date": date_ymd, "num": num, "samai": samai, "g_num": g_num, "avg": avg})
    return data

def get_slot_data_from_table(soup, date_ymd: str) -> List[Dict]:
    data = []
    for table in soup.select("div.table_wrap > table, table.table, .result_table table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "台番" in headers and "差枚" in headers and "G数" in headers:
            try:
                avg_idx = headers.index("合成")
            except ValueError:
                avg_idx = -1
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 3:
                    num_tag = tds[0].find("a")
                    num = num_tag.text.strip() if num_tag else tds[0].text.strip()
                    if not num.isdigit(): continue
                    samai = tds[1].text.strip().replace(",", "")
                    g_num = tds[2].text.strip().replace(",", "")
                    avg = 0
                    if avg_idx != -1 and len(tds) > avg_idx:
                        avg = _parse_avg(tds[avg_idx].get_text(strip=True))
                    data.append({"date": date_ymd, "num": num, "samai": samai, "g_num": g_num, "avg": avg})
            break
    return data

NEXT_TEXT_PAT = re.compile(r"(翌日|次.?日|次へ|Next|翌|明日|翌ページ)", re.I)
def find_next_url(soup) -> str:
    a = soup.find("a", attrs={"rel": "next"})
    if a and a.get("href"):
        href = a.get("href"); return BASE_URL + href if href.startswith("/") else href
    for a in soup.select("div.prev_next_link a, .prev_next a, .pager a, .pagination a"):
        t = a.get_text(" ", strip=True)
        if NEXT_TEXT_PAT.search(t) or re.match(r"^\d{1,2}/\d{1,2}$", t):
            href = a.get("href")
            if href: return BASE_URL + href if href.startswith("/") else href
    for a in soup.find_all("a"):
        t = a.get_text(" ", strip=True)
        if NEXT_TEXT_PAT.search(t) or re.match(r"^\d{1,2}/\d{1,2}$", t):
            href = a.get("href")
            if href: return BASE_URL + href if href.startswith("/") else href
    return None

def get_resume_url(start_url: str, last_url_file: str) -> str:
    if os.path.exists(last_url_file):
        with open(last_url_file, "r", encoding="utf-8") as f:
            url = f.read().strip()
            if url: return normalize_url(url)
    return normalize_url(start_url)

def save_resume_url(url: str, last_url_file: str):
    with open(last_url_file, "w", encoding="utf-8") as f:
        f.write(normalize_url(url))

def _to_int_str(x) -> str:
    try:
        v = float(str(x).replace(",", ""))
        return str(int(round(v)))
    except Exception:
        return "0"

def merge_write(csv_path: Path, series: str, rows: List[Dict]):
    if not rows: return
    df_new = pd.DataFrame(rows)
    df_new["series"] = series
    df_new["samai"] = df_new["samai"].apply(_to_int_str)
    df_new["g_num"] = df_new["g_num"].apply(_to_int_str)
    df_new["avg"]   = df_new["avg"].apply(_to_int_str)
    df_new = df_new[["date","num","samai","g_num","avg","series"]]

    if csv_path.exists() and csv_path.stat().st_size > 0:
        cur = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        need_cols = ["date","num","samai","g_num","avg","series"]
        for c in need_cols:
            if c not in cur.columns: cur[c] = ""
        cur = cur[need_cols]
        merged = pd.concat([cur, df_new], ignore_index=True)
    else:
        merged = df_new

    merged = merged.drop_duplicates(subset=["date","num"]).sort_values(["date","num"]).reset_index(drop=True)
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(csv_path, "r", encoding="utf-8-sig") as rf:
        txt = rf.read()
        if re.search(r",(?:-?\d+)\.0(?:,|\n|\r)", txt):
            raise AssertionError(f"{csv_path.name} に .0 が含まれています")
    log(f"[WRITE] {csv_path.name} rows={len(merged)}")

def produce_v14(series: str, all_csv: Path):
    if not all_csv.exists(): return
    df = pd.read_csv(all_csv, dtype=str, keep_default_na=False)
    need = ["date","num","samai","g_num","avg","series"]
    for c in need:
        if c not in df.columns: df[c] = ""
    df = df[need]
    df["samai"] = df["samai"].apply(_to_int_str)
    df["avg"]   = df["avg"].apply(_to_int_str)
    df = df.sort_values(["date","num"]).reset_index(drop=True)

    dt = pd.to_datetime(df["date"])
    train = df.loc[dt <= pd.to_datetime(TRAIN_CUTOFF)].copy()
    test  = df.loc[dt >= pd.to_datetime(TEST_FROM)].copy()

    out_train = all_csv.with_name(f"{series}_plazahakata_all_days_v14.csv")
    out_test  = all_csv.with_name(f"{series}_test_ge_{TEST_FROM}_v14.csv")

    if not train.empty:
        train.to_csv(out_train, index=False, encoding="utf-8-sig")
        log(f"[WRITE] {out_train.name} rows={len(train)}")
    else:
        log(f"[WARN] empty train v14: {out_train.name}]")
    if not test.empty:
        test.to_csv(out_test, index=False, encoding="utf-8-sig")
        log(f"[WRITE] {out_test.name} rows={len(test)}")
    else:
        log(f"[WARN] empty test v14: {out_test.name}]")

def scrape_one(cfg: Dict):
    name   = cfg["name"]
    series = cfg["series"]
    csv_file = Path(cfg["csv_file"])
    last_url_file = cfg["last_url_file"]
    url = get_resume_url(cfg["start_url"], last_url_file)

    log(f"=== {name} ({series}) スクレイピング開始 ===")
    visited = set()
    appended: List[Dict] = []

    while url and url not in visited:
        visited.add(url)
        url = normalize_url(url)
        log(f"[GET] {url}")
        r = requests.get(url, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(r.text, "html.parser")

        date_ymd = get_date(soup)
        log(f"[PAGE DATE] {date_ymd or 'N/A'}")
        if date_ymd:
            rows = get_slot_data_from_slump_list(soup, date_ymd) or get_slot_data_from_table(soup, date_ymd)
            if rows:
                log(f"[COLLECT] {len(rows)} rows on {date_ymd}")
                appended.extend(rows)
        else:
            log("[WARN] date parse failed, skip")

        next_url = find_next_url(soup)
        log(f"[NEXT] {next_url}")
        if not next_url: break
        url = next_url
        save_resume_url(url, last_url_file)
        time.sleep(1)

    if appended:
        merge_write(csv_file, series, appended)
        produce_v14(series, csv_file)
    else:
        log("[INFO] no new data; keep files as-is]")

def main():
    for cfg in KISHU_CONFIGS:
        try:
            scrape_one(cfg)
        except Exception as e:
            log(f"[ERROR] {cfg.get('name','?')}: {e}")
    log("all done.")

if __name__ == "__main__":
    main()
