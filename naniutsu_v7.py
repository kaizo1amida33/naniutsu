# -*- coding: utf-8 -*-
"""
統合版パチンコ機械学習パイプライン
main.pyとsub.pyを統合し、特徴量生成の問題を修正
Windows環境（Anacondaプロンプト）対応
"""
import optuna
from optuna.samplers import TPESampler
import json
import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from joblib import dump, load
from sklearn.metrics import average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from typing import List, Optional, Tuple

# KMeansのメモリリーク警告を回避
os.environ['OMP_NUM_THREADS'] = '1'

# または、sklearn関連の警告を全て抑制する場合
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak*')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# より具体的にKMeansの警告のみを抑制する場合
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak*')

# ====================
# 定数・設定
# ====================
REQ_COLS = {"date", "num", "samai", "g_num", "avg"}
SPECIAL_DAYS = {7, 8, 17, 18, 27, 28}

# 角台の台番号セット (sub.pyより)
CORNER_IDS_BY_DATE = {
    "hokuto": {
        "default": {800, 811, 830, 812},
        "2025-09-26": {831, 849, 869, 850}, 
    },
    "ghoul": {
        "default": {1090, 1101},
        "2025-06-24": {1088, 1101},
    },
    "monkey": {
        "default": {881, 889, 890, 911},
        "2025-06-06": {884, 889, 890, 911},
    },
    "jugg": {
        "default": {264, 273, 274, 286},
    },
}

# 特徴量グループ (sub.pyより)
TAIL_COL = "tail_eq_dom_d1"
P90_COLS = ["p90_gap_avg_14", "p90_z14", "stretch_p90p50_14"]
WEEK_TE_COLS = ["hit_rate_te_wk", "hit_rate_te_wk_ns", "sxw_ema14_prev", "sxw_ema14_resid"]
CORNER_COLS = ["corner_flag", "corner_x_special"] + [f"corner_wd_{i}" for i in range(7)]
ADD_ALL = [TAIL_COL] + WEEK_TE_COLS + P90_COLS + CORNER_COLS

# ====================
# ログ・ユーティリティ関数
# ====================
def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

# ====================
# データ検証・解析
# ====================
def _assert_non_empty(df, where="train"):
    if len(df) == 0 or df["date"].isna().all():   
        sample = (df.head(10).to_dict(orient="records") if len(df) else [])
        raise RuntimeError(f"[FATAL] empty dataframe at {where}. "
                           f"Check date parsing & cutoff. sample_head10={sample}")

def _load_drop_from_rank_report(path: str, thresh: float) -> set:
    try:
        if not path:
            return set()
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if "feature" not in cols or ("rank_score" not in cols and "importance" not in cols):
            return set()
        score_col = cols.get("rank_score", cols.get("importance"))
        feats = df.loc[df[score_col] < thresh, cols["feature"]].astype(str).tolist()
        return set(feats)
    except Exception as e:
        log(f"[WARN] failed to read rank report: {e}")
        return set()

def _apply_feature_drop(feat: list, drop_set: set, manual_drop_str: str = "") -> list:
    manual = set([s.strip() for s in str(manual_drop_str).split(",") if s.strip()])
    drop = set(drop_set) | manual
    kept = [c for c in feat if c not in drop]
    removed = [c for c in feat if c in drop]
    log(f"[PRUNE] requested drop={len(drop)} -> removed={len(removed)} kept={len(kept)}")
    if removed:
        log(f"[PRUNE] removed (head): {removed[:10]}{'...' if len(removed)>10 else ''}")
    return kept

# ====================
# sub.py由来のヘルパー関数
# ====================
def _get_series_col(df: pd.DataFrame) -> str:
    for c in ["series", "title", "name"]:
        if c in df.columns:
            return c
    # 予防的にseriesを作成
    if "series" not in df.columns:
        df["series"] = "NA"
    return "series"

def _get_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["machine_id", "slot_id", "unit_no", "sid", "machine", "id", "num"]:
        if c in df.columns:
            return c
    return None

def _get_samai_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).lower().startswith("samai"):
            return c
    return None

# ====================
# 日付・データ処理関数
# ====================
def _force_parse_date_col(df, col="date"):
    s = df[col].astype(str)
    m = s.str.extract(r'([0-9]{4})[^0-9]?([0-9]{1,2})[^0-9]?([0-9]{1,2})', expand=True)
    parsed = (m[0].fillna('') + '-' + m[1].fillna('').str.zfill(2) + '-' + m[2].fillna('').str.zfill(2))
    df[col] = pd.to_datetime(parsed, errors='coerce')
    return df

def parse_date_unified(s) -> pd.Timestamp:
    s = str(s).strip()
    d = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if pd.isna(d):
        d = pd.to_datetime(s, format="%Y/%m/%d", errors="coerce")
    if pd.isna(d):
        d = pd.to_datetime("2025-" + s, format="%Y-%m/%d", errors="coerce")
    return d

def norm_series_key_fixed(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0].lower()
    known_series = {
        "monkey": "monkey",
        "hokuto": "hokuto", 
        "ghoul": "ghoul",
        "myjuggler": "myjugglerv",
        "myjugglerv": "myjugglerv"
    }
    for key, normalized in known_series.items():
        if key in base:
            return normalized
    return re.sub(r"(_2|_all.*|_test.*)$", "", base)

def read_and_stack(csv_list, src_tag: str) -> pd.DataFrame:
    frames = []
    for p in csv_list:
        df = pd.read_csv(p)
        miss = REQ_COLS - set(df.columns)
        if miss:
            raise ValueError(f"{p} に必要列がありません: {miss}")
        df["date"] = df["date"].map(parse_date_unified)
        df["num"] = pd.to_numeric(df["num"], errors="coerce").astype(float)
        df["samai"] = pd.to_numeric(df["samai"], errors="coerce").astype(float)
        df["g_num"] = pd.to_numeric(df["g_num"], errors="coerce").astype(float)
        df["avg"] = pd.to_numeric(df["avg"], errors="coerce").astype(float)
        

        # CSVにseries列がある場合はそれを使用、ない場合のみファイル名から抽出
        if "series" not in df.columns or df["series"].isna().all():
            df["series"] = norm_series_key_fixed(p)

        df["machine_id"] = df["num"].astype(int).astype(str) + "_" + df["series"]
        df["target"] = (df["samai"] >= 2000).astype(int)
        df["src"] = src_tag
        frames.append(df[["date", "series", "num", "machine_id", "target", "g_num", "avg", "samai", "src"]])
    df = pd.concat(frames, ignore_index=True).dropna(subset=["date"]).sort_values(["series", "machine_id", "date"]).reset_index(drop=True)
    log(f"[READ] {src_tag}: {len(df)} rows, {df['series'].nunique()} series")
    return df

def is_special_date(ts) -> bool:
    try:
        d = pd.Timestamp(ts).day
    except Exception:
        return False
    return d in SPECIAL_DAYS

# ====================
# カレンダー特徴量
# ====================
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dts = pd.to_datetime(df["date"]).dt
    df["is_special"] = dts.day.astype(int).isin(list(SPECIAL_DAYS)).astype(int)
    df["is_weekend"] = dts.weekday.isin([5, 6]).astype(int)
    df["weekday"] = dts.weekday.astype("Int64")
    w = dts.weekday.astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * w / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * w / 7.0)
    dom = dts.day.astype(int)
    df["dom_sin"] = np.sin(2 * np.pi * dom / 31.0)
    df["dom_cos"] = np.cos(2 * np.pi * dom / 31.0)
    return df

# ====================
# sub.py由来の高度特徴量
# ====================
def add_tail_eq_dom_d1(df: pd.DataFrame) -> pd.DataFrame:
    """台番号末尾=日付末尾のフラグ（修正版）"""
    df = df.copy()
    idc = _get_id_col(df)
    dd = pd.to_datetime(df["date"]).dt.day % 10
    
    if idc is None:
        df[TAIL_COL] = 0
        return df
    
    s = df[idc].astype(str).str.replace(r"[^0-9]", "", regex=True)
    ld = pd.to_numeric(s.str[-1], errors="coerce")
    ld_mod = ld % 10
    
    # 修正: より明確なNaN処理
    df[TAIL_COL] = ((ld_mod == dd) & ld_mod.notna()).astype(int)
    return df

def add_weekday_te(trn: pd.DataFrame, val: pd.DataFrame, serc: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Weekday TE（修正版：履歴データなしでも動作）"""
    trn = trn.copy()
    val = val.copy()

    for D in (trn, val):
        if "weekday" not in D.columns:
            D["weekday"] = pd.to_datetime(D["date"]).dt.weekday

    # trnが空の場合はvalのデータを使用
    if len(trn) == 0 or trn["target"].isna().all():
        # デフォルト値を設定
        gm = 0.1  # デフォルトの平均的中率
        
        for D in [val]:
            D["hit_rate_te_wk"] = gm
            D["hit_rate_te_wk_ns"] = gm
            D["sxw_ema14_prev"] = gm
            D["sxw_ema14_resid"] = 0.0
        
        return trn, val

    gm = float(trn["target"].mean())
    
    # 全体
    ag = trn.groupby([serc, "weekday"])["target"].agg(["sum", "count"]).reset_index()
    ag["hit_rate_te_wk"] = (ag["sum"] + 5 * gm) / (ag["count"] + 5)
    
    # 非特日
    ag_ns = (
        trn[trn["is_special"] != 1]
        .groupby([serc, "weekday"])["target"]
        .agg(["sum", "count"])
        .reset_index()
    )
    ag_ns["hit_rate_te_wk_ns"] = (ag_ns["sum"] + 5 * gm) / (ag_ns["count"] + 5)

    def _merge_te(D: pd.DataFrame) -> pd.DataFrame:
        D = D.merge(ag[[serc, "weekday", "hit_rate_te_wk"]], on=[serc, "weekday"], how="left")
        D = D.merge(ag_ns[[serc, "weekday", "hit_rate_te_wk_ns"]], on=[serc, "weekday"], how="left")
        D["hit_rate_te_wk"] = D["hit_rate_te_wk"].fillna(gm)
        D["hit_rate_te_wk_ns"] = D["hit_rate_te_wk_ns"].fillna(gm)
        return D

    trn = _merge_te(trn)
    val = _merge_te(val)

    # weekday-EMA14
    tmp = trn[[serc, "weekday", "date", "target"]].sort_values("date").copy()
    tmp["sxw_ema14_prev"] = (
        tmp.groupby([serc, "weekday"])["target"].transform(lambda x: x.ewm(span=14, adjust=False).mean())
    )
    ema_tbl = tmp.groupby([serc, "weekday"])["sxw_ema14_prev"].last().reset_index()

    def _merge_ema(D: pd.DataFrame) -> pd.DataFrame:
        D = D.merge(ema_tbl, on=[serc, "weekday"], how="left")
        D["sxw_ema14_prev"] = D["sxw_ema14_prev"].fillna(gm)
        return D

    trn = _merge_ema(trn)
    val = _merge_ema(val)

    trn["sxw_ema14_resid"] = trn["sxw_ema14_prev"] - trn["hit_rate_te_wk"]
    val["sxw_ema14_resid"] = val["sxw_ema14_prev"] - val["hit_rate_te_wk"]

    return trn, val


def add_p90_block(trn: pd.DataFrame, val: pd.DataFrame, serc: str, sam_col: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """p90系特徴量（修正版）"""
    trn = trn.copy()
    val = val.copy()

    # sam_colがない、またはtrnが空の場合
    if sam_col is None or sam_col not in trn.columns or len(trn) == 0:
        # デフォルト値を設定
        for D in (trn, val):
            for c in P90_COLS:
                if c not in D.columns:
                    D[c] = 0.0
        return trn, val

    # 訓練データから特徴量を計算
    t = trn[[serc, "date", sam_col]].dropna().sort_values([serc, "date"]).copy()
    
    if len(t) == 0:
        for D in (trn, val):
            for c in P90_COLS:
                if c not in D.columns:
                    D[c] = 0.0
        return trn, val

    # グループ化して統計量を計算
    gp = t.groupby(serc)[sam_col]
    
    # 各統計量を計算（shift(1)で未来データの漏洩を防ぐ）
    q90 = gp.transform(lambda x: x.shift(1).rolling(14, min_periods=1).quantile(0.9))
    q50 = gp.transform(lambda x: x.shift(1).rolling(14, min_periods=1).quantile(0.5))
    m14 = gp.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    s14 = gp.transform(lambda x: x.shift(1).rolling(14, min_periods=1).std())
    
    # 特徴量を作成
    t["tmp_p90_gap_avg_14"] = (q90 - m14).fillna(0.0)
    t["tmp_p90_z14"] = ((q90 - m14) / s14.replace(0, np.nan)).fillna(0.0)
    t["tmp_stretch_p90p50_14"] = (q90 / q50.replace(0, np.nan)).fillna(1.0)
    
    # 各グループの最新値を取得
    tbl = (
        t.groupby(serc)[["tmp_p90_gap_avg_14", "tmp_p90_z14", "tmp_stretch_p90p50_14"]]
        .last()
        .reset_index()
    )

    def _merge_p90(D: pd.DataFrame) -> pd.DataFrame:
        D = D.merge(tbl, on=serc, how="left")
        
        # 既存値がある場合は保持、ない場合は新しい値を使用
        if "p90_gap_avg_14" in D.columns:
            D["p90_gap_avg_14"] = D["p90_gap_avg_14"].fillna(D["tmp_p90_gap_avg_14"]).fillna(0.0)
        else:
            D["p90_gap_avg_14"] = D["tmp_p90_gap_avg_14"].fillna(0.0)
            
        if "p90_z14" in D.columns:
            D["p90_z14"] = D["p90_z14"].fillna(D["tmp_p90_z14"]).fillna(0.0)
        else:
            D["p90_z14"] = D["tmp_p90_z14"].fillna(0.0)
            
        if "stretch_p90p50_14" in D.columns:
            D["stretch_p90p50_14"] = D["stretch_p90p50_14"].fillna(D["tmp_stretch_p90p50_14"]).fillna(1.0)
        else:
            D["stretch_p90p50_14"] = D["tmp_stretch_p90p50_14"].fillna(1.0)
        
        D = D.drop(columns=["tmp_p90_gap_avg_14", "tmp_p90_z14", "tmp_stretch_p90p50_14"], errors='ignore')
        return D

    trn = _merge_p90(trn)
    val = _merge_p90(val)
    return trn, val

def add_corners(trn: pd.DataFrame, val: pd.DataFrame, serc: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """角台フラグ群（日付対応版）"""
    def _flag(D: pd.DataFrame) -> pd.DataFrame:
        out = D.copy()
        idc = _get_id_col(out)
        if "weekday" not in out.columns:
            out["weekday"] = pd.to_datetime(out["date"]).dt.weekday
            
        if idc is None:
            out["corner_flag"] = 0
        else:
            s = out[idc].astype(str).str.replace(r"[^0-9]", "", regex=True)
            idn = pd.to_numeric(s, errors="coerce")
            
            c_hok = out[serc].astype(str).str.contains("hokuto", case=False, na=False)
            c_ghl = out[serc].astype(str).str.contains("ghoul", case=False, na=False)
            c_mky = out[serc].astype(str).str.contains("monkey", case=False, na=False)
            c_jug = out[serc].astype(str).str.contains("jugg", case=False, na=False)
            
            # 日付対応の角台ID取得
            date = out["date"].iloc[0] if len(out) > 0 else pd.Timestamp.now()
            hokuto_ids = get_corner_ids_for_date("hokuto", date)
            ghoul_ids = get_corner_ids_for_date("ghoul", date)
            monkey_ids = get_corner_ids_for_date("monkey", date)
            jugg_ids = get_corner_ids_for_date("jugg", date)
            
            f = (
                (c_hok & idn.isin(hokuto_ids)) |
                (c_ghl & idn.isin(ghoul_ids)) |
                (c_mky & idn.isin(monkey_ids)) |
                (c_jug & idn.isin(jugg_ids))
            ).astype(int)
            out["corner_flag"] = f
            
        # 特別日との交互作用
        if "is_special" not in out.columns:
            out = add_calendar_features(out)
            
        out["corner_x_special"] = out["corner_flag"] * (out["is_special"] == 1).astype(int)
        
        # 曜日別角台フラグ
        for i in range(7):
            out[f"corner_wd_{i}"] = out["corner_flag"] * (out["weekday"] == i).astype(int)
            
        return out

    trn = _flag(trn)
    val = _flag(val)
    return trn, val

def get_corner_ids_for_date(series: str, date) -> set:
    """日付対応角台ID取得"""
    series_lower = series.lower()
    if series_lower not in CORNER_IDS_BY_DATE:
        return set()
    
    dates_dict = CORNER_IDS_BY_DATE[series_lower]
    applicable_date = "default"
    target_date = pd.to_datetime(date)
    
    for date_key in sorted(dates_dict.keys()):
        if date_key != "default" and target_date >= pd.to_datetime(date_key):
            applicable_date = date_key
    
    return dates_dict[applicable_date]

# ====================
# main.py由来の基本特徴量
# ====================
def add_exante_features(df: pd.DataFrame) -> pd.DataFrame:
    """基本特徴量の追加（lag特徴量修正版）"""
    df = df.copy()
    
    if "machine_id" not in df.columns:
        df["machine_id"] = df["series"].astype(str) + "_" + df["num"].astype(str)
    
    dcol = "date_used" if "date_used" in df.columns else "date"
    
    # 重要：ソート後にインデックスをリセット
    df = df.sort_values(["machine_id", dcol], kind="mergesort").reset_index(drop=True)
    
    # デバッグ出力
    log(f"[DEBUG] After sort - shape: {df.shape}, machine_id count: {df['machine_id'].nunique()}")
    
    g = df.groupby("machine_id", sort=False)
    
    for prefix in ["samai", "g_num", "avg"]:
        if prefix not in df.columns:
            log(f"[WARNING] Column {prefix} not found, skipping")
            continue
            
        df[prefix] = pd.to_numeric(df[prefix], errors='coerce')
        
        # lag特徴量（修正版）
        df[f"{prefix}_lag1"] = g[prefix].shift(1)
        df[f"{prefix}_lag2"] = g[prefix].shift(2)
        df[f"{prefix}_lag3"] = g[prefix].shift(3)
        
        # rolling特徴量（修正版）
        df[f"{prefix}_roll3"] = g[prefix].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        df[f"{prefix}_roll7"] = g[prefix].transform(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
        
        # デバッグ出力
        lag1_nulls = df[f"{prefix}_lag1"].isna().sum()
        log(f"[DEBUG] {prefix}_lag1 - null count: {lag1_nulls}/{len(df)}")

    # 過去7日間でsamai>=2000だった割合
    df["recent_hit_rate"] = g["samai"].transform(
        lambda s: (s.shift(1) >= 2000).astype(float).rolling(7, min_periods=2).mean()
    )

    # 最後のヒットからの経過日数
    def _days_since_last_2000_vec(s: pd.Series) -> pd.Series:
        hits = (s >= 2000).astype(int).values
        out = [np.nan] * len(hits)
        last = None
        for i in range(len(hits)):
            if last is None:
                out[i] = np.nan
            else:
                out[i] = last
            last = 0 if hits[i] == 1 else (0 if last is None else last + 1)
        return pd.Series(out, index=s.index, dtype="float")

    df["days_since_last_2000"] = g["samai"].transform(_days_since_last_2000_vec)

    # 変化量（MoM）
    df["samai_mom"] = df["samai_lag1"] - df["samai_lag2"]
    
    # 簡単な検証
    if len(df) > 0:
        test_id = df["machine_id"].iloc[0]
        test_data = df[df["machine_id"] == test_id].head(5)
        log(f"[VALIDATION] Sample from {test_id}:")
        for i in range(1, min(len(test_data), 4)):
            if i < len(test_data):
                orig = test_data["samai"].iloc[i-1] if i > 0 else np.nan
                lag1 = test_data["samai_lag1"].iloc[i]
                log(f"  Row {i}: samai[{i-1}]={orig} -> lag1[{i}]={lag1}")
    
    return df

def _validate_lag_features(df: pd.DataFrame):
    """lag特徴量の整合性をチェック"""
    
    # テストケース：特定のmachine_idで確認
    if len(df) > 0:
        test_id = df["machine_id"].iloc[0]
        test_data = df[df["machine_id"] == test_id].sort_values("date").head(10)
        
        log(f"[VALIDATION] Testing machine_id: {test_id}")
        log(f"[VALIDATION] Sample data shape: {test_data.shape}")
        
        for prefix in ["samai", "g_num", "avg"]:
            if f"{prefix}_lag1" in df.columns:
                # 元の値とlag1の関係をチェック
                for i in range(1, min(len(test_data), 5)):
                    original = test_data[prefix].iloc[i-1]
                    lag1 = test_data[f"{prefix}_lag1"].iloc[i]
                    
                    if pd.notna(original) and pd.notna(lag1):
                        if abs(original - lag1) > 1e-6:  # 浮動小数点の誤差を考慮
                            log(f"[ERROR] Lag validation failed for {prefix}")
                            log(f"[ERROR] Row {i}: original={original}, lag1={lag1}")
                            return False
                        else:
                            log(f"[OK] {prefix} lag1 validation passed for row {i}")

def add_series_monthweek_features(df: pd.DataFrame) -> pd.DataFrame:
    """シリーズ別月週特徴量（FutureWarning修正版）"""
    log("[FEAT] add_series_monthweek_features")
    if df is None or len(df) == 0:
        return df

    needs = {"series", "machine_id", "date", "samai"}
    if not needs.issubset(df.columns):
        log("[FEAT] add_series_monthweek_features: skip (missing columns)")
        return df

    d = df.sort_values(["series", "machine_id", "date"]).copy()
    mkey = d["date"].dt.to_period("M")
    wkey = d["date"].dt.to_period("W")

    # 修正：groupby後にcumsumし、その後machine_id別にshift
    g_m = d.groupby(["series", "machine_id", mkey], sort=False)["samai"]
    g_w = d.groupby(["series", "machine_id", wkey], sort=False)["samai"]

    # 累積和を計算してからmachine_id別にshift
    month_cumsum = g_m.cumsum()
    week_cumsum = g_w.cumsum()
    
    # group_keys=Falseを追加してFutureWarningを解消
    d["series_month_cumsum_d1"] = d.groupby("machine_id", sort=False, group_keys=False)[month_cumsum.name].apply(lambda x: month_cumsum[x.index].shift(1))
    d["series_week_cumsum_d1"] = d.groupby("machine_id", sort=False, group_keys=False)[week_cumsum.name].apply(lambda x: week_cumsum[x.index].shift(1))

    # その"グループ内でのこれまでの日数"（=分母）も作る
    d["_m_days"] = g_m.cumcount()
    d["_w_days"] = g_w.cumcount()

    d["series_month_cumsum_perday_d1"] = (d["series_month_cumsum_d1"] / d["_m_days"].replace(0, np.nan)).fillna(0.0)
    d["series_week_cumsum_perday_d1"] = (d["series_week_cumsum_d1"] / d["_w_days"].replace(0, np.nan)).fillna(0.0)

    d = d.drop(columns=["_m_days", "_w_days"])
    return d

def add_compat_feature_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    互換性のある特徴量エイリアス作成（修正版）
    """
    log("[FEAT] add_compat_feature_aliases")
    df = df.copy()
    n = len(df)
    
    # machine_prior_hit: 前日ヒットフラグの代替作成
    if "machine_prior_hit" not in df.columns:
        if "samai_lag1" in df.columns:
            df["machine_prior_hit"] = (
                pd.to_numeric(df["samai_lag1"], errors="coerce") >= 2000
            ).astype("Int64")
        else:
            df["machine_prior_hit"] = pd.Series([pd.NA] * n, dtype="Int64")
    
    # gnum_prior_mean: 事前平均値の代替作成
    if "gnum_prior_mean" not in df.columns:
        # 優先順位順にチェック
        candidates = ["g_num_roll3", "g_num_roll7", "g_num_lag1"]
        cand = None
        for c in candidates:
            if c in df.columns:
                cand = c
                break
        
        if cand:
            df["gnum_prior_mean"] = pd.to_numeric(df[cand], errors="coerce")
        else:
            df["gnum_prior_mean"] = np.nan
    
    # MoM (Month over Month) 特徴量の作成
    def _create_mom_feature(lag1_col, lag2_col, output_col):
        """MoM特徴量作成のヘルパー関数"""
        if output_col not in df.columns:
            if lag1_col in df.columns and lag2_col in df.columns:
                val1 = pd.to_numeric(df[lag1_col], errors="coerce")
                val2 = pd.to_numeric(df[lag2_col], errors="coerce")
                df[output_col] = val1 - val2
            else:
                df[output_col] = np.nan
    
    # MoM特徴量を作成
    _create_mom_feature("samai_lag1", "samai_lag2", "samai_mom1")
    _create_mom_feature("g_num_lag1", "g_num_lag2", "gnum_mom1")
    _create_mom_feature("avg_lag1", "avg_lag2", "avg_mom1")
    
    return df


def debug_lag_generation(df: pd.DataFrame, machine_id: str = None) -> None:
    """
    lag特徴量生成のデバッグ関数
    """
    print("\n" + "="*60)
    print("LAG特徴量生成デバッグ")
    print("="*60)
    
    if machine_id is None:
        # 最初のmachine_idを使用
        machine_id = df["machine_id"].iloc[0]
    
    # 特定のマシンのデータを抽出
    sample_data = df[df["machine_id"] == machine_id].sort_values("date").head(10).copy()
    
    print(f"Machine ID: {machine_id}")
    print(f"Sample size: {len(sample_data)}")
    print(f"Date range: {sample_data['date'].min()} to {sample_data['date'].max()}")
    
    # 主要列を表示
    display_cols = ["date", "samai", "samai_lag1", "samai_lag2", "g_num", "g_num_lag1"]
    available_cols = [col for col in display_cols if col in sample_data.columns]
    
    print(f"\nColumns to display: {available_cols}")
    print("\nSample data:")
    for i, (_, row) in enumerate(sample_data[available_cols].iterrows()):
        print(f"Row {i:2d}: " + " | ".join([f"{col}={row[col]}" for col in available_cols]))
    
    # 検証
    print(f"\nLag validation:")
    for i in range(1, min(len(sample_data), 5)):
        if "samai_lag1" in sample_data.columns:
            prev_samai = sample_data["samai"].iloc[i-1] 
            curr_lag1 = sample_data["samai_lag1"].iloc[i]
            status = "✓" if pd.notna(prev_samai) and pd.notna(curr_lag1) and abs(prev_samai - curr_lag1) < 1e-6 else "✗"
            print(f"  Row {i}: samai[{i-1}]={prev_samai} vs lag1[{i}]={curr_lag1} {status}")

def _debug_check_monthweek(df: pd.DataFrame, where: str):
    cols = ["series_month_cumsum_d1", "series_month_cumsum_perday_d1", 
            "series_week_cumsum_d1", "series_week_cumsum_perday_d1"]
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    log(f"[FEAT] monthweek_present@{where}: {present} missing={missing}")
    for c in present:
        nulls = int(df[c].isna().sum())
        log(f"[FEAT] monthweek_nulls@{where}: {c} nulls={nulls}")


def add_series_aggregates(df: pd.DataFrame, target_date=None, production_mode=False) -> pd.DataFrame:
    log("[FEAT] add_series_aggregates (production_mode supported, leak-free)")

    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.sort_values(["series", "date", "machine_id"]).copy()

    # targetの再計算（前日samai使用）
    df["target_d1"] = (df.groupby("machine_id")["samai"].shift(1) >= 2000).astype(int).fillna(0)

    # === 重要：常に前日までのデータで集約を作成 ===
    # 各日付に対して、その日より前のデータのみで集約を計算
    
    # 日付でソート
    unique_dates = sorted(df["date"].unique())
    
    # 各日付の集約結果を格納
    ag_list = []
    hall_list = []
    sd_list = []
    
    for current_date in unique_dates:
        # 当日より前のデータのみを使用（未来リーク防止）
        fit_df = df[df["date"] < current_date].copy()
        
        if len(fit_df) == 0:
            # 初日など履歴がない場合はスキップ
            continue
        
        # シリーズ集約（前日までのデータで計算）
        ag_day = (
            fit_df.groupby("series").agg(
                hits=("target_d1", "sum"),
                gnum_p90=("g_num_lag1", lambda s: float(np.nanquantile(s.dropna(), 0.9)) if len(s.dropna()) > 0 else np.nan),
                avg_p90=("avg_lag1", lambda s: float(np.nanquantile(s.dropna(), 0.9)) if len(s.dropna()) > 0 else np.nan),
                samai_p90=("samai_lag1", lambda s: float(np.nanquantile(s.dropna(), 0.9)) if len(s.dropna()) > 0 else np.nan),
                hit_rate2000=("target_d1", "mean"),
                spike_g=("g_num_lag1", lambda s: np.nanmax(s.dropna()) if len(s.dropna()) > 0 else np.nan),
            ).reset_index()
        )
        ag_day["date"] = current_date
        ag_list.append(ag_day)
        
        # ホール全体集約（前日までのデータで計算）
        hall_day = pd.DataFrame({
            "date": [current_date],
            "hall_samai_p90": [float(np.nanquantile(fit_df["samai_lag1"].dropna(), 0.9)) if len(fit_df["samai_lag1"].dropna()) > 0 else np.nan],
            "hall_samai_med": [fit_df["samai_lag1"].median()],
            "hall_hit_rate2000": [fit_df["target_d1"].mean()],
        })
        hall_list.append(hall_day)
        
        # 非特日のHR集約（前日までのデータで計算）
        nonsp = fit_df[fit_df["is_special"] == 0]
        if len(nonsp) > 0:
            sd_day = (
                nonsp.groupby("series").agg(
                    hr_nonsp_ma14=("target_d1", lambda s: s.iloc[-14:].mean() if len(s) > 0 else np.nan),
                    hr_nonsp_wk_ma=("target_d1", lambda s: s.iloc[-7:].mean() if len(s) > 0 else np.nan),
                ).reset_index()
            )
            sd_day["date"] = current_date
            sd_list.append(sd_day)
    
    # 集約結果を結合
    if ag_list:
        ag = pd.concat(ag_list, ignore_index=True)
        df = df.merge(ag, on=["series", "date"], how="left")
    
    if hall_list:
        hall = pd.concat(hall_list, ignore_index=True)
        df = df.merge(hall, on="date", how="left")
    
    if sd_list:
        sd = pd.concat(sd_list, ignore_index=True)
        df = df.merge(sd, on=["series", "date"], how="left")
    
    # Leave-One-Out処理（自分自身の影響を除外）
    df["_grp_n"] = df.groupby(["series", "date"])["machine_id"].transform("size")
    df["_hits_all"] = df["hits"].fillna(0)
    df["hits"] = (df["_hits_all"] - df["target_d1"]).clip(lower=0)
    df["hit_rate2000"] = np.where(
        df["_grp_n"] > 1,
        df["hits"] / (df["_grp_n"] - 1),
        0.0
    )
    df.drop(columns=["_grp_n", "_hits_all"], inplace=True, errors='ignore')
    
    # fill対象の列
    fill_cols = [
        "hits", "gnum_p90", "avg_p90", "samai_p90", "hit_rate2000", "spike_g",
        "hall_samai_p90", "hall_samai_med", "hall_hit_rate2000",
        "hr_nonsp_ma14", "hr_nonsp_wk_ma"
    ]
    
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    log("[FEAT] add_series_aggregates completed (leak-free)")
    return df

def add_series_weekday_resid(df: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    """
    シリーズ×曜日ごとの差枚平均からの乖離を計算（未来リーク修正版）
    
    修正点：
    - rolling -> expanding に変更（過去全データを使用）
    - または、明示的に日付フィルタリングを追加
    """
    if df is None or len(df) == 0:
        return df
    
    d = df.copy()
    
    # weekdayカラムの準備
    if "weekday" not in d.columns:
        d["weekday"] = pd.to_datetime(d["date"]).dt.weekday
    
    # 日付でソート（重要）
    d = d.sort_values(["series", "machine_id", "date"]).reset_index(drop=True)
    
    if "samai" not in d.columns:
        d[f"samai_serieswk_mean{window}_d1"] = 0.0
        d["samai_lag1_dev_serieswk_d1"] = 0.0
        return d
    
    # === 方法1: 日付ベースの明示的フィルタリング（推奨） ===
    unique_dates = sorted(d["date"].unique())
    mean_values = []
    
    for current_date in unique_dates:
        # 当日より前のデータのみ
        past_data = d[d["date"] < current_date].copy()
        
        if len(past_data) == 0:
            # 初日はデータなし
            current_rows = d[d["date"] == current_date]
            mean_values.extend([0.0] * len(current_rows))
            continue
        
        # series × weekday ごとの平均を計算
        weekday_means = (
            past_data.groupby(["series", "weekday"])["samai"]
            .apply(lambda s: s.iloc[-window:].mean() if len(s) > 0 else 0.0)
            .to_dict()
        )
        
        # 当日の各行に対応する平均値を取得
        current_rows = d[d["date"] == current_date].copy()
        for _, row in current_rows.iterrows():
            key = (row["series"], row["weekday"])
            mean_values.append(weekday_means.get(key, 0.0))
    
    d[f"samai_serieswk_mean{window}_d1"] = mean_values
    
    # === 方法2: expanding計算（より簡潔だが計算量多い） ===
    # d = d.sort_values(["series", "weekday", "date"]).reset_index(drop=True)
    # g = d.groupby(["series", "weekday"], group_keys=False)
    # 
    # # expanding().mean() を使用（過去全データの平均）
    # mean_d1 = g["samai"].apply(
    #     lambda s: s.shift(1).expanding(min_periods=max(3, min(window//4, 7))).mean()
    # )
    # d[f"samai_serieswk_mean{window}_d1"] = mean_d1
    
    # 偏差の計算
    if "samai_lag1" in d.columns:
        d["samai_lag1_dev_serieswk_d1"] = d["samai_lag1"] - d[f"samai_serieswk_mean{window}_d1"]
    else:
        d["samai_lag1_dev_serieswk_d1"] = 0.0
    
    # 欠損値を0で埋める
    d[f"samai_serieswk_mean{window}_d1"] = d[f"samai_serieswk_mean{window}_d1"].fillna(0.0)
    d["samai_lag1_dev_serieswk_d1"] = d["samai_lag1_dev_serieswk_d1"].fillna(0.0)
    
    log(f"[FEAT] add_series_weekday_resid completed (window={window}, leak-free)")
    return d

def add_num_end_digit_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["num_end_digit"] = (pd.to_numeric(d["num"], errors="coerce").astype("Int64") % 10).astype("float")
    dom = pd.to_datetime(d["date"]).dt.day.astype(int)
    d["enddigit_eq_dom"] = (d["num_end_digit"].astype("Int64") == (dom % 10)).astype(int)
    
    if "is_special" not in d.columns:
        d = add_calendar_features(d)
    
    d["enddigit_eq_dom_x_special"] = d["enddigit_eq_dom"] * d["is_special"]
    return d

    
def add_neighbor_features(df: pd.DataFrame, radius_list=(1, 2)) -> pd.DataFrame:
    """近隣特徴量（角台対応・特徴量拡張版）"""
    log("[FEAT] add_neighbor_features (enhanced)")
    df = df.sort_values(["series", "date", "num"]).copy()
    
    # 前日特徴量を準備（samaiを削除、prev2000を復元）
    df["prev2000_d1"] = (df.groupby("machine_id", group_keys=False)["samai"].shift(1) >= 2000).astype(int).fillna(0)
    df["gnum_d1"] = df.groupby("machine_id", group_keys=False)["g_num"].shift(1).fillna(0.0)
    df["avg_d1"] = df.groupby("machine_id", group_keys=False)["avg"].shift(1).fillna(0.0)
    # df["samai_d1"] = ... # 削除
    
    out_frames = []
    
    for (s, d), g in df.groupby(["series", "date"], sort=False):
        g = g.sort_values("num").copy()
        n = len(g)
        
        # 角台判定
        corner_ids = get_corner_ids_for_date(s, d)
        corner_flags = g["num"].astype(int).isin(corner_ids).values
        
        # 各特徴量の値を取得（samaiを除外）
        features = {
            "prev2000": g["prev2000_d1"].values.astype(float),  # prev2000復活
            "gnum": g["gnum_d1"].values.astype(float),
            "avg": g["avg_d1"].values.astype(float),
            # "samai": g["samai_d1"].values.astype(float)  # 削除
        }
        
        # 以下のループ処理は同じ
        for r in radius_list:
            for feat_name, values in features.items():
                # 近隣集約計算
                kernel = np.ones(2 * r + 1, dtype=float)
                pad_values = np.pad(values, (r, r), mode="constant", constant_values=0.0)
                csum_values = np.convolve(pad_values, kernel, mode="valid")
                neigh_sum = csum_values - values
                
                # 角台考慮の有効近隣台数
                avail = np.zeros(n, dtype=float)
                for i in range(n):
                    if corner_flags[i]:
                        # 角台：片側のみ
                        if i == 0:
                            avail[i] = min(r, n - 1 - i)
                        elif i == n - 1:
                            avail[i] = min(r, i)
                        else:
                            avail[i] = min(r, i) + min(r, n - 1 - i)
                    else:
                        # 通常台：両側
                        avail[i] = min(r, i) + min(r, n - 1 - i)
                
                neigh_rate = np.divide(neigh_sum, avail, out=np.zeros_like(neigh_sum), where=avail > 0)
                
                g[f"neigh_{feat_name}_pm{r}_sum_d1"] = neigh_sum
                g[f"neigh_{feat_name}_pm{r}_rate_d1"] = neigh_rate
        
        out_frames.append(g)
    
    df = pd.concat(out_frames, ignore_index=True)
    
    # prev2000関連の削除は削除（復活させるため）
    # if "prev2000_d1" in df.columns:
    #     df = df.drop(columns=["prev2000_d1"])
    
    return df


def get_corner_ids_for_date(series: str, date) -> set:
    # 日付対応角台ID取得
    series_lower = series.lower()
    if series_lower not in CORNER_IDS_BY_DATE:
        return set()
    
    dates_dict = CORNER_IDS_BY_DATE[series_lower]
    applicable_date = "default"
    target_date = pd.to_datetime(date)
    
    for date_key in sorted(dates_dict.keys()):
        if date_key != "default" and target_date >= pd.to_datetime(date_key):
            applicable_date = date_key
    
    return dates_dict[applicable_date]

def add_renzoku_hekomi_features(df: pd.DataFrame) -> pd.DataFrame:
    log("[FEAT] add_renzoku_hekomi_features")
    df = df.sort_values(["machine_id", "date"]).copy()
    
    # 前日のsamaiを使用してヒット判定
    prev_samai = df.groupby("machine_id")["samai"].shift(1)
    df["is_hit"] = (prev_samai >= 2000).astype(int).fillna(0)
    df["is_hekomi"] = (prev_samai < 2000).astype(int).fillna(0)
    
    def _calculate_consecutive_count(series: pd.Series, target_value: int) -> pd.Series:
        ##連続カウント計算
        result = pd.Series(0, index=series.index, dtype=int)
        current_count = 0
        
        for i, value in enumerate(series):
            if value == target_value:
                current_count += 1
                result.iloc[i] = current_count
            else:
                current_count = 0
                result.iloc[i] = 0
        
        return result
    
    # machine_id別にグループ化して連続カウント計算
    g = df.groupby("machine_id", sort=False)
    
    # 連続ヘコミ日数（2000枚未満の連続日数）
    df["renzoku_hekomi_count"] = g["is_hekomi"].transform(_calculate_consecutive_count, 1)
    
    # 連続当たり日数（2000枚以上の連続日数）
    df["renzoku_hit_count"] = g["is_hit"].transform(_calculate_consecutive_count, 1)
    
    # 前日までの連続ヘコミ/当たり日数（リーク防止）
    df["renzoku_hekomi_count_d1"] = g["renzoku_hekomi_count"].shift(1).fillna(0).astype(int)
    df["renzoku_hit_count_d1"] = g["renzoku_hit_count"].shift(1).fillna(0).astype(int)
    
    # 最大連続ヘコミ記録（過去の最長記録）
    df["max_renzoku_hekomi_d1"] = g["renzoku_hekomi_count_d1"].transform(lambda x: x.expanding().max()).fillna(0).astype(int)
    df["max_renzoku_hit_d1"] = g["renzoku_hit_count_d1"].transform(lambda x: x.expanding().max()).fillna(0).astype(int)
    
    # 連続ヘコミからの回復フラグ（前日まで連続ヘコミで今日当たり）
    df["recovery_from_hekomi"] = ((df["renzoku_hekomi_count_d1"] > 0) & (df["is_hit"] == 1)).astype(int)
    
    # 長期ヘコミフラグ（5日以上連続）
    df["long_hekomi_d1"] = (df["renzoku_hekomi_count_d1"] >= 5).astype(int)
    
    # デバッグ出力
    hekomi_features = ["renzoku_hekomi_count_d1", "renzoku_hit_count_d1", "max_renzoku_hekomi_d1", 
                      "recovery_from_hekomi", "long_hekomi_d1"]
    
    for feat in hekomi_features:
        non_zero = (df[feat] > 0).sum()
        log(f"[DEBUG] {feat}: {non_zero}/{len(df)} non-zero values")
    
    # 一時的な列を削除
    df = df.drop(columns=["is_hit", "is_hekomi", "renzoku_hekomi_count", "renzoku_hit_count"], errors='ignore')
    
    return df 

def add_wave_trend_features(df: pd.DataFrame, wins=(7, 14, 28), cols=("samai", "g_num", "avg"), 
                           z_thr=0.8, r2_min=0.05) -> pd.DataFrame:
    df = df.sort_values(["series", "machine_id", "date"]).copy()
    g = df.groupby("machine_id", group_keys=False)

    for col in cols:
        if col not in df.columns:
            continue
            
        for H in wins:
            t = np.arange(H, dtype=float)
            t_mean = t.mean()
            Sxx = np.sum((t - t_mean) ** 2)

            def _slope(arr, t=t, t_mean=t_mean, Sxx=Sxx):
                y = np.asarray(arr, dtype=float)
                if np.isnan(y).any():
                    return np.nan
                y_mean = y.mean()
                num = np.dot(t - t_mean, y - y_mean)
                return float(num / Sxx) if Sxx > 0 else 0.0

            def _r2(arr, t=t, t_mean=t_mean, Sxx=Sxx):
                y = np.asarray(arr, dtype=float)
                if np.isnan(y).any():
                    return np.nan
                y_mean = y.mean()
                num = np.dot(t - t_mean, y - y_mean)
                denom = np.sqrt(Sxx * np.sum((y - y_mean) ** 2))
                if denom <= 0:
                    return 0.0
                r = num / denom
                return float(r * r)

            slope = g[col].apply(lambda s: s.shift(1).rolling(H, min_periods=H).apply(_slope, raw=True))
            r2 = g[col].apply(lambda s: s.shift(1).rolling(H, min_periods=H).apply(_r2, raw=True))
            std = g[col].apply(lambda s: s.shift(1).rolling(H, min_periods=H).std(ddof=0))

            z = slope * np.sqrt(Sxx) / std.replace(0, np.nan)
            z = z.replace([np.inf, -np.inf], np.nan)

            reg = pd.Series(0.0, index=z.index)
            reg = reg.where(~((z > z_thr) & (r2 >= r2_min)), 1.0)
            reg = reg.where(~((z < -z_thr) & (r2 >= r2_min)), -1.0)

            slope = slope.reset_index(level=0, drop=True)
            r2 = r2.reset_index(level=0, drop=True)
            z = z.reset_index(level=0, drop=True)
            reg = reg.reset_index(level=0, drop=True)

            df[f"wave_slope_{col}_w{H}_d1"] = slope.values
            df[f"wave_r2_{col}_w{H}_d1"] = r2.values
            df[f"wave_z_{col}_w{H}_d1"] = z.values
            df[f"wave_reg_{col}_w{H}_d1"] = reg.values

    # 欠損値の埋め合わせ
    fill_cols = [c for c in df.columns if c.startswith("wave_")]
    if fill_cols:
        df[fill_cols] = df[fill_cols].fillna(0.0)
    
    return df

def add_series_percentile_features(df: pd.DataFrame, target_date=None) -> pd.DataFrame:
    """シリーズ内パーセンタイル特徴量（修正版）"""
    log("[FEAT] add_series_percentile_features")
    
    if df is None or len(df) == 0:
        return df
    
    df = df.sort_values(["series", "date"]).copy()
    
    # lag特徴量の順位付けのみ実施（当日データは使用しない）
    for col in ["samai", "g_num", "avg"]:
        # 当日データの順位付けは削除（未来リーク）
        # df[f"{col}_rank_in_series_d"] = ... ← 削除
        
        # lag特徴量があれば、それを順位付け（安全）
        lag_col = f"{col}_lag1"
        if lag_col in df.columns:
            df[f"{lag_col}_pct_in_series_d"] = df.groupby(["series", "date"])[lag_col].rank(pct=True, method="min")
    
    return df

# ジニ係数と上位Kシェア関数
def gini_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return 0.0
    s = x.sum()
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    g = (2.0 * (np.arange(1, n + 1) * xs).sum()) / (n * s) - (n + 1.0) / n
    return float(g)

def topk_share(x: np.ndarray, k: int = 10) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    s = x.sum()
    if s <= 0 or x.size == 0:
        return 0.0
    kk = int(min(k, x.size))
    return float(np.sort(x)[-kk:].sum() / s)

def build_hall_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """ホール日次特徴量（修正版：前日データのみ使用）"""
    
    def q(s, p):
        s = np.asarray(s, dtype=float)
        s_valid = s[~np.isnan(s)]
        return float(np.nanquantile(s_valid, p)) if len(s_valid) > 0 else np.nan
    
    # 重要：lag特徴量が既に存在することを前提とする
    # add_exante_featuresの後に呼ぶ必要がある
    
    agg = (
        df.groupby("date").agg(
            # g_num_lag1を使用（前日のゲーム数）
            gnum_p90=("g_num_lag1", lambda s: q(s, 0.90)),
            gnum_p75=("g_num_lag1", lambda s: q(s, 0.75)),
            gnum_med=("g_num_lag1", "median"),
            gnum_var=("g_num_lag1", lambda s: s.var() if len(s.dropna()) > 1 else np.nan),
            gnum_gini=("g_num_lag1", lambda s: gini_np(s.dropna())),
            gnum_top10=("g_num_lag1", lambda s: topk_share(s.dropna(), k=10)),
            
            # avg_lag1を使用（前日の平均）
            avg_p90=("avg_lag1", lambda s: q(s, 0.90)),
            avg_p75=("avg_lag1", lambda s: q(s, 0.75)),
            avg_med=("avg_lag1", "median"),
            avg_var=("avg_lag1", lambda s: s.var() if len(s.dropna()) > 1 else np.nan),
            avg_gini=("avg_lag1", lambda s: gini_np(s.dropna())),
            avg_top10=("avg_lag1", lambda s: topk_share(s.dropna(), k=10)),
        ).reset_index()
    ).sort_values("date").reset_index(drop=True)
    
    return agg


def add_day_state_features(df: pd.DataFrame, fit: bool = False, 
                          scaler=None, kmeans=None, n_clusters: int = 5,
                          soft_assignment: bool = False, temperature: float = 1.0):
    """修正版：前日データのみを使用"""
    # lag特徴量が存在することを確認
    required_lags = ['g_num_lag1', 'avg_lag1']
    missing = [col for col in required_lags if col not in df.columns]
    if missing:
        log(f"[WARNING] Missing lag columns for day_state: {missing}")
        # lag特徴量がない場合は0で埋める
        for col in missing:
            df[col] = 0.0

    # 修正版のhall特徴量構築
    hall = build_hall_day_features(df)
    
    # 特徴量の準備
    feat_cols = [c for c in hall.columns if c != "date"]
    X = hall[feat_cols].fillna(0.0).values
    
    # 学習フェーズ
    if fit:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(Xs)
    
    # スケーラーまたはクラスタリングモデルがない場合
    if scaler is None or kmeans is None:
        # デフォルト値で埋める
        for i in range(n_clusters):
            df[f"ds_prob{i}_d1"] = 0.0
        return df, scaler, kmeans
    
    # 推論フェーズ
    Xs = scaler.transform(X)
    
    if soft_assignment:
        # ソフト割り当て
        D = kmeans.transform(Xs)
        W = np.exp(-D / temperature)
        W = W / W.sum(axis=1, keepdims=True)
        prob_data = W
    else:
        # ハード割り当て
        cluster_labels = kmeans.predict(Xs)
        prob_data = np.eye(n_clusters)[cluster_labels]
    
    # 確率データフレームの作成
    prob = pd.DataFrame(prob_data, columns=[f"ds_prob{i}" for i in range(n_clusters)])
    prob["date"] = hall["date"].values
    
    # 定休日を考慮した前営業日マッピング
    unique_dates = sorted(df["date"].unique())
    date_mapping = {}
    for i, current_date in enumerate(unique_dates):
        if i > 0:
            prev_date = unique_dates[i-1]
            date_mapping[current_date] = prev_date

    # 前営業日データを正しくマッピング（prob）
    prob_d1_rows = []
    for current_date, prev_date in date_mapping.items():
        prev_prob = prob[prob["date"] == prev_date]
        if len(prev_prob) > 0:
            prev_prob = prev_prob.copy()
            prev_prob["date"] = current_date
            prob_d1_rows.append(prev_prob)

    if prob_d1_rows:
        prob_d1 = pd.concat(prob_d1_rows, ignore_index=True)
    else:
        prob_d1 = pd.DataFrame(columns=prob.columns)

    prob_d1 = prob_d1.add_prefix("ds_")
    prob_d1 = prob_d1.rename(columns={"ds_date": "date"})

    # 前営業日データを正しくマッピング（hall）
    hall_d1_rows = []
    for current_date, prev_date in date_mapping.items():
        prev_hall = hall[hall["date"] == prev_date]
        if len(prev_hall) > 0:
            prev_hall = prev_hall.copy()
            prev_hall["date"] = current_date
            hall_d1_rows.append(prev_hall)

    if hall_d1_rows:
        hall_d1 = pd.concat(hall_d1_rows, ignore_index=True)
    else:
        hall_d1 = pd.DataFrame(columns=hall.columns)

    hall_d1 = hall_d1.add_prefix("hall_")
    hall_d1 = hall_d1.rename(columns={"hall_date": "date"})
    
    # マージ
    df = df.merge(hall_d1, on="date", how="left")
    df = df.merge(prob_d1, on="date", how="left")
    
    # 欠損値の埋め
    fill_cols = [c for c in df.columns if c.startswith("hall_") or c.startswith("ds_ds_prob")]
    if fill_cols:
        df[fill_cols] = df[fill_cols].fillna(0.0)
    
    return df, scaler, kmeans

def add_nonsp_power_features(df: pd.DataFrame) -> pd.DataFrame:
    need = {'date', 'series', 'num', 'samai', 'g_num', 'avg'}
    if not need.issubset(df.columns):
        return df

    df = df.sort_values(['series', 'num', 'date']).copy()
    key = ['series', 'num']
    
    # インデックスをリセットして互換性を確保
    df = df.reset_index(drop=True)
    
    # 基本移動平均特徴量を事前生成
    prefixes_and_cols = [
        ('gnum', 'g_num'),
        # 他のカラムが存在する場合は追加
    ]
    
    for prefix, base_col in prefixes_and_cols:
        if base_col in df.columns:
            # 方法1: transform を使用してインデックス互換性を確保
            df[f'{prefix}_ma3'] = df.groupby(key)[base_col].transform(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
            )
            df[f'{prefix}_ma7'] = df.groupby(key)[base_col].transform(
                lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
            )
            
            # ma3_over_ma7比率を計算（ゼロ除算対策）
            df[f'{prefix}_ma3_over_ma7'] = df[f'{prefix}_ma3'] / (df[f'{prefix}_ma7'] + 1e-8)
            
            print(f"[FEAT] Generated {prefix}_ma3, {prefix}_ma7, {prefix}_ma3_over_ma7")
    
    # 遅延特徴量を生成
    for prefix, base_col in prefixes_and_cols:
        if base_col in df.columns:
            # shift も transform を使用
            ma3_over_ma7_col = f'{prefix}_ma3_over_ma7'
            if ma3_over_ma7_col in df.columns:
                df[f'{prefix}_ma3_over_ma7_d1'] = df.groupby(key)[ma3_over_ma7_col].transform(
                    lambda x: x.shift(1)
                )
                print(f"[FEAT] Generated {prefix}_ma3_over_ma7_d1")
    
    return df

def add_bayesian_machine_ctr(df: pd.DataFrame, prior_strength: float = 10.0) -> pd.DataFrame:
    df = df.sort_values(["series", "machine_id", "date"]).copy()
    g = df.groupby("machine_id", group_keys=False)
    
    cum_pos = g["target"].shift(1).fillna(0).groupby(df["machine_id"], group_keys=False).cumsum()
    cum_n = g.cumcount()
    
    prior_rate = df.get("hr_nonsp_ma14_d1", pd.Series(0.1, index=df.index))
    prior_rate = prior_rate.where(prior_rate.notna(), 
                                 df.get("series_hit_rate2000_d1", pd.Series(0.1, index=df.index)))
    prior_rate = prior_rate.fillna(0.1)
    
    alpha = 1.0 + cum_pos + prior_rate * prior_strength
    beta = 1.0 + (cum_n - cum_pos).clip(lower=0) + (1.0 - prior_rate) * prior_strength
    df["machine_ctr_beta_d1"] = (alpha / (alpha + beta)).astype(float)
    
    return df

# ====================
# sub.pyの統合特徴量追加関数
# ====================
def add_all_new_features(trn: pd.DataFrame, val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """sub.pyの全特徴量をtrn/valに付与し、追加列名リストも返す"""
    serc = _get_series_col(trn)
    sam_col = _get_samai_col(trn)

    # 末尾桁一致特徴量
    trn = add_tail_eq_dom_d1(trn)
    val = add_tail_eq_dom_d1(val)

    # 高度特徴量の追加
    trn, val = add_weekday_te(trn, val, serc)
    trn, val = add_p90_block(trn, val, serc, sam_col)
    trn, val = add_corners(trn, val, serc)

    # 明示的に0埋め（保険）
    for D in (trn, val):
        for c in ADD_ALL:
            if c not in D.columns:
                D[c] = 0.0
            else:
                D[c] = D[c].fillna(0.0)

    return trn, val, ADD_ALL

# ====================
# 機械学習関連関数
# ====================
def filter_present_features(df: pd.DataFrame, feat: List[str]) -> List[str]:
    present = [c for c in feat if c in df.columns]
    missing = [c for c in feat if c not in df.columns]
    if missing:
        try:
            msg = "[WARN] dropped_missing_features: " + ", ".join(missing[:10])
            if len(missing) > 10:
                msg += ", ..."
            log(msg)
        except Exception:
            pass
    return present

def features_list(columns) -> list:
    base = [c for c in columns if re.match(r"(samai|g_num|avg)_lag[1-3]$", c)]
    base += [c for c in columns if c.endswith("_roll3") or c.endswith("_roll7")]
    # 実際に存在する特徴量に修正
    base += [
        "machine_prior_hit", "gnum_prior_mean", "days_since_last_2000", "weekday",
        "gnum_mom1", "avg_mom1", "samai_mom1", 
    ]
    # series集約系：実際の列名に修正
    base += [
        "hits",                # series_prev_hits の代替（当日除外は後段で対応）
        "gnum_p90",           # series_gnum_p90_d1 → 実名
        "avg_p90",            # series_avg_p90_d1  → 実名
        "samai_p90",          # series_samai_p90_d1→ 実名
        "hit_rate2000",       # series_hit_rate2000_d1 → 実名
        "spike_g",            # series_spike_g_d1 → 実名
        "hall_samai_p90",     # hall_samai_p90_d1 → 実名
        "hall_samai_med",     # hall_samai_med_d1 → 実名
        "hall_hit_rate2000",  # hall_hit_rate2000_d1 → 実名
        "hr_nonsp_ma14",      # hr_nonsp_ma14_d1 → 実名
        "hr_nonsp_wk_ma",     # hr_nonsp_wk_ma_d1 → 実名
    ]
    base += [c for c in columns if c.startswith("hall_")]
    base += [c for c in columns if c.startswith("ds_ds_prob")]
    base += [c for c in columns if c.startswith("wave_z_")]
    base += [c for c in columns if c.startswith("wave_r2_")]
    base += [c for c in columns if c.startswith("wave_reg_")]
    base += [c for c in columns if c.startswith("neigh_")]
    base += [
        "samai_lag1_pct_in_series_d", "g_num_lag1_pct_in_series_d", "avg_lag1_pct_in_series_d",
        "machine_ctr_beta_d1",
    ]
    base += ["is_special", "is_weekend", "dow_sin", "dow_cos", "dom_sin", "dom_cos"]
    
    # nonsp_power関連
    base += [c for c in ['samai_lag1_dev_serieswk_d1', 'gnum_ma3_over_ma7_d1', 'avg_ma3_over_ma7_d1'] if c in columns]
    
    # sub.pyの特徴量
    base += ADD_ALL
       # 重複除去して返す
    seen, out = set(), []
    for c in base:
        if c in columns and c not in seen:
            seen.add(c); out.append(c)
    return base

    # 旧名→実装名のマップ（あなたの指摘そのまま）
LEGACY_ALIAS = {
    "series_prev_hits":        "hits",
    "series_gnum_p90_d1":      "gnum_p90",
    "series_avg_p90_d1":       "avg_p90",
    "series_samai_p90_d1":     "samai_p90",
    "series_hit_rate2000_d1":  "hit_rate2000",
    "series_spike_g_d1":       "spike_g",
    "hall_samai_p90_d1":       "hall_samai_p90",
    "hall_samai_med_d1":       "hall_samai_med",
    "hall_hit_rate2000_d1":    "hall_hit_rate2000",
    "hr_nonsp_ma14_d1":        "hr_nonsp_ma14",
    "hr_nonsp_wk_ma_d1":       "hr_nonsp_wk_ma",
}

def apply_legacy_alias(names: list) -> list:
    return [LEGACY_ALIAS.get(n, n) for n in names]

def get_feature_groups(df, ADD_ALL):
    """DataFrameに基づいて特徴量グループを定義する関数"""
    
    feature_groups = {
        # 結果出力用:
        # - test_topk_k1.csv: K=1での選択台リスト（期待値最高の台）
        # - test_pk_k1.csv: 日別精度結果
        # - val_pk_summary.csv: 検証用P@K要約
        'roll': [c for c in df.columns if c.endswith('_roll3') or c.endswith('_roll7')],
        'series_agg': [c for c in df.columns if c.startswith('series_') and not c.startswith('series_month') and not c.startswith('series_week')],
        'monthweek': [c for c in df.columns if 'month' in c or 'week' in c],
        'hall': [c for c in df.columns if c.startswith('hall_')],
        'wave': [c for c in df.columns if c.startswith('wave_')],
        'neighbor': [c for c in df.columns if c.startswith('neigh_')],
        'power': [c for c in df.columns if c in ['neg_streak_d1', 'drawdown_log1p_d1', 'samai_lag1_dev_serieswk_d1']],
        'sub_advanced': ADD_ALL,
        'percentile': [c for c in df.columns if '_pct_in_series_d' in c],
        'machine_ctr': [c for c in df.columns if 'machine_ctr' in c],
        'day_state': [c for c in df.columns if c.startswith('ds_')],
    }
    
    return feature_groups
        
    # サンプルデータの出力（各シリーズから最新5件）
    sample_rows = []
    for series, group in df.groupby('series'):
        sample = group.sort_values('date').head(7)
        sample_rows.append(sample)
    
    if sample_rows:
        sample_df = pd.concat(sample_rows, ignore_index=True)
        sample_path = os.path.join(output_dir, f"feature_sample_{filename}")
        sample_df.to_csv(sample_path, index=False)
        log(f"[EXPORT] Feature sample exported to: {sample_path}")
        
def _parse_series_date_map(s: str) -> dict:
    mp = {}
    if not s:
        return mp
    for part in re.split(r"[;,]", s):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        try:
            mp[k] = pd.to_datetime(v.strip()).normalize()
        except Exception:
            pass
    return mp

def enforce_series_start_dates(df: pd.DataFrame, date_map: dict) -> pd.DataFrame:
    if not date_map:
        return df
    df = df.copy()
    df["date_norm"] = pd.to_datetime(df["date"]).dt.normalize()
    keep = []
    for s, g in df.groupby("series", sort=False):
        start = date_map.get(str(s), None)
        if start is None:
            keep.append(g)
        else:
            keep.append(g[g["date_norm"] >= start])
    out = pd.concat(keep, ignore_index=True)
    out = out.drop(columns=["date_norm"])
    return out

def drop_na_lagk_for_training(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    ks = max(1, int(k))
    req = [f"samai_lag{i}" for i in range(1, ks+1)] + [f"g_num_lag{i}" for i in range(1, ks+1)] + [f"avg_lag{i}" for i in range(1, ks+1)]
    cols = [c for c in req if c in df.columns]
    return df.dropna(subset=cols).copy()

def split_train_valid(df: pd.DataFrame, cutoff_str: str, valid_days: int):
    cutoff = pd.Timestamp(cutoff_str)
    all_tr = df[df["date"] <= cutoff].copy()
    uniq = np.sort(all_tr["date"].unique())
    if len(uniq) < valid_days + 5:
        raise ValueError("Not enough days to split: increase valid_days or extend training window.")
    valid_start = uniq[-valid_days]
    trn = all_tr[all_tr["date"] < valid_start].copy()
    val = all_tr[all_tr["date"] >= valid_start].copy()
    vstart = pd.Timestamp(valid_start).strftime("%Y-%m-%d")
    vend = pd.Timestamp(all_tr["date"].max()).strftime("%Y-%m-%d")
    return trn, val, vstart, vend

def make_feature_scaler(method: str = "standard"):
    if method == "none":
        class _Identity:
            def fit(self, X): return self
            def transform(self, X): return X
        return _Identity()
    if method == "minmax":
        return MinMaxScaler()
    return StandardScaler()

def transform_safe(scaler, X):  
    X2 = np.asarray(X, dtype=float)
    X2 = np.where(np.isnan(X2), 0.0, X2)
    return scaler.transform(X2)

def scale_pos_weight(y: np.ndarray) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return max(1.0, float(neg) / max(1, pos))

def train_hgb(Xtr: np.ndarray, ytr: np.ndarray, trn_dates: pd.Series, emphasize_nonsp: bool = False) -> HistGradientBoostingClassifier:
    cw = scale_pos_weight(ytr)
    if emphasize_nonsp:
        is_sp = pd.to_datetime(trn_dates).dt.day.astype(int).isin(list(SPECIAL_DAYS)).values
        w_ns = np.where(is_sp, 1.0, 1.2) 
    else:
        w_ns = 1.0
    sw = np.where(ytr == 1, cw, 1.0) * w_ns
    clf = HistGradientBoostingClassifier(
        learning_rate=0.03,
        l2_regularization=0.1,
        max_iter=350,
        validation_fraction=None,
        early_stopping=False,
        random_state=42,
    )
    clf.fit(Xtr, ytr, sample_weight=sw)
    return clf

def calculate_comprehensive_metrics(df: pd.DataFrame, score_col: str = "score") -> dict:
    """包括的なK=1メトリクス計算"""
    daily_results = []
    
    for date, group in df.groupby("date"):
        # K=1で最高スコアの台を選択
        top1 = group.sort_values(score_col, ascending=False).iloc[0]
        
        is_special = pd.to_datetime(date).day in SPECIAL_DAYS
        
        daily_results.append({
            "date": date,
            "hit": int(top1["target"]),
            "samai": float(top1["samai"]) if "samai" in top1 else 0.0,
            "is_special": is_special,
            "score": float(top1[score_col])
        })
    
    results_df = pd.DataFrame(daily_results)
    
    # 基本勝率
    win_rate_all = results_df["hit"].mean()
    win_rate_special = results_df[results_df["is_special"]]["hit"].mean() if results_df["is_special"].any() else 0.0
    win_rate_normal = results_df[~results_df["is_special"]]["hit"].mean() if (~results_df["is_special"]).any() else 0.0
    
    # 差枚関連
    cumulative_samai = results_df["samai"].sum()
    avg_daily_samai = results_df["samai"].mean()
    positive_days = (results_df["samai"] > 0).sum()
    
    # 投資効率（1000円/日投資想定）
    total_investment = len(results_df) * 1000
    roi = (cumulative_samai * 4) / total_investment * 100 if total_investment > 0 else 0.0
    
    # 安定性指標
    samai_std = results_df["samai"].std()
    win_consistency = positive_days / len(results_df) if len(results_df) > 0 else 0.0
    
    return {
        "win_rate_all": win_rate_all,
        "win_rate_special": win_rate_special,
        "win_rate_normal": win_rate_normal,
        "cumulative_samai": cumulative_samai,
        "avg_daily_samai": avg_daily_samai,
        "roi_percent": roi,
        "positive_days_ratio": win_consistency,
        "samai_volatility": samai_std,
        "total_days": len(results_df),
        "avg_confidence": results_df["score"].mean()
    }

def advanced_objective_function(trial, X_train, y_train, X_val, y_val, val_df, train_dates):
    """高度な最適化目的関数"""
    
    # 拡張されたハイパーパラメーター空間
    params = {
        # 基本パラメーター
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0001, 5.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 50, 1000, step=25),
        
        # 木構造パラメーター
        "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 7, 10, 15, 20]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [None, 15, 31, 63, 127, 255, 511]),
        
        # ビニング・サンプリング
        "max_bins": trial.suggest_categorical("max_bins", [63, 127, 255]),
        
        # 正則化強化
        "monotonic_cst": None,  # 単調制約は複雑すぎるためNone
    }
    
    # 動的重み調整
    base_weight = scale_pos_weight(y_train)
    nonsp_emphasis = trial.suggest_float("nonsp_emphasis", 1.0, 2.0)
    special_emphasis = trial.suggest_float("special_emphasis", 0.8, 1.5)
    
    # 高度な重み計算
    is_sp = pd.to_datetime(train_dates).dt.day.astype(int).isin(list(SPECIAL_DAYS)).values
    dynamic_weights = np.where(is_sp, special_emphasis, nonsp_emphasis)
    sample_weights = np.where(y_train == 1, base_weight, 1.0) * dynamic_weights
    
    try:
        # モデル訓練
        model = HistGradientBoostingClassifier(
            random_state=42,
            validation_fraction=None,
            early_stopping=False,
            **params
        )
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # 予測と校正
        raw_pred = model.predict_proba(X_val)[:, 1]
        
        # 高度な校正
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_pred, y_val)
        calibrated_pred = iso.transform(raw_pred)
        
        # 検証データフレームにスコア追加
        val_df_copy = val_df.copy()
        val_df_copy["score"] = calibrated_pred
        
        # 包括的メトリクス計算
        metrics = calculate_comprehensive_metrics(val_df_copy)
        
        # 多目的最適化関数（重み調整可能）
        primary_weight = 0.50    # 全体勝率
        special_weight = 0.20    # 特別日勝率
        roi_weight = 0.15        # ROI
        consistency_weight = 0.10 # 安定性
        volatility_weight = 0.05  # ボラティリティ（低い方が良い）
        
        # 正規化されたROI（-50%〜+50%を0〜1にマップ）
        normalized_roi = max(0.0, min(1.0, (metrics["roi_percent"] + 50) / 100))
        
        # 正規化されたボラティリティ（低い方が良いので反転）
        max_volatility = 3000  # 想定最大ボラティリティ
        normalized_volatility = max(0.0, 1.0 - (metrics["samai_volatility"] / max_volatility))
        
        # 複合目的関数
        objective_score = (
            primary_weight * metrics["win_rate_all"] +
            special_weight * metrics["win_rate_special"] +
            roi_weight * normalized_roi +
            consistency_weight * metrics["positive_days_ratio"] +
            volatility_weight * normalized_volatility
        )
        
        # 制約条件（実用性を確保）
        if metrics["win_rate_all"] < 0.25:  # 最低勝率制約
            objective_score *= 0.5
        
        if metrics["avg_daily_samai"] < -500:  # 極端な損失を避ける
            objective_score *= 0.3
        
        # メトリクスを記録
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        
        trial.set_user_attr("composite_score", objective_score)
        
        return objective_score
        
    except Exception as e:
        log(f"[TRIAL ERROR] {str(e)[:100]}")
        return 0.0

def run_advanced_hyperparameter_tuning(feat, trn, val, feat_scaler, n_trials=50):
    """高度なベイズ最適化実行"""
    log(f"[TUNING] Starting advanced Bayesian optimization with {n_trials} trials")
    log(f"[TUNING] Target: Maximize K=1 win rate and cumulative samai")
    
    # データ準備
    X_train = trn.reindex(columns=feat, fill_value=0.0).values
    y_train = trn["target"].values
    X_val = val.reindex(columns=feat, fill_value=0.0).values
    y_val = val["target"].values
    
    # スケーリング
    feat_scaler.fit(np.nan_to_num(X_train, nan=0.0))
    X_train_scaled = transform_safe(feat_scaler, X_train)
    X_val_scaled = transform_safe(feat_scaler, X_val)
    
    # Optuna study作成（高度な設定）
    sampler = TPESampler(
        seed=42,
        n_startup_trials=10,  # ランダム探索期間
        n_ei_candidates=24,   # 期待改善候補数
        multivariate=True,    # 多変数相関考慮
        warn_independent_sampling=False
    )
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="pachislot_advanced_optimization",
        load_if_exists=False
    )
    
    # 最適化実行
    log("[TUNING] Starting optimization trials...")
    study.optimize(
        lambda trial: advanced_objective_function(
            trial, X_train_scaled, y_train, X_val_scaled, y_val, val, trn["date"]
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[lambda study, trial: log(f"Trial {trial.number}: {trial.value:.4f}")]
    )
    
    # 詳細結果表示
    best_trial = study.best_trial
    log(f"\n{'='*60}")
    log("ベイズ最適化結果")
    log(f"{'='*60}")
    log(f"Best objective score: {best_trial.value:.4f}")
    
    log(f"\n最適パラメーター:")
    for key, value in best_trial.params.items():
        log(f"  {key}: {value}")
    
    log(f"\n性能メトリクス:")
    log(f"  全体勝率: {best_trial.user_attrs['win_rate_all']:.4f} ({best_trial.user_attrs['win_rate_all']*100:.1f}%)")
    log(f"  特別日勝率: {best_trial.user_attrs['win_rate_special']:.4f} ({best_trial.user_attrs['win_rate_special']*100:.1f}%)")
    log(f"  通常日勝率: {best_trial.user_attrs['win_rate_normal']:.4f} ({best_trial.user_attrs['win_rate_normal']*100:.1f}%)")
    log(f"  累積差枚: {best_trial.user_attrs['cumulative_samai']:+.0f}枚")
    log(f"  平均日次差枚: {best_trial.user_attrs['avg_daily_samai']:+.1f}枚")
    log(f"  ROI: {best_trial.user_attrs['roi_percent']:+.1f}%")
    log(f"  プラス日率: {best_trial.user_attrs['positive_days_ratio']:.3f}")
    log(f"  ボラティリティ: {best_trial.user_attrs['samai_volatility']:.1f}")
    
    # 上位試行の分析
    log(f"\n上位5試行の比較:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
    
    for i, trial in enumerate(top_trials, 1):
        if trial.value:
            log(f"  {i}位: Score={trial.value:.4f}, 勝率={trial.user_attrs.get('win_rate_all', 0)*100:.1f}%, ROI={trial.user_attrs.get('roi_percent', 0):+.1f}%")
    
    return study, best_trial.params

def train_hgb_optimized(Xtr: np.ndarray, ytr: np.ndarray, trn_dates: pd.Series, 
                       best_params: dict, emphasize_nonsp: bool = False) -> HistGradientBoostingClassifier:
    """最適化されたパラメーターでモデル訓練"""
    
    # 重み計算
    base_weight = scale_pos_weight(ytr)
    
    if emphasize_nonsp:
        is_sp = pd.to_datetime(trn_dates).dt.day.astype(int).isin(list(SPECIAL_DAYS)).values
        nonsp_emphasis = best_params.get("nonsp_emphasis", 1.2)
        special_emphasis = best_params.get("special_emphasis", 1.0)
        dynamic_weights = np.where(is_sp, special_emphasis, nonsp_emphasis)
    else:
        dynamic_weights = 1.0
    
    sample_weights = np.where(ytr == 1, base_weight, 1.0) * dynamic_weights
    
    # 最適化パラメーターをコピー（重み関連は除外）
    model_params = {k: v for k, v in best_params.items() 
                   if k not in ["nonsp_emphasis", "special_emphasis"]}
    
    clf = HistGradientBoostingClassifier(
        learning_rate=0.03,
        l2_regularization=0.1,
        max_iter=350,
        random_state=42,
        validation_fraction=0.1,
        early_stopping=True,
        n_iter_no_change=10,
        **model_params
    )
    
    clf.fit(Xtr, ytr, sample_weight=sample_weights)
    return clf

# parse_args関数への追加（これを既存のparse_args関数に追加）
def add_tuning_arguments(pt):
    """チューニング用引数を追加"""
    pt.add_argument("--tune_hyperparams", action="store_true", help="Run Bayesian hyperparameter optimization")
    pt.add_argument("--n_trials", type=int, default=50, help="Number of optimization trials")
    pt.add_argument("--save_study", action="store_true", help="Save optimization study results")
    return pt

def pk_by_day(df: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("date"):
        g = g.sort_values(score_col, ascending=False)
        k_ = min(k, len(g))
        pick = g.head(k_)
        rows.append({
            "date": str(pd.Timestamp(d).date()),
            "picked": int(k_),
            "positives": int(pick["target"].sum()) if k_ > 0 else 0,
            "precision_at_k": float(pick["target"].mean()) if k_ > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("date")

def finalize_scoring_with_cumulative_analysis_fixed(perday_df, output_dir):
    """スコアリング結果の最終分析（エラー修正版）"""
    
    try:
        df = perday_df.copy()
        
        # samai列の検索
        samai_col = None
        possible_cols = ['samai', 'total_samai', 'samai_sum', 'picked_samai']
        
        for col in possible_cols:
            if col in df.columns:
                samai_col = col
                break
        
        # topkファイルから取得を試行
        if samai_col is None:
            topk_path = os.path.join(output_dir, "test_topk_k1.csv")
            if os.path.exists(topk_path):
                try:
                    topk_df = pd.read_csv(topk_path)
                    if 'samai' in topk_df.columns:
                        daily_samai = topk_df.groupby('date')['samai'].sum().reset_index()
                        df = df.merge(daily_samai, on='date', how='left')
                        samai_col = 'samai'
                except Exception:
                    pass
        
        # 最終手段：ダミー値
        if samai_col is None:
            print("[WARNING] samai data not found, using dummy values")
            df['samai'] = 0
            samai_col = 'samai'
        
        print(f"\n{'='*60}")
        print("TOPK=1 実運用シミュレーション")
        print(f"{'='*60}")
        
        # 日付でソート
        df = df.sort_values('date')
        
        # 累積差枚計算
        df['cumulative_samai'] = df[samai_col].cumsum()
        
        # 表示
        print(f"{'Date':<12} {'Daily':<10} {'累積':<12} {'的中率':<8}")
        print("-" * 50)
        
        for _, row in df.iterrows():
            date_str = str(row['date'])[-5:] if isinstance(row['date'], str) else str(row['date'])[:10]
            daily = f"{row[samai_col]:+.0f}枚"
            cumulative = f"{row['cumulative_samai']:+.0f}枚"
            precision = f"{row['precision_at_k']:.3f}"
            print(f"{date_str:<12} {daily:<10} {cumulative:<12} {precision:<8}")
        
        # 統計
        total_days = len(df)
        final_cumulative = df['cumulative_samai'].iloc[-1] if len(df) > 0 else 0
        avg_daily = final_cumulative / total_days if total_days > 0 else 0
        overall_precision = df['precision_at_k'].mean()
        positive_days = (df[samai_col] > 0).sum()
        
        print("-" * 50)
        print(f"運用期間: {total_days}日")
        print(f"最終累積差枚: {final_cumulative:+.0f}枚")
        print(f"平均日次差枚: {avg_daily:+.1f}枚")
        print(f"全体的中率: {overall_precision:.3f}")
        print(f"プラス日数: {positive_days}/{total_days}日 ({positive_days/total_days*100:.1f}%)")
        
        # 必ず2つの値を返す
        return df, df
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        # エラー時も2つの値を返す
        return perday_df, perday_df

def analyze_samai_distributions(output_dir: str):
    """差枚分布の包括的分析"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('差枚分布分析：データリーク検証', fontsize=16)
    
    try:
        # 1. 予測結果の差枚分布（topk_k1.csv）
        topk_path = os.path.join(output_dir, "test_topk_k1.csv")
        if os.path.exists(topk_path):
            topk_df = pd.read_csv(topk_path)
            if 'samai' in topk_df.columns:
                predicted_samai = topk_df['samai'].values
                
                # ヒストグラム
                axes[0, 0].hist(predicted_samai, bins=50, alpha=0.7, color='red', edgecolor='black')
                axes[0, 0].set_title('予測結果の差枚分布')
                axes[0, 0].set_xlabel('差枚')
                axes[0, 0].set_ylabel('頻度')
                axes[0, 0].axvline(x=2000, color='blue', linestyle='--', label='2000枚閾値')
                axes[0, 0].legend()
                
                # 統計情報
                print(f"\n予測結果の差枚統計:")
                print(f"  件数: {len(predicted_samai)}")
                print(f"  平均: {predicted_samai.mean():.1f}枚")
                print(f"  中央値: {np.median(predicted_samai):.1f}枚")
                print(f"  標準偏差: {predicted_samai.std():.1f}枚")
                print(f"  最小値: {predicted_samai.min():.1f}枚")
                print(f"  最大値: {predicted_samai.max():.1f}枚")
                print(f"  2000枚以上: {(predicted_samai >= 2000).sum()}件 ({(predicted_samai >= 2000).mean()*100:.1f}%)")
                print(f"  0-2000枚: {((predicted_samai >= 0) & (predicted_samai < 2000)).sum()}件")
                print(f"  0枚未満: {(predicted_samai < 0).sum()}件")
            else:
                axes[0, 0].text(0.5, 0.5, 'samaiデータなし', ha='center', va='center')
                predicted_samai = None
        else:
            axes[0, 0].text(0.5, 0.5, 'topk_k1.csvなし', ha='center', va='center')
            predicted_samai = None
        
        # 2. 通常営業データの差枚分布（全履歴データ）
        history_files = [
            "hokuto_plazahakata_all_days_v14.csv",
            "monkey_plazahakata_all_days_v14.csv", 
            "ghoul_plazahakata_all_days_v14.csv",
            "myjugglerV_plazahakata_all_days_v14.csv"
        ]
        
        all_historical_samai = []
        for file in history_files:
            if os.path.exists(file):
                try:
                    df = pd.read_csv(file)
                    if 'samai' in df.columns:
                        samai_data = pd.to_numeric(df['samai'], errors='coerce').dropna()
                        all_historical_samai.extend(samai_data.values)
                except Exception as e:
                    print(f"Warning: Could not read {file}: {e}")
        
        if all_historical_samai:
            historical_samai = np.array(all_historical_samai)
            
            # ヒストグラム
            axes[0, 1].hist(historical_samai, bins=100, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('通常営業の差枚分布（全履歴）')
            axes[0, 1].set_xlabel('差枚')
            axes[0, 1].set_ylabel('頻度')
            axes[0, 1].axvline(x=2000, color='blue', linestyle='--', label='2000枚閾値')
            axes[0, 1].legend()
            
            # 統計情報
            print(f"\n通常営業の差枚統計:")
            print(f"  件数: {len(historical_samai)}")
            print(f"  平均: {historical_samai.mean():.1f}枚")
            print(f"  中央値: {np.median(historical_samai):.1f}枚")
            print(f"  標準偏差: {historical_samai.std():.1f}枚")
            print(f"  最小値: {historical_samai.min():.1f}枚")
            print(f"  最大値: {historical_samai.max():.1f}枚")
            print(f"  2000枚以上: {(historical_samai >= 2000).sum()}件 ({(historical_samai >= 2000).mean()*100:.1f}%)")
            print(f"  0-2000枚: {((historical_samai >= 0) & (historical_samai < 2000)).sum()}件")
            print(f"  0枚未満: {(historical_samai < 0).sum()}件")
        else:
            axes[0, 1].text(0.5, 0.5, '履歴データなし', ha='center', va='center')
            historical_samai = None
        
        # 3. 重ね合わせ比較
        if predicted_samai is not None and historical_samai is not None:
            # 正規化されたヒストグラム
            axes[0, 2].hist(historical_samai, bins=50, alpha=0.5, color='green', 
                           label='通常営業', density=True, edgecolor='black')
            axes[0, 2].hist(predicted_samai, bins=20, alpha=0.7, color='red', 
                           label='予測結果', density=True, edgecolor='black')
            axes[0, 2].set_title('分布比較（正規化）')
            axes[0, 2].set_xlabel('差枚')
            axes[0, 2].set_ylabel('密度')
            axes[0, 2].axvline(x=2000, color='blue', linestyle='--', label='2000枚閾値')
            axes[0, 2].legend()
        
        # 4. 範囲別詳細分析
        ranges = [(-5000, 0), (0, 2000), (2000, 5000), (5000, 10000)]
        range_labels = ['マイナス', '0-2000', '2000-5000', '5000+']
        
        if predicted_samai is not None:
            pred_counts = []
            for (low, high) in ranges:
                count = ((predicted_samai >= low) & (predicted_samai < high)).sum()
                pred_counts.append(count)
            
            axes[1, 0].bar(range_labels, pred_counts, alpha=0.7, color='red')
            axes[1, 0].set_title('予測結果：範囲別件数')
            axes[1, 0].set_ylabel('件数')
            for i, v in enumerate(pred_counts):
                axes[1, 0].text(i, v + 0.1, str(v), ha='center')
        
        if historical_samai is not None:
            hist_counts = []
            for (low, high) in ranges:
                count = ((historical_samai >= low) & (historical_samai < high)).sum()
                hist_counts.append(count)
            
            axes[1, 1].bar(range_labels, hist_counts, alpha=0.7, color='green')
            axes[1, 1].set_title('通常営業：範囲別件数')
            axes[1, 1].set_ylabel('件数')
            for i, v in enumerate(hist_counts):
                axes[1, 1].text(i, v + max(hist_counts)*0.01, str(v), ha='center')
        
        # 5. KSテスト（分布の同一性検定）
        if predicted_samai is not None and historical_samai is not None:
            from scipy import stats
            
            # サンプルサイズを合わせる
            min_size = min(len(predicted_samai), 1000, len(historical_samai))
            pred_sample = np.random.choice(predicted_samai, min_size, replace=False)
            hist_sample = np.random.choice(historical_samai, min_size, replace=False)
            
            ks_stat, p_value = stats.ks_2samp(pred_sample, hist_sample)
            
            axes[1, 2].text(0.1, 0.8, f'Kolmogorov-Smirnov検定', fontsize=14, weight='bold')
            axes[1, 2].text(0.1, 0.6, f'KS統計量: {ks_stat:.4f}', fontsize=12)
            axes[1, 2].text(0.1, 0.4, f'p値: {p_value:.6f}', fontsize=12)
            
            if p_value < 0.001:
                result_text = "分布は有意に異なる\n（データリークの可能性）"
                color = 'red'
            elif p_value < 0.05:
                result_text = "分布に有意差あり\n（要注意）"
                color = 'orange'
            else:
                result_text = "分布に有意差なし\n（正常）"
                color = 'green'
            
            axes[1, 2].text(0.1, 0.2, result_text, fontsize=12, color=color, weight='bold')
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].axis('off')
            
            print(f"\n統計的検定結果:")
            print(f"  KS統計量: {ks_stat:.4f}")
            print(f"  p値: {p_value:.6f}")
            print(f"  結論: {result_text}")
        
        plt.tight_layout()
        
        # 保存
        plot_path = os.path.join(output_dir, "samai_distribution_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n分布分析グラフを保存: {plot_path}")
        
        plt.show()
        
        # 要約レポート
        print(f"\n{'='*60}")
        print("データリーク分析要約")
        print(f"{'='*60}")
        
        if predicted_samai is not None:
            zero_to_2000 = ((predicted_samai >= 0) & (predicted_samai < 2000)).sum()
            total_pred = len(predicted_samai)
            
            if zero_to_2000 == 0:
                print("⚠️  警告: 予測結果に0-2000枚の範囲が存在しません")
                print("   これはデータリークの強い兆候です")
            elif zero_to_2000 < total_pred * 0.1:
                print("⚠️  注意: 0-2000枚の範囲が異常に少ないです")
                print("   データリークの可能性があります")
            else:
                print("✓  0-2000枚の範囲に十分なデータが存在します")
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()

# 使用方法
def run_distribution_analysis():
    """
    スコアリング実行後に以下を実行:
    analyze_samai_distributions("score_output_0817-0906")
    """
    pass


def handle_production_data(df):
    """
    本番稼働用データ（samai, avg, g_numが空白）の処理
    """
    production_mode = False
    
    # 本番モードの検出
    if 'samai' in df.columns and df['samai'].isna().all():
        production_mode = True
        log("[INFO] Production mode detected: samai column is empty")
    
    if 'avg' in df.columns and df['avg'].isna().all():
        production_mode = True
        log("[INFO] Production mode detected: avg column is empty")
        
    if 'g_num' in df.columns and df['g_num'].isna().all():
        production_mode = True
        log("[INFO] Production mode detected: g_num column is empty")
    
    if production_mode:
        log("[PRODUCTION] Running in production mode - no ground truth available")
        
        # ターゲット列を0で埋める（予測専用）
        if 'target' not in df.columns:
            df['target'] = 0
        
        # 必要なカラムをダミー値で埋める
        if 'samai' in df.columns:
            df['samai'] = 0.0
        if 'avg' in df.columns:
            df['avg'] = df.get('num', 0).astype(float)  # numを代替値として使用
        if 'g_num' in df.columns:
            df['g_num'] = df.get('num', 0).astype(float)
    
    return df, production_mode
def export_production_picks(df, predictions, output_dir, filename_base="production_picks"):
    """
    本番稼働用の選択結果をCSVで出力
    """
    
    # 予測スコアを追加
    df = df.copy()
    df['prediction_score'] = predictions
    
    # 日別にTOPK=1を選択
    daily_picks = []
    all_predictions = []
    
    for date, group in df.groupby('date'):
        # スコア順にソート
        group_sorted = group.sort_values('prediction_score', ascending=False)
        
        # 全予測データを保存（features.csv用）
        for i, (_, row) in enumerate(group_sorted.iterrows()):
            all_predictions.append({
                'date': date,
                'series': row['series'],
                'num': row['num'],
                'prediction_score': row['prediction_score'],
                'rank': i + 1
            })
        
        # TOPK=1を選択
        top1 = group_sorted.iloc[0]
        daily_picks.append({
            'date': date,
            'series': top1['series'],
            'num': top1['num'],
            'machine_id': top1.get('machine_id', f"{top1['series']}_{top1['num']}"),
            'prediction_score': top1['prediction_score'],
            'recommended_investment': 1000  # デフォルト推奨投資額
        })
    
    # TOPK=1選択結果を保存
    picks_df = pd.DataFrame(daily_picks)
    picks_path = os.path.join(output_dir, f"{filename_base}_topk1.csv")
    picks_df.to_csv(picks_path, index=False)
    log(f"[EXPORT] Production TOPK=1 picks saved: {picks_path}")
    
    # 全予測結果を保存（features.csv相当）
    all_predictions_df = pd.DataFrame(all_predictions)
    features_path = os.path.join(output_dir, f"{filename_base}_all_predictions.csv")
    all_predictions_df.to_csv(features_path, index=False)
    log(f"[EXPORT] All predictions saved: {features_path}")
    
    # サンプル表示
    print(f"\n{'='*50}")
    print("本番稼働用選択結果 (TOPK=1)")
    print(f"{'='*50}")
    print(f"{'Date':<12} {'Series':<10} {'台番':<6} {'Score':<8}")
    print("-" * 40)
    
    for _, row in picks_df.iterrows():
        date_str = pd.to_datetime(row['date']).strftime('%m-%d')
        print(f"{date_str:<12} {row['series']:<10} {row['num']:<6} {row['prediction_score']:.4f}")
    
    return picks_df, all_predictions_df
# ====================
# 特徴量確認用CSV出力関数
# ====================
def export_feature_verification_csv(df: pd.DataFrame, output_dir: str, filename: str = "feature_verification.csv"):
    """特徴量の動作確認用CSV出力（修正版）"""
    ensure_dir(output_dir)
    
    # 基本情報
    log("[EXPORT] Creating feature verification CSV...")
    
    # 特徴量グループごとに列を整理
    feature_groups = {
        'basic': ['date', 'series', 'machine_id', 'target', 'samai', 'g_num', 'avg', 'num'],
        'calendar': [c for c in df.columns if c in ['is_special', 'is_weekend', 'weekday', 'dow_sin', 'dow_cos', 'dom_sin', 'dom_cos']],
        'lag': [c for c in df.columns if re.match(r'.+_lag[1-3]', c)],
        'roll': [c for c in df.columns if c.endswith('_roll3') or c.endswith('_roll7')],
        'series_agg': [c for c in df.columns if c.startswith('series_') and not c.startswith('series_month') and not c.startswith('series_week')],
        'monthweek': [c for c in df.columns if 'month' in c or 'week' in c],
        'hall': [c for c in df.columns if c.startswith('hall_')],
        'wave': [c for c in df.columns if c.startswith('wave_')],
        'neighbor': [c for c in df.columns if c.startswith('neigh_')],
        'power': [c for c in df.columns if c in ['neg_streak_d1', 'drawdown_log1p_d1', 'samai_lag1_dev_serieswk_d1']],
        'sub_advanced': ADD_ALL,
        'percentile': [c for c in df.columns if '_pct_in_series_d' in c],
        'machine_ctr': [c for c in df.columns if 'machine_ctr' in c],
        'day_state': [c for c in df.columns if c.startswith('ds_')],
    }
    
    # 統計情報を計算
    stats_rows = []
    
    for group_name, cols in feature_groups.items():
        present_cols = [c for c in cols if c in df.columns]
        if not present_cols:
            continue
            
        for col in present_cols:
            if col not in df.columns:
                continue
                
            series_data = df[col]
            stats = {
                'feature_group': group_name,
                'feature_name': col,
                'dtype': str(series_data.dtype),
                'total_count': len(series_data),
                'null_count': int(series_data.isna().sum()),
                'null_percentage': float(series_data.isna().sum() / len(series_data) * 100),
                'unique_count': int(series_data.nunique()),
                'min_value': float(series_data.min()) if pd.api.types.is_numeric_dtype(series_data) else None,
                'max_value': float(series_data.max()) if pd.api.types.is_numeric_dtype(series_data) else None,
                'mean_value': float(series_data.mean()) if pd.api.types.is_numeric_dtype(series_data) else None,
                'std_value': float(series_data.std()) if pd.api.types.is_numeric_dtype(series_data) else None,
            }
            stats_rows.append(stats)
    
    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(output_dir, f"feature_stats_{filename}")
    stats_df.to_csv(stats_path, index=False)
    
    # サンプルデータの出力（修正版）
    sample_rows = []
    
    # 各シリーズから先頭7日分を取得
    for series, group in df.groupby('series'):
        # 日付順でソート
        group_sorted = group.sort_values('date')
        
        # ユニークな日付を取得
        unique_dates = group_sorted['date'].unique()
        
        # 先頭から7日分を選択
        # target_dates = unique_dates[:7] if len(unique_dates) >= 7 else unique_dates
        target_dates = unique_dates[-7:] if len(unique_dates) >= 7 else unique_dates
        
        # 各日付から代表的なサンプルを取得
        for date in target_dates:
            date_group = group_sorted[group_sorted['date'] == date]
            
            # 各日付から最大3台のサンプルを取得
            sample_count = min(3, len(date_group))
            sample = date_group.head(sample_count)
            sample_rows.append(sample)
    
    if sample_rows:
        sample_df = pd.concat(sample_rows, ignore_index=True)
        sample_path = os.path.join(output_dir, f"feature_sample_{filename}")
        sample_df.to_csv(sample_path, index=False)
        log(f"[EXPORT] Feature sample exported to: {sample_path}")
        log(f"[EXPORT] Sample contains {len(sample_df)} rows from {sample_df['date'].nunique()} days")
    
    # 詳細サンプル：各シリーズの先頭1週間を完全出力
    detailed_sample_rows = []
    for series, group in df.groupby('series'):
        group_sorted = group.sort_values('date')
        unique_dates = group_sorted['date'].unique()
        
        # 先頭1週間の全データ
        week_dates = unique_dates[:7] if len(unique_dates) >= 7 else unique_dates
        week_data = group_sorted[group_sorted['date'].isin(week_dates)]
        detailed_sample_rows.append(week_data)
    
    if detailed_sample_rows:
        detailed_sample_df = pd.concat(detailed_sample_rows, ignore_index=True)
        detailed_path = os.path.join(output_dir, f"feature_sample_week1_{filename}")
        detailed_sample_df.to_csv(detailed_path, index=False)
        log(f"[EXPORT] Detailed week1 sample exported to: {detailed_path}")
        log(f"[EXPORT] Week1 sample contains {len(detailed_sample_df)} rows from {detailed_sample_df['date'].nunique()} days")
    
    # 特徴量生成成功/失敗レポート
    feature_report = {
        'total_features': len([c for c in df.columns if c not in feature_groups['basic']]),
        'calendar_features': len(feature_groups['calendar']),
        'lag_features': len([c for c in df.columns if re.match(r'.+_lag[1-3]', c)]),
        'sub_advanced_features': len([c for c in ADD_ALL if c in df.columns]),
        'missing_sub_features': [c for c in ADD_ALL if c not in df.columns],
        'wave_features': len([c for c in df.columns if c.startswith('wave_')]),
        'neighbor_features': len([c for c in df.columns if c.startswith('neigh_')]),
        'hall_features': len([c for c in df.columns if c.startswith('hall_')]),
        'pctins_features': 0,  # 削除済み
    }
    
    report_path = os.path.join(output_dir, f"feature_report_{filename.replace('.csv', '.txt')}")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 特徴量生成レポート ===\n")
        f.write(f"総特徴量数: {feature_report['total_features']}\n")
        f.write(f"カレンダー特徴量: {feature_report['calendar_features']}\n")
        f.write(f"ラグ特徴量: {feature_report['lag_features']}\n")
        f.write(f"高度特徴量(sub.py): {feature_report['sub_advanced_features']}\n")
        f.write(f"波動・トレンド特徴量: {feature_report['wave_features']}\n")
        f.write(f"近隣特徴量: {feature_report['neighbor_features']}\n")
        f.write(f"ホール状態特徴量: {feature_report['hall_features']}\n")
        f.write(f"pctins特徴量: {feature_report['pctins_features']}\n")
        
        if feature_report['missing_sub_features']:
            f.write(f"\n未生成のsub.py特徴量: {feature_report['missing_sub_features']}\n")
        
        f.write(f"\n=== サンプルデータ情報 ===\n")
        f.write(f"サンプル期間: {df['date'].min()} ～ {df['date'].max()}\n")
        f.write(f"総日数: {df['date'].nunique()}日\n")
        f.write(f"シリーズ数: {df['series'].nunique()}\n")
        f.write(f"機種別データ:\n")
        for series, group in df.groupby('series'):
            f.write(f"  {series}: {len(group)}行, {group['date'].nunique()}日\n")
    
    log(f"[EXPORT] Feature statistics exported to: {stats_path}")
    log(f"[EXPORT] Feature report exported to: {report_path}")
    
    return stats_path, sample_path if 'sample_path' in locals() else None, report_path

# ====================
# メイン実行部分
# ====================
def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--series_start_dates", default="", help="ghoul=2025-04-23;hokuto=2025-01-01;monkey=2025-01-01")
    pt.add_argument("--lag_required", type=int, default=3)
    pt.add_argument("--feat_scale", choices=["none","standard","minmax"], default="standard")
    pt.add_argument("--train_files", required=True)
    pt.add_argument("--cutoff", default="2025-08-16")
    pt.add_argument("--valid_days", type=int, default=14)
    pt.add_argument("--k_list", default="7,8,10")
    pt.add_argument("--out_dir", default="out_series_dual")
    pt.add_argument("--calib_mode", choices=["global", "split"], default="split")
    pt.add_argument("--dual_models", action="store_true")
    pt.add_argument("--rank_report", type=str, default="", help="CSV with columns [feature, rank_score] to drop low-importance features")
    pt.add_argument("--rank_threshold", type=float, default=0.10)
    pt.add_argument("--drop_features", type=str, default="", help="Comma-separated feature names to drop additionally")
    pt.add_argument("--export_features", action="store_true", help="Export feature verification CSV")
    pt.add_argument("--tune_hyperparams", action="store_true", help="Run Bayesian hyperparameter optimization")
    pt.add_argument("--n_trials", type=int, default=50, help="Number of optimization trials")

    ps = sub.add_parser("score")
    ps.add_argument("--feat_scale", choices=["none","standard","minmax"], default="standard")
    ps.add_argument("--model_dir", required=True)
    ps.add_argument("--day_gate", action="store_true")
    ps.add_argument("--gate_cov_min", type=float, default=0.35)
    ps.add_argument("--gate_cov_max", type=float, default=0.60)
    ps.add_argument("--test_files", required=True)
    ps.add_argument("--history_files", required=True)
    ps.add_argument("--out_dir", default="out_series_score_v4x")
    ps.add_argument("--auto_k", action="store_true")
    ps.add_argument("--autok_mode", choices=["p", "p_and_lift"], default="p_and_lift")
    ps.add_argument("--alloc_by_series", choices=["greedy", "none"], default="greedy")
    ps.add_argument("--k_min", type=int, default=3)
    ps.add_argument("--k_max", type=int, default=8)
    ps.add_argument("--tau_special", type=float, default=0.60)
    ps.add_argument("--tau_normal", type=float, default=0.45)
    ps.add_argument("--lift_tau_special", type=float, default=1.20)
    ps.add_argument("--lift_tau_normal", type=float, default=1.50)
    ps.add_argument("--fallback_policy", choices=["skip", "min_entry"], default="skip")
    ps.add_argument("--export_features", action="store_true", help="Export feature verification CSV")
    ps.add_argument("--production_mode", action="store_true", help="Enable production mode with date gate")
    ps.add_argument("--target_date", type=str, default=None, help="Target date for production (YYYY-MM-DD)")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    if args.mode == "train":
        log("[START] TRAIN")
        try:
            # データ読み込み
            train_files = [s.strip() for s in args.train_files.split(",") if s.strip()]
            df = read_and_stack(train_files, "train")
            
            # 基本特徴量生成
            log("[FEAT] Adding basic features...")
            df = add_calendar_features(df)
            df = add_exante_features(df)
            df = add_series_monthweek_features(df)
            _debug_check_monthweek(df, 'after_exante')
            
            # 高度特徴量生成
            log("[FEAT] Adding advanced features...")
            df = add_series_aggregates(df)
            df = add_nonsp_power_features(df)
            df = add_series_weekday_resid(df)
            df = add_num_end_digit_features(df)
            df = add_neighbor_features(df)
            df = add_wave_trend_features(df)
            df = add_series_percentile_features(df)
            df = add_bayesian_machine_ctr(df)
            df = add_renzoku_hekomi_features(df)
            # 日付状態特徴量
            df, scaler, kmeans = add_day_state_features(df, fit=True, n_clusters=5)
            
            # フィルタリングと分割
            start_map = _parse_series_date_map(args.series_start_dates)
            df = enforce_series_start_dates(df, start_map)
            df = drop_na_lagk_for_training(df, k=args.lag_required)
            
            # 分割前に特徴量エイリアス作成
            df = add_compat_feature_aliases(df)
            
            # 訓練・検証分割
            trn, val, vstart, vend = split_train_valid(df, args.cutoff, args.valid_days)
            
            # sub.pyの高度特徴量を追加（分割後）
            log("[FEAT] Adding sub.py advanced features...")
            trn, val, sub_features = add_all_new_features(trn, val)
            
            # 特徴量確認用CSV出力（オプション）
            if getattr(args, 'export_features', False):
                log("[EXPORT] Exporting feature verification files...")
                export_feature_verification_csv(trn, args.out_dir, "train_features.csv")
                export_feature_verification_csv(val, args.out_dir, "val_features.csv")
            
            # 特徴量選択
            feat = features_list(trn.columns)
            feat = apply_legacy_alias(feat)  # ★ 追加
            feat = filter_present_features(trn, feat)
            drop_set = _load_drop_from_rank_report(args.rank_report, args.rank_threshold)
            feat = _apply_feature_drop(feat, drop_set, args.drop_features)
            
            log(f"[FEAT] Selected {len(feat)} features")
            
            # スケーリング準備
            feat_scaler = make_feature_scaler(args.feat_scale)
            trn = trn.sort_values(["date", "series", "machine_id"]).reset_index(drop=True)
            val = val.sort_values(["date", "series", "machine_id"]).reset_index(drop=True)
            
            bundles = {}
            
            if args.tune_hyperparams:
                log("[TUNING] Running advanced Bayesian optimization...")
                study, best_params = run_advanced_hyperparameter_tuning(feat, trn, val, feat_scaler, args.n_trials)
                # 結果保存
                with open(os.path.join(args.out_dir, "best_hyperparams.json"), "w") as f:
                    json.dump(best_params, f, indent=2)
            else:
                best_params = {
                    "learning_rate": 0.03,
                    "l2_regularization": 0.1, 
                    "max_iter": 350,
                    "max_depth": None,
                    "min_samples_leaf": 1
                }

            bundles = {}

            if args.dual_models:
                # デュアルモデル（特別日・非特別日別）
                for flag, name in [(0, "nonsp"), (1, "sp")]:
                    tr_sub = trn[trn["is_special"] == flag].copy()
                    va_sub = val[val["is_special"] == flag].copy()
                    
                    if len(tr_sub) == 0 or len(va_sub) == 0:
                        # フォールバック：subsetが空なら全体で
                        Xtr_raw, ytr = trn.reindex(columns=feat, fill_value=0.0).values, trn["target"].values
                        Xva_raw, yva = val.reindex(columns=feat, fill_value=0.0).values, val["target"].values
                        tr_dates = trn["date"]
                    else:
                        Xtr_raw, ytr = tr_sub.reindex(columns=feat, fill_value=0.0).values, tr_sub['target'].values
                        Xva_raw, yva = va_sub.reindex(columns=feat, fill_value=0.0).values, va_sub['target'].values
                        tr_dates = tr_sub["date"]
                    
                    feat_scaler.fit(np.nan_to_num(Xtr_raw, nan=0.0))
                    Xtr, Xva = transform_safe(feat_scaler, Xtr_raw), transform_safe(feat_scaler, Xva_raw)
                    
                    model = train_hgb(Xtr, ytr, tr_dates, emphasize_nonsp=(flag==0))
                    raw_val = model.predict_proba(Xva)[:, 1]
                    
                    # 校正（各subsetで独立に保存）              
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(raw_val, yva)
                    proba_val = iso.transform(raw_val)
                    pr = average_precision_score(yva, proba_val) if len(yva) > 0 else np.nan
                    
                    bundles[name] = {
                        "model": model, 
                        "calibrator": {"type": "isotonic", "params": iso}, 
                        "pr_auc": float(pr) if pr==pr else np.nan
                    }

                # 検証：日ごとに該当モデルで推論            
                vdf = val.copy()
                preds = np.zeros(len(vdf), dtype=float)
                
                for d, g in vdf.groupby("date"):
                    is_sp = is_special_date(d)
                    sub = bundles["sp"] if is_sp else bundles["nonsp"]
                    raw = sub['model'].predict_proba(transform_safe(feat_scaler, g.reindex(columns=feat, fill_value=0.0).values))[:, 1]
                    iso = sub["calibrator"]["params"]
                    sc = iso.transform(raw) if iso is not None else raw
                    preds[g.index.values] = sc
                    
                vdf["score"] = preds
            else:
                # 単一モデル
                Xtr_raw, ytr = trn.reindex(columns=feat, fill_value=0.0).values, trn['target'].values
                Xva_raw, yva = val.reindex(columns=feat, fill_value=0.0).values, val['target'].values
                feat_scaler.fit(np.nan_to_num(Xtr_raw, nan=0.0))
                Xtr, Xva = transform_safe(feat_scaler, Xtr_raw), transform_safe(feat_scaler, Xva_raw)
                
                model = train_hgb(Xtr, ytr, trn["date"], emphasize_nonsp=True)
                raw_val = model.predict_proba(Xva)[:, 1]
                
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw_val, yva)
                proba_val = iso.transform(raw_val)
                
                vdf = val.copy()
                vdf["score"] = proba_val
                bundles["all"] = {"model": model, "calibrator": {"type": "isotonic", "params": iso}}

            # P@K評価
            rows = []
            k_list = [int(s) for s in args.k_list.split(",")]
            for k in k_list:
                pk = pk_by_day(vdf, "score", k)
                sp = pd.to_datetime(pk["date"]).dt.day.astype(int).isin(list(SPECIAL_DAYS))
                rows.append({
                    "k": k,
                    "mean_P@k_all": float(pk["precision_at_k"].mean()),
                    "mean_P@k_special(7/8/17/18/27/28)": float(pk.loc[sp, "precision_at_k"].mean()),
                    "days": int(len(pk)),
                    "special_days": int(sp.sum()),
                })
                
            summ = pd.DataFrame(rows)
            summ.to_csv(os.path.join(args.out_dir, "val_pk_summary.csv"), index=False)

            # モデル保存
            dump({
                "feat": feat,
                "feat_used": feat,
                "algo": "hgb_dual" if args.dual_models else "hgb_single",
                "day_state": {"scaler": scaler, "kmeans": kmeans},
                "feat_scaler": feat_scaler,
                "bundles": bundles,
            }, os.path.join(args.out_dir, "model_dir.joblib"))

            log(f"[TRAIN] saved -> {os.path.join(args.out_dir, 'model_dir.joblib')}")
            log(f"[VALID] window: {vstart} ~ {vend}")
            log(summ.to_string(index=False))
            log("[END] TRAIN")
            
        except Exception as e:
            log(f"[ERROR] Training failed: {e}")
            import traceback
            log(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise

    else:  # score mode
        log("[START] SCORE")
        try:
            # データ読み込み
            hist_files = [s.strip() for s in args.history_files.split(",") if s.strip()]
            test_files = [s.strip() for s in args.test_files.split(",") if s.strip()]

            # history_filesが空の場合の処理
            if hist_files:
                hist = read_and_stack(hist_files, "hist")
            else:
                log("[INFO] No history files provided - using test data only")
                hist = pd.DataFrame()

            test = read_and_stack(test_files, "test")
            
            if len(hist) > 0:
                log(f"[READ] hist: {len(hist)} rows, {hist['series'].nunique()} series")
            else:
                log(f"[READ] hist: 0 rows (no history files)")
            log(f"[READ] test: {len(test)} rows, {test['series'].nunique()} series")
            
            if len(hist) > 0:
                comb = pd.concat([hist, test], ignore_index=True).sort_values(["series", "machine_id", "date"]).reset_index(drop=True)
            else:
                comb = test.sort_values(["series", "machine_id", "date"]).reset_index(drop=True)
            
            # ===== 本番モード処理 =====
            target_date = getattr(args, 'target_date', None)
            production_mode = getattr(args, 'production_mode', False)
            
            if production_mode and target_date:
                target_date = pd.to_datetime(target_date)
                log(f"[PRODUCTION MODE] Target date: {target_date}")
                log(f"[PRODUCTION MODE] Using date gate for feature generation")
            
            # 特徴量生成（訓練と同じ順序）
            log("[FEAT] Adding features for scoring...")
            try:
                comb = add_calendar_features(comb)
                log("[FEAT] add_calendar_features completed")

                comb = add_exante_features(comb)
                log("[FEAT] add_exante_features completed")

                comb = add_series_monthweek_features(comb)
                log("[FEAT] add_series_monthweek_features completed")
                # キーで並べ替え（rolling/shiftの前提）
                comb = comb.sort_values(["series", "num", "date"]).reset_index(drop=True)

                # target_d1 を一度だけ作る（無ければ）
                if "target_d1" not in comb.columns:
                    if "target" in comb.columns:
                        # 既存の target を d-1 シフト
                        comb["target_d1"] = (
                            comb.groupby(["series", "num"], sort=False)["target"]
                                .shift(1)
                                .astype("float64")
                        )
                    elif "samai" in comb.columns:
                        # samai から target を作って d-1 シフト（例：samai>=2000 を陽性）
                        comb["target"] = (comb["samai"] >= 2000).astype("int8")
                        comb["target_d1"] = (
                            comb.groupby(["series", "num"], sort=False)["target"]
                                .shift(1)
                                .astype("float64")
                        )
                    else:
                        raise KeyError("Column(s) ['target_d1'] do not exist and cannot be derived (need 'target' or 'samai').")

                # 本番モードでは日付ゲート付きで実行
                if production_mode and target_date:
                    comb = add_series_aggregates(comb, target_date=target_date)
                else:
                    comb = add_series_aggregates(comb)
                log("[FEAT] add_series_aggregates completed")

                comb = add_nonsp_power_features(comb)
                log("[FEAT] add_nonsp_power_features completed")

                comb = add_series_weekday_resid(comb)
                log("[FEAT] add_series_weekday_resid completed")

                comb = add_num_end_digit_features(comb)
                log("[FEAT] add_num_end_digit_features completed")

                comb = add_neighbor_features(comb)
                log("[FEAT] add_neighbor_features completed")

                comb = add_wave_trend_features(comb)
                log("[FEAT] add_wave_trend_features completed")

                comb = add_series_percentile_features(comb)
                log("[FEAT] add_series_percentile_features completed")

                comb = add_bayesian_machine_ctr(comb)
                log("[FEAT] add_bayesian_machine_ctr completed")
                
                comb = add_renzoku_hekomi_features(comb)
                log("[FEAT] add_renzoku_hekomi_features completed")

                comb = add_compat_feature_aliases(comb)
                log("[FEAT] add_compat_feature_aliases completed")

            except Exception as e:
                log(f"[ERROR] Feature generation failed: {e}")
                import traceback
                log(f"[ERROR] Traceback: {traceback.format_exc()}")
                raise
                
            # モデル読み込み
            model_path = args.model_dir
            if not model_path.endswith(".joblib"):
                model_path = os.path.join(model_path, "model_dir.joblib")
            
            bundle_dir = load(model_path)
            scaler = bundle_dir.get("day_state", {}).get("scaler", None)
            kmeans = bundle_dir.get("day_state", {}).get("kmeans", None)
            comb, _, _ = add_day_state_features(comb, fit=False, scaler=scaler, kmeans=kmeans, n_clusters=5)

            # テストデータの分離と検証
            test_comb = comb[comb["src"] == "test"].copy()
            val_comb = comb[comb["src"] == "hist"].copy()
            
            # sub.pyの高度特徴量を追加
            if len(val_comb) > 0:
                val_comb, test_comb, sub_features = add_all_new_features(val_comb, test_comb)
            else:
                test_comb = add_tail_eq_dom_d1(test_comb)
                for c in ADD_ALL:
                    if c not in test_comb.columns:
                        test_comb[c] = 0.0

            # 特徴量確認用CSV出力（オプション）
            if getattr(args, 'export_features', False):
                log("[EXPORT] Exporting feature verification files for scoring...")
                export_feature_verification_csv(test_comb, args.out_dir, "test_features.csv")

            # ラグ特徴量の欠損チェックと補完
            miss = test_comb[["samai_lag1", "g_num_lag1", "avg_lag1"]].isna().any(axis=1).sum()
            if miss > 0:
                log(f"[WARNING] Missing lag1 features in {miss} test rows - filling with 0")
                test_comb["samai_lag1"] = test_comb["samai_lag1"].fillna(0)
                test_comb["g_num_lag1"] = test_comb["g_num_lag1"].fillna(0)
                test_comb["avg_lag1"] = test_comb["avg_lag1"].fillna(0)
            
            feat = bundle_dir["feat"]
            bundles = bundle_dir["bundles"]
            algo = bundle_dir.get("algo", "hgb_single")
            feat_scaler = bundle_dir.get("feat_scaler", make_feature_scaler(args.feat_scale))

            df = test_comb.copy()
            ensure_dir(args.out_dir)

            # スコアリング実行
            per_rows = []
            top_rows = []

            for d, g in df.groupby("date"):
                is_sp = is_special_date(d)
                
                if algo == "hgb_dual":
                    sub = bundles["sp"] if is_sp else bundles["nonsp"]
                else:
                    sub = bundles["all"]
                    
                model = sub["model"]
                iso = sub["calibrator"]["params"] if sub.get("calibrator") else None
                
                missing = [c for c in feat if c not in g.columns]
                if missing and len(missing) <= 3:
                    log(f'[WARN] score missing cols: {", ".join(missing[:6])}{"..." if len(missing)>6 else ""}')
                
                Xraw = g.reindex(columns=feat, fill_value=0.0).values
                X = transform_safe(feat_scaler, Xraw)
                raw = model.predict_proba(X)[:, 1]
                score = iso.transform(raw) if iso is not None else raw
                
                g = g.copy()
                g["score"] = score
                
                # K=1で最高スコアの台を選択
                k = min(1, len(g))
                pick = g.sort_values("score", ascending=False).head(k).copy()
                
                if len(pick) > 0:
                    pick = pick.sort_values("score", ascending=False).copy()
                    pick["rank"] = np.arange(1, len(pick) + 1)
                    
                    top_rows.append(pick[["date", "series", "machine_id", "score", "target", "samai", "g_num", "avg", "num"] + (["rank"] if len(pick)>0 else [])])

                prec = float(pick["target"].mean()) if len(pick) > 0 else np.nan
                per_rows.append({
                    "date": str(pd.Timestamp(d).date()),
                    "is_special": bool(is_sp),
                    "picked": int(len(pick)),
                    "positives": int(pick["target"].sum()) if len(pick) > 0 else 0,
                    "precision_at_k": prec,
                    "k": k,
                })
            # === 全台予測スコアを保存（K=3用） ===
            all_predictions_rows = []
            for d, g in df.groupby("date"):
                g_scored = g.copy()
                # このループ内でscoreが計算されているはず
                g_scored["score"] = score  # または該当する変数名
                all_predictions_rows.append(g_scored[["date", "series", "num", "machine_id", "score", "target"]])

            if all_predictions_rows:
                all_pred = pd.concat(all_predictions_rows, ignore_index=True)
                all_pred_path = os.path.join(args.out_dir, "all_predictions.csv")
                all_pred.to_csv(all_pred_path, index=False)
                log(f"[EXPORT] All predictions: {all_pred_path}")
                
            topk = pd.concat(top_rows) if len(top_rows) > 0 else pd.DataFrame()
            perday = pd.DataFrame(per_rows).sort_values("date")

            # 結果出力
            topk_path = os.path.join(args.out_dir, "test_topk_k1.csv")
            per_path = os.path.join(args.out_dir, "test_pk_k1.csv")
            
            if len(topk) > 0:
                topk.to_csv(topk_path, index=False)
            perday.to_csv(per_path, index=False)

            sp_mask = pd.to_datetime(perday["date"]).dt.day.astype(int).isin(list(SPECIAL_DAYS))
            mean_all = float(perday["precision_at_k"].mean())
            mean_sp = float(perday.loc[sp_mask, "precision_at_k"].mean()) if sp_mask.any() else np.nan

            log(f"[SCORE] saved -> {topk_path}")
            log(f"[SCORE] saved -> {per_path}")
            log(f"[SCORE] mean P@1 (ALL) = {mean_all:.4f}")
            log(f"[SCORE] mean P@1 (SPECIAL 7/8/17/18/27/28) = {mean_sp:.4f}")
            
            # 累積差枚分析
            log("[ANALYSIS] Performing comprehensive P@K and cumulative analysis...")
            pk_summary, topk1_detail = finalize_scoring_with_cumulative_analysis_fixed(perday, args.out_dir)
            
            log("[END] SCORE")
            
        except Exception as e:
            log(f"[ERROR] Scoring failed: {e}")
            import traceback
            log(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_samai_distributions(sys.argv[2])
    else:
        main()
# ====================
# 実際のデータファイル用コマンド例
# ====================
"""
Windows Anaconda Prompt での使用例:

## 1. モデル訓練（2025-08-16までのデータで訓練、特徴量確認CSV出力あり）
python naniutsu_v1.py train ^
  --train_files "hokuto_plazahakata_all_days_v14.csv,ghoul_plazahakata_all_days_v14.csv,monkey_plazahakata_all_days_v14.csv,myjugglerV_plazahakata_all_days_v14.csv" ^
  --cutoff "2025-08-16" ^
  --valid_days 14 ^
  --out_dir "model_output" ^
  --dual_models ^
  --export_features ^
  --series_start_dates "hokuto=2025-01-01;ghoul=2025-04-23;monkey=2025-01-01;myjugglerV=2025-01-01"

## 2. 2025-08-17～2025-09-06の予測スコアリング（K=1表示）
python naniutsu_v1.py score ^
  --model_dir "model_output/model_dir.joblib" ^
  --history_files "hokuto_plazahakata_all_days_v14.csv,ghoul_plazahakata_all_days_v14.csv,monkey_plazahakata_all_days_v14.csv,myjugglerV_plazahakata_all_days_v14.csv" ^
  --test_files "hokuto_test_ge_2025-08-17_v14.csv,monkey_test_ge_2025-08-17_v14.csv,ghoul_test_ge_2025-08-17_v14.csv,myjugglerV_test_ge_2025-08-17_v14.csv" ^
  --out_dir "score_output_0817-0906" ^
  --export_features

## 3. 2025-09-07の予測スコアリング（K=1表示）
python naniutsu_v1.py score ^
  --model_dir "model_output/model_dir.joblib" ^
  --history_files "hokuto_plazahakata_all_days_v14.csv,ghoul_plazahakata_all_days_v14.csv,monkey_plazahakata_all_days_v14.csv,myjugglerV_plazahakata_all_days_v14.csv" ^
  --test_files "zzz_test_ge_2025-09-07_v14.csv" ^
  --out_dir "score_output_0907" ^
  --export_features

## データファイル構成:
学習用データ（訓練に使用）:
- hokuto_plazahakata_all_days_v14.csv
- ghoul_plazahakata_all_days_v14.csv  
- monkey_plazahakata_all_days_v14.csv
- myjugglerV_plazahakata_all_days_v14.csv

テスト用データ（2025-08-17～2025-09-06）:
- hokuto_test_ge_2025-08-17_v14.csv
- monkey_test_ge_2025-08-17_v14.csv
- ghoul_test_ge_2025-08-17_v14.csv
- myjugglerV_test_ge_2025-08-17_v14.csv

予測用データ（2025-09-07）:
- zzz_test_ge_2025-09-07_v14.csv

## 出力ファイル:
特徴量確認用:
- feature_stats_*.csv: 各特徴量の統計情報
- feature_sample_*.csv: サンプルデータ
- feature_report_*.txt: 特徴量生成レポート
"""