import argparse
import os
import re
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
import re
def _force_parse_date_col(df, col="date"):
    # 文字列化
    s = df[col].astype(str)
    # 年-月-日を強制抽出（数字以外は区切り扱い）
    m = s.str.extract(r'([0-9]{4})[^0-9]?([0-9]{1,2})[^0-9]?([0-9]{1,2})', expand=True)
    # 3列すべて取れなかった行はNaNになる → そのままcoerceでNaTに
    parsed = (m[0].fillna('') + '-' + m[1].fillna('').str.zfill(2) + '-' + m[2].fillna('').str.zfill(2))
    df[col] = pd.to_datetime(parsed, errors='coerce')
    return df

def _assert_non_empty(df, where="train"):
    if len(df) == 0 or df["date"].isna().all():
        # # 直近の10件を出して原因を明示（例: dateに数字が無い）      
        sample = (df.head(10).to_dict(orient="records") if len(df) else [])
        raise RuntimeError(f"[FATAL] empty dataframe at {where}. "
                           f"Check date parsing & cutoff. sample_head10={sample}")
REQ_COLS = {"date", "num", "samai", "g_num", "avg"}
SPECIAL_DAYS = {7, 8, 17, 18, 27, 28}

# === Added in v13: feature pruning + lightweight new features ===

def _load_drop_from_rank_report(path: str, thresh: float) -> set:
    try:
        if not path:
            return set()
        import pandas as _pd
        df = _pd.read_csv(path)
        # Find columns robustly
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

def add_series_weekday_resid(df: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    # series×weekday のd-1条件付き平均からの残差
    if df is None or len(df) == 0:
        return df
    d = df.copy()
    if "weekday" not in d.columns:
        d["weekday"] = pd.to_datetime(d["date"]).dt.weekday
    d = d.sort_values(["series", "weekday", "date"]).reset_index(drop=True)
    g = d.groupby(["series", "weekday"], group_keys=False)
    mean_d1 = g["samai"].apply(lambda s: s.shift(1).rolling(window, min_periods=max(3, min(window//4, 7))).mean())
    d[f"samai_serieswk_mean{window}_d1"] = mean_d1
    d["samai_lag1_dev_serieswk_d1"] = d["samai_lag1"] - mean_d1
    return d
def add_num_end_digit_features(df: pd.DataFrame) -> pd.DataFrame:
    
    d = df.copy()
    d["num_end_digit"] = (pd.to_numeric(d["num"], errors="coerce").astype("Int64") % 10).astype("float")
    dom = pd.to_datetime(d["date"]).dt.day.astype(int)
    d["enddigit_eq_dom"] = (d["num_end_digit"].astype("Int64") == (dom % 10)).astype(int)
    # 特日 × 末尾一致
    if "is_special" not in d.columns:
        d = add_calendar_features(d)
    d["enddigit_eq_dom_x_special"] = d["enddigit_eq_dom"] * d["is_special"]
    return d



def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def parse_date_unified(s) -> pd.Timestamp:
    s = str(s).strip()
    d = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if pd.isna(d):
        d = pd.to_datetime(s, format="%Y/%m/%d", errors="coerce")
    if pd.isna(d):
        d = pd.to_datetime("2025-" + s, format="%Y-%m/%d", errors="coerce")
    return d


def norm_series_key(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0].lower()
    for key in ("monkey", "hokuto", "ghoul", "myjugglerv", "myjugglerv", "myjuggler"):
        if key in base:
            return "myjugglerv" if key in ("myjugglerv","myjugglerv","myjuggler") else key
    return re.sub(r"(_2|_all.*|_test.*)$", "", base)


def read_and_stack(csv_list, src_tag: str) -> pd.DataFrame:
    frames = []
    for p in csv_list:
        df = pd.read_csv(p)
        miss = REQ_COLS - set(df.columns)
        if miss:
            raise ValueError(f"{p} に必要列がありません: {miss}")
        df["date"] = df["date"].map(parse_date_unified)
        df["samai"] = pd.to_numeric(df["samai"], errors="coerce")
        df["g_num"] = pd.to_numeric(df["g_num"], errors="coerce")
        df["avg"] = pd.to_numeric(df["avg"], errors="coerce")
        df["num"] = pd.to_numeric(df["num"], errors="coerce")
        df["series"] = norm_series_key(p)
        df["machine_id"] = df["num"].astype(int).astype(str) + "_" + df["series"]
        df["target"] = (df["samai"] >= 2000).astype(int)
        df["src"] = src_tag
        frames.append(df[["date", "series", "num", "machine_id", "target", "g_num", "avg", "samai", "src"]])
    df = pd.concat(frames, ignore_index=True).dropna(subset=["date"]).sort_values(["series", "machine_id", "date"]).reset_index(drop=True)
    log(f"[READ] {src_tag}: {len(df)} rows, {df['series'].nunique()} series")
    return df



# ---- Safe feature subset helper (repaired) ----
from typing import List

def filter_present_features(df: pd.DataFrame, feat: List[str]) -> List[str]:

    present = [c for c in feat if c in df.columns]
    missing = [c for c in feat if c not in df.columns]
    if missing:
        try:
            msg = "[WARN] dropped_missing_features: " + ", ".join(missing[:10])
            if len(missing) > 10:
                msg += ", ..."
            print(msg)
        except Exception:
            pass
    return present

# ---- helpers ----

def _days_since_flag(values):
    out = []
    last = -1
    for i, v in enumerate(values):
        if v == 1:
            last = i
        out.append(i - last if last != -1 else 9999)
    return out


def is_special_date(ts) -> bool:
    try:
        d = pd.Timestamp(ts).day
    except Exception:
        return False
    return d in SPECIAL_DAYS


# ---- feature engineering ----

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dts = pd.to_datetime(df["date"]).dt
    df["is_special"] = dts.day.astype(int).isin(list(SPECIAL_DAYS)).astype(int)
    df["is_weekend"] = dts.weekday.isin([5, 6]).astype(int)
    w = dts.weekday.astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * w / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * w / 7.0)
    dom = dts.day.astype(int)
    df["dom_sin"] = np.sin(2 * np.pi * dom / 31.0)
    df["dom_cos"] = np.cos(2 * np.pi * dom / 31.0)
    return df


def add_series_monthweek_features(df: pd.DataFrame) -> pd.DataFrame:
    
    log("[FEAT] add_series_monthweek_features")
    if df is None or len(df) == 0:
        return df
    needs = {"series","machine_id","date","samai"}
    if not needs.issubset(df.columns):
        log("[FEAT] add_series_monthweek_features: skip (missing columns)")
        return df
    df = df.sort_values(["series","machine_id","date"]).copy()
    month_key = df["date"].dt.to_period("M")
    week_key  = df["date"].dt.to_period("W")
    g_m = df.groupby(["series","machine_id", month_key])
    df["series_month_cumsum_d1"] = g_m["samai"].cumsum().shift(1).values
    m_days = g_m.cumcount().replace(0, np.nan)
    df["series_month_cumsum_perday_d1"] = (df["series_month_cumsum_d1"] / m_days).fillna(0.0)
    g_w = df.groupby(["series","machine_id", week_key])
    df["series_week_cumsum_d1"] = g_w["samai"].cumsum().shift(1).values
    w_days = g_w.cumcount().replace(0, np.nan)
    df["series_week_cumsum_perday_d1"] = (df["series_week_cumsum_d1"] / w_days).fillna(0.0)
    for c in ["series_month_cumsum_d1","series_month_cumsum_perday_d1","series_week_cumsum_d1","series_week_cumsum_perday_d1"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    return df

def _debug_check_monthweek(df: pd.DataFrame, where: str):
    cols = ["series_month_cumsum_d1","series_month_cumsum_perday_d1","series_week_cumsum_d1","series_week_cumsum_perday_d1"]
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    log(f"[FEAT] monthweek_present@{where}: {present} missing={missing}")
    for c in present:
        nulls = int(df[c].isna().sum())
        log(f"[FEAT] monthweek_nulls@{where}: {c} nulls={nulls}")



def add_exante_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    if "machine_id" not in df.columns:
        df["machine_id"] = df["series"].astype(str) + "_" + df["num"].astype(str)

    dcol = "date_used" if "date_used" in df.columns else "date"
    df = df.sort_values(["machine_id", dcol], kind="mergesort")
    g = df.groupby("machine_id", sort=False)

    # distinct lags (index-safe)
    for prefix in ["samai", "g_num", "avg"]:
        df[f"{prefix}_lag1"] = g[prefix].shift(1)
        df[f"{prefix}_lag2"] = g[prefix].shift(2)
        df[f"{prefix}_lag3"] = g[prefix].shift(3)
        # rolling means via transform to keep original index
        df[f"{prefix}_roll3"] = g[prefix].transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
        df[f"{prefix}_roll7"] = g[prefix].transform(lambda s: s.rolling(7, min_periods=2).mean().shift(1))

    # recent_hit_rate (d-1)
    df["recent_hit_rate"] = g["samai"].transform(
        lambda s: (s >= 2000).astype(float).rolling(7, min_periods=2).mean().shift(1)
    )

    # days_since_last_2000 (d-1)
    def _days_since_last_2000_vec(s: pd.Series) -> pd.Series:
        # compute distance to previous hit (>=2000), ex-ante so shift at start
        hits = (s >= 2000).astype(int).values
        out = [np.nan] * len(hits)
        last = None
        for i in range(len(hits)):
            if last is None:
                out[i] = np.nan
            else:
                out[i] = last
            # update last for next row
            last = 0 if hits[i] == 1 else (0 if last is None else last + 1)
        return pd.Series(out, index=s.index, dtype="float")

    df["days_since_last_2000"] = g["samai"].transform(_days_since_last_2000_vec)

    # MoM
    df["samai_mom"] = df["samai_lag1"] - df["samai_lag2"]
    return df




def add_series_percentile_features(df: pd.DataFrame) -> pd.DataFrame:
    # 同一日・同一シリーズ内でのlag1値の順位パーセンタイル
    d = df.copy()
    for col in ["samai_lag1", "g_num_lag1", "avg_lag1"]:
        if col in d.columns:
            rank = d.groupby(["series", "date"])[col].rank(pct=True)
            d[f"{col}_pct_in_series_d"] = rank.fillna(0.0)
    return d
def add_bayesian_machine_ctr(df: pd.DataFrame, prior_strength: float = 10.0) -> pd.DataFrame:
    # 機械別の事前分布付きCTR（>=2000率）のベイズ推定（d-1）
    df = df.sort_values(["series", "machine_id", "date"]).copy()
    g = df.groupby("machine_id", group_keys=False)
    cum_pos = g["target"].shift(1).fillna(0).groupby(df["machine_id"], group_keys=False).cumsum()
    cum_n = g.cumcount()  # その日までの出現回数（当日含まない=shift(1)相当）
    # シリーズの非特日14日率 or 前日シリーズ率を事前に
    prior_rate = df["hr_nonsp_ma14_d1"].where(df["hr_nonsp_ma14_d1"].notna(), df["series_hit_rate2000_d1"]).fillna(0.1)
    alpha = 1.0 + cum_pos + prior_rate * prior_strength
    beta = 1.0 + (cum_n - cum_pos).clip(lower=0) + (1.0 - prior_rate) * prior_strength
    df["machine_ctr_beta_d1"] = (alpha / (alpha + beta)).astype(float)
    return df

def add_series_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    log("[FEAT] add_series_aggregates")
    df = df.sort_values(["series", "date", "machine_id"]).copy()

    ag = (
        df.groupby(["series", "date"]).agg(
            hits=("target", "sum"),
            gnum_p90=("g_num", lambda s: float(np.nanquantile(s, 0.9)) if len(s) > 0 else np.nan),
            avg_p90=("avg", lambda s: float(np.nanquantile(s, 0.9)) if len(s) > 0 else np.nan),
            gnum_mean=("g_num", "mean"),
            samai_p90=("samai", lambda s: float(np.nanquantile(s, 0.9)) if len(s) > 0 else np.nan),
            samai_med=("samai", "median"),
            hit_rate_2000=("samai", lambda s: float((s >= 2000).sum()) / max(1, len(s)))
        ).reset_index().sort_values(["series", "date"]).reset_index(drop=True)
    )

    ag["series_prev_hits"] = ag.groupby("series", group_keys=False)["hits"].shift(1)
    ag["series_roll3_hits"] = ag.groupby("series", group_keys=False)["hits"].shift(1)
    ag["series_roll3_hits"] = ag.groupby("series", group_keys=False)["series_roll3_hits"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    ag["series_gnum_p90_d1"] = ag.groupby("series", group_keys=False)["gnum_p90"].shift(1)
    ag["series_avg_p90_d1"] = ag.groupby("series", group_keys=False)["avg_p90"].shift(1)
    ag["series_samai_p90_d1"] = ag.groupby("series", group_keys=False)["samai_p90"].shift(1)
    ag["series_hit_rate2000_d1"] = ag.groupby("series", group_keys=False)["hit_rate_2000"].shift(1)

    ag["gm1"] = ag.groupby("series", group_keys=False)["gnum_mean"].shift(1)
    ag["mu14"] = ag.groupby("series", group_keys=False)["gm1"].transform(lambda s: s.rolling(14, min_periods=3).mean())
    ag["sd14"] = ag.groupby("series", group_keys=False)["gm1"].transform(lambda s: s.rolling(14, min_periods=3).std())
    ag["series_spike_g_d1"] = (ag["gm1"] - ag["mu14"]) / ag["sd14"].replace(0, np.nan)
    ag["series_spike_g_d1"] = ag["series_spike_g_d1"].fillna(0.0)

    use_cols = [
        "series", "date", "series_prev_hits", "series_roll3_hits",
        "series_gnum_p90_d1", "series_avg_p90_d1", "series_samai_p90_d1",
        "series_hit_rate2000_d1", "series_spike_g_d1"
    ]
    df = df.merge(ag[use_cols], on=["series", "date"], how="left")

    hall = (
        df.groupby("date").agg(
            tail_share_1000=("samai", lambda s: float((s >= 1000).sum()) / max(1, len(s))),
            hall_samai_p90=("samai", lambda s: float(np.nanquantile(s, 0.9)) if len(s) > 0 else np.nan),
            hall_samai_med=("samai", "median"),
            hall_hit_rate2000=("samai", lambda s: float((s >= 2000).sum()) / max(1, len(s)))
        ).reset_index().sort_values("date")
    )
    hall["tail_share_1000_d1"] = hall["tail_share_1000"].shift(1)
    hall["hall_samai_p90_d1"] = hall["hall_samai_p90"].shift(1)
    hall["hall_samai_med_d1"] = hall["hall_samai_med"].shift(1)
    hall["hall_hit_rate2000_d1"] = hall["hall_hit_rate2000"].shift(1)
    df = df.merge(hall[["date", "tail_share_1000_d1", "hall_samai_p90_d1", "hall_samai_med_d1", "hall_hit_rate2000_d1"]], on="date", how="left")

    # non-special only rolling rate (d-1) and weekday-conditioned expanding mean (d-1)
    sd = df[["series", "date"]].drop_duplicates().sort_values(["series", "date"]).reset_index(drop=True)
    sd = sd.merge(ag[["series", "date", "hit_rate_2000"]], on=["series", "date"], how="left")
    sd["is_special_flag"] = sd["date"].dt.day.astype(int).isin(list(SPECIAL_DAYS)).astype(int)
    sd["weekday"] = sd["date"].dt.weekday
    sd["hr_nonsp"] = sd["hit_rate_2000"].where(sd["is_special_flag"] == 0)
    sd = sd.sort_values(["series", "date"]).reset_index(drop=True)
    sd["hr_nonsp_ma14_d1"] = sd.groupby("series", group_keys=False)["hr_nonsp"].apply(lambda s: s.shift(1).rolling(14, min_periods=3).mean())
    sd["hr_nonsp_wk_ma_d1"] = sd.groupby(["series", "weekday"], group_keys=False)["hr_nonsp"].apply(lambda s: s.shift(1).expanding(min_periods=3).mean())
    df = df.merge(sd[["series", "date", "hr_nonsp_ma14_d1", "hr_nonsp_wk_ma_d1"]], on=["series", "date"], how="left")

    fill_cols = [
        "series_prev_hits", "series_roll3_hits", "series_gnum_p90_d1", "series_avg_p90_d1",
        "series_samai_p90_d1", "series_hit_rate2000_d1", "series_spike_g_d1",
        "tail_share_1000_d1", "hall_samai_p90_d1", "hall_samai_med_d1", "hall_hit_rate2000_d1",
        "hr_nonsp_ma14_d1", "hr_nonsp_wk_ma_d1"
    ]
    df[fill_cols] = df[fill_cols].fillna(0.0)
    return df


def add_neighbor_features(df: pd.DataFrame, radius_list=(1, 2)) -> pd.DataFrame:
    log("[FEAT] add_neighbor_features")
    df = df.sort_values(["series", "date", "num"]).copy()
    df["prev2000_d1"] = df.groupby("machine_id", group_keys=False)["target"].shift(1).fillna(0).astype(int)
    df["gnum_d1"] = df.groupby("machine_id", group_keys=False)["g_num"].shift(1).fillna(0.0)

    out_frames = []
    for (s, d), g in df.groupby(["series", "date"], sort=False):
        g = g.sort_values("num").copy()
        v_hit = g["prev2000_d1"].values.astype(float)
        v_gnum = g["gnum_d1"].values.astype(float)
        n = len(v_hit)
        for r in radius_list:
            kernel = np.ones(2 * r + 1, dtype=float)
            # hits
            pad_h = np.pad(v_hit, (r, r), mode="constant", constant_values=0.0)
            csum_h = np.convolve(pad_h, kernel, mode="valid")
            neigh_sum = csum_h - v_hit
            avail = np.minimum(np.arange(n) + r, r) + np.minimum((n - 1 - np.arange(n)) + r, r)
            neigh_rate = np.divide(neigh_sum, avail, out=np.zeros_like(neigh_sum), where=avail > 0)
            g[f"neigh_prev2000_pm{r}_sum"] = neigh_sum
            g[f"neigh_prev2000_pm{r}_rate"] = neigh_rate
            # g_num
            pad_g = np.pad(v_gnum, (r, r), mode="constant", constant_values=0.0)
            csum_g = np.convolve(pad_g, kernel, mode="valid")
            neigh_gsum = csum_g - v_gnum
            neigh_gmean = np.divide(neigh_gsum, avail, out=np.zeros_like(neigh_gsum), where=avail > 0)
            g[f"neigh_gnum_pm{r}_sum_d1"] = neigh_gsum
            g[f"neigh_gnum_pm{r}_mean_d1"] = neigh_gmean
        out_frames.append(g)
    df = pd.concat(out_frames, ignore_index=True)
    return df


def add_wave_trend_features(df: pd.DataFrame, wins=(7, 14, 28), cols=("samai", "g_num", "avg"), z_thr=0.8, r2_min=0.05) -> pd.DataFrame:
    df = df.sort_values(["series", "machine_id", "date"]).copy()
    g = df.groupby("machine_id", group_keys=False)

    for col in cols:
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

    fill_cols = [c for c in df.columns if c.startswith("wave_")]
    if fill_cols:
        df[fill_cols] = df[fill_cols].fillna(0.0)
    return df


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
    def q(s, p):
        s = np.asarray(s, dtype=float)
        return float(np.nanquantile(s, p)) if s.size > 0 else np.nan
    agg = (
        df.groupby("date").agg(
            gnum_p90=("g_num", lambda s: q(s, 0.90)),
            gnum_p75=("g_num", lambda s: q(s, 0.75)),
            gnum_med=("g_num", "median"),
            gnum_var=("g_num", "var"),
            gnum_gini=("g_num", gini_np),
            gnum_top10=("g_num", topk_share),
            avg_p90=("avg", lambda s: q(s, 0.90)),
            avg_p75=("avg", lambda s: q(s, 0.75)),
            avg_med=("avg", "median"),
            avg_var=("avg", "var"),
            avg_gini=("avg", gini_np),
            avg_top10=("avg", topk_share),
        ).reset_index()
    ).sort_values("date").reset_index(drop=True)
    return agg

def add_day_state_features(df: pd.DataFrame, fit: bool = False, scaler=None, kmeans=None, n_clusters: int = 5):
    hall = build_hall_day_features(df)

    hall_d1 = hall.copy()
    hall_d1["date"] = hall_d1["date"] + pd.Timedelta(days=1)
    hall_d1 = hall_d1.add_prefix("hall_")
    hall_d1 = hall_d1.rename(columns={"hall_date": "date"})

    feat_cols = [c for c in hall.columns if c != "date"]
    X = hall[feat_cols].fillna(0.0).values

    if fit:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(Xs)
    if scaler is None or kmeans is None:
        df = df.merge(hall_d1, on="date", how="left")
        for c in [f"ds_prob{i}" for i in range(n_clusters)]:
            df[f"ds_{c}"] = 0.0
        return df, scaler, kmeans

    Xs = scaler.transform(X)
    D = kmeans.transform(Xs)
    W = 1.0 / (D + 1e-6)
    W = W / W.sum(axis=1, keepdims=True)
    prob = pd.DataFrame(W, columns=[f"ds_prob{i}" for i in range(W.shape[1])])
    prob["date"] = hall["date"].values
    prob_d1 = prob.copy()
    prob_d1["date"] = prob_d1["date"] + pd.Timedelta(days=1)
    prob_d1 = prob_d1.add_prefix("ds_")
    prob_d1 = prob_d1.rename(columns={"ds_date": "date"})

    df = df.merge(hall_d1, on="date", how="left")
    df = df.merge(prob_d1, on="date", how="left")

    fill = [c for c in df.columns if c.startswith("hall_") or c.startswith("ds_ds_prob")]
    if fill:
        df[fill] = df[fill].fillna(0.0)

    return df, scaler, kmeans



# ---- nonsp power features & day-gate (added in v11) ----

import numpy as _np
import pandas as _pd


def add_nonsp_power_features(df: _pd.DataFrame) -> _pd.DataFrame:

    need = {'date','series','num','samai','g_num','avg'}
    if not need.issubset(df.columns):
        return df

    df = df.sort_values(['series','num','date']).copy()
    key = ['series','num']

    # へこみ連続日数
    def _neg_streak(x):
        hit = (x['samai'] >= 2000).astype(int)
        lose = 1 - hit
        streak = _np.zeros(len(lose), dtype=int)
        c = 0
        for i, v in enumerate(lose):
            c = c + 1 if v == 1 else 0
            streak[i] = c
        return _pd.Series(streak, index=x.index, name='neg_streak')

    # 累積ドローダウンの log1p（非負）
    def _drawdown(x):
        cum = x['samai'].cumsum()
        peak = cum.cummax()
        dd = cum - peak  # <= 0
        return _pd.Series(_np.log1p((-dd).clip(lower=0)), index=x.index, name='drawdown_log1p')

    grp = df.groupby(key, sort=False, group_keys=False)
    df['neg_streak'] = grp.apply(_neg_streak).values
    df['drawdown_log1p'] = grp.apply(_drawdown).values

    # シリーズ×曜日平均との差（lag1基準）
    df['weekday'] = df['date'].dt.weekday
    base = df.groupby(['series','weekday'], sort=False)['samai'].mean().rename('series_weekday_mean')
    df = df.join(base, on=['series','weekday'])
    df['samai_lag1'] = grp['samai'].shift(1)
    df['samai_lag1_dev_serieswk'] = df['samai_lag1'] - df['series_weekday_mean']

    # 短中期クロス（transformで安定に算出）→ d-1 シフト
    for col, prefix in [('g_num','gnum'),('avg','avg')]:
        ma3 = df.groupby(key)[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        ma7 = df.groupby(key)[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
        ratio = (ma3 / ma7).replace([_np.inf, -_np.inf], _np.nan)
        df[f'{prefix}_ma3_over_ma7'] = ratio
        df[f'{prefix}_ma3_over_ma7_d1'] = df.groupby(key)[f'{prefix}_ma3_over_ma7'].shift(1)

    #  d-1 版を確定（リーク防止）
    for c in ['neg_streak','drawdown_log1p','samai_lag1_dev_serieswk']:
        df[f'{c}_d1'] = df.groupby(key)[c].shift(1)

    return df

    df = df.sort_values(['series','num','date']).copy()
    key = ['series','num']

    def _neg_streak(x):
        hit = (x['samai'] >= 2000).astype(int)
        lose = 1 - hit
        streak = _np.zeros(len(lose), dtype=int)
        c = 0
        for i, v in enumerate(lose):
            c = c + 1 if v == 1 else 0
            streak[i] = c
        return _pd.Series(streak, index=x.index, name='neg_streak')

    def _drawdown(x):
        cum = x['samai'].cumsum()
        peak = cum.cummax()
        dd = cum - peak
        dd_log1p = _np.log1p((-dd).clip(lower=0))
        return _pd.Series(dd_log1p, index=x.index, name='drawdown_log1p')

    grp = df.groupby(key, sort=False, group_keys=False)
    df['neg_streak'] = grp.apply(_neg_streak).values
    df['drawdown_log1p'] = grp.apply(_drawdown).values

    df['weekday'] = df['date'].dt.weekday
    base = df.groupby(['series','weekday'], sort=False)['samai'].mean().rename('series_weekday_mean')
    df = df.join(base, on=['series','weekday'])
    df['samai_lag1'] = grp['samai'].shift(1)
    df['samai_lag1_dev_serieswk'] = df['samai_lag1'] - df['series_weekday_mean']

    for col, prefix in [('g_num','gnum'),('avg','avg')]:
        ma3 = grp[col].apply(lambda s: s.rolling(3, min_periods=1).mean())
        ma7 = grp[col].apply(lambda s: s.rolling(7, min_periods=1).mean())
        ratio = (ma3 / ma7).replace([_np.inf, -_np.inf], _np.nan)
        df[f'{prefix}_ma3_over_ma7'] = ratio
        df[f'{prefix}_ma3_over_ma7'] = grp[f'{prefix}_ma3_over_ma7'].shift(1)

    for c in ['neg_streak','drawdown_log1p','samai_lag1_dev_serieswk']:
        df[f'{c}_d1'] = grp[c].shift(1)
    for c in ['gnum_ma3_over_ma7','avg_ma3_over_ma7']:
        df[f'{c}_d1'] = df[c]
    return df


def day_gate_score_rows(df: _pd.DataFrame) -> _pd.Series:
    cols = {
        'series_prev_hits': 0.30,
        'series_roll3_hits': 0.25,
        'hall_gnum_p90': 0.12,
        'hall_avg_med': 0.10,
        'hall_gnum_gini': -0.10,
        'hall_avg_top10': 0.10,
        'wave_slope_samai_w7_d1': 0.10,
        'wave_slope_samai_w14_d1': 0.08,
        'wave_slope_samai_w28_d1': 0.06,
    }
    s = _np.zeros(len(df), dtype=float)
    for c, w in cols.items():
        v = df[c] if c in df.columns else 0.0
        try:
            s = s + v.values * w
        except Exception:
            s = s + 0.0
    return _pd.Series(s, index=df.index, name="score_gate")


def compute_play_dates(df_rows: _pd.DataFrame, cov_min: float = 0.35, cov_max: float = 0.60) -> set:
    cov = float((cov_min + cov_max) / 2.0)
    df = df_rows.copy()
    if 'is_special' not in df.columns:
        df['is_special'] = df['date'].dt.day.astype(int).isin(list(SPECIAL_DAYS)).astype(int)
    df['score_gate'] = day_gate_score_rows(df)
    daily = df.groupby('date', sort=False).agg(
        score_gate=('score_gate','mean'),
        is_special=('is_special','max')
    ).reset_index()
    sp_dates = set(daily.loc[daily['is_special']==1, 'date'].astype(str))
    nonsp = daily[daily['is_special']==0].copy()
    if len(nonsp) == 0:
        return sp_dates
    thr = nonsp['score_gate'].quantile(max(0.0, min(1.0, 1.0 - cov)))
    play_nonsp = set(nonsp.loc[nonsp['score_gate'] >= thr, 'date'].astype(str))
    return sp_dates | play_nonsp

# ---- feature list ----

# v12_fixlag11

def features_list(columns) -> list:
    base = [c for c in columns if re.match(r"(samai|g_num|avg)_lag[1-3]$", c)]
    base += [c for c in columns if c.endswith("_roll3_mean") or c.endswith("_roll7_mean")]
    base += [
        "machine_prior_hit", "gnum_prior_mean", "days_since_last_2000", "weekday",
        "gnum_mom1", "avg_mom1", "samai_mom1",
    ]
    base += [
        "series_prev_hits", "series_roll3_hits", "series_gnum_p90_d1", "series_avg_p90_d1",
        "series_samai_p90_d1", "series_hit_rate2000_d1", "series_spike_g_d1",
        "tail_share_1000_d1", "hall_samai_p90_d1", "hall_samai_med_d1", "hall_hit_rate2000_d1",
        "hr_nonsp_ma14_d1", "hr_nonsp_wk_ma_d1",
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
    # v11 added nonsp-power features
    base += [c for c in ['neg_streak_d1','drawdown_log1p_d1','samai_lag1_dev_serieswk_d1','gnum_ma3_over_ma7_d1','avg_ma3_over_ma7_d1'] if c in columns]

    return base


def drop_na_lag1_for_training(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["samai_lag1", "g_num_lag1", "avg_lag1"]).copy()


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


def scale_pos_weight(y: np.ndarray) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return max(1.0, float(neg) / max(1, pos))


def train_hgb(Xtr: np.ndarray, ytr: np.ndarray, trn_dates: pd.Series, emphasize_nonsp: bool = False) -> HistGradientBoostingClassifier:
    cw = scale_pos_weight(ytr)
    if emphasize_nonsp:
        is_sp = pd.to_datetime(trn_dates).dt.day.astype(int).isin(list(SPECIAL_DAYS)).values
        w_ns = np.where(is_sp, 1.0, 1.2)  # 非特日をやや強調
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


# ---- auto-k helpers ----

def choose_k_greedy_by_threshold(probs_sorted: np.ndarray, k_min: int, k_max: int, tau_p: float) -> int:
    if probs_sorted.size == 0:
        return 0
    chosen = 0
    s = 0.0
    for p in probs_sorted[:k_max]:
        mean_after = (s + p) / (chosen + 1)
        if mean_after >= tau_p or chosen + 1 < k_min:
            s += p
            chosen += 1
        else:
            break
    return chosen


def choose_k_greedy_by_p_and_lift(probs_sorted: np.ndarray, k_min: int, k_max: int, tau_p: float, base_proxy: float, tau_lift: float) -> int:
    if probs_sorted.size == 0:
        return 0
    base_proxy = float(max(1e-6, base_proxy))
    chosen = 0
    s = 0.0
    for p in probs_sorted[:k_max]:
        mean_after = (s + p) / (chosen + 1)
        cond_p = mean_after >= tau_p
        cond_lift = (mean_after / base_proxy) >= tau_lift
        if (chosen + 1 < k_min) or (cond_p and cond_lift):
            s += p
            chosen += 1
        else:
            break
    return chosen


# === Added utilities for start-date filtering & feature scaling (v12_lagstd) ===
from sklearn.preprocessing import MinMaxScaler


# --- COMPAT SHIM GUARD (always available) ---
try:
    add_compat_feature_aliases
except NameError:
    import pandas as pd, numpy as np
    def add_compat_feature_aliases(df: pd.DataFrame) -> pd.DataFrame:
        print("[FEAT] add_compat_feature_aliases<guard>")
        df = df.copy()
        n = len(df)
        # machine_prior_hit
        if "machine_prior_hit" not in df.columns:
            if "samai_lag1" in df.columns:
                df["machine_prior_hit"] = (pd.to_numeric(df["samai_lag1"], errors="coerce") >= 2000).astype("Int64")
            else:
                df["machine_prior_hit"] = pd.Series([pd.NA]*n, dtype="Int64")
        # gnum_prior_mean
        if "gnum_prior_mean" not in df.columns:
            cand = None
            for c in ("g_num_roll3","g_num_roll7","g_num_lag1"):
                if c in df.columns:
                    cand = c
                    break
            df["gnum_prior_mean"] = pd.to_numeric(df[cand], errors="coerce") if cand else np.nan
        # *_mom1 = lag1 - lag2
        def _mom(l1,l2,out):
            if out not in df.columns:
                if l1 in df.columns and l2 in df.columns:
                    df[out] = pd.to_numeric(df[l1], errors="coerce") - pd.to_numeric(df[l2], errors="coerce")
                else:
                    df[out] = np.nan
        _mom("samai_lag1","samai_lag2","samai_mom1")
        _mom("g_num_lag1","g_num_lag2","gnum_mom1")
        _mom("avg_lag1","avg_lag2","avg_mom1")
        return df
# --- END GUARD ---

# Safe feature subset helper: drop features not present in df (prevents KeyError)

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
    # HGB can handle NaN, but scalersはNaNが苦手→一時的に0で埋めてスケール後に0に戻す必要はない（スケール値をそのまま利用）
    # ここでは単純に NaN→0 で統一（学習・推論で整合）    
    import numpy as _np
    X2 = _np.asarray(X, dtype=float)
    X2 = _np.where(_np.isnan(X2), 0.0, X2)
    return scaler.transform(X2)

def _align_to_scaler(X, scaler):
    import numpy as np
    X = np.array(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    exp = getattr(scaler, "n_features_in_", None)
    if exp is not None and X.ndim == 2:
        cur = X.shape[1]
        if cur < exp:
            X = np.hstack([X, np.zeros((X.shape[0], exp-cur), dtype=X.dtype)])
            try: print(f"[WARN] pre-align pad: {cur}->{exp}")
            except Exception: pass
        elif cur > exp:
            X = X[:, :exp]
            try: print(f"[WARN] pre-align trunc: {cur}->{exp}")
            except Exception: pass
    return X


# ---- CLI ----

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--series_start_dates", default="", help="萓・ghoul=2025-04-23;hokuto=2025-01-01;monkey=2025-01-01")
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

    return p.parse_args()


# ---- main ----

def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    if args.mode == "train":
        log("[START] TRAIN")
        train_files = [s.strip() for s in args.train_files.split(",") if s.strip()]
        df = read_and_stack(train_files, "train")
        df = add_calendar_features(df)
        df = add_exante_features(df)
        df = add_series_monthweek_features(df)
        _debug_check_monthweek(df, 'df_after_add_exante_features')
        df = add_series_aggregates(df)
        df = add_nonsp_power_features(df)
        df = add_series_monthweek_features(df)
        df = add_series_weekday_resid(df)
        df = add_num_end_digit_features(df)
        _debug_check_monthweek(df, 'df_after_add_series_aggregates')
        df = add_neighbor_features(df)
        df = add_series_monthweek_features(df)
        _debug_check_monthweek(df, 'df_after_add_neighbor_features')
        df = add_wave_trend_features(df)
        df = add_series_percentile_features(df)
        from pk_add_feat_thickness_v14p import add_thickness_and_interactions
        df = add_bayesian_machine_ctr(df)
        df, scaler, kmeans = add_day_state_features(df, fit=True, n_clusters=5)
        # Apply series start-date filter then require lagK
        start_map = _parse_series_date_map(args.series_start_dates)
        df = enforce_series_start_dates(df, start_map)
        df = drop_na_lagk_for_training(df, k=args.lag_required)
        trn, val, vstart, vend = split_train_valid(df, args.cutoff, args.valid_days)
        print("[FEAT] add_compat_feature_aliases")
        df = add_compat_feature_aliases(df)
        print("[FEAT] add_compat_feature_aliases")
        df = add_compat_feature_aliases(df)
        print("[FEAT] add_compat_feature_aliases")
        df = add_compat_feature_aliases(df)
        feat = features_list(df.columns)
        feat = [c for c in feat if c in df.columns]
        # prune by rank report/manual
        drop_set = _load_drop_from_rank_report(args.rank_report, args.rank_threshold)
        feat = _apply_feature_drop(feat, drop_set, args.drop_features)        
        feat_used = feat
        feat = list(feat_used)
        # Feature scaling
        feat_scaler = make_feature_scaler(args.feat_scale)
        # prepare splits

        trn = trn.sort_values(["date", "series", "machine_id"]).reset_index(drop=True)
        val = val.sort_values(["date", "series", "machine_id"]).reset_index(drop=True)

        bundles = {}
        if args.dual_models:
            for flag, name in [(0, "nonsp"), (1, "sp")]:
                tr_sub = trn[trn["is_special"] == flag].copy()
                va_sub = val[val["is_special"] == flag].copy()
                Xtr_raw, ytr = tr_sub.reindex(columns=feat, fill_value=0.0).values, tr_sub['target'].values
                Xva_raw, yva = va_sub.reindex(columns=feat, fill_value=0.0).values, va_sub['target'].values
                feat_scaler.fit(np.nan_to_num(Xtr_raw, nan=0.0))
                Xtr, Xva = transform_safe(feat_scaler, Xtr_raw), transform_safe(feat_scaler, Xva_raw)
                if Xtr.shape[0] == 0 or Xva.shape[0] == 0:
                    # # フォールバック：subsetが空なら全体で
                    Xtr, ytr = trn[feat].values, trn["target"].values
                    Xva, yva = val[feat].values, val["target"].values
                    tr_dates = trn["date"]
                else:
                    tr_dates = tr_sub["date"]
                model = train_hgb(Xtr, ytr, tr_dates, emphasize_nonsp=(flag==0))
                raw_val = model.predict_proba(Xva)[:, 1]
                # 校正（各subsetで独立に保存）              
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw_val, yva)
                proba_val = iso.transform(raw_val)
                pr = average_precision_score(yva, proba_val) if len(yva) > 0 else np.nan
                bundles[name] = {"model": model, "calibrator": {"type": "isotonic", "params": iso}, "pr_auc": float(pr) if pr==pr else np.nan}

            # 検証：日ごとに該当モデルで推論            
            vdf = val.copy()
            preds = np.zeros(len(vdf), dtype=float)
            for d, g in vdf.groupby("date"):
                is_sp = is_special_date(d)
                sub = bundles["sp"] if is_sp else bundles["nonsp"]
                raw = sub['model'].predict_proba(g.reindex(columns=feat, fill_value=0.0).values)[:, 1]
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

        dump({
            "feat": feat,
            "feat_used": feat,
            "algo": "hgb_dual" if args.dual_models else "hgb_single",
            "day_state": {"scaler": scaler, "kmeans": kmeans},
            "feat_scaler": feat_scaler,
            "bundles": bundles,
        }, os.path.join(args.out_dir, "model_dir.joblib"))

        log(f"[TRAIN] saved -> {os.path.join(args.out_dir, 'model_dir.joblib')}")
        log(f"[VALID] window: {vstart} 縲・{vend}")
        log(summ.to_string(index=False))
        log("[END] TRAIN")

    else:
        log("[START] SCORE")
        hist_files = [s.strip() for s in args.history_files.split(",") if s.strip()]
        test_files = [s.strip() for s in args.test_files.split(",") if s.strip()]
        hist = read_and_stack(hist_files, "hist")
        test = read_and_stack(test_files, "test")
        comb = pd.concat([hist, test], ignore_index=True).sort_values(["series", "machine_id", "date"]).reset_index(drop=True)
        comb = add_calendar_features(comb)
        comb = add_exante_features(comb)
        comb = add_series_monthweek_features(comb)
        _debug_check_monthweek(comb, 'comb_after_add_exante_features')
        comb = add_series_aggregates(comb)
        comb = add_nonsp_power_features(comb)
        comb = add_series_monthweek_features(comb)
        _debug_check_monthweek(comb, 'comb_after_add_series_aggregates')
        comb = add_series_weekday_resid(comb)
        comb = add_num_end_digit_features(comb)
        comb = add_neighbor_features(comb)
        comb = add_series_monthweek_features(comb)
        _debug_check_monthweek(comb, 'comb_after_add_neighbor_features')
        comb = add_wave_trend_features(comb)
        comb = add_series_percentile_features(comb)
        comb = add_compat_feature_aliases(comb)
        comb = add_bayesian_machine_ctr(comb)

        bundle_dir = load(args.model_dir + ("" if args.model_dir.endswith(".joblib") else "/model_dir.joblib"))
        scaler = bundle_dir.get("day_state", {}).get("scaler", None)
        kmeans = bundle_dir.get("day_state", {}).get("kmeans", None)
        comb, _, _ = add_day_state_features(comb, fit=False, scaler=scaler, kmeans=kmeans, n_clusters=5)

        miss = comb[comb["src"] == "test"][ ["samai_lag1", "g_num_lag1", "avg_lag1"] ].isna().any(axis=1).sum()
        if miss > 0:
            raise ValueError(f"[SCORE] Missing lag1 features in test rows: {miss}. Provide history_files to warm up lags.")

        feat = bundle_dir["feat"]
        bundles = bundle_dir["bundles"]
        algo = bundle_dir.get("algo", "hgb_single")
        feat_scaler = bundle_dir.get("feat_scaler", make_feature_scaler(args.feat_scale))

        df = comb[comb["src"] == "test"].copy()
        ensure_dir(args.out_dir)
        play_dates = None
        if getattr(args, "day_gate", False):
            play_dates = compute_play_dates(df, cov_min=args.gate_cov_min, cov_max=args.gate_cov_max)
            _dg = pd.DataFrame({'date': sorted(set(df['date'].astype(str)))})
            _dg['played'] = _dg['date'].isin(play_dates)
            _dg.to_csv(os.path.join(args.out_dir, "day_gate_played.csv"), index=False)

        # 日別 base_proxy（d-1 hall hit率の平滑）
        proxy = df.groupby("date").agg(base_proxy=("hall_hit_rate2000_d1", "mean")).reset_index()
        proxy["base_proxy"] = proxy["base_proxy"].fillna(0.0).clip(lower=0.0)
        proxy.to_csv(os.path.join(args.out_dir, "daily_base_proxy.csv"), index=False)
        base_map = dict(zip(proxy["date"].astype(str), proxy["base_proxy"].astype(float)))

        per_rows = []
        top_rows = []

        for d, g in df.groupby("date"):
            is_sp = is_special_date(d)
            if getattr(args, "day_gate", False) and (not is_sp):
                if (play_dates is not None) and (str(pd.Timestamp(d).date()) not in play_dates):
                    if args.fallback_policy == "skip":
                        per_rows.append({"date": str(pd.Timestamp(d).date()), "is_special": bool(is_sp), "picked": 0, "positives": 0, "precision_at_k": float("nan"), "tau": float(args.tau_normal), "lift_tau": float(args.lift_tau_normal), "base_proxy": float(base_map.get(str(pd.Timestamp(d).date()), 0.0)), "auto_k": bool(args.auto_k), "played": False})
                        top_rows.append(g.head(0))
                        continue
            played_flag = True
            if algo == "hgb_dual":
                sub = bundles["sp"] if is_sp else bundles["nonsp"]
            else:
                sub = bundles["all"]
            model = sub["model"]
            iso = sub["calibrator"]["params"] if sub.get("calibrator") else None
            missing = [c for c in feat if c not in g.columns]
            if missing:
                try: print('[WARN] score missing cols:', ', '.join(missing[:6]) + (' ...' if len(missing)>6 else ''))
                except Exception: pass
            Xraw = g.reindex(columns=feat, fill_value=0.0).values
            X = transform_safe(feat_scaler, Xraw)
            raw = model.predict_proba(X)[:, 1]
            score = iso.transform(raw) if iso is not None else raw
            g = g.copy()
            g["score"] = score

            # 配分：シリーズごとの次点pを見ながら貪欲
            series_groups = {s: grp.sort_values("score", ascending=False).copy() for s, grp in g.groupby("series")}
            series_idx = {s: 0 for s in series_groups.keys()}
            chosen_rows = []
            ssum = 0.0
            k = 0
            k_min, k_max = args.k_min, min(args.k_max, len(g))
            tau_p = args.tau_special if is_sp else args.tau_normal
            tau_lift = args.lift_tau_special if is_sp else args.lift_tau_normal
            base_p = float(base_map.get(str(pd.Timestamp(d).date()), 0.0))

            def next_best():
                best_s = None
                best_p = -1.0
                for s, grp in series_groups.items():
                    idx = series_idx[s]
                    if idx < len(grp):
                        p = float(grp.iloc[idx]["score"])
                        if p > best_p:
                            best_p = p
                            best_s = s
                return best_s, best_p

            while k < k_max:
                s_name, p_next = next_best()
                if s_name is None:
                    break
                mean_after = (ssum + p_next) / (k + 1)
                cond_p = (mean_after >= tau_p)
                cond_lift = (mean_after / float(max(1e-6, base_p))) >= tau_lift if args.autok_mode == "p_and_lift" else True
                if (k + 1 < k_min) or (not args.auto_k) or (cond_p and cond_lift):
                    row = series_groups[s_name].iloc[series_idx[s_name]]
                    chosen_rows.append(row)
                    series_idx[s_name] += 1
                    ssum += p_next
                    k += 1
                    if not args.auto_k and k >= k_max:
                        break
                else:
                    break

            if k == 0:
                if args.fallback_policy == "skip":
                    chosen = g.head(0).copy()
                else:
                    chosen = g.sort_values("score", ascending=False).head(3 if not is_sp else 5).copy()
                    k = len(chosen)
                chosen_rows = [chosen] if len(chosen) > 0 else []

            if len(chosen_rows) > 0 and not isinstance(chosen_rows[0], pd.DataFrame):
                pick = pd.DataFrame(chosen_rows)
            elif len(chosen_rows) > 0:
                pick = pd.concat(chosen_rows, ignore_index=True)
            else:
                pick = g.head(0).copy()

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
                "tau": float(tau_p),
                "lift_tau": float(tau_lift),
                "base_proxy": float(base_p),
                "auto_k": bool(args.auto_k),
            })

        topk = pd.concat(top_rows) if len(top_rows) > 0 else pd.DataFrame()
        perday = pd.DataFrame(per_rows).sort_values("date")

        topk_path = os.path.join(args.out_dir, "test_topk_autok.csv")
        per_path = os.path.join(args.out_dir, "test_pk_autok.csv")
        if len(topk) > 0:
            topk.to_csv(topk_path, index=False)
        perday.to_csv(per_path, index=False)

        sp_mask = pd.to_datetime(perday["date"]).dt.day.astype(int).isin(list(SPECIAL_DAYS))
        mean_all = float(perday["precision_at_k"].mean())
        mean_sp = float(perday.loc[sp_mask, "precision_at_k"].mean()) if sp_mask.any() else np.nan
        log(f"[SCORE] saved -> {topk_path}")
        log(f"[SCORE] saved -> {per_path}")
        log(f"[SCORE] mean P@ (ALL) = {mean_all:.4f}")
        log(f"[SCORE] mean P@ (SPECIAL 7/8/17/18/27/28) = {mean_sp:.4f}")
        if perday["date"].isin(["2025-08-17", "2025-08-18"]).any():
            log("[IMPORTANT] 2025-08-17 / 2025-08-18")
            log(perday[perday["date"].isin(["2025-08-17", "2025-08-18"])].to_string(index=False))
        log("[END] SCORE")


if __name__ == "__main__":
    main()


def add_compat_feature_aliases(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    n = len(df)
    # machine_prior_hit
    if "machine_prior_hit" not in df.columns:
        if "samai_lag1" in df.columns:
            df["machine_prior_hit"] = (pd.to_numeric(df["samai_lag1"], errors="coerce") >= 2000).astype("Int64")
        else:
            df["machine_prior_hit"] = pd.Series([pd.NA]*n, dtype="Int64")
    # gnum_prior_mean
    if "gnum_prior_mean" not in df.columns:
        cand = None
        for c in ["g_num_roll3", "g_num_roll7", "g_num_lag1"]:
            if c in df.columns:
                cand = c
                break
        df["gnum_prior_mean"] = pd.to_numeric(df[cand], errors="coerce") if cand else np.nan
    # mom1 features
    def _make_mom(name_l1, name_l2, out_name):
        if out_name not in df.columns:
            if name_l1 in df.columns and name_l2 in df.columns:
                df[out_name] = pd.to_numeric(df[name_l1], errors="coerce") - pd.to_numeric(df[name_l2], errors="coerce")
            else:
                df[out_name] = np.nan
    _make_mom("samai_lag1", "samai_lag2", "samai_mom1")
    _make_mom("g_num_lag1", "g_num_lag2", "gnum_mom1")
    _make_mom("avg_lag1", "avg_lag2", "avg_mom1")
    return df


