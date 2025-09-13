# -*- coding: utf-8 -*-
# Streamlit — 翌日K=1予測（BASE/DROP/BLEND）
# - 学習CSV(v14×4) + テストCSV(*_test_ge_*_v14)をアップロード
# - 特徴量は学習時パイプラインを再現（リーク防止・ウォームアップ付き）
# - αは固定 or 自動スイープ
# - K=1トップを表示＆CSV保存

import sys, platform, os, io, glob, zipfile, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import sklearn, joblib as jb
import importlib.util as iu

APP_TITLE = "K=1 BLEND Predictor"

# ---------- ここを一番最初の Streamlit 呼び出しにする ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
# --------------------------------------------------------------

# バージョン情報表示（任意）
st.title(APP_TITLE)
st.caption(
    f"Python {sys.version.split()[0]} | "
    f"sklearn {sklearn.__version__} | joblib {jb.__version__} | "
    f"numpy {np.__version__} | pandas {pd.__version__} | {platform.platform()}"
)

# ====== 設定 ======
PK_FILE   = "pk_series_rank_v_4x_dual_mw_v13_featprune_fix7.py"  # リポジトリ同梱想定
WARMUP_DAYS_DEFAULT = 7

# ====== ヘルパ ======
@st.cache_resource(show_spinner=False)
def load_module(pk_path: str):
    if not os.path.isfile(pk_path):
        raise FileNotFoundError(f"{pk_path} が見つかりません（リポジトリに同梱してください）")
    S = iu.spec_from_file_location("m", pk_path)
    m = iu.module_from_spec(S); S.loader.exec_module(m)
    return m

def ensure_min_schema(df: pd.DataFrame) -> pd.DataFrame:
    D = df.copy()
    if "date" not in D.columns:
        raise ValueError("date 列が必要です")
    D["date"] = pd.to_datetime(D["date"], errors="coerce")

    for c in ["samai","g_num","avg"]:
        if c in D.columns:
            D[c] = pd.to_numeric(D[c], errors="coerce")

    if "series" not in D.columns:
        D["series"] = "NA"
    if "is_special" not in D.columns:
        D["is_special"] = 0

    # 学習CSVに target が無い時のフォールバック（2000枚以上で1）
    if "target" not in D.columns and "samai" in D.columns:
        D["target"] = (D["samai"] >= 2000).astype(int)

    # 評価可視化用
    if "target_eval" not in D.columns and "samai" in D.columns:
        D["target_eval"] = (D["samai"] >= 2000).astype(int)

    return D

def get_samai_col(df: pd.DataFrame):
    for c in df.columns:
        if str(c).lower().startswith("samai"):
            return c
    return None

def _resolve_model(bundles: dict, prefer_key: str):
    entry = bundles.get(prefer_key)
    if entry is None:
        entry = next(iter(bundles.values()))
    if isinstance(entry, dict):
        entry = entry.get("model", entry)
    return entry

def _score_proba(mdl, X: np.ndarray) -> np.ndarray:
    if hasattr(mdl, "predict_proba"):
        p = mdl.predict_proba(X); return p[:, 1] if p.ndim == 2 else p
    if hasattr(mdl, "decision_function"):
        s = mdl.decision_function(X); s = np.asarray(s).ravel()
        return 1.0 / (1.0 + np.exp(-s))
    y = mdl.predict(X); return (np.asarray(y).ravel() >= 0.5).astype(float)

def predict_bundle(model_blob, feat_list, D: pd.DataFrame):
    X = D.reindex(columns=feat_list, fill_value=0.0).values
    bundles = model_blob["bundles"]
    if ("nonsp" in bundles) or ("sp" in bundles):
        sp_mask = (D["is_special"] == 1).values
        p = np.zeros(len(D), dtype=float)
        mdl_ns = _resolve_model(bundles, "nonsp" if "nonsp" in bundles else "all")
        idx_ns = np.where(~sp_mask)[0]
        if idx_ns.size: p[idx_ns] = _score_proba(mdl_ns, X[idx_ns])
        mdl_sp = _resolve_model(bundles, "sp" if "sp" in bundles else "all")
        idx_sp = np.where(sp_mask)[0]
        if idx_sp.size: p[idx_sp] = _score_proba(mdl_sp, X[idx_sp])
        return p
    mdl = _resolve_model(bundles, "all")
    return _score_proba(mdl, X)

def eval_k1(D: pd.DataFrame, scores: np.ndarray, sam_col: str | None):
    V = D.copy(); V["score"] = scores
    g = (V.sort_values(["date","score"], ascending=[True,False])
           .groupby("date", as_index=False).head(1))
    has_target = "target" in g.columns and g["target"].notna().any()
    p1_all = float(g["target"].mean()) if has_target else float("nan")
    p1_sp  = float(g[g["is_special"]==1]["target"].mean()) if has_target else float("nan")
    cum    = float(g[sam_col].sum()) if sam_col else float("nan")
    return p1_all, p1_sp, cum, g

def sweep_alphas(D: pd.DataFrame, p0: np.ndarray, p1: np.ndarray, step=0.05):
    is_sp = (D["is_special"] == 1).values
    best = {"sp":0.0,"ns":0.0,"p1":-1.0}
    for a_sp in np.arange(0.0, 1.0 + 1e-9, step):
        for a_ns in np.arange(0.0, 1.0 + 1e-9, step):
            blend = np.where(is_sp, a_sp*p1 + (1-a_sp)*p0, a_ns*p1 + (1-a_ns)*p0)
            p1_all, _, _, _ = eval_k1(D, blend, get_samai_col(D))
            if p1_all > best["p1"]:
                best = {"sp":float(a_sp), "ns":float(a_ns), "p1":float(p1_all)}
    return best

def resolve_window(days: int, end_date_str: str | None):
    end = (pd.to_datetime(end_date_str).normalize()
           if end_date_str else pd.to_datetime("today").normalize())
    start  = end - pd.Timedelta(days=days-1)
    cutoff = start - pd.Timedelta(days=1)
    return start, end, cutoff

def concat_csv_files(uploaded_files) -> pd.DataFrame:
    dfs = []
    for f in uploaded_files or []:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            st.error(f"{getattr(f,'name',str(f))}: 読み込み失敗 ({e})")
    if not dfs: return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def build_features_train(m, df: pd.DataFrame) -> tuple[pd.DataFrame, object, object]:
    D = df.copy()
    if hasattr(m, "add_calendar_features"):         D = m.add_calendar_features(D)
    if hasattr(m, "add_exante_features"):           D = m.add_exante_features(D)
    D = m.add_series_aggregates(D)
    D = m.add_neighbor_features(D)
    if hasattr(m, "add_wave_trend_features"):       D = m.add_wave_trend_features(D)
    if hasattr(m, "add_series_percentile_features"):D = m.add_series_percentile_features(D)
    if hasattr(m, "add_bayesian_machine_ctr"):      D = m.add_bayesian_machine_ctr(D)
    if hasattr(m, "add_day_state_features"):        D, sc, km = m.add_day_state_features(D, fit=True, n_clusters=5)
    else:                                           sc, km = None, None
    must = ["samai_lag1", "g_num_lag1", "avg_lag1"]
    if all(c in D.columns for c in must):
        D = m.drop_na_lag1_for_training(D)
    return D, sc, km

def build_features_val(m, df: pd.DataFrame, sc=None, km=None) -> pd.DataFrame:
    D = df.copy()
    if hasattr(m, "add_calendar_features"):         D = m.add_calendar_features(D)
    if hasattr(m, "add_exante_features"):           D = m.add_exante_features(D)
    D = m.add_series_aggregates(D)
    D = m.add_neighbor_features(D)
    if hasattr(m, "add_wave_trend_features"):       D = m.add_wave_trend_features(D)
    if hasattr(m, "add_series_percentile_features"):D = m.add_series_percentile_features(D)
    if hasattr(m, "add_bayesian_machine_ctr"):      D = m.add_bayesian_machine_ctr(D)
    if hasattr(m, "add_day_state_features"):
        D, _, _ = m.add_day_state_features(D, fit=True, n_clusters=5)
    return D

# ====== UI ======
st.title(APP_TITLE)

with st.sidebar:
    st.header("① モデル")
    base_model = st.file_uploader("BASE model_dir.joblib", type=["joblib","pkl"])
    drop_model = st.file_uploader("DROP model_dir.joblib", type=["joblib","pkl"])

    st.header("② データ（CSV）")
    tr_files = st.file_uploader("学習CSV v14（4機種まとめてOK）", type=["csv"], accept_multiple_files=True)
    te_files = st.file_uploader("テストCSV（*_test_ge_*_v14）", type=["csv"], accept_multiple_files=True)

    st.header("③ 期間設定")
    days = st.number_input("遡り日数（評価窓）", 7, 60, 14, 1)
    end_date = st.date_input("評価最終日（未指定なら当日）", value=None)
    warmup_days = st.number_input("ウォームアップ日数（学習直前履歴）", 0, 30, WARMUP_DAYS_DEFAULT, 1)

    st.header("④ ブレンド")
    sweep = st.toggle("α 自動スイープ", value=True)
    a_sp = st.slider("α (Special)", 0.0, 1.0, 0.25, 0.05, disabled=sweep)
    a_ns = st.slider("α (Non-Special)", 0.0, 1.0, 0.75, 0.05, disabled=sweep)

run = st.button("予測を実行", type="primary")

if run:
    try:
        if base_model is None or drop_model is None:
            st.error("BASE / DROP モデルをアップロードしてください。"); st.stop()
        if not tr_files or not te_files:
            st.error("学習CSV（v14）とテストCSV（*_test_ge_*_v14）をアップロードしてください。"); st.stop()

        # モデル＆特徴量
        b0, b1 = jb.load(base_model), jb.load(drop_model)
        F0, F1 = list(b0["feat"]), list(b1["feat"])

        df_train = concat_csv_files(tr_files)
        df_test  = concat_csv_files(te_files)
        if df_train.empty or df_test.empty:
            st.error("CSVの読み込みに失敗しました。"); st.stop()

        df_train = ensure_min_schema(df_train)
        df_test  = ensure_min_schema(df_test)

        # 評価窓
        end_str = None if end_date is None else pd.to_datetime(end_date).strftime("%Y-%m-%d")
        start, end, cutoff = resolve_window(int(days), end_str)

        mask_tr = (df_train["date"] <= cutoff) | (
            (df_train["date"] < start) & (df_train["date"] >= (start - pd.Timedelta(days=int(warmup_days))))
        )
        trn = df_train[mask_tr].copy()
        val = df_test[(df_test["date"] >= start) & (df_test["date"] <= end)].copy()
        if val.empty:
            st.error(f"評価期間 {start.date()}..{end.date()} でテストCSVにデータがありません。"); st.stop()

        # 特徴量
        m = load_module(PK_FILE)
        st.write(f"[WINDOW] train <= {cutoff.date()} | eval {start.date()}..{end.date()}")
        trn, sc, km = build_features_train(m, trn)
        val = build_features_val(m, val, sc=sc, km=km)

        # 予測
        p0 = predict_bundle(b0, F0, val)
        p1 = predict_bundle(b1, F1, val)
        sam_col = get_samai_col(val)

        ba, bs, bc, _ = eval_k1(val, p0, sam_col)
        da, ds, dc, _ = eval_k1(val, p1, sam_col)
        st.write(f"BASE  P1_ALL={ba:.4f} SP={bs:.4f} CUM={int(bc)}")
        st.write(f"DROP  P1_ALL={da:.4f} SP={ds:.4f} CUM={int(dc)}")

        # α
        if sweep:
            best = sweep_alphas(val, p0, p1, step=0.05)
            st.write(f"[SWEEP] Best α => SP:{best['sp']:.2f}  NS:{best['ns']:.2f}  (metric=P1)")
            alpha_sp, alpha_ns = best["sp"], best["ns"]
        else:
            alpha_sp, alpha_ns = float(a_sp), float(a_ns)
            st.write(f"[BLEND] Fixed α => SP:{alpha_sp:.2f}  NS:{alpha_ns:.2f}")

        is_sp = (val["is_special"] == 1).values
        final = np.where(is_sp, alpha_sp*p1 + (1-alpha_sp)*p0,
                               alpha_ns*p1 + (1-alpha_ns)*p0)

        fa, fs, fc, _ = eval_k1(val, final, sam_col)
        st.success(f"BLEND P1_ALL={fa:.4f} SP={fs:.4f} CUM={int(fc)}")

        out = val.copy()
        out["pred_base"]  = p0
        out["pred_drop"]  = p1
        out["pred_blend"] = final

        top1 = (out.sort_values(["date","pred_blend"], ascending=[True, False])
                    .groupby("date", as_index=False).head(1))

        cols = ["date","series","num"]
        if sam_col: cols.append(sam_col)
        for c in ["target","target_eval","pred_base","pred_drop","pred_blend"]:
            if c in top1.columns: cols.append(c)

        st.subheader("K=1（各日トップ）")
        st.dataframe(top1[cols], use_container_width=True)

        csv_bytes = top1.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSVダウンロード（top1_by_date_blend.csv）",
                           data=csv_bytes, file_name="top1_by_date_blend.csv",
                           mime="text/csv")
    except Exception as e:
        st.exception(e)
