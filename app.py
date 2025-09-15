# -*- coding: utf-8 -*-
# Streamlit — K=1ミニマル配信UI（特日/角バッジ・月次チャート・軽量認証・CSS幅固定）
# ルール厳守:
# - 相対パスのみ（絶対パス禁止）
# - st.set_page_config は最上部で一度だけ
# - inplace=False
# - ユーザー画面は「日付・機種・台番号」中心。内部CSVのUpload/Downloadは出さない

from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st

# ===== ページ設定（先頭で一度だけ） =====
st.set_page_config(page_title="K=1日次予測", page_icon="🎯", layout="wide")

# ===== 軽量認証（Secrets） =====
# Secrets 例（.streamlit/secrets.toml）:
# APP_PASSCODE = "123456"
# ALLOWED_EMAILS = ["you@example.com", "ops@example.com"]
# BASE_MODEL_DIR = "models/out_series_dual_v14_7_nf_plus_p90_pct_corner/model_dir.joblib"
# DROP_MODEL_DIR = "models/out_series_dual_v14_7_dropcand_v1/model_dir.joblib"
# PICKS_DIR = "outputs/picks"
# METRICS_FILE = "outputs/metrics/metrics_history.csv"

def _get_allowed_emails():
    v = st.secrets.get("ALLOWED_EMAILS", [])
    if isinstance(v, str):
        v = [x.strip() for x in v.split(",") if x.strip()]
    return set(v)

def require_auth():
    allowed = _get_allowed_emails()
    passcode = st.secrets.get("APP_PASSCODE", None)
    if not allowed and not passcode:
        st.info("デモモード（認証無効）。本番は Secrets に ALLOWED_EMAILS と APP_PASSCODE を設定してください。")
        return
    with st.sidebar:
        st.header("認証")
        email = st.text_input("メール", value="")
        code = st.text_input("パスコード", type="password", value="")
        ok = st.button("入室")
    if ok:
        email_ok = True if not allowed else (email in allowed)
        code_ok = True if not passcode else (str(code) == str(passcode))
        st.session_state["_authed"] = bool(email_ok and code_ok)
        if not st.session_state["_authed"]:
            st.error("認証失敗：許可メールまたはパスコードが不正です。")
    if not st.session_state.get("_authed", False):
        st.stop()

# ===== サイドバーCSS（幅固定） =====
def inject_sidebar_css(width_px=320):
    css = "<style>[data-testid='stSidebar']{min-width:%dpx;width:%dpx;}</style>" % (width_px, width_px)
    st.markdown(css, unsafe_allow_html=True)

# ===== 設定（Secrets優先・相対パスのみ） =====
BASE_MODEL_DIR = st.secrets.get("BASE_MODEL_DIR", "models/out_series_dual_v14_7_nf_plus_p90_pct_corner/model_dir.joblib")
DROP_MODEL_DIR = st.secrets.get("DROP_MODEL_DIR", "models/out_series_dual_v14_7_dropcand_v1/model_dir.joblib")
PICKS_DIR      = Path(st.secrets.get("PICKS_DIR", "outputs/picks"))
METRICS_FILE   = Path(st.secrets.get("METRICS_FILE", "outputs/metrics/metrics_history.csv"))

# ===== 画面ヘッダ =====
require_auth()
inject_sidebar_css(320)
st.title("K=1 日次予測（プラザ博多 4機種）")

# ===== サイドバー：対象日 =====
with st.sidebar:
    st.header("対象日")
    target_dt = st.date_input("日付", date.today())
    st.caption("K=1の『日付・機種・台番号』のみ表示（特日/角バッジ付き）。")

# ===== データ読込 =====
def _coerce_bool_int(s):
    try:
        return s.astype("int64")
    except Exception:
        return s.fillna(0).astype("int64")

def load_k1_picks_for_day(dt: date) -> pd.DataFrame:
    # 既定: outputs/picks/YYYY-MM-DD_k1.csv
    fp = PICKS_DIR / (str(dt) + "_k1.csv")
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)

    # 必須列: date, series, num
    must = {"date", "series", "num"}
    missing = sorted(list(must - set(df.columns)))
    if missing:
        raise RuntimeError("必須列が不足: " + ", ".join(missing))

    # 型整備
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["series"] = d["series"].astype(str)
    d["num"] = pd.to_numeric(d["num"], errors="coerce").astype("Int64")

    # バッジ用（無ければ0で生成）
    if "is_special" not in d.columns:
        d["is_special"] = 0
    if "is_corner" not in d.columns:
        d["is_corner"] = 0
    d["is_special"] = _coerce_bool_int(d["is_special"])
    d["is_corner"]  = _coerce_bool_int(d["is_corner"])

    # 複数日が入っている場合の安全対策：対象日で絞る
    day_mask = d["date"].dt.date == dt
    d = d[day_mask].copy()

    # 去重（最新を残す）
    d = d.sort_values(["date"]).drop_duplicates(subset=["date", "series", "num"], keep="last")
    return d

# ===== 月次KPI集計 =====
def compute_monthly_kpis(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    g = df.groupby("month", as_index=False)
    out = g.agg({
        "p_at_1": "mean",
        "cum_samai": "sum",
        "n_picked": "sum",
        "n_positives": "sum",
    })
    return out

# ===== 表示本体 =====
def render_badge(row):
    tags = []
    if int(row.get("is_special", 0)) == 1:
        tags.append("特日")
    if int(row.get("is_corner", 0)) == 1:
        tags.append("角")
    if not tags:
        return ""
    return " ".join(["[" + x + "]" for x in tags])

def render_k1_table(dt: date):
    try:
        picks = load_k1_picks_for_day(dt)
        if picks.empty:
            st.warning("対象日のK=1結果ファイルが見つかりません。日次バッチ（run_blend_daily.py）の完了を確認してください。")
            return

        # 自己検証ログ
        n_rows = len(picks)
        day_min = str(picks["date"].min().date()) if not picks.empty else "-"
        day_max = str(picks["date"].max().date()) if not picks.empty else "-"
        st.caption("自己検証: rows=%d, date_range=%s..%s" % (n_rows, day_min, day_max))

        view = picks.copy()
        view["badge"] = view.apply(render_badge, axis=1)
        view = view[["date", "series", "num", "badge"]].rename(columns={"num": "台番号"})
        st.subheader(str(dt) + " のK=1")
        st.dataframe(view, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error("表示処理で例外が発生しました。内部ログを確認してください。")
        st.exception(e)

render_k1_table(target_dt)

# ===== 月次チャート =====
if METRICS_FILE.exists():
    try:
        monthly = compute_monthly_kpis(METRICS_FILE)
        if monthly.empty:
            st.info("メトリクス履歴が空です。日次バッチ完了後に表示されます。")
        else:
            st.subheader("月次サマリ：P@1（平均）")
            st.line_chart(monthly.set_index("month")["p_at_1"], height=240)
            st.subheader("月次サマリ：累積差枚（合計）")
            st.line_chart(monthly.set_index("month")["cum_samai"], height=240)
    except Exception as e:
        st.warning("月次可視化に失敗しました。")
        st.exception(e)
else:
    st.info("メトリクス履歴が未作成です（outputs/metrics/metrics_history.csv）。初回日次完了後に表示されます。")
