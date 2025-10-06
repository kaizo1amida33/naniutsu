#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workflow_predict_777_fixed.py
スクレイピングデータから明日の予測を実行
"""
import argparse, os, sys, subprocess, shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

JST = timezone(timedelta(hours=9))

def log(msg: str):
    ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def get_next_prediction_date(test_file: str) -> str:
    """テストCSVの最終日+1日を取得"""
    try:
        df = pd.read_csv(test_file)
        date_col = 'date'
        if date_col not in df.columns:
            date_col = [c for c in df.columns if 'date' in c.lower()][0]
        
        df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
        max_date = df['date_parsed'].max()
        next_date = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")
        log(f"[DATE] Test max: {max_date.date()} → Next: {next_date}")
        return next_date
    except Exception as e:
        log(f"[WARN] Failed to get date from {test_file}: {e}")
        return datetime.now(JST).strftime("%Y-%m-%d")

def find_test_files() -> list:
    """スクレイピング済みのテストファイルを検索"""
    test_files = []
    for series in ['hokuto', 'monkey', 'ghoul', 'myjugglerV']:
        pattern = f"{series}_test_ge_"
        files = list(Path('.').glob(f"{pattern}*_v14.csv"))
        if files:
            latest = max(files, key=lambda p: p.stat().st_mtime)
            test_files.append(str(latest))
            log(f"[FOUND] {series}: {latest.name}")
    return test_files

def create_stub_csv(test_file: str, target_date: str, output_path: str) -> str:
    """テストファイルから次の日のスタブCSVを作成"""
    try:
        df = pd.read_csv(test_file, dtype=str)
        log(f"[READ] {test_file}: {len(df)} rows")
        
        date_col = 'date'
        if date_col not in df.columns:
            possible = [c for c in df.columns if 'date' in c.lower()]
            if possible:
                date_col = possible[0]
            else:
                raise ValueError(f"日付列が見つかりません: {test_file}")
        
        df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
        max_date = df['date_parsed'].max()
        latest_data = df[df['date_parsed'] == max_date].copy()
        
        log(f"[LATEST] {max_date.date()}: {len(latest_data)} machines")
        
        next_date = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")
        log(f"[NEXT] Stub date: {next_date}")
        
        stub = pd.DataFrame()
        stub['date'] = [next_date] * len(latest_data)
        stub['num'] = latest_data['num'].values
        stub['samai'] = 0
        stub['g_num'] = 0
        stub['avg'] = 0
        
        if 'series' in latest_data.columns:
            stub['series'] = latest_data['series'].values
        else:
            series_name = Path(test_file).stem.split('_')[0]
            stub['series'] = series_name
        
        stub = stub[['date', 'num', 'samai', 'g_num', 'avg', 'series']]
        stub.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        log(f"[STUB] Created: {output_path} ({len(stub)} rows, date={next_date})")
        return output_path
        
    except Exception as e:
        log(f"[ERROR] Failed to create stub from {test_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_warmup_files(series_list: list) -> list:
    """全履歴から直近14日分のwarmupファイルを作成"""
    warmup_files = []
    
    for series in series_list:
        all_file = f"{series}_plazahakata_all_days_v14.csv"
        warmup_file = f"{series}_warmup_14days.csv"
        
        if not os.path.exists(all_file):
            log(f"[WARN] {all_file} not found, skipping warmup")
            continue
        
        try:
            df = pd.read_csv(all_file)
            df['date'] = pd.to_datetime(df['date'])
            
            num_machines = df['num'].nunique()
            recent = df.sort_values('date').tail(14 * num_machines)
            
            recent.to_csv(warmup_file, index=False, encoding='utf-8-sig')
            warmup_files.append(warmup_file)
            log(f"[WARMUP] {warmup_file}: {len(recent)} rows")
            
        except Exception as e:
            log(f"[ERROR] Failed to create warmup for {series}: {e}")
    
    return warmup_files

def run_prediction(naniutsu: str, model_dir: str, history_files: list, 
                  stub_files: list, out_dir: str, target_date: str) -> bool:
    """naniutsu_v7.pyでスコアリング実行"""
    
    if not stub_files:
        log("[ERROR] No stub files to predict")
        return False
    
    cmd = [
        sys.executable, naniutsu, "score",
        "--model_dir", model_dir,
        "--history_files", ",".join(history_files),
        "--test_files", ",".join(stub_files),
        "--production_mode",
        "--target_date", target_date,
        "--out_dir", out_dir,
        "--export_features",
    ]
    
    log(f"[CMD] {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode != 0:
            log(f"[ERROR] Prediction failed with code {result.returncode}")
            return False
        
        log("[SUCCESS] Prediction completed")
        return True
        
    except subprocess.TimeoutExpired:
        log("[ERROR] Prediction timeout (5 minutes)")
        return False
    except Exception as e:
        log(f"[ERROR] Prediction failed: {e}")
        return False

def find_prediction_csv(out_dir: str) -> str:
    """予測結果CSVを検索"""
    all_csvs = list(Path(out_dir).glob("*.csv"))
    log(f"[INFO] Found {len(all_csvs)} CSV files in {out_dir}")
    
    best_file = None
    max_rows = 0
    
    for path in all_csvs:
        try:
            if path.stat().st_size < 100:
                log(f"[SKIP] {path.name}: too small")
                continue
            
            df = pd.read_csv(path, nrows=5)
            cols = set(df.columns)
            log(f"[CHECK] {path.name}: columns={list(cols)[:5]}...")
            
            has_score = 'score' in cols
            has_date = 'date' in cols
            has_series = 'series' in cols
            has_id = 'machine_id' in cols or 'num' in cols
            
            if has_score and has_id:
                full_df = pd.read_csv(path)
                rows = len(full_df)
                log(f"[VALID] {path.name}: {rows} rows with predictions")
                
                if rows > max_rows:
                    max_rows = rows
                    best_file = str(path)
            elif has_date and has_series and has_id and len(cols) > 10:
                full_df = pd.read_csv(path)
                rows = len(full_df)
                log(f"[CANDIDATE] {path.name}: {rows} rows (feature file)")
                
                score_cols = [c for c in cols if any(x in c.lower() for x in ['score', 'pred', 'prob'])]
                if score_cols:
                    log(f"[FOUND SCORE] {path.name}: using column '{score_cols[0]}'")
                    if rows > max_rows:
                        max_rows = rows
                        best_file = str(path)
            else:
                log(f"[SKIP] {path.name}: score={has_score}, id={has_id}, cols={len(cols)}")
                
        except Exception as e:
            log(f"[ERROR] Failed to read {path.name}: {e}")
            continue
    
    if best_file:
        log(f"[SELECTED] {Path(best_file).name} with {max_rows} predictions")
        return best_file
    
    log("[ERROR] No valid prediction file found")
    return None

def normalize_prediction_output(pred_file: str, output_file: str, all_output_file: str):
    """予測結果をpredict_777.csv形式に正規化し、all_predictions.csvも出力"""
    try:
        df = pd.read_csv(pred_file)
        
        log(f"[DEBUG] Input file: {Path(pred_file).name}")
        log(f"[DEBUG] Shape: {df.shape}, columns: {list(df.columns)}")
        
        if len(df) == 1 and 'topk' in pred_file.lower():
            log("[WARN] Input file contains only K=1. Need naniutsu_v7.py modification to output all predictions.")
        
        score_col = None
        for col in ['score', 'prediction_score', 'pred', 'prob', 'probability']:
            if col in df.columns:
                score_col = col
                log(f"[SCORE] Using column: {score_col}")
                break
        
        if score_col is None:
            log("[ERROR] No score column found")
            return False
        
        if 'date' not in df.columns:
            log("[INFO] Adding date from stub")
            stub_files = list(Path('.').glob('*_stub.csv'))
            if stub_files:
                stub_df = pd.read_csv(stub_files[0])
                df['date'] = stub_df['date'].iloc[0]
        
        if 'series' not in df.columns:
            log("[WARN] No series column, trying to extract from machine_id")
            if 'machine_id' in df.columns:
                df['series'] = df['machine_id'].str.split('_').str[-1]
        
        if 'machine_id' in df.columns:
            if 'num' not in df.columns:
                df['num'] = df['machine_id'].str.extract(r'^(\d+)', expand=False)
        elif 'num' in df.columns:
            df['machine_id'] = df['num'].astype(str) + '_' + df.get('series', 'unknown')
        else:
            log("[ERROR] No machine_id or num column")
            return False
        
        # スコア列を正規化
        df = df.rename(columns={score_col: 'score'})
        
        # K3用（上位3件）
        df_sorted = df.sort_values('score', ascending=False)
        k = min(3, len(df_sorted))
        result_k3 = df_sorted.head(k)
        
        out_cols = ['date', 'series', 'num', 'machine_id', 'score']
        result_k3 = result_k3[[c for c in out_cols if c in result_k3.columns]]
        result_k3.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        pred_date = result_k3['date'].iloc[0] if 'date' in result_k3.columns else 'N/A'
        log(f"[OUTPUT] {output_file}: {len(result_k3)} predictions (K={k}) for {pred_date}")
        
        # 全台データ（all_predictions.csv）
        df_all = df_sorted[[c for c in out_cols if c in df_sorted.columns]]
        df_all.to_csv(all_output_file, index=False, encoding='utf-8-sig')
        log(f"[OUTPUT] {all_output_file}: {len(df_all)} predictions (all machines) for {pred_date}")
        
        print(f"\n=== K={k} 予測結果 ===")
        for idx, row in result_k3.iterrows():
            date_str = row.get('date', 'N/A')
            num_str = row.get('num', row.get('machine_id', 'N/A'))
            series_str = row.get('series', 'N/A')
            print(f"[{date_str}] {series_str:<12} 台番={num_str:<6} score={row['score']:.4f}")
        
        return True
        
    except Exception as e:
        log(f"[ERROR] Failed to normalize output: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Daily prediction workflow")
    parser.add_argument("--naniutsu", default="naniutsu_v7.py")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_dir", default="prediction_output")
    args = parser.parse_args()
    
    log("=== Prediction Workflow Start ===")
    
    test_files = find_test_files()
    if not test_files:
        log("[ERROR] No test files found. Run scraping first.")
        sys.exit(1)
    
    stub_files = []
    for test_file in test_files:
        series_name = Path(test_file).stem.split('_')[0]
        stub_path = f"{series_name}_stub.csv"
        stub = create_stub_csv(test_file, None, stub_path)
        if stub:
            stub_files.append(stub)
    
    if not stub_files:
        log("[ERROR] Failed to create stub files")
        sys.exit(1)
    
    log("\n=== Stub Preview ===")
    for stub_file in stub_files:
        df_preview = pd.read_csv(stub_file)
        log(f"{stub_file}: date={df_preview['date'].iloc[0]}, machines={len(df_preview)}")
        print(df_preview.head(3).to_string(index=False))
        print()
    
    # シリーズリスト定義
    series_list = ['hokuto', 'monkey', 'ghoul', 'myjugglerV']
    
    # 既存の固定warmupファイルを使用
    warmup_files = []
    for series in series_list:
        warmup_file = f"{series}_warmup_14days.csv"
        if os.path.exists(warmup_file):
            warmup_files.append(warmup_file)
            log(f"[WARMUP] Found {warmup_file}")
        else:
            log(f"[WARN] {warmup_file} not found")
    
    # warmup（固定：8/03-8/16） + test（更新：8/17-10/05）を履歴として使用
    if warmup_files:
        history_files = warmup_files + test_files
        log(f"[HISTORY] Using {len(warmup_files)} warmup files + {len(test_files)} test files")
    else:
        history_files = test_files
        log(f"[HISTORY] Using {len(test_files)} test files only (no warmup found)")
    
    target_date = pd.read_csv(stub_files[0])['date'].iloc[0]
    log(f"[TARGET DATE] {target_date}")
    
    success = run_prediction(
        naniutsu=args.naniutsu,
        model_dir=args.model_dir,
        history_files=history_files,
        stub_files=stub_files,
        out_dir=args.out_dir,
        target_date=target_date
    )
    
    if not success:
        log("[ERROR] Prediction failed")
        sys.exit(1)
    
    pred_file = find_prediction_csv(args.out_dir)
    if not pred_file:
        log("[ERROR] No prediction output found")
        sys.exit(1)
    
    output_file = "predict_777.csv"
    all_output_file = "all_predictions.csv"
    
    success = normalize_prediction_output(pred_file, output_file, all_output_file)
    
    if success:
        log(f"[SUCCESS] {output_file} and {all_output_file} created")
    else:
        log("[ERROR] Failed to create final output")
        sys.exit(1)
    
    for stub_file in stub_files:
        try:
            os.remove(stub_file)
            log(f"[CLEANUP] Removed {stub_file}")
        except:
            pass
    
    log("=== Workflow Complete ===")

if __name__ == "__main__":
    main()
