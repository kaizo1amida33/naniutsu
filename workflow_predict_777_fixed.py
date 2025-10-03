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
        # フォールバック: 今日の日付
        return datetime.now(JST).strftime("%Y-%m-%d")

def find_test_files() -> list:
    """スクレイピング済みのテストファイルを検索"""
    test_files = []
    for series in ['hokuto', 'monkey', 'ghoul', 'myjugglerV']:
        pattern = f"{series}_test_ge_"
        files = list(Path('.').glob(f"{pattern}*_v14.csv"))
        if files:
            # 最新のファイルを選択
            latest = max(files, key=lambda p: p.stat().st_mtime)
            test_files.append(str(latest))
            log(f"[FOUND] {series}: {latest.name}")
    return test_files

def create_stub_csv(test_file: str, target_date: str, output_path: str) -> str:
    """テストファイルから次の日のスタブCSVを作成（date,num,seriesのみ）"""
    try:
        df = pd.read_csv(test_file, dtype=str)
        log(f"[READ] {test_file}: {len(df)} rows")
        
        # 日付列の正規化
        date_col = 'date'
        if date_col not in df.columns:
            possible = [c for c in df.columns if 'date' in c.lower()]
            if possible:
                date_col = possible[0]
            else:
                raise ValueError(f"日付列が見つかりません: {test_file}")
        
        # 最終日のデータを抽出
        df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
        max_date = df['date_parsed'].max()
        latest_data = df[df['date_parsed'] == max_date].copy()
        
        log(f"[LATEST] {max_date.date()}: {len(latest_data)} machines")
        
        # 次の日の日付を計算
        next_date = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")
        log(f"[NEXT] Stub date: {next_date}")
        
        # 必須列を含むスタブ作成（samai, g_num, avgは0で埋める）
        stub = pd.DataFrame()
        stub['date'] = [next_date] * len(latest_data)
        stub['num'] = latest_data['num'].values
        stub['samai'] = 0  # ダミー値
        stub['g_num'] = 0  # ダミー値
        stub['avg'] = 0    # ダミー値
        
        # series列の取得
        if 'series' in latest_data.columns:
            stub['series'] = latest_data['series'].values
        else:
            # ファイル名から推定
            series_name = Path(test_file).stem.split('_')[0]
            stub['series'] = series_name
        
        # 列順を固定: date, num, samai, g_num, avg, series
        stub = stub[['date', 'num', 'samai', 'g_num', 'avg', 'series']]
        
        # 上書き保存
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
            
            # 直近14日分（各台ごとに14レコード）
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
        "--export_features",  # 全台データを出力
    ]
    
    log(f"[CMD] {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # ログ出力
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
    """予測結果CSVを検索（全台データを含むファイルを優先）"""
    
    # まずディレクトリ内の全CSVを確認
    all_csvs = list(Path(out_dir).glob("*.csv"))
    log(f"[INFO] Found {len(all_csvs)} CSV files in {out_dir}")
    
    best_file = None
    max_rows = 0
    
    for path in all_csvs:
        try:
            # ファイルサイズチェック（空ファイル除外）
            if path.stat().st_size < 100:
                log(f"[SKIP] {path.name}: too small")
                continue
            
            df = pd.read_csv(path, nrows=5)
            
            # 列名を確認
            cols = set(df.columns)
            log(f"[CHECK] {path.name}: columns={list(cols)[:5]}...")
            
            # 必要な列があるか確認
            has_score = 'score' in cols
            has_date = 'date' in cols
            has_series = 'series' in cols
            has_id = 'machine_id' in cols or 'num' in cols
            
            # scoreがある場合のみ考慮（予測結果）
            if has_score and has_id:
                full_df = pd.read_csv(path)
                rows = len(full_df)
                log(f"[VALID] {path.name}: {rows} rows with predictions")
                
                if rows > max_rows:
                    max_rows = rows
                    best_file = str(path)
            # scoreがなくても、日付・シリーズ・台番・特徴量があれば使える
            elif has_date and has_series and has_id and len(cols) > 10:
                full_df = pd.read_csv(path)
                rows = len(full_df)
                log(f"[CANDIDATE] {path.name}: {rows} rows (feature file)")
                
                # スコア列を探す（pred, prediction, prob等）
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

def normalize_prediction_output(pred_file: str, output_path: str):
    """予測結果をpredict_777.csv形式に正規化（日付は元データから取得）"""
    try:
        df = pd.read_csv(pred_file)
        
        log(f"[DEBUG] Input file: {Path(pred_file).name}")
        log(f"[DEBUG] Shape: {df.shape}, columns: {list(df.columns)}")
        
        # test_topk_k1.csv（1台のみ）の場合の警告
        if len(df) == 1 and 'topk' in pred_file.lower():
            log("[WARN] Input file contains only K=1. Need naniutsu_v7.py modification to output all predictions.")
            log("[WARN] Current workaround: outputting single prediction as K=1")
        
        # スコア列を探す
        score_col = None
        for col in ['score', 'prediction_score', 'pred', 'prob', 'probability']:
            if col in df.columns:
                score_col = col
                log(f"[SCORE] Using column: {score_col}")
                break
        
        if score_col is None:
            log("[ERROR] No score column found")
            log(f"[DEBUG] Available columns: {list(df.columns)}")
            return False
        
        # date列の確認
        if 'date' not in df.columns:
            log("[INFO] Adding date from stub")
            stub_files = list(Path('.').glob('*_stub.csv'))
            if stub_files:
                stub_df = pd.read_csv(stub_files[0])
                df['date'] = stub_df['date'].iloc[0]
        
        # series列の確認
        if 'series' not in df.columns:
            log("[WARN] No series column, trying to extract from machine_id")
            if 'machine_id' in df.columns:
                df['series'] = df['machine_id'].str.split('_').str[-1]
        
        # machine_id or num
        if 'machine_id' in df.columns:
            if 'num' not in df.columns:
                df['num'] = df['machine_id'].str.extract(r'^(\d+)', expand=False)
        elif 'num' in df.columns:
            df['machine_id'] = df['num'].astype(str) + '_' + df.get('series', 'unknown')
        else:
            log("[ERROR] No machine_id or num column")
            return False
        
        # スコアでソートして上位K=3を取得（可能な場合）
        df_sorted = df.sort_values(score_col, ascending=False)
        
        k = min(3, len(df_sorted))
        result = df_sorted.head(k)
        log(f"[INFO] Selected top {k} from {len(df_sorted)} predictions")
        
        # 出力用にリネーム
        result = result.rename(columns={score_col: 'score'})
        
        # 出力列を整理
        out_cols = ['date', 'series', 'num', 'machine_id', 'score']
        result = result[[c for c in out_cols if c in result.columns]]
        result.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        pred_date = result['date'].iloc[0] if 'date' in result.columns else 'N/A'
        log(f"[OUTPUT] {output_path}: {len(result)} predictions (K={k}) for {pred_date}")
        
        # 結果表示
        print(f"\n=== K={k} 予測結果 ===")
        if k < 3:
            print(f"[注意] naniutsu_v7.pyの出力が{len(df)}台のため、K={k}となっています")
            print("[推奨] naniutsu_v7.pyで all_predictions.csv を出力するように修正してください")
        
        for idx, row in result.iterrows():
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
    
    # 1. テストファイル検索
    test_files = find_test_files()
    if not test_files:
        log("[ERROR] No test files found. Run scraping first.")
        sys.exit(1)
    
    # 2. スタブCSV作成（各シリーズごとに上書き保存）
    stub_files = []
    for test_file in test_files:
        # シリーズ名を取得
        series_name = Path(test_file).stem.split('_')[0]
        stub_path = f"{series_name}_stub.csv"  # 上書き用の固定ファイル名
        
        stub = create_stub_csv(test_file, None, stub_path)
        if stub:
            stub_files.append(stub)
    
    if not stub_files:
        log("[ERROR] Failed to create stub files")
        sys.exit(1)
    
    # スタブの内容確認
    log("\n=== Stub Preview ===")
    for stub_file in stub_files:
        df_preview = pd.read_csv(stub_file)
        log(f"{stub_file}: date={df_preview['date'].iloc[0]}, machines={len(df_preview)}")
        print(df_preview.head(3).to_string(index=False))
        print()
    
    # 3. Warmupファイル作成
    series_list = ['hokuto', 'monkey', 'ghoul', 'myjugglerV']
    warmup_files = create_warmup_files(series_list)
    
    if not warmup_files:
        log("[WARN] No warmup files created, using stubs only")
        warmup_files = stub_files  # フォールバック
    
    # 4. 予測対象日を取得（最初のスタブから）
    target_date = pd.read_csv(stub_files[0])['date'].iloc[0]
    log(f"[TARGET DATE] {target_date}")
    
    # 5. 予測実行
    success = run_prediction(
        naniutsu=args.naniutsu,
        model_dir=args.model_dir,
        history_files=warmup_files,
        stub_files=stub_files,
        out_dir=args.out_dir,
        target_date=target_date
    )
    
    if not success:
        log("[ERROR] Prediction failed")
        sys.exit(1)
    
    # 6. 結果ファイル検索
    pred_file = find_prediction_csv(args.out_dir)
    if not pred_file:
        log("[ERROR] No prediction output found")
        sys.exit(1)
    
    # 7. 結果を正規化してpredict_777.csvに出力（上書き）
    output_file = "predict_777.csv"
    success = normalize_prediction_output(pred_file, output_file)
    
    if success:
        log(f"[SUCCESS] {output_file} created")
    else:
        log("[ERROR] Failed to create final output")
        sys.exit(1)
    
    # 8. スタブファイル削除（クリーンアップ）
    for stub_file in stub_files:
        try:
            os.remove(stub_file)
            log(f"[CLEANUP] Removed {stub_file}")
        except:
            pass
    
    log("=== Workflow Complete ===")

if __name__ == "__main__":
    main()