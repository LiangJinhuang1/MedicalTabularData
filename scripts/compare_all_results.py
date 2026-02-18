#!/usr/bin/env python3
"""汇总所有 output 下的 best_results.csv，结合 losses.txt 与 log 检查 loss 是否收敛。"""
import statistics
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
LAST_N_EPOCHS = 50
VAL_LOSS_STD_THRESH = 0.02
MIN_EPOCHS_FOR_CONVERGENCE = 100

def parse_float(s):
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "N/A" or s == "":
        return None
    if isinstance(s, str) and s.strip().lower() == "nan":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def check_log_complete(experiment_dir):
    log_file = experiment_dir / "logs" / "train.log"
    if not log_file.exists():
        return False
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return "Training complete!" in content or "training complete" in content.lower()
    except Exception:
        return False

def check_loss_converged(experiment_dir, best_model_name):
    """解析 losses.txt 中 best_model 的 Val 列，判断最后一段是否收敛。返回 (converged, msg)。"""
    losses_file = experiment_dir / "losses.txt"
    if not losses_file.exists():
        return False, "no losses.txt"
    try:
        with open(losses_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception as e:
        return False, str(e)
    if len(lines) < 2:
        return False, "empty"
    sep = " | "
    header = lines[0].split(sep)
    model_upper = best_model_name.upper().replace("-", "_")
    val_col_idx = None
    for i, h in enumerate(header):
        if h.strip() == (model_upper + " Val"):
            val_col_idx = i
            break
    if val_col_idx is None:
        return False, "model not in losses header"
    val_losses = []
    for ln in lines[1:]:
        if ln.startswith("-") or ln.startswith("=") or "BEST" in ln.upper():
            continue
        parts = ln.split(sep)
        if len(parts) <= val_col_idx:
            continue
        v = parse_float(parts[val_col_idx].strip())
        if v is not None and 0 < v < 1e6:
            val_losses.append(v)
    if len(val_losses) < MIN_EPOCHS_FOR_CONVERGENCE:
        return False, "epochs=%d" % len(val_losses)
    n = len(val_losses)
    last_n = min(LAST_N_EPOCHS, n // 2)
    if last_n < 10:
        return False, "tail too short"
    tail = val_losses[-last_n:]
    std_tail = statistics.stdev(tail) if len(tail) > 1 else 0.0
    mean_tail = statistics.mean(tail)
    converged = std_tail <= VAL_LOSS_STD_THRESH or (mean_tail < 0.5 and std_tail <= 0.05)
    msg = "epochs=%d std=%.4f mean=%.4f" % (n, std_tail, mean_tail)
    return converged, msg

def get_run_best(csv_path: Path) -> dict:
    """读取单个 best_results.csv，返回该 run 的最佳指标；并检查 losses.txt 与 log。"""
    df = pd.read_csv(csv_path)
    run_name = csv_path.parent.name
    experiment_dir = csv_path.parent

    r2_col = "Val R2 (Best)" if "Val R2 (Best)" in df.columns else None
    auc_col = "Val AUC (Best)" if "Val AUC (Best)" in df.columns else None
    loss_col = "Val Loss (Best)" if "Val Loss (Best)" in df.columns else None

    best_r2 = None
    best_auc = None
    best_loss = None
    best_r2_model = None
    best_auc_model = None

    for _, row in df.iterrows():
        model = row.get("Model", "")
        if r2_col:
            v = parse_float(row.get(r2_col))
            if v is not None and (best_r2 is None or v > best_r2):
                best_r2 = v
                best_r2_model = model
        if auc_col:
            v = parse_float(row.get(auc_col))
            if v is not None and (best_auc is None or v > best_auc):
                best_auc = v
                best_auc_model = model
        if loss_col:
            v = parse_float(row.get(loss_col))
            if v is not None and v > 0 and (best_loss is None or v < best_loss):
                best_loss = v

    is_classification = "LACa" in run_name or "LACm" in run_name
    if is_classification and best_auc is not None:
        primary_metric = "AUC"
        primary_value = best_auc
        primary_model = best_auc_model
    else:
        primary_metric = "R2"
        primary_value = best_r2
        primary_model = best_r2_model

    log_ok = check_log_complete(experiment_dir)
    loss_ok, loss_msg = check_loss_converged(experiment_dir, primary_model) if primary_model else (False, "no model")

    return {
        "run": run_name,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "primary_model": primary_model,
        "best_r2": best_r2,
        "best_auc": best_auc,
        "best_loss": best_loss,
        "log_complete": log_ok,
        "loss_converged": loss_ok,
        "loss_converged_msg": loss_msg,
    }

def main():
    rows = []
    for sub in OUTPUT_DIR.iterdir():
        if not sub.is_dir():
            continue
        csv_path = sub / "best_results.csv"
        if not csv_path.exists():
            continue
        try:
            row = get_run_best(csv_path)
            if row["primary_value"] is not None:
                rows.append(row)
        except Exception as e:
            print("Skip %s: %s" % (sub.name, e), file=__import__("sys").stderr)

    if not rows:
        print("未找到任何有效的 best_results.csv")
        return

    df = pd.DataFrame(rows)

    def task_type(r):
        if "LACa" in r:
            return "LACa"
        if "LACm" in r:
            return "LACm"
        return "LVEF"

    df["task"] = df["run"].map(task_type)

    # 先按 primary 排序，再优先选 log_complete & loss_converged
    df = df.sort_values(
        ["task", "log_complete", "loss_converged", "primary_value"],
        ascending=[True, False, False, False],
    )

    print("=" * 80)
    print("各任务下 primary 指标最佳的前 5 个 run（要求 log 完成 + loss 收敛时优先）")
    print("=" * 80)
    for task in ["LVEF", "LACa", "LACm"]:
        subset = df[df["task"] == task].head(5)
        if subset.empty:
            continue
        print("\n--- %s ---" % task)
        for _, r in subset.iterrows():
            conv = "converged" if r["loss_converged"] else "not_converged"
            log = "log_ok" if r["log_complete"] else "no_log"
            print("  %.4f (%s) model=%s  [%s, %s]" % (
                r["primary_value"], r["primary_metric"], r["primary_model"], log, conv))
            print("    run: %s" % r["run"])
            if r["best_r2"] is not None:
                print("    R2=%.4f" % r["best_r2"], end="")
            if r["best_auc"] is not None:
                print("  AUC=%.4f" % r["best_auc"], end="")
            if r["best_loss"] is not None:
                print("  Loss=%.4f" % r["best_loss"], end="")
            if r["loss_converged_msg"]:
                print("  (%s)" % r["loss_converged_msg"], end="")
            print()

    # 仅“收敛且 log 完成”的最佳
    df_converged = df[(df["log_complete"]) & (df["loss_converged"])]
    print("\n" + "=" * 80)
    print("各任务中「log 完成 + loss 收敛」的最佳 run")
    print("=" * 80)
    if df_converged.empty:
        print("没有同时满足 log 完成且 loss 收敛的 run；上表为按 primary 排序的前 5。")
    else:
        df_c = df_converged.sort_values(["task", "primary_value"], ascending=[True, False])
        best_converged = df_c.groupby("task").first().reset_index()
        for _, r in best_converged.iterrows():
            print("\n%s: %s" % (r["task"], r["run"]))
            print("  %s = %.4f, best model = %s" % (r["primary_metric"], r["primary_value"], r["primary_model"]))
            print("  loss: %s" % r["loss_converged_msg"])

    # 全局最佳（不要求收敛）
    print("\n" + "=" * 80)
    print("全局最佳（按 primary 指标，不要求收敛）")
    print("=" * 80)
    best_per_task = df.groupby("task").first().reset_index()
    for _, r in best_per_task.iterrows():
        print("\n%s: %s" % (r["task"], r["run"]))
        print("  %s = %.4f, best model = %s" % (r["primary_metric"], r["primary_value"], r["primary_model"]))
        print("  log_complete=%s, loss_converged=%s" % (r["log_complete"], r["loss_converged"]))

    out_csv = OUTPUT_DIR.parent / "output_summary_best_runs.csv"
    df.to_csv(out_csv, index=False)
    print("\n汇总已保存: %s" % out_csv)

if __name__ == "__main__":
    main()
