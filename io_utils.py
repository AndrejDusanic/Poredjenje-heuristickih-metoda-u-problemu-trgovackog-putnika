import os, csv, json
import numpy as np

def to_py(x):
    if isinstance(x, (np.integer,)):     return int(x)
    if isinstance(x, (np.floating,)):    return float(x)
    if isinstance(x, (np.ndarray,)):     return x.tolist()
    if isinstance(x, (list, tuple)):     return [to_py(i) for i in x]
    if isinstance(x, dict):              return {str(k): to_py(v) for k, v in x.items()}
    return x

def write_pergen_csv(path, best_hist, mean_hist, median_hist):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_km", "mean_km", "median_km"])
        for g, (b, m, d) in enumerate(zip(best_hist, mean_hist, median_hist), start=1):
            w.writerow([g, round(b, 4), round(m, 4), round(d, 4)])

def write_summary_csv(path, summary_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in summary_dict.items():
            vv = to_py(v)
            if isinstance(vv, (list, dict)):
                vv = json.dumps(vv, ensure_ascii=False)
            w.writerow([k, vv])
