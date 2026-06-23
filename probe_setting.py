#!/usr/bin/env python3
"""Care-setting confound probes: acrophase by hospitalization status.
Run from pipeline root: python3 probe_setting.py"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from obf_psychiatric_pipeline.config import load_config
from obf_psychiatric_pipeline.data.loader import load_all
from obf_psychiatric_pipeline.data.preprocess import preprocess
from obf_psychiatric_pipeline.data.raw_loader import load_all_actigraphy
from obf_psychiatric_pipeline.features.extract import extract_all_features
from obf_psychiatric_pipeline.models.aggregate import to_participant_level

cfg = load_config(Path("config/config.yaml"))

# acrophase per analyzed participant (same n=76 chain as the figure)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    metadata, features = load_all(cfg.data.root)
    _, features = preprocess(metadata, features, cfg)
dist = to_participant_level(features)
temp = extract_all_features(load_all_actigraphy(cfg.data.actigraphy_root))
temp = temp.rename_axis("user").reset_index().drop(columns=["label"], errors="ignore")
df = dist.merge(temp, on="user", how="inner")
df = df[df["cosinor_r_squared"] >= 0.5].copy()
df = df[["user", "class", "cosinor_acrophase_hours", "cosinor_r_squared"]]


# hospitalization flag from depression metadata
dep_meta = metadata["depression"].copy()
key = "number" if "number" in dep_meta.columns else dep_meta.columns[0]
hosp_col = next((c for c in dep_meta.columns if "hosp" in c.lower() or "inpatient" in c.lower()), None)
print("depression metadata columns:", list(dep_meta.columns))
print("hospitalization column detected:", hosp_col)

if hosp_col:
    dep_meta = dep_meta.rename(columns={key: "user", hosp_col: "hosp"})
    df = df.merge(dep_meta[["user", "hosp"]], on="user", how="left")

acro = "cosinor_acrophase_hours"
def summ(s, label):
    s = s.dropna()
    print(f"  {label:32s} n={len(s):2d}  median={s.median():.2f}  mean={s.mean():.2f}  ({s.min():.2f}-{s.max():.2f})")

print("\n=== PROBE 1: within depression, by hospitalization ===")
if hosp_col:
    dep = df[df["class"] == "depression"]
    for v, lbl in [(1, "inpatient (hosp=1)"), (2, "outpatient (hosp=2)")]:
        summ(dep.loc[dep["hosp"] == v, acro], lbl)
    print("  NOTE: confirm coding (1=inpatient/2=outpatient) against dataset docs.")
else:
    print("  hospitalization column not found; inspect columns printed above.")

print("\n=== PROBE 2: setting-matched, outpatient depression vs controls ===")
summ(df.loc[df["class"] == "control", acro], "control (community)")
if hosp_col:
    summ(df.loc[(df["class"] == "depression") & (df["hosp"] == 2), acro], "depression outpatient")
summ(df.loc[df["class"] == "schizophrenia", acro], "schizophrenia (inpatient, ref)")