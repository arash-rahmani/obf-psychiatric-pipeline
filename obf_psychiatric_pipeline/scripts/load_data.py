"""Smoke test: load everything and print a sanity summary."""
from pathlib import Path

from obf_psychiatric_pipeline.config import load_config
from obf_psychiatric_pipeline.data.loader import load_all


def main() -> None:
    cfg = load_config(Path("config/config.yaml"))
    metadata, features = load_all(cfg.data.root)

    print("=== Metadata cohorts ===")
    for cohort, df in metadata.items():
        print(f"  {cohort:15s} n={len(df):3d}  cols={len(df.columns)}")

    print("\n=== Features matrix ===")
    print(f"  rows={len(features)}, unique users={features['user'].nunique()}")
    print(features["class"].value_counts())


if __name__ == "__main__":
    main()