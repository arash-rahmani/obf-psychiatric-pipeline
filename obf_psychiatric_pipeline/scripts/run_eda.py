"""Run all EDA plots."""
from pathlib import Path

from obf_psychiatric_pipeline.config import load_config
from obf_psychiatric_pipeline.data.loader import load_all
from obf_psychiatric_pipeline.viz.eda import run_eda


def main() -> None:
    cfg = load_config(Path("config/config.yaml"))
    metadata, features = load_all(cfg.data.root)
    run_eda(metadata, features, Path(cfg.output.eda_dir))
    print(f"EDA plots written to {cfg.output.eda_dir}")


if __name__ == "__main__":
    main()