"""Run the full 2x2x3 modeling experiment grid."""
from pathlib import Path

from obf_psychiatric_pipeline.config import load_config
from obf_psychiatric_pipeline.data.loader import load_all
from obf_psychiatric_pipeline.data.preprocess import preprocess
from obf_psychiatric_pipeline.models.train import run_experiments


def main() -> None:
    cfg = load_config(Path("config/config.yaml"))
    metadata, features = load_all(cfg.data.root)
    _, features = preprocess(metadata, features, cfg)

    out_dir = Path(cfg.output.results_dir) / "models"
    summary = run_experiments(features, cfg, out_dir)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()