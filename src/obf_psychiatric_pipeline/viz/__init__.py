"""Project-wide plotting style constants."""
from pathlib import Path
import matplotlib.pyplot as plt

CLASS_PALETTE = {
    "control":       "#4C72B0",
    "depression":    "#C44E52",
    "schizophrenia": "#8172B2",
}

FIGSIZE_DEFAULT = (8, 5)


def save_fig(fig: plt.Figure, out_path: Path, dpi: int = 150) -> None:
    """Tight-layout, save, close."""
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)