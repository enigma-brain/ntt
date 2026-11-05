"""Plot-output helpers shared between the eval jobs and replot scripts.

The robustness eval jobs (``ntt.jobs.eval_robustness_input``,
``ntt.jobs.eval_robustness_latent``) and the ``scripts/replot_*``
CLIs all need to write the same figure under multiple formats so we
can later embed PDFs in papers without losing fidelity.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

# Extensions we treat as a replaceable image suffix on ``base_path``. Anything else
# (e.g. a dot inside a session id preserved by ``_safe_filename``) is kept verbatim.
_KNOWN_PLOT_SUFFIXES = frozenset({".png", ".svg", ".pdf", ".jpg", ".jpeg", ".eps"})


def save_fig_multi(
    fig,
    base_path: Path | str,
    *,
    formats: Iterable[str] = ("png", "svg", "pdf"),
    dpi: int = 150,
    bbox_inches: str | None = "tight",
) -> list[Path]:
    """Save ``fig`` under ``base_path`` (sans extension) in each requested
    format.

    Default formats are PNG (web/preview), SVG (lossless vector for
    notebooks), and PDF (paper-ready). Non-existent parent directories
    are created.

    Args:
        fig: ``matplotlib.figure.Figure`` instance.
        base_path: target path. A trailing known image suffix (e.g. ``.png``) is
            replaced per format; any other dots (e.g. in a session id) are preserved.
        formats: extensions to emit, without the leading dot.
        dpi: rasterizer DPI (used by PNG; ignored by vector formats).
        bbox_inches: passed through to ``fig.savefig``.

    Returns:
        List of absolute paths actually written, in the order of
        ``formats``.
    """
    p = Path(base_path)
    base = p.with_suffix("") if p.suffix.lower() in _KNOWN_PLOT_SUFFIXES else p
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in formats:
        out = base.parent / f"{base.name}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches)
        written.append(out.resolve())
        logger.info("Wrote %s", out)
    return written
