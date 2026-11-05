"""Tests for ntt.evaluation.plotting.io.save_fig_multi."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ntt.evaluation.plotting.io import save_fig_multi


def test_save_fig_multi_preserves_dotted_stem(tmp_path) -> None:
    """A base path whose final component contains dots must not be truncated.

    Callers build stems from ``_safe_filename(session_id)``, which preserves '.',
    so a dotted id (or numeric suffix) must survive into the output filename rather
    than being mistaken for a file extension and stripped.
    """
    fig = plt.figure()
    base = tmp_path / "input-corruption_allen.neuropixels_0.5"
    written = save_fig_multi(fig, base, formats=("png",))
    plt.close(fig)

    expected = tmp_path / "input-corruption_allen.neuropixels_0.5.png"
    assert expected.exists()
    assert written == [expected.resolve()]


def test_save_fig_multi_normalizes_explicit_known_extension(tmp_path) -> None:
    """A base path passed *with* a known image extension is normalized per format."""
    fig = plt.figure()
    base = tmp_path / "plot.png"
    written = save_fig_multi(fig, base, formats=("png", "svg"))
    plt.close(fig)

    assert (tmp_path / "plot.png").exists()
    assert (tmp_path / "plot.svg").exists()
    # The explicit ".png" must not survive as a stem segment ("plot.png.svg").
    assert not (tmp_path / "plot.png.svg").exists()
    assert written == [(tmp_path / "plot.png").resolve(), (tmp_path / "plot.svg").resolve()]
