"""Tests for _flush_segment() in validation_collation.py."""

import logging

import numpy as np
import pytest
import torch

from ntt.evaluation.utils.validation_collation import _concatenate_segments, _flush_segment


def _make_window(
    n_units: int = 10,
    n_time: int = 100,
    n_tokens: int = 5,
    latent_dim: int = 1,
    absolute_start: float = 0.0,
    absolute_end: float = 1.0,
    session_id: str = "sess_0",
    include_behavioral: bool = False,
    include_token_times: bool = False,
) -> dict:
    """Create a synthetic per-window dict for testing _flush_segment."""
    window = {
        "original": torch.randn(n_units, n_time),
        "reconstructed": torch.randn(n_units, n_time),
        "mask": torch.ones(n_units, n_time, dtype=torch.bool),
        "latents": torch.randn(n_tokens, n_units, latent_dim),
        "absolute_start": absolute_start,
        "absolute_end": absolute_end,
        "session_id": session_id,
        "metadata": {"sampling_rate": 100.0},
    }
    if include_token_times:
        token_duration = (absolute_end - absolute_start) / n_tokens
        starts = torch.tensor(
            [absolute_start + i * token_duration for i in range(n_tokens)],
            dtype=torch.float64,
        )
        ends = starts + token_duration
        window["token_start_times"] = starts
        window["token_end_times"] = ends
    if include_behavioral:
        window["behavioral_data_token_res"] = {
            "velocity": np.random.randn(n_tokens, 2),
            "position": np.random.randn(n_tokens, 3),
        }
    return window


class TestFlushSegmentThreeWindows:
    """3 contiguous windows concatenated correctly."""

    def test_shapes_after_concatenation(self):
        n_units, n_time, n_tokens, latent_dim = 10, 100, 5, 1
        windows = [
            _make_window(
                n_units=n_units,
                n_time=n_time,
                n_tokens=n_tokens,
                latent_dim=latent_dim,
                absolute_start=float(i),
                absolute_end=float(i + 1),
            )
            for i in range(3)
        ]
        segment = _flush_segment(windows)

        # original/reconstructed: concatenated along time (dim=1)
        assert segment.original.shape == (n_units, n_time * 3)
        assert segment.reconstructed.shape == (n_units, n_time * 3)

        # latents: stacked along new window dimension (dim=0)
        assert segment.latents.shape == (3, n_tokens, n_units, latent_dim)

        # mask concatenated along time
        assert segment.pad_and_gap_mask.shape == (n_units, n_time * 3)

    def test_session_id_preserved(self):
        windows = [
            _make_window(absolute_start=float(i), absolute_end=float(i + 1)) for i in range(3)
        ]
        segment = _flush_segment(windows)
        assert segment.session_id == "sess_0"


class TestFlushSegmentGapDetection:
    """Between-window gaps are concatenated across and recorded in metadata.

    Multi-interval sampling domains (trial-based sessions) leave time gaps
    between buffered windows; the session must still yield one segment.
    """

    @staticmethod
    def _gapped_windows(bounds: list[tuple[float, float]], n_time: int = 100, n_tokens: int = 5):
        windows = []
        for start, end in bounds:
            w = _make_window(
                n_time=n_time,
                n_tokens=n_tokens,
                absolute_start=start,
                absolute_end=end,
                include_token_times=True,
            )
            w["token_timestamps"] = (w["token_start_times"] + w["token_end_times"]) / 2
            # Behavior rows tagged with each bin's absolute time, so the test can
            # check they stay aligned with the binned data across the joins.
            bin_times = start + (np.arange(n_time) + 0.5) * (end - start) / n_time
            w["behavioral_data_input_res"] = {"cursor_pos": bin_times[:, None]}
            w["behavioral_data_token_res"] = {
                "trial_id": np.full(n_tokens, int(start), dtype=np.int64)
            }
            windows.append(w)
        return windows

    def test_gap_concatenates_with_metadata(self):
        windows = self._gapped_windows([(0.0, 1.0), (1.5, 2.5)])  # 0.5 s gap
        segment = _flush_segment(windows)
        mask, latents = segment.pad_and_gap_mask, segment.latents
        starts, centers = segment.token_start_times, segment.token_timestamps
        behavior_in, behavior_tok = (
            segment.behavioral_data_input_res,
            segment.behavioral_data_token_res,
        )
        assert mask is not None and isinstance(latents, torch.Tensor)
        assert starts is not None and centers is not None
        assert behavior_in is not None and behavior_tok is not None

        torch.testing.assert_close(
            segment.original, torch.cat([w["original"] for w in windows], dim=1)
        )
        torch.testing.assert_close(
            segment.reconstructed, torch.cat([w["reconstructed"] for w in windows], dim=1)
        )
        assert mask.shape == (10, 200)
        assert latents.shape == (2, 5, 10, 1)
        assert segment.segment_absolute_start == 0.0
        # Token times keep their real values across the gap.
        torch.testing.assert_close(starts, torch.cat([w["token_start_times"] for w in windows]))
        assert float(starts[5]) == 1.5
        assert float(centers[4]) == pytest.approx(0.9)
        assert float(centers[5]) == pytest.approx(1.6)
        # Behavior stays row-aligned with the binned time axis.
        cursor = behavior_in["cursor_pos"]
        assert cursor.shape == (200, 1)
        assert cursor[99, 0] == pytest.approx(0.995)
        assert cursor[100, 0] == pytest.approx(1.505)
        np.testing.assert_array_equal(behavior_tok["trial_id"], np.repeat([0, 1], 5))
        assert segment.metadata["n_gaps"] == 1
        assert segment.metadata["gap_seconds"] == pytest.approx(0.5)

    def test_no_gap_when_contiguous(self):
        w0 = _make_window(absolute_start=0.0, absolute_end=1.0)
        w1 = _make_window(absolute_start=1.0, absolute_end=2.0)
        segment = _flush_segment([w0, w1])
        assert segment.original is not None
        assert segment.original.shape[1] == 200
        torch.testing.assert_close(
            segment.original, torch.cat([w0["original"], w1["original"]], 1)
        )
        assert segment.metadata["n_gaps"] == 0
        assert segment.metadata["gap_seconds"] == 0.0

    def test_multiple_gaps(self):
        bounds = [(0.0, 1.0), (1.25, 2.25), (2.25, 3.25), (4.0, 5.0)]
        segment = _flush_segment(self._gapped_windows(bounds))
        assert segment.original is not None and isinstance(segment.latents, torch.Tensor)
        assert segment.original.shape[1] == 400
        assert segment.latents.shape[0] == 4
        assert segment.metadata["n_gaps"] == 2
        assert segment.metadata["gap_seconds"] == pytest.approx(0.25 + 0.75)

    def test_gap_at_tolerance_boundary(self):
        """2 ms tolerance: exactly 2 ms (and FP jitter) is contiguous, above is a gap."""
        # Exactly 2e-3 apart in float64 (a window ending at 0.0 avoids rounding).
        exact = [
            _make_window(absolute_start=-1.0, absolute_end=0.0),
            _make_window(absolute_start=2e-3, absolute_end=1.002),
        ]
        assert _flush_segment(exact).metadata["n_gaps"] == 0

        jitter = [
            _make_window(absolute_start=0.0, absolute_end=1.0),
            _make_window(absolute_start=1.0 + 1e-10, absolute_end=2.0),
        ]
        assert _flush_segment(jitter).metadata["n_gaps"] == 0

        above = [
            _make_window(absolute_start=0.0, absolute_end=1.0),
            _make_window(absolute_start=1.0021, absolute_end=2.0021),
        ]
        segment = _flush_segment(above)
        assert segment.metadata["n_gaps"] == 1
        assert segment.metadata["gap_seconds"] == pytest.approx(0.0021)

    def test_skip_gap_check_unchanged(self):
        """skip_gap_check=True concatenates without counting gaps (M1 replica path)."""
        windows = self._gapped_windows([(0.0, 1.0), (1.5, 2.5)])
        segment = _flush_segment(windows, skip_gap_check=True)
        assert segment.original is not None
        assert segment.original.shape[1] == 200
        assert "n_gaps" not in segment.metadata
        assert "gap_seconds" not in segment.metadata

    def test_gap_logged_at_info_not_warning(self, caplog):
        windows = self._gapped_windows([(0.0, 1.0), (1.5, 2.5)])
        with caplog.at_level(logging.INFO, logger="ntt.evaluation.utils.validation_collation"):
            _flush_segment(windows)
        gap_records = [r for r in caplog.records if "temporal gap" in r.getMessage()]
        assert len(gap_records) == 1
        assert gap_records[0].levelno == logging.INFO


class TestFlushSegmentSingleWindow:
    """Single window works without stacking issues."""

    def test_single_window(self):
        n_units, n_time, n_tokens, latent_dim = 10, 100, 5, 1
        window = _make_window(
            n_units=n_units,
            n_time=n_time,
            n_tokens=n_tokens,
            latent_dim=latent_dim,
        )
        segment = _flush_segment([window])

        assert segment.original.shape == (n_units, n_time)
        assert segment.reconstructed.shape == (n_units, n_time)
        # Single window: latents should still have window dim
        assert segment.latents.shape == (1, n_tokens, n_units, latent_dim)


class TestFlushSegmentBehavioralData:
    """Behavioral dict values concatenated along time (token dim)."""

    def test_behavioral_concatenated(self):
        n_tokens = 5
        windows = [
            _make_window(
                n_tokens=n_tokens,
                absolute_start=float(i),
                absolute_end=float(i + 1),
                include_behavioral=True,
            )
            for i in range(3)
        ]
        segment = _flush_segment(windows)

        assert segment.behavioral_data_token_res is not None
        assert segment.behavioral_data_token_res["velocity"].shape == (n_tokens * 3, 2)
        assert segment.behavioral_data_token_res["position"].shape == (n_tokens * 3, 3)

    def test_window_group_id_token_res_concatenated(self):
        n_tokens = 5
        windows = []
        for group_id in range(3):
            window = _make_window(
                n_tokens=n_tokens,
                absolute_start=float(group_id),
                absolute_end=float(group_id + 1),
                include_behavioral=True,
            )
            window["behavioral_data_token_res"]["window_group_id_token_res"] = np.full(
                n_tokens, group_id, dtype=np.int32
            )
            windows.append(window)

        segment = _flush_segment(windows)

        assert segment.behavioral_data_token_res is not None
        np.testing.assert_array_equal(
            segment.behavioral_data_token_res["window_group_id_token_res"],
            np.concatenate([np.full(n_tokens, group_id, dtype=np.int32) for group_id in range(3)]),
        )


def test_concatenate_segments_offsets_window_group_id_token_ress():
    first_window = _make_window(
        n_tokens=4,
        absolute_start=0.0,
        absolute_end=1.0,
        session_id="sess_a",
        include_behavioral=True,
    )
    first_window["behavioral_data_token_res"]["window_group_id_token_res"] = np.zeros(
        4, dtype=np.int32
    )
    first_segment = _flush_segment([first_window])

    second_window = _make_window(
        n_tokens=4,
        absolute_start=1.0,
        absolute_end=2.0,
        session_id="sess_b",
        include_behavioral=True,
    )
    second_window["behavioral_data_token_res"]["window_group_id_token_res"] = np.zeros(
        4, dtype=np.int32
    )
    second_segment = _flush_segment([second_window])

    combined = _concatenate_segments([first_segment, second_segment])

    assert combined.behavioral_data_token_res is not None
    np.testing.assert_array_equal(
        combined.behavioral_data_token_res["window_group_id_token_res"],
        np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
    )


class TestFlushSegmentTokenTimestamps:
    """token_start_times, token_end_times concatenated."""

    def test_token_times_concatenated(self):
        n_tokens = 5
        windows = [
            _make_window(
                n_tokens=n_tokens,
                absolute_start=float(i),
                absolute_end=float(i + 1),
                include_token_times=True,
            )
            for i in range(3)
        ]
        segment = _flush_segment(windows)

        assert segment.token_start_times is not None
        assert segment.token_end_times is not None
        assert segment.token_start_times.shape == (n_tokens * 3,)
        assert segment.token_end_times.shape == (n_tokens * 3,)

        # Check monotonicity
        all_starts = segment.token_start_times.numpy()
        assert np.all(np.diff(all_starts) >= -1e-9), "token_start_times not monotonic"

    def test_no_token_times_yields_none(self):
        windows = [
            _make_window(
                absolute_start=float(i),
                absolute_end=float(i + 1),
                include_token_times=False,
            )
            for i in range(2)
        ]
        segment = _flush_segment(windows)
        assert segment.token_start_times is None
        assert segment.token_end_times is None
