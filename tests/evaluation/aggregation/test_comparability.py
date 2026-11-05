"""Tests for the cross-tokenizer comparability check (``comparability.py``)."""

from __future__ import annotations

import json

import pytest

# ── find_mixed_member_rows ─────────────────────────────────────────────────────
from ntt.evaluation.aggregation.comparability import (
    COMPARABILITY_ASPECTS,
    COMPARABILITY_VERSION,
    NO_INPUT,
    find_mixed_member_rows,
    provenance_record,
)


def _comparable_row(tokenizer: str, **fields) -> dict:
    """A current-version row in which every comparability field has the same recorded
    value, every score is present, and ``fields`` overrides some of them."""
    row = {"tokenizer": tokenizer, "comparability_version": str(COMPARABILITY_VERSION)}
    row |= dict.fromkeys(
        (
            "single_neuron_recon",
            "population_structure",
            "latent_structure_and_robustness",
            "decoding",
            "downstream_encoding",
            "generalization",
            "compression",
        ),
        0.5,
    )
    for field_name in COMPARABILITY_ASPECTS:
        row[field_name] = "{}" if field_name.endswith("_settings") else "same"
    row["recon_settings"] = '{"metrics": {}}'
    row["recon_chunk_window_ms"] = "{}"
    row["decoding_provenance"] = "poyo=absent;ridge=0.1 (2 targets)"
    return row | fields


def _note_start(lines: list[str]) -> int:
    """Index where the "could NOT check" note begins (``len(lines)`` if there is none)."""
    starts = [i for i, ln in enumerate(lines) if ln.startswith("WARNING: could NOT check")]
    return starts[0] if starts else len(lines)


def test_find_mixed_member_rows_names_tokenizer_without_m2():
    rows = [
        _comparable_row("tok_a", m2_provenance="included"),
        _comparable_row("tok_b", m2_provenance="excluded_no_report"),
    ]
    lines = find_mixed_member_rows(rows)
    assert lines[0].startswith("WARNING: tokenizers in this table were averaged DIFFERENTLY")
    m2_line = next(line for line in lines if "m2_provenance" in line)
    assert "excluded: ['tok_b (excluded_no_report)']" in m2_line
    assert "included: ['tok_a']" in m2_line
    # Identical rows -> no message at all.
    assert find_mixed_member_rows([rows[0], {**rows[0], "tokenizer": "tok_c"}]) == []


def test_two_exclusion_reasons_are_the_same_member_set():
    # "M2 key missing" and "no M2 report" both leave M2 out of the mean.
    rows = [
        _comparable_row("tok_a", m2_provenance="excluded_missing_key"),
        _comparable_row("tok_b", m2_provenance="excluded_no_report"),
    ]
    assert find_mixed_member_rows(rows) == []


def test_unknown_is_a_note_never_a_difference_or_an_all_clear():
    # Nobody saved it: not silent, and not "different".
    rows = [
        _comparable_row("tok_a", m1_sessions=None),
        _comparable_row("tok_b", m1_sessions=float("nan")),  # empty CSV cell
    ]
    lines = find_mixed_member_rows(rows)
    assert not any("DIFFERENTLY" in ln for ln in lines)
    assert lines[0] == (
        "WARNING: could NOT check whether these tokenizers were averaged the same way:"
    )
    assert any(
        ln.startswith(
            "  m1_sessions for ['tok_a', 'tok_b'], because the input-corruption robustness run"
        )
        for ln in lines
    )
    # One saved, one not: still only a note.
    rows[0]["m1_sessions"] = "2 #abc"
    lines = find_mixed_member_rows(rows)
    assert not any("DIFFERENTLY" in ln for ln in lines)
    assert any("m1_sessions for ['tok_b']" in ln for ln in lines)


def test_no_input_is_neither_a_difference_nor_a_note():
    rows = [
        _comparable_row("tok_a", m2_settings='{"epsilons": [0.1]}'),
        _comparable_row("tok_b", m2_sessions=NO_INPUT, m2_settings=NO_INPUT),
    ]
    assert find_mixed_member_rows(rows) == []


def test_recorded_run_settings_are_compared_key_by_key():
    rows = [
        _comparable_row("tok_a", recon_settings='{"max_eval_seconds": 600.0, "split": "val"}'),
        _comparable_row("tok_b", recon_settings='{"max_eval_seconds": null}'),
    ]
    lines = find_mixed_member_rows(rows)
    differing = [ln for ln in lines[: _note_start(lines)] if ln.startswith("  ")]
    assert differing == [
        "  recon_settings.max_eval_seconds (affects single_neuron_recon, population_structure, "
        "latent_structure_and_robustness, decoding, generalization, compression): "
        "600.0: ['tok_a'] | null: ['tok_b']"
    ]
    # tok_b's report did not save `split` while tok_a's did -> note, not difference.
    assert any("recon_settings.split for ['tok_b']" in ln for ln in lines[_note_start(lines) :])


def test_find_mixed_member_rows_flags_different_poisson_overlap():
    def row(tok, ids):
        cs = {
            "metadata": {
                "comparability_version": COMPARABILITY_VERSION,
                "poisson_norm_sessions": {"source": "precomputed", "session_ids": ids},
            }
        }
        record = provenance_record(cs)
        poisson_fields = {k: record[k] for k in ("poisson_norm_sessions", "poisson_norm_source")}
        return _comparable_row(tok, **poisson_fields)

    lines = find_mixed_member_rows([row("tok_a", ["s1", "s2"]), row("tok_b", ["s1"])])
    line = next(ln for ln in lines if "poisson_norm_sessions" in ln)
    assert "(affects single_neuron_recon, population_structure)" in line
    assert "['tok_a']" in line and "['tok_b']" in line
    # Same overlap -> silent; unrecorded overlap -> a note, not treated as a match.
    assert find_mixed_member_rows([row("tok_a", ["s1"]), row("tok_b", ["s1"])]) == []
    lines = find_mixed_member_rows([row("tok_a", None), row("tok_b", None)])
    assert lines[0].startswith("WARNING: could NOT check")
    assert any("poisson_norm_sessions for ['tok_a', 'tok_b']" in ln for ln in lines)


# ── score parts (M1, M2, POYO, ridge) ─────────────────────────────────────────

from ntt.evaluation.aggregation.comparability import (  # noqa: E402
    MixedScorePartsError,
    find_mixed_score_parts,
    raise_on_mixed_score_parts,
)


def test_mixed_score_parts_names_the_part_and_the_tokenizers():
    rows = [
        _comparable_row("tok_a", m2_provenance="included"),
        _comparable_row("tok_b", m2_provenance="excluded_missing_key"),
        _comparable_row(
            "tok_c",
            m2_provenance="included",
            decoding_provenance="poyo=0.5 (2 sessions);ridge=0.1",
        ),
    ]
    assert find_mixed_score_parts(rows) == {
        "m2": {"included": ["tok_a", "tok_c"], "left_out": ["tok_b"]},
        "poyo": {"included": ["tok_c"], "left_out": ["tok_a", "tok_b"]},
    }
    with pytest.raises(MixedScorePartsError, match="use the flag"):
        raise_on_mixed_score_parts(rows, how_to_fix="use the flag")


def test_a_part_left_out_for_everyone_is_not_mixed():
    rows = [
        _comparable_row("tok_a", m2_provenance="off"),
        _comparable_row("tok_b", m2_provenance="excluded_no_report"),
    ]
    assert find_mixed_score_parts(rows) == {}
    raise_on_mixed_score_parts(rows, how_to_fix="unused")  # no error


def test_provenance_record_carries_where_downstream_scores_came_from():
    cs = {
        "metadata": {
            "downstream": {
                "encoding": {"split": "test"},
                "decoding": {
                    "split": "test",
                    "complete": False,
                    "mixed_provenance": True,
                },
            }
        }
    }
    rec = provenance_record(cs)
    assert rec["encoding_split"] == "test"
    assert rec["decoding_mixed_provenance"] == "True"
    assert rec["decoding_complete"] == "False"
    assert provenance_record({})["encoding_split"] is None


def test_selected_chunk_sizes_are_compared_per_session_and_metric():
    from ntt.evaluation.aggregation.comparability import (
        raise_on_mixed_chunk_sizes,
        selected_chunk_windows,
    )

    def row(tokenizer, size):
        report = {
            "per_session_per_segment": {
                "s1": [
                    {
                        "results": {
                            "behavioral_regression_cursor_vel": {
                                "name": "behavioral_regression",
                                "metadata": {
                                    "primary_chunk_window_ms": size,
                                    "chunk_window_ms": 200,
                                },
                            }
                        }
                    }
                ]
            }
        }
        selected = selected_chunk_windows(report)
        assert selected == {"s1/behavioral_regression_cursor_vel": [size] if size else None}
        saved = {"metadata": {"recon_chunk_window_ms": selected}}
        return _comparable_row(
            tokenizer,
            **{
                "recon_chunk_window_ms": provenance_record(saved)["recon_chunk_window_ms"],
                "recon_settings": json.dumps(
                    {
                        "metrics": {
                            "behavioral_regression_cursor_vel": {
                                "chunk_window_ms": [200, 1000] if tokenizer == "tok_a" else 200,
                                "behavioral_bin_size_ms": 10,
                            }
                        }
                    }
                ),
            },
        )

    first = row("tok_a", 200)
    assert find_mixed_member_rows([first, row("tok_b", 200)]) == []
    raise_on_mixed_chunk_sizes([first, row("tok_b", 200)])
    different_parameters = row("tok_b", 200)
    settings = json.loads(different_parameters["recon_settings"])
    settings["metrics"]["behavioral_regression_cursor_vel"]["behavioral_bin_size_ms"] = 20
    different_parameters["recon_settings"] = json.dumps(settings)
    assert any(
        "recon_settings.metrics" in line
        for line in find_mixed_member_rows([first, different_parameters])
    )
    lines = find_mixed_member_rows([first, row("tok_b", 1000)])
    assert any(
        "recon_chunk_window_ms.s1/behavioral_regression_cursor_vel" in line
        and "[200.0]" in line
        and "[1000.0]" in line
        for line in lines
    )
    lines = find_mixed_member_rows([first, row("tok_b", None)])
    assert any(
        "recon_chunk_window_ms.s1/behavioral_regression_cursor_vel for ['tok_b']" in line
        for line in lines
    )
    assert not any("DIFFERENTLY" in line for line in lines)
    unused_regression = [row("tok_a", 200), row("tok_b", 1000)]
    for entry in unused_regression:
        entry["decoding_provenance"] = "poyo=0.5 (2 sessions);ridge=off"
    raise_on_mixed_chunk_sizes(unused_regression)
    position_rows = [row("tok_a", 200), row("tok_b", 1000)]
    for entry in position_rows:
        entry["recon_chunk_window_ms"] = entry["recon_chunk_window_ms"].replace(
            "cursor_vel", "cursor_pos"
        )
    position_rows[1]["position_regression_excluded"] = "True"
    raise_on_mixed_chunk_sizes(position_rows)


def test_a_session_metric_without_a_result_is_not_an_unknown_chunk_size():
    """A report with no result for one session/metric (e.g. a metric skipped for that
    session) contributes nothing there, so there is no chunk size to compare. A row whose
    chunk sizes are not saved at all stays unknown."""

    def row(tokenizer, selected):
        saved = {"metadata": {"recon_chunk_window_ms": selected}}
        return _comparable_row(
            tokenizer, recon_chunk_window_ms=provenance_record(saved)["recon_chunk_window_ms"]
        )

    both = {
        "s1/behavioral_regression_cursor_vel": [200.0],
        "s2/behavioral_regression_cursor_vel": [200.0],
    }
    only_s1 = {"s1/behavioral_regression_cursor_vel": [200.0]}
    lines = find_mixed_member_rows([row("tok_a", both), row("tok_b", only_s1)])
    assert not any("recon_chunk_window_ms.s2" in line for line in lines)
    unsaved = _comparable_row("tok_c", recon_chunk_window_ms=None)
    lines = find_mixed_member_rows([row("tok_a", both), unsaved])
    assert any("recon_chunk_window_ms" in line and "'tok_c'" in line for line in lines)


_VEL = "behavioral_regression_cursor_vel"
_POS = "behavioral_regression_cursor_pos"


def _ridge_row(tokenizer: str, chunk_sizes: dict, *, position_excluded: bool = False) -> dict:
    """A row whose ridge fields come from ``provenance_record``. ``chunk_sizes`` maps
    "<session>/<metric>" to the selected chunk sizes; with ``position_excluded``, the
    position rule left position regression out of the score."""
    provenance = {"poyo": "absent", "ridge": "0.1 (2 targets)"}
    if position_excluded:
        provenance["position_regression"] = "excluded_no_200ms"
    targets = {key.rsplit("/", 1)[1] for key in chunk_sizes}
    per_metric = {
        f"ridge/{target}": {} for target in targets if not (position_excluded and target == _POS)
    }
    record = provenance_record(
        {
            "metadata": {"recon_chunk_window_ms": chunk_sizes},
            "diagnostics": {"decoding": {"per_metric": per_metric, "provenance": provenance}},
        }
    )
    fields = (
        "decoding_members",
        "decoding_provenance",
        "recon_chunk_window_ms",
        "position_regression_excluded",
    )
    return _comparable_row(tokenizer, **{field: record[field] for field in fields})


_BOTH_SESSIONS = {f"s1/{_VEL}": [200.0], f"s2/{_VEL}": [200.0]}
# Each case: two rows whose scores were computed differently.
_NOT_COMPARABLE = {
    "different eval sessions": [
        _comparable_row("tok_a", recon_sessions="2 #aaaa"),
        _comparable_row("tok_b", recon_sessions="2 #bbbb"),
    ],
    "velocity missing in one session": [
        _ridge_row("tok_a", _BOTH_SESSIONS),
        _ridge_row("tok_b", {f"s1/{_VEL}": [200.0]}),
    ],
    "different regression targets (position counted for one tokenizer only)": [
        _ridge_row("tok_a", {f"s1/{_POS}": [200.0], f"s1/{_VEL}": [200.0]}),
        _ridge_row("tok_b", {f"s1/{_VEL}": [200.0]}),
    ],
    "different ridge sessions": [
        _ridge_row("tok_a", _BOTH_SESSIONS),
        _ridge_row("tok_b", {f"s1/{_VEL}": [200.0], f"s3/{_VEL}": [200.0]}),
    ],
}


@pytest.mark.parametrize("case", _NOT_COMPARABLE)
def test_scores_computed_differently_stop_the_build(case):
    from ntt.evaluation.aggregation.comparability import (
        UncheckedComparabilityError,
        raise_if_not_comparable,
    )

    with pytest.raises(UncheckedComparabilityError, match="averaged DIFFERENTLY"):
        raise_if_not_comparable(_NOT_COMPARABLE[case], common_parts_only=False)


def test_common_parts_only_turns_differences_into_a_warning():
    from ntt.evaluation.aggregation.comparability import (
        raise_if_not_comparable,
        raise_on_mixed_chunk_sizes,
    )

    rows = _NOT_COMPARABLE["velocity missing in one session"]
    raise_on_mixed_chunk_sizes(rows)
    raise_if_not_comparable(rows, common_parts_only=True)
    assert any("DIFFERENTLY" in line for line in find_mixed_member_rows(rows))


def test_position_left_out_by_the_position_rule_does_not_stop_the_build():
    from ntt.evaluation.aggregation.comparability import raise_if_not_comparable

    rows = [
        _ridge_row("tok_a", {f"s1/{_POS}": [200.0], f"s1/{_VEL}": [200.0]}),
        _ridge_row(
            "tok_b", {f"s1/{_POS}": [1000.0], f"s1/{_VEL}": [200.0]}, position_excluded=True
        ),
    ]
    raise_if_not_comparable(rows, common_parts_only=False)


def test_recorded_settings_takes_each_key_from_the_first_source_that_has_it():
    from ntt.evaluation.aggregation.comparability import recorded_settings

    assert recorded_settings({"split": "test"}, {"split": "val", "n_replicas": 10}) == {
        "split": "test",
        "n_replicas": 10,
    }
    assert recorded_settings(None) is None
    assert recorded_settings({"other": 1}) == {}


def test_recorded_settings_reads_metrics_from_a_run_meta_eval_config():
    """``metrics`` saved only in run_meta.json's eval_config is read from there; a source
    that records it on top wins."""
    from ntt.evaluation.aggregation.comparability import recorded_settings

    metrics = {"robustness_input": {"metric_class": "robustness_input"}}
    run_meta = {"n_replicas": 10, "eval_config": {"metrics": metrics, "max_eval_seconds": 1.0}}
    assert recorded_settings({"split": "test"}, run_meta) == {
        "split": "test",
        "n_replicas": 10,
        "metrics": metrics,
    }
    assert recorded_settings({"metrics": {"other": {}}}, run_meta)["metrics"] == {"other": {}}


def test_settings_in_the_report_config_and_in_run_meta_compare_equal():
    """The same robustness metric parameters saved in the two layouts are one value, so
    mixing the layouts neither differs nor leaves a field unchecked."""
    import json

    from ntt.evaluation.aggregation.comparability import (
        find_mixed_member_rows,
        recorded_settings,
    )

    metrics = {"robustness_input": {"metric_class": "robustness_input", "epsilons": [0.1]}}
    sidecar = {"n_replicas": 10, "eval_config": {"metrics": metrics}}
    in_report = recorded_settings({"split": "test", "metrics": metrics}, {"n_replicas": 10})
    in_sidecar = recorded_settings({"split": "test"}, sidecar)
    assert in_report == in_sidecar
    rows = [
        _comparable_row("report_layout", m2_settings=json.dumps(in_report, sort_keys=True)),
        _comparable_row("sidecar_layout", m2_settings=json.dumps(in_sidecar, sort_keys=True)),
    ]
    lines = find_mixed_member_rows(rows)
    assert not any("could NOT check" in ln or "DIFFERENTLY" in ln for ln in lines)


def test_derive_scored_window_from_the_all_neuron_path():
    from ntt.evaluation.aggregation.comparability import derive_scored_window

    config = {
        "neuron_selection": {"policy": "eval_unit_selection"},
        "evaluate_all_neurons": True,
        "eval_window_seconds": 0.5,
    }
    assert derive_scored_window(config) == 0.5
    # eval_window_seconds may sit in a nested eval_config; the first source holding a key wins.
    assert derive_scored_window({}, config) == 0.5
    assert derive_scored_window({"eval_config": config}) == 0.5


def test_derive_scored_window_on_the_padded_path_needs_the_tokenizer_window_to_agree():
    from ntt.evaluation.aggregation.comparability import derive_scored_window

    padded = {"eval_window_seconds": 1.0, "tokenizer_config": {"window_length_seconds": 1.0}}
    assert derive_scored_window(padded) == 1.0
    other = {"eval_window_seconds": 1.0, "tokenizer_config": {"window_length_seconds": 5.0}}
    assert derive_scored_window(other) is None


def test_derive_scored_window_is_none_without_a_numeric_eval_window():
    from ntt.evaluation.aggregation.comparability import derive_scored_window

    assert derive_scored_window(None, {}) is None
    assert derive_scored_window({"eval_window_seconds": "1.0"}) is None
    assert derive_scored_window({"eval_window_seconds": True}) is None


def test_with_scored_window_fills_a_missing_window_and_keeps_a_recorded_one():
    from ntt.evaluation.aggregation.comparability import with_scored_window

    derivable = {"eval_window_seconds": 2.0, "tokenizer_config": {"window_length_seconds": 2.0}}
    assert with_scored_window({"split": "test"}, derivable) == {
        "split": "test",
        "scored_window_seconds": 2.0,
    }
    recorded = {"scored_window_seconds": 1.0}
    assert with_scored_window(recorded, derivable) is recorded
    assert with_scored_window({"split": "test"}, {}) == {"split": "test"}
    assert with_scored_window(None, derivable) is None


def test_aggregated_only_chunk_sizes_stay_unknown_next_to_per_session_ones():
    """A report without a per-session breakdown saves its chunk sizes under
    ``aggregated/<metric>``; its missing session keys mean "not broken down", not "no
    result", so they must stay unknown instead of being skipped."""
    rows = [
        _comparable_row(
            tok,
            recon_chunk_window_ms=provenance_record(
                {"metadata": {"recon_chunk_window_ms": selected}}
            )["recon_chunk_window_ms"],
        )
        for tok, selected in (
            ("tok_a", {"s1/behavioral_regression_cursor_vel": [200.0]}),
            ("tok_b", {"aggregated/behavioral_regression_cursor_vel": [1000.0]}),
        )
    ]
    lines = find_mixed_member_rows(rows)
    assert any(
        "recon_chunk_window_ms.s1/behavioral_regression_cursor_vel for ['tok_b']" in line
        for line in lines
    )
