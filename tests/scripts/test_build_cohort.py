"""Tests for scripts/eval/build_cohort.py — offline discovery paths and the CSV contract.

The MLflow discovery path is exercised against a real tracker separately; here we
cover the parts that must hold without a network: name derivation, list/dir
discovery, duplicate handling, and the column contract the aggregation layer reads.
"""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import pytest


def _module():
    """Import scripts/eval/build_cohort.py the way the sibling script tests do."""
    script_dir = Path(__file__).resolve().parents[2] / "scripts" / "eval"
    sys.path.insert(0, str(script_dir))
    try:
        import build_cohort

        importlib.reload(build_cohort)
    finally:
        sys.path.pop(0)
    return build_cohort


@pytest.fixture
def bc():
    return _module()


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), [dict(r) for r in reader]


# ── name derivation ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        # identity lives in a parent dir, because the file is always called last.ckpt
        ("/data/ckpts/tok_alpha/checkpoints/last.ckpt", "tok_alpha"),
        ("/data/ckpts/tok_beta/last.ckpt", "tok_beta"),
        # versioned last-vN directories are skipped when walking up
        ("/data/ckpts/tok_gamma/last-v2/model.ckpt", "tok_gamma"),
        # a genuinely named file keeps its stem
        ("/data/ckpts/epoch_37_step_1000.ckpt", "epoch_37_step_1000"),
        # tracker URIs are parsed, not treated as opaque strings
        ("mlflow-artifacts:/12/abc123/artifacts/checkpoints/last.ckpt", "abc123"),
    ],
)
def test_name_from_path(bc, uri, expected):
    assert bc.name_from_path(uri) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("My Tokenizer B", "my_tokenizer_b"),
        ("tok.alpha-01", "tok_alpha_01"),
        ("__leading_and_trailing__", "leading_and_trailing"),
        ("ALLCAPS", "allcaps"),
    ],
)
def test_slugify_is_filesystem_safe(bc, raw, expected):
    assert bc.slugify(raw) == expected


def test_slugify_matches_aggregation_rule(bc):
    """The rule must match the aggregation layer's, or names change on round-trip."""
    import re

    for name in ["Tok Alpha", "a.b-c", "x__y", "Mixed.Case-1"]:
        expected = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
        assert bc.slugify(name) == expected


# ── list discovery ──────────────────────────────────────────────────────────────
def test_read_list_handles_comments_blanks_and_explicit_names(bc, tmp_path):
    listing = tmp_path / "ckpts.txt"
    listing.write_text(
        "\n".join(
            [
                "# a comment line",
                "",
                "   ",
                "/data/run_one/checkpoints/last.ckpt",
                "My Tokenizer B,/data/whatever/last.ckpt",
                "tab_named\t/data/other/last.ckpt",
                "/data/run_two/last.ckpt  # trailing comment",
            ]
        )
    )
    rows = bc.read_list(listing)
    assert [r["tokenizer"] for r in rows] == [
        "run_one",
        "my_tokenizer_b",
        "tab_named",
        "run_two",
    ]
    assert rows[3]["ckpt_uri"] == "/data/run_two/last.ckpt", "trailing comment not stripped"


def test_read_list_preserves_uris_verbatim(bc, tmp_path):
    listing = tmp_path / "ckpts.txt"
    uri = "mlflow-artifacts:/7/deadbeef/artifacts/last/model.ckpt"
    listing.write_text(uri + "\n")
    assert bc.read_list(listing)[0]["ckpt_uri"] == uri


# ── directory discovery ─────────────────────────────────────────────────────────
def test_scan_dir_finds_both_layouts(bc, tmp_path):
    for name in ("tok_a", "tok_b"):
        d = tmp_path / name / "checkpoints"
        d.mkdir(parents=True)
        (d / "last.ckpt").touch()
    vd = tmp_path / "tok_c" / "last-v2"
    vd.mkdir(parents=True)
    (vd / "model.ckpt").touch()
    (tmp_path / "notes.txt").touch()  # must be ignored

    rows = bc.scan_dir(tmp_path, "**/*.ckpt")
    assert sorted(r["tokenizer"] for r in rows) == ["tok_a", "tok_b", "tok_c"]
    assert all(r["ckpt_uri"].endswith(".ckpt") for r in rows)


def test_scan_dir_empty_is_not_an_error(bc, tmp_path):
    assert bc.scan_dir(tmp_path, "**/*.ckpt") == []


# ── duplicate handling ──────────────────────────────────────────────────────────
def test_dedupe_disambiguates_rather_than_dropping(bc):
    rows = [
        {"tokenizer": "dup", "ckpt_uri": "/a.ckpt", "_run_id": "aaaaaaaabbbb"},
        {"tokenizer": "dup", "ckpt_uri": "/b.ckpt", "_run_id": "ccccccccdddd"},
        {"tokenizer": "solo", "ckpt_uri": "/c.ckpt"},
    ]
    out = bc.dedupe(rows)
    names = [r["tokenizer"] for r in out]
    assert len(out) == 3, "a duplicate name must never drop a cohort member"
    assert len(set(names)) == 3, f"names still collide: {names}"
    assert names[0] == "dup"
    assert names[1].startswith("dup_")
    # the surviving rows keep their own checkpoints
    assert {r["ckpt_uri"] for r in out} == {"/a.ckpt", "/b.ckpt", "/c.ckpt"}


def test_dedupe_without_run_id_still_disambiguates(bc):
    rows = [{"tokenizer": "x", "ckpt_uri": f"/{i}.ckpt"} for i in range(3)]
    out = bc.dedupe(rows)
    assert len({r["tokenizer"] for r in out}) == 3


# ── the CSV column contract ─────────────────────────────────────────────────────
def test_every_axis_column_is_read_by_aggregation(bc):
    """A run URL in each axis column must reach aggregation's row->URI mapping."""
    from ntt.evaluation.aggregation.tracker_io import row_to_source_uris

    url = "https://mlflow.example/#/experiments/7/runs/0123456789abcdef"
    row = {"tokenizer": "t", "ckpt_uri": "/a.ckpt", **dict.fromkeys(bc.AXIS_COLUMNS, url)}
    uris = row_to_source_uris(row)
    assert uris is not None
    assert len(uris) == len(bc.AXIS_COLUMNS), f"an axis column is not read: {sorted(uris)}"


def test_written_columns_match_the_aggregation_schema(bc, tmp_path):
    out = tmp_path / "cohort.csv"
    bc.write_cohort([{"tokenizer": "t", "ckpt_uri": "/a.ckpt"}], out)
    fieldnames, rows = _read(out)

    assert fieldnames[:2] == ["tokenizer", "ckpt_uri"], "the two human-filled columns come first"
    # These are the names the aggregation layer's row->URI mapping reads.
    for required in (
        "mlflow_recon",
        "mlflow_latent_input_corruption",
        "mlflow_latent_latent_corruption",
        "mlflow_encoding",
        "mlflow_decoding",
    ):
        assert required in fieldnames, f"{required} missing; aggregation would drop that axis"
    assert "mlflow_latent_snr" in fieldnames, "SNR column, as tracker_io reads it"

    assert len(rows) == 1
    assert all(rows[0][c] == "" for c in fieldnames[2:]), "axis columns start empty"


def test_axis_columns_are_empty_not_absent(bc, tmp_path):
    """An empty cell is skipped-with-a-warning downstream; a missing column is silent."""
    out = tmp_path / "cohort.csv"
    bc.write_cohort([{"tokenizer": "t", "ckpt_uri": "/a.ckpt"}], out)
    assert out.read_text().splitlines()[1].endswith("," * len(bc.AXIS_COLUMNS))


def test_write_cohort_drops_internal_fields(bc, tmp_path):
    out = tmp_path / "cohort.csv"
    bc.write_cohort([{"tokenizer": "t", "ckpt_uri": "/a.ckpt", "_run_id": "secretish"}], out)
    assert "_run_id" not in out.read_text()


def test_write_cohort_creates_parent_dirs(bc, tmp_path):
    out = tmp_path / "nested" / "deeper" / "cohort.csv"
    bc.write_cohort([{"tokenizer": "t", "ckpt_uri": "/a.ckpt"}], out)
    assert out.exists()


# ── messy list input ────────────────────────────────────────────────────────────
def test_a_cohort_csv_round_trips_through_from_list(bc, tmp_path):
    """Re-reading a generated table must not rename rows or mangle URIs.

    The trailing empty axis columns are the hazard: naive line-splitting absorbs them
    into ckpt_uri, silently corrupting every row.
    """
    original = tmp_path / "cohort.csv"
    uri = "mlflow-artifacts:/12/abc123/artifacts/last/last.ckpt"
    bc.write_cohort([{"tokenizer": "keep_my_name", "ckpt_uri": uri}], original)

    rows = bc.read_list(original)
    assert len(rows) == 1
    assert rows[0]["tokenizer"] == "keep_my_name", "name must survive the round trip"
    assert rows[0]["ckpt_uri"] == uri, f"URI corrupted: {rows[0]['ckpt_uri']!r}"


def test_cohort_csv_skips_rows_without_a_checkpoint(bc, tmp_path):
    table = tmp_path / "cohort.csv"
    table.write_text("tokenizer,ckpt_uri,mlflow_recon\na,/a.ckpt,\nb,,\n")
    assert [r["tokenizer"] for r in bc.read_list(table)] == ["a"]


def test_quoted_field_containing_a_comma_survives(bc, tmp_path):
    listing = tmp_path / "l.txt"
    listing.write_text('"name, with comma",/data/b/last.ckpt\n')
    rows = bc.read_list(listing)
    assert rows[0]["tokenizer"] == "name_with_comma"
    assert rows[0]["ckpt_uri"] == "/data/b/last.ckpt"


def test_crlf_line_endings_are_handled(bc, tmp_path):
    """Spreadsheet exports and Windows editors produce these."""
    listing = tmp_path / "l.txt"
    listing.write_bytes(b"tok_a,/data/a/last.ckpt\r\ntok_b,/data/b/last.ckpt\r\n")
    rows = bc.read_list(listing)
    assert [r["tokenizer"] for r in rows] == ["tok_a", "tok_b"]
    assert all(not r["ckpt_uri"].endswith("\r") for r in rows)


def test_local_paths_are_absolutised(bc, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    listing = tmp_path / "l.txt"
    listing.write_text("rel/d/last.ckpt\n")
    got = bc.read_list(listing)[0]["ckpt_uri"]
    assert Path(got).is_absolute(), f"relative path left unresolved: {got!r}"
    assert got.endswith("rel/d/last.ckpt")


def test_tracker_uris_are_not_mangled_by_normalisation(bc):
    for uri in (
        "mlflow-artifacts:/12/abc/artifacts/last.ckpt",
        "mlflow:/12/abc/artifacts/last.ckpt",
        "s3://bucket/x.ckpt",
    ):
        assert bc.normalise_uri(uri) == uri


# ── table validation ────────────────────────────────────────────────────────────
def test_check_rows_accepts_a_minimal_two_column_table(bc, tmp_path):
    ck = tmp_path / "a.ckpt"
    ck.touch()
    assert bc.check_rows([{"tokenizer": "tok_one", "ckpt_uri": str(ck)}]) == []


def test_check_rows_flags_unsafe_name(bc, tmp_path):
    ck = tmp_path / "a.ckpt"
    ck.touch()
    problems = bc.check_rows([{"tokenizer": "My Tokenizer", "ckpt_uri": str(ck)}])
    assert any("filesystem-safe" in p for p in problems)


def test_check_rows_flags_duplicates_missing_and_bad_schemes(bc, tmp_path):
    ck = tmp_path / "a.ckpt"
    ck.touch()
    problems = bc.check_rows(
        [
            {"tokenizer": "dup", "ckpt_uri": str(ck)},
            {"tokenizer": "dup", "ckpt_uri": str(ck)},
            {"tokenizer": "gone", "ckpt_uri": "/nope/nowhere.ckpt"},
            {"tokenizer": "s3", "ckpt_uri": "s3://bucket/x.ckpt"},
            {"tokenizer": "", "ckpt_uri": str(ck)},
            {"tokenizer": "nouri", "ckpt_uri": ""},
        ]
    )
    joined = " | ".join(problems)
    assert "duplicate tokenizer 'dup'" in joined
    assert "does not exist" in joined
    assert "s3://" in joined
    assert "empty tokenizer" in joined
    assert "empty ckpt_uri" in joined


def test_check_rows_accepts_resolvable_tracker_uris(bc):
    rows = [{"tokenizer": "t", "ckpt_uri": "mlflow-artifacts:/1/a/artifacts/last.ckpt"}]
    assert bc.check_rows(rows) == []


def test_check_rows_empty_table_is_a_problem(bc):
    assert bc.check_rows([]) == ["table has no rows"]


def test_check_table_exit_codes(bc, tmp_path):
    ck = tmp_path / "a.ckpt"
    ck.touch()
    good = tmp_path / "good.csv"
    good.write_text(f"tokenizer,ckpt_uri\ntok_one,{ck}\n")
    assert bc.check_table(good) == 0

    bad = tmp_path / "bad.csv"
    bad.write_text("tokenizer,ckpt_uri\ndup,/nope.ckpt\ndup,/nope.ckpt\n")
    assert bc.check_table(bad) == 1


def test_check_table_reports_missing_required_column(bc, tmp_path):
    t = tmp_path / "t.csv"
    t.write_text("tokenizer\nonly_a_name\n")
    assert bc.check_table(t) == 1


def test_main_check_requires_no_out(bc, tmp_path):
    ck = tmp_path / "a.ckpt"
    ck.touch()
    t = tmp_path / "t.csv"
    t.write_text(f"tokenizer,ckpt_uri\ntok_one,{ck}\n")
    assert bc.main(["--check", str(t)]) == 0


def test_main_requires_out_for_discovery(bc, tmp_path):
    assert bc.main(["--from-dir", str(tmp_path)]) == 2


# ── checkpoint-artifact layout resolution ───────────────────────────────────────
class _Art:
    def __init__(self, path: str, is_dir: bool = False) -> None:
        self.path = path
        self.is_dir = is_dir


def _lister(tree: dict[str | None, list[_Art]]):
    """Fake list_artifacts(path=None) with MLflow's semantics, counting the calls made."""
    calls: list[str | None] = []

    def ls(path=None):
        calls.append(path)
        return tree.get(path, [])

    ls.calls = calls  # type: ignore[attr-defined]
    return ls


def test_layout_checkpoints_last(bc):
    ls = _lister(
        {None: [_Art("checkpoints", True)], "checkpoints": [_Art("checkpoints/last.ckpt")]}
    )
    assert bc.resolve_ckpt_artifact(ls, "rid") == "checkpoints/last.ckpt"


def test_layout_top_level_last(bc):
    ls = _lister({None: [_Art("last.ckpt")]})
    assert bc.resolve_ckpt_artifact(ls, "rid") == "last.ckpt"


def test_layout_last_dir(bc):
    """The layout the real nuclr-byol-r4 experiment actually uses."""
    ls = _lister({None: [_Art("last", True)], "last": [_Art("last/last.ckpt")]})
    assert bc.resolve_ckpt_artifact(ls, "rid") == "last/last.ckpt"


def test_layout_highest_version_wins(bc):
    ls = _lister(
        {
            None: [_Art("last", True), _Art("last-v1", True), _Art("last-v2", True)],
            "last": [_Art("last/last.ckpt")],
            "last-v1": [_Art("last-v1/last.ckpt")],
            "last-v2": [_Art("last-v2/last.ckpt")],
        }
    )
    assert bc.resolve_ckpt_artifact(ls, "rid") == "last-v2/last.ckpt"


def test_layout_version_sort_is_numeric_not_lexical(bc):
    ls = _lister(
        {
            None: [_Art("last-v9", True), _Art("last-v10", True)],
            "last-v9": [_Art("last-v9/last.ckpt")],
            "last-v10": [_Art("last-v10/last.ckpt")],
        }
    )
    assert bc.resolve_ckpt_artifact(ls, "rid") == "last-v10/last.ckpt"


def test_layout_empty_highest_version_falls_back(bc):
    ls = _lister(
        {
            None: [_Art("last-v1", True), _Art("last-v2", True)],
            "last-v1": [_Art("last-v1/last.ckpt")],
            "last-v2": [],
        }
    )
    assert bc.resolve_ckpt_artifact(ls, "rid") == "last-v1/last.ckpt"


def test_layout_canonical_beats_top_level(bc):
    ls = _lister(
        {
            None: [_Art("last.ckpt"), _Art("checkpoints", True)],
            "checkpoints": [_Art("checkpoints/last.ckpt")],
        }
    )
    assert bc.resolve_ckpt_artifact(ls, "rid") == "checkpoints/last.ckpt"


def test_layout_single_noncanonical_ckpt_is_accepted(bc):
    ls = _lister({None: [_Art("epoch3", True)], "epoch3": [_Art("epoch3/epoch3.ckpt")]})
    assert bc.resolve_ckpt_artifact(ls, "rid") == "epoch3/epoch3.ckpt"


def test_layout_ambiguous_warns_rather_than_guessing(bc, caplog):
    ls = _lister(
        {
            None: [_Art("epoch3", True), _Art("epoch8", True)],
            "epoch3": [_Art("epoch3/epoch3.ckpt")],
            "epoch8": [_Art("epoch8/epoch8.ckpt")],
        }
    )
    with caplog.at_level("WARNING"):
        assert bc.resolve_ckpt_artifact(ls, "rid") is None
    assert "none canonical" in caplog.text


def test_layout_deep_miss_is_noisy_not_silent(bc, caplog):
    """A checkpoint nested deeper than one level must not look like 'no checkpoint'."""
    ls = _lister(
        {
            None: [_Art("checkpoints", True)],
            "checkpoints": [_Art("checkpoints/run1", True)],
        }
    )
    with caplog.at_level("WARNING"):
        assert bc.resolve_ckpt_artifact(ls, "rid") is None
    assert "--ckpt-name" in caplog.text, "a deep miss should tell you how to recover"


def test_layout_no_artifacts_at_all_is_quiet(bc, caplog):
    with caplog.at_level("WARNING"):
        assert bc.resolve_ckpt_artifact(_lister({None: []}), "rid") is None
    assert caplog.text == "", "nothing to search is not a warning-worthy condition"


def test_resolution_does_not_list_irrelevant_directories(bc):
    """Healthy runs must not pay for listing large sibling dirs (train/ has 1000s of PNGs)."""
    ls = _lister(
        {
            None: [_Art("last", True), _Art("train", True), _Art("val", True)],
            "last": [_Art("last/last.ckpt")],
            "train": [_Art(f"train/{i}.png") for i in range(3566)],
            "val": [_Art("val/0.png")],
        }
    )
    assert bc.resolve_ckpt_artifact(ls, "rid") == "last/last.ckpt"
    assert "train" not in ls.calls, f"listed train/ unnecessarily; calls={ls.calls}"
    assert "val" not in ls.calls


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("last/last.ckpt", "last/last.ckpt"),
        ("last.ckpt", "last.ckpt"),
        ("nope/nope.ckpt", None),
    ],
)
def test_ckpt_name_bypasses_probing(bc, name, expected):
    ls = _lister({None: [_Art("last.ckpt"), _Art("last", True)], "last": [_Art("last/last.ckpt")]})
    assert bc.resolve_ckpt_artifact(ls, "rid", name) == expected


def test_ckpt_name_lists_only_the_named_parent(bc):
    ls = _lister({"last": [_Art("last/last.ckpt")]})
    bc.resolve_ckpt_artifact(ls, "rid", "last/last.ckpt")
    assert ls.calls == ["last"], f"should list only the parent; got {ls.calls}"


# ── CLI wiring ──────────────────────────────────────────────────────────────────
def test_main_round_trip_from_dir(bc, tmp_path):
    ck = tmp_path / "tok_one" / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "last.ckpt").touch()
    out = tmp_path / "cohort.csv"

    rc = bc.main(["--from-dir", str(tmp_path), "--out", str(out)])
    assert rc == 0
    _, rows = _read(out)
    assert [r["tokenizer"] for r in rows] == ["tok_one"]


def test_main_returns_nonzero_when_nothing_found(bc, tmp_path):
    out = tmp_path / "cohort.csv"
    assert bc.main(["--from-dir", str(tmp_path), "--out", str(out)]) == 1
    assert not out.exists(), "no cohort file should be written when discovery found nothing"


def test_discovery_sources_are_mutually_exclusive(bc, tmp_path):
    with pytest.raises(SystemExit):
        bc.parse_args(["--from-dir", str(tmp_path), "--from-mlflow", "exp", "--out", "x.csv"])


def test_a_discovery_source_is_required(bc):
    with pytest.raises(SystemExit):
        bc.parse_args(["--out", "x.csv"])
