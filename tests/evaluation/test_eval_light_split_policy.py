"""Guards the per-tier split policy of the eval_light data configs:
xtime sessions load split-respecting (ignore_splits=False) so the
generalization baseline is leak-free; OOD sessions load full-recording.

`ntt_eval_light` is the default light-eval data config."""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from ntt.core.data.components.dataset import NTTDataset

REPO = Path(__file__).resolve().parents[2]
COMPOSITE = REPO / "config/data/selections/composite"

PUBLIC_BRAINSETS = frozenset(
    {
        "IBL_brainwide_2023",
        "allen_visual_behaviour_neuropixels",
        "allen_visual_coding_neuropixels",
        "c4_dataset_cerebellum_2025",
        "churchland_shenoy_neural_2012",
        "cowley_v4_compact_2024",
        "flint_slutzky_accurate_2012",
        "odoherty_sabes_nonhuman_2017",
        "perich_miller_population_2018",
        "wingert_auditory_cortex_2026",
    }
)

EXPECTED_XTIME = frozenset(
    {
        "IBL_brainwide_2023/csh_zad_011_20200321_csh_zad_011",
        "allen_visual_behaviour_neuropixels/mouse521466_session1049273528",
        "allen_visual_coding_neuropixels/specimen699733581_session715093703",
        "churchland_shenoy_neural_2012/jenkins_20090918_center_out_reaching",
        "flint_slutzky_accurate_2012/monkey_c_e4_reaching",
        "odoherty_sabes_nonhuman_2017/indy_20160407_02",
        "perich_miller_population_2018/c_20131029_random_target_reaching",
    }
)
EXPECTED_OOD = frozenset(
    {
        "IBL_brainwide_2023/csh_zad_022_20200526_csh_zad_022",
        "IBL_brainwide_2023/ucla005_20210802_ucla005",
        "allen_visual_behaviour_neuropixels/mouse544456_session1064400234",
        "allen_visual_coding_neuropixels/specimen763808604_session789848216",
        "c4_dataset_cerebellum_2025/hausser_19-01-24_YC013_probe1",
        "churchland_shenoy_neural_2012/nitschke_20090819_center_out_reaching",
        "cowley_v4_compact_2024/PE_201015",
        "flint_slutzky_accurate_2012/monkey_c_e1_reaching",
        "odoherty_sabes_nonhuman_2017/indy_20160419_01",
        "odoherty_sabes_nonhuman_2017/loco_20170210_03",
        "perich_miller_population_2018/c_20151104_center_out_reaching",
        "perich_miller_population_2018/j_20160406_center_out_reaching",
        "wingert_auditory_cortex_2026/CLT027c",
    }
)

# (data config, xtime composite, ood composite, expected xtime, expected ood)
VARIANTS = [
    pytest.param(
        "ntt_eval_light",
        "ntt_eval_light_xtime",
        "ntt_eval_light_ood",
        EXPECTED_XTIME,
        EXPECTED_OOD,
        id="light",
    ),
]


def _ignore_splits(path: Path) -> bool:
    cfg = OmegaConf.load(path)
    return bool(cfg.get("config", {}).get("ignore_splits", False))


def _session_ids(path: Path) -> set[str]:
    cfg = OmegaConf.load(path)
    out: set[str] = set()
    for sel in cfg.get("selection", []):
        bs = sel.get("brainset", "")
        for s in sel.get("sessions", []) or []:
            out.add(f"{bs}/{s}")
    return out


@pytest.mark.parametrize(("data", "xtime", "ood", "exp_xtime", "exp_ood"), VARIANTS)
def test_xtime_composite_is_split_respecting(data, xtime, ood, exp_xtime, exp_ood):
    assert _ignore_splits(COMPOSITE / f"{xtime}.yaml") is False


@pytest.mark.parametrize(("data", "xtime", "ood", "exp_xtime", "exp_ood"), VARIANTS)
def test_ood_composite_is_full_recording(data, xtime, ood, exp_xtime, exp_ood):
    assert _ignore_splits(COMPOSITE / f"{ood}.yaml") is True


@pytest.mark.parametrize(("data", "xtime", "ood", "exp_xtime", "exp_ood"), VARIANTS)
def test_data_config_references_both_blocks(data, xtime, ood, exp_xtime, exp_ood):
    text = (REPO / f"config/data/{data}.yaml").read_text()
    assert f"composite/{xtime}\n" in text
    assert f"composite/{ood}\n" in text


@pytest.mark.parametrize(("data", "xtime", "ood", "exp_xtime", "exp_ood"), VARIANTS)
def test_exact_session_membership(data, xtime, ood, exp_xtime, exp_ood):
    """Pin the canonical per-tier membership. Catches a session being added,
    dropped, OR swapped between blocks (a same-count swap would corrupt the
    generalization baseline and is invisible to a count-only check). This
    test is the source of truth for the eval_light session split."""
    got_xtime = _session_ids(COMPOSITE / f"{xtime}.yaml")
    got_ood = _session_ids(COMPOSITE / f"{ood}.yaml")
    assert got_xtime == exp_xtime
    assert got_ood == exp_ood
    assert got_xtime.isdisjoint(got_ood)


def test_light_counts_and_public_brainsets_only():
    """The default holds 7 xtime + 13 OOD sessions, all from public brainsets,
    including in the train selections used for tokenizer spec resolution."""
    assert len(EXPECTED_XTIME) == 7 and len(EXPECTED_OOD) == 13
    brainsets = {
        entry.brainset
        for name in _defaults_selections("ntt_eval_light", "selections@")
        for entry in OmegaConf.load(SELECTIONS / f"{name}.yaml").selection
    }
    assert brainsets and brainsets <= PUBLIC_BRAINSETS


EXPECTED_ROBUSTNESS = frozenset(
    {
        "perich_miller_population_2018/t_20130830_random_target_reaching",
        "cowley_v4_compact_2024/PE_201014",
        "c4_dataset_cerebellum_2025/hausser_18-08-30_YC001_probe1",
        "IBL_brainwide_2023/ibl_witten_17_20200723_ibl_witten_17",
    }
)

# (selection / data config name, expected membership)
ROBUSTNESS_VARIANTS = [
    pytest.param("robustness_fast_eval", EXPECTED_ROBUSTNESS, id="fast"),
]


@pytest.mark.parametrize(("name", "expected"), ROBUSTNESS_VARIANTS)
def test_robustness_selection_membership_and_split_policy(name, expected):
    """M1/M2 robustness selection: exact membership, full-recording loading."""
    path = REPO / f"config/data/selections/{name}.yaml"
    assert _session_ids(path) == expected
    assert _ignore_splits(path) is True


@pytest.mark.parametrize(("name", "expected"), ROBUSTNESS_VARIANTS)
def test_robustness_data_config_uses_its_selection(name, expected):
    text = (REPO / f"config/data/{name}.yaml").read_text()
    assert f"train_selections.0: {name}\n" in text
    assert f"val_selections.0: {name}\n" in text


def test_robustness_sessions_are_in_test_composites():
    """Every robustness session must come from a held-out test
    composite (ignore_splits=true scoring is only leak-free for never-trained
    sessions). Also checks none overlaps the eval_light composites."""
    held_out: set[str] = set()
    for name in (
        "ntt_full_test_xsession",
        "ntt_full_test_xsubject",
        "ntt_full_test_xsource",
    ):
        held_out |= _session_ids(COMPOSITE / f"{name}.yaml")
    assert EXPECTED_ROBUSTNESS <= held_out
    light = _session_ids(COMPOSITE / "ntt_eval_light_xtime.yaml") | _session_ids(
        COMPOSITE / "ntt_eval_light_ood.yaml"
    )
    assert EXPECTED_ROBUSTNESS.isdisjoint(light)


def test_robustness_names_only_public_brainsets():
    for rel in (
        "config/data/selections/robustness_fast_eval.yaml",
        "config/data/robustness_fast_eval.yaml",
    ):
        names = set(re.findall(r"brainset: (\w+)", (REPO / rel).read_text()))
        assert names <= PUBLIC_BRAINSETS, rel


class _StubDataset(NTTDataset):
    """Bypasses NTTDataset.__init__ to exercise get_sampling_intervals in
    isolation. get_sampling_intervals only touches recording_dict, split,
    _get_data_config, _should_ignore_splits and _get_data_object."""

    def __init__(self, recording_dict, split):  # noqa: D401 - test stub
        self.recording_dict = recording_dict
        self.split = split

    def _get_data_object(self, recording_id):
        return self.recording_dict[recording_id]["data"]


def _make_recording(ignore_splits: bool):
    data = SimpleNamespace(
        domain="FULL",
        valid_domain="VALID",
        test_domain="TEST",
    )
    return {"rec": {"config": {"ignore_splits": ignore_splits}, "data": data}}


def test_split_respecting_loads_valid_domain():
    ds = _StubDataset(_make_recording(ignore_splits=False), split="valid")
    assert ds.get_sampling_intervals() == {"rec": "VALID"}


def test_split_respecting_loads_test_domain():
    ds = _StubDataset(_make_recording(ignore_splits=False), split="test")
    assert ds.get_sampling_intervals() == {"rec": "TEST"}


def test_ignore_splits_loads_full_domain():
    ds = _StubDataset(_make_recording(ignore_splits=True), split="valid")
    assert ds.get_sampling_intervals() == {"rec": "FULL"}


# ---------------------------------------------------------------------------
# split=test wiring: test selections of every intrinsic-eval data config
# ---------------------------------------------------------------------------
SELECTIONS = REPO / "config/data/selections"
TEST_COMPOSITES = (
    "composite/ntt_full_test_xsession",
    "composite/ntt_full_test_xsubject",
    "composite/ntt_full_test_xsource",
)

# 13 brainset x tier cells of ntt_eval_light_ood, each filled from the test composites.
EXPECTED_OOD_TEST = frozenset(
    {
        "odoherty_sabes_nonhuman_2017/indy_20161011_03",
        "perich_miller_population_2018/c_20131220_center_out_reaching",
        "IBL_brainwide_2023/cshl045_20200226_cshl045",
        "allen_visual_behaviour_neuropixels/mouse536480_session1064415305",
        "churchland_shenoy_neural_2012/jenkins_20090923_center_out_reaching",
        "flint_slutzky_accurate_2012/monkey_c_e2_reaching",
        "odoherty_sabes_nonhuman_2017/loco_20170217_02",
        "perich_miller_population_2018/j_20160407_center_out_reaching",
        "IBL_brainwide_2023/ucla011_20210720_ucla011",
        "allen_visual_coding_neuropixels/specimen756578435_session771990200",
        "c4_dataset_cerebellum_2025/hausser_18-12-13_YC008_probe1",
        "cowley_v4_compact_2024/PE_201016",
        "wingert_auditory_cortex_2026/CLT034c",
    }
)
EXPECTED_SNR_TEST = frozenset({"cowley_v4_compact_2024/PE_201014"})


def _sel(rel: str) -> set[str]:
    return _session_ids(SELECTIONS / f"{rel}.yaml")


def _held_out_test_sessions() -> set[str]:
    out: set[str] = set()
    for name in TEST_COMPOSITES:
        out |= _sel(name)
    return out


def _cell(session_id: str) -> tuple[str, str]:
    """(tier, brainset) of a session, by the test composite listing it."""
    bs = session_id.split("/")[0]
    for tier, name in zip(("xsession", "xsubject", "xsource"), TEST_COMPOSITES, strict=True):
        if session_id in _sel(name):
            return tier, bs
    raise AssertionError(session_id)


def _defaults_selections(data: str, prefix: str) -> list[str]:
    """Selection names a data config's defaults list pulls in under `prefix`."""
    cfg = OmegaConf.load(REPO / f"config/data/{data}.yaml")
    return [
        str(value)
        for item in cfg.defaults
        if not isinstance(item, str)
        for key, value in item.items()
        if str(key).startswith(prefix)
    ]


def _data_config_test_ids(data: str) -> set[str]:
    """Session ids of the test_selections a data config pulls in."""
    ids: set[str] = set()
    for name in _defaults_selections(data, "selections@test_selections."):
        ids |= _sel(name)
    return ids


def test_ood_test_composite_mirrors_val_cells():
    got = _sel("composite/ntt_eval_light_ood_test")
    assert got == EXPECTED_OOD_TEST
    assert _ignore_splits(COMPOSITE / "ntt_eval_light_ood_test.yaml") is True
    # same (tier, brainset) cell structure as the val OOD block: 6 xsession, 4 xsubject, 3 xsource
    val = _sel("composite/ntt_eval_light_ood")
    val_cells = sorted((_tier_of_val(s), s.split("/")[0]) for s in val)
    assert sorted(_cell(s) for s in got) == val_cells
    # the val picks themselves are not reused
    assert got.isdisjoint(val)


def _tier_of_val(session_id: str) -> str:
    """Tier of a val OOD pick, from the val composites."""
    for tier, name in zip(
        ("xsession", "xsubject", "xsource"),
        (
            "composite/ntt_full_val_xsession",
            "composite/ntt_full_val_xsubject",
            "composite/ntt_full_val_xsource",
        ),
        strict=True,
    ):
        if session_id in _sel(name):
            return tier
    raise AssertionError(session_id)


def test_eval_light_test_selections_membership():
    xtime = _sel("composite/ntt_eval_light_xtime")
    assert _data_config_test_ids("ntt_eval_light") == xtime | EXPECTED_OOD_TEST
    assert _ignore_splits(COMPOSITE / "ntt_eval_light_xtime.yaml") is False


def test_other_eval_data_configs_test_selections():
    assert _data_config_test_ids("cowley_snr_fast_eval") == EXPECTED_SNR_TEST
    assert _data_config_test_ids("robustness_fast_eval") == EXPECTED_ROBUSTNESS
    assert _data_config_test_ids("cowley_v4_compact_val") == _sel(
        "cowley_v4_compact_2024_test_snr"
    )
    # the cowley test selections are only xsource-test sessions
    assert _sel("cowley_v4_compact_2024_test") <= _sel("composite/ntt_full_test_xsource")
    assert _sel("cowley_v4_compact_2024_test_snr") <= _sel("composite/ntt_full_test_xsource")


def test_cowley_snr_test_selection_drops_only_close_stimulus_sessions():
    """The precision-SNR test set is the cowley test set minus exactly the 3
    sessions whose stimulus spacing the per-stimulus sampler rejects."""
    assert _sel("cowley_v4_compact_2024_test_snr") == _sel("cowley_v4_compact_2024_test") - {
        "cowley_v4_compact_2024/PE_210303",
        "cowley_v4_compact_2024/RA_211025",
        "cowley_v4_compact_2024/WE_190927",
    }
    assert _ignore_splits(SELECTIONS / "cowley_v4_compact_2024_test_snr.yaml") is True


def test_cowley_snr_test_sessions_fit_per_stimulus_window():
    """Data-dependent: every SNR test session's stimulus starts are at least
    fixed_window_duration_ms - cross_stim_padding_ms apart, so the sampler's
    overlap check cannot fire. Skips when the h5 files are absent."""
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    pipeline = OmegaConf.load(REPO / "config/evaluation/pipelines/cowley_per_stimulus.yaml")
    min_gap_s = (pipeline.fixed_window_duration_ms - pipeline.cross_stim_padding_ms) / 1000.0
    root = os.environ.get("NTT_TEST_DATA_ROOT") or str(REPO / "data" / "brainsets" / "processed")
    data_dir = Path(root)
    for sid in sorted(_sel("cowley_v4_compact_2024_test_snr")):
        path = data_dir / f"{sid}.h5"
        if not path.exists():
            pytest.skip(f"data not available: {path}")
        with h5py.File(path, "r") as f:
            starts = np.sort(f["stimuli/start"][:])
        assert np.diff(starts).min() >= min_gap_s - 1e-9, sid


def _is_trained(session_id: str) -> bool:
    """True if the ntt_full train selections include the session.

    Only the full config is checked: the medium/small train lists are not yet
    subsets of full train (being fixed separately), so they would flag
    sessions that the full model never sees.

    Exclude-list selections train on everything except the listed sessions;
    include-list selections train on exactly the listed ones. Brainsets with
    no public train selection (cowley, c4, wingert) are never trained on."""
    brainset, session = session_id.split("/", 1)
    for data in ("ntt_full",):
        for name in _defaults_selections(data, "selections@train_selections."):
            for entry in OmegaConf.load(SELECTIONS / f"{name}.yaml").selection:
                if entry.brainset != brainset:
                    continue
                if "exclude_sessions" in entry:
                    if session not in {str(x) for x in entry.exclude_sessions}:
                        return True
                elif session in {str(x) for x in entry.get("sessions", [])}:
                    return True
    return False


def test_full_recording_test_sessions_are_held_out():
    """Every ignore_splits=true test session lives in an ntt_full_test_*
    composite (so aggregation's tier lookup finds it) and is not trained on."""
    held_out = _held_out_test_sessions()
    full_recording = (
        EXPECTED_OOD_TEST
        | EXPECTED_SNR_TEST
        | EXPECTED_ROBUSTNESS
        | _sel("cowley_v4_compact_2024_test")
        | _sel("cowley_v4_compact_2024_test_snr")
    )
    assert full_recording <= held_out
    assert not [sid for sid in full_recording if _is_trained(sid)]


def test_test_selections_only_public_brainsets():
    names = [
        "composite/ntt_eval_light_ood_test",
        "cowley_snr_fast_eval_test",
        "cowley_v4_compact_2024_test",
        "cowley_v4_compact_2024_test_snr",
        "robustness_fast_eval",
    ]
    for name in names:
        assert {sid.split("/")[0] for sid in _sel(name)} <= PUBLIC_BRAINSETS, name
    for data in ("ntt_eval_light", "cowley_snr_fast_eval", "robustness_fast_eval"):
        assert {sid.split("/")[0] for sid in _data_config_test_ids(data)} <= PUBLIC_BRAINSETS, data
