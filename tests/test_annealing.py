"""Tests for the general annealing schedule and callback."""

from unittest.mock import MagicMock

import pytest
from torch import nn

from ntt.contrib.tokenizer.wanneural.bottleneck import (
    BSQZetaAnnealingCallback,
    VAEBetaAnnealingCallback,
    VAEBottleneck,
)
from ntt.core.model.components.annealing import (
    AnnealingCallback,
    compute_annealed_value,
    find_target_submodules,
)


class TestComputeAnnealedValue:
    """Test the core schedule computation."""

    def test_linear_start(self):
        """Value should be value_min at the start of a cycle."""
        v = compute_annealed_value(
            global_step=0,
            max_steps=100,
            schedule="linear",
            value_min=0.0,
            value_max=1.0,
            n_cycles=1,
            ratio=0.5,
            delay_steps=0,
        )
        assert v == pytest.approx(0.0)

    def test_linear_midpoint_one_cycle(self):
        """Linear schedule at midpoint of annealing phase (R=0.5, M=1)."""
        v = compute_annealed_value(
            global_step=25,
            max_steps=100,
            schedule="linear",
            value_min=0.0,
            value_max=1.0,
            n_cycles=1,
            ratio=0.5,
            delay_steps=0,
        )
        assert v == pytest.approx(0.5)

    def test_linear_fixed_phase(self):
        """Value should be value_max during the fixed phase."""
        v = compute_annealed_value(
            global_step=60,
            max_steps=100,
            schedule="linear",
            value_min=0.0,
            value_max=1.0,
            n_cycles=1,
            ratio=0.5,
            delay_steps=0,
        )
        assert v == pytest.approx(1.0)

    def test_cosine_start_and_end(self):
        """Cosine schedule: starts at value_min, reaches value_max at end of annealing."""
        v_start = compute_annealed_value(
            global_step=0,
            max_steps=100,
            schedule="cosine",
            value_min=0.0,
            value_max=1.0,
            n_cycles=1,
            ratio=0.5,
            delay_steps=0,
        )
        assert v_start == pytest.approx(0.0)

        v_end = compute_annealed_value(
            global_step=49,
            max_steps=100,
            schedule="cosine",
            value_min=0.0,
            value_max=1.0,
            n_cycles=1,
            ratio=0.5,
            delay_steps=0,
        )
        assert v_end == pytest.approx(1.0, abs=0.05)

    def test_sigmoid_midpoint(self):
        """Sigmoid at p=0.5 should give ~0.5 (inflection point: 12*0.5-6=0)."""
        v = compute_annealed_value(
            global_step=25,
            max_steps=100,
            schedule="sigmoid",
            value_min=0.0,
            value_max=1.0,
            n_cycles=1,
            ratio=0.5,
            delay_steps=0,
        )
        assert v == pytest.approx(0.5, abs=0.01)

    def test_delay_steps(self):
        """Value should stay at value_min during delay period."""
        v = compute_annealed_value(
            global_step=50,
            max_steps=200,
            schedule="linear",
            value_min=0.0,
            value_max=1.0,
            n_cycles=1,
            ratio=0.5,
            delay_steps=100,
        )
        assert v == pytest.approx(0.0)

    def test_delay_then_anneal(self):
        """After delay, annealing should start from value_min."""
        v = compute_annealed_value(
            global_step=125,
            max_steps=200,
            schedule="linear",
            value_min=0.0,
            value_max=1.0,
            n_cycles=1,
            ratio=0.5,
            delay_steps=100,
        )
        assert v == pytest.approx(0.5)

    def test_cyclical_resets(self):
        """With M=4 cycles, value should reset at each cycle boundary."""
        v_cycle2_start = compute_annealed_value(
            global_step=25,
            max_steps=100,
            schedule="linear",
            value_min=0.0,
            value_max=1.0,
            n_cycles=4,
            ratio=0.5,
            delay_steps=0,
        )
        assert v_cycle2_start == pytest.approx(0.0)

    def test_cyclical_fixed_phase(self):
        """In cyclical mode, fixed phase within each cycle should be value_max."""
        v = compute_annealed_value(
            global_step=15,
            max_steps=100,
            schedule="linear",
            value_min=0.0,
            value_max=1.0,
            n_cycles=4,
            ratio=0.5,
            delay_steps=0,
        )
        assert v == pytest.approx(1.0)

    def test_custom_range(self):
        """Schedule should interpolate between value_min and value_max."""
        v = compute_annealed_value(
            global_step=25,
            max_steps=100,
            schedule="linear",
            value_min=1e-7,
            value_max=1e-3,
            n_cycles=1,
            ratio=0.5,
            delay_steps=0,
        )
        expected = 1e-7 + (1e-3 - 1e-7) * 0.5
        assert v == pytest.approx(expected, rel=1e-5)

    def test_ramp_down(self):
        """value_min > value_max ramps DOWN (the BSQ-zeta case)."""
        v = compute_annealed_value(
            global_step=25,
            max_steps=100,
            schedule="linear",
            value_min=1.0,
            value_max=0.0,
            n_cycles=1,
            ratio=0.5,
            delay_steps=0,
        )
        assert v == pytest.approx(0.5)

    def test_invalid_schedule_raises(self):
        """Unknown schedule name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown schedule"):
            compute_annealed_value(
                global_step=0,
                max_steps=100,
                schedule="invalid",
                value_min=0.0,
                value_max=1.0,
                n_cycles=1,
                ratio=0.5,
                delay_steps=0,
            )


class FakeTokenizer(nn.Module):
    """Minimal tokenizer containing a real VAEBottleneck."""

    def __init__(self):
        super().__init__()
        self.bottleneck = VAEBottleneck(z_dim=4, beta=1e-6)


class TestFindTargetSubmodules:
    """Test the standalone submodule-location helper."""

    class _Tokenizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = VAEBottleneck(z_dim=4, beta=1e-6)
            self.b = VAEBottleneck(z_dim=4, beta=1e-6)
            self.other = nn.Linear(2, 2)

    def test_returns_all_matches_in_tree_order(self):
        tok = self._Tokenizer()
        found = find_target_submodules(tok, VAEBottleneck)
        assert found == [("a", tok.a), ("b", tok.b)]

    def test_single_name(self):
        tok = self._Tokenizer()
        assert find_target_submodules(tok, VAEBottleneck, "b") == [("b", tok.b)]

    def test_list_of_names_preserves_order(self):
        tok = self._Tokenizer()
        found = find_target_submodules(tok, VAEBottleneck, ["b", "a"])
        assert found == [("b", tok.b), ("a", tok.a)]

    def test_unknown_name_raises(self):
        tok = self._Tokenizer()
        with pytest.raises(RuntimeError, match=r"named \['other'\]"):
            # 'other' exists but is not a VAEBottleneck, so it does not match.
            find_target_submodules(tok, VAEBottleneck, ["a", "other"])

    def test_no_match_raises(self):
        with pytest.raises(RuntimeError, match="no VAEBottleneck submodule"):
            find_target_submodules(nn.Linear(1, 1), VAEBottleneck)

    def test_unwraps_torch_compile(self):
        tok = self._Tokenizer()
        wrapper = nn.Module()
        wrapper._orig_mod = tok  # mimic torch.compile's OptimizedModule
        assert find_target_submodules(wrapper, VAEBottleneck) == [("a", tok.a), ("b", tok.b)]


class TestAnnealingCallback:
    """Test the Lightning callback integration."""

    def _make_trainer_and_module(self, max_steps=-1, max_epochs=1, global_step=0):
        trainer = MagicMock()
        trainer.max_steps = max_steps
        trainer.max_epochs = max_epochs
        trainer.estimated_stepping_batches = 100
        trainer.global_step = global_step
        pl_module = MagicMock()
        pl_module.tokenizer = FakeTokenizer()
        pl_module.log = MagicMock()
        return trainer, pl_module

    def _beta_cb(self, **kwargs):
        defaults = {
            "schedule": "linear",
            "value_max": 1.0,
            "value_min": 0.0,
            "n_cycles": 1,
            "ratio": 0.5,
            "delay_steps": 0,
        }
        defaults.update(kwargs)
        return VAEBetaAnnealingCallback(**defaults)

    def test_base_class_cannot_be_instantiated(self):
        """AnnealingCallback is abstract: target_class/target_param unset."""
        with pytest.raises(TypeError, match="abstract"):
            AnnealingCallback(schedule="linear")

    def test_subclass_missing_target_fails_at_definition(self):
        """__init_subclass__ enforces the contract at import/definition time."""
        with pytest.raises(TypeError, match="target_class.*target_param"):

            class BadCallback(AnnealingCallback):
                pass  # neither target_class nor target_param set

    def test_on_train_start_finds_module_by_class(self):
        cb = self._beta_cb(n_cycles=4)
        trainer, pl_module = self._make_trainer_and_module()
        cb.on_train_start(trainer, pl_module)
        assert [m for _, m in cb._targets] == [pl_module.tokenizer.bottleneck]

    def test_on_train_start_uses_estimated_stepping_batches(self):
        cb = self._beta_cb()
        trainer, pl_module = self._make_trainer_and_module(max_steps=-1)
        trainer.estimated_stepping_batches = 200
        cb.on_train_start(trainer, pl_module)
        assert cb._max_steps == 200

    def test_on_train_start_uses_max_steps_when_set(self):
        cb = self._beta_cb()
        trainer, pl_module = self._make_trainer_and_module(max_steps=500)
        cb.on_train_start(trainer, pl_module)
        assert cb._max_steps == 500

    def test_on_train_batch_start_sets_param(self):
        cb = self._beta_cb()
        trainer, pl_module = self._make_trainer_and_module()
        cb.on_train_start(trainer, pl_module)
        trainer.global_step = 25
        cb.on_train_batch_start(trainer, pl_module, None, 0)
        assert pl_module.tokenizer.bottleneck.beta == pytest.approx(0.5)

    def test_on_train_batch_start_logs_default_key(self):
        cb = self._beta_cb()
        trainer, pl_module = self._make_trainer_and_module()
        cb.on_train_start(trainer, pl_module)
        trainer.global_step = 25
        cb.on_train_batch_start(trainer, pl_module, None, 0)
        pl_module.log.assert_any_call("train/beta", pytest.approx(0.5))

    def test_custom_log_key(self):
        cb = self._beta_cb(log_key="custom/metric")
        trainer, pl_module = self._make_trainer_and_module()
        cb.on_train_start(trainer, pl_module)
        trainer.global_step = 25
        cb.on_train_batch_start(trainer, pl_module, None, 0)
        pl_module.log.assert_any_call("custom/metric", pytest.approx(0.5))

    def test_missing_target_class_raises(self):
        cb = self._beta_cb()
        trainer = MagicMock()
        trainer.max_steps = 100
        pl_module = MagicMock()
        pl_module.tokenizer = nn.Linear(1, 1)
        with pytest.raises(RuntimeError, match="no VAEBottleneck submodule"):
            cb.on_train_start(trainer, pl_module)

    def test_checkpoint_resume_uses_global_step(self):
        cb = self._beta_cb()
        trainer, pl_module = self._make_trainer_and_module()
        cb.on_train_start(trainer, pl_module)
        trainer.global_step = 50
        cb.on_train_batch_start(trainer, pl_module, None, 0)
        assert pl_module.tokenizer.bottleneck.beta == pytest.approx(1.0)

    def test_torch_compile_unwrapping(self):
        cb = self._beta_cb()
        trainer, pl_module = self._make_trainer_and_module()
        real_tokenizer = pl_module.tokenizer
        compiled_wrapper = MagicMock()
        compiled_wrapper._orig_mod = real_tokenizer
        pl_module.tokenizer = compiled_wrapper
        cb.on_train_start(trainer, pl_module)
        assert [m for _, m in cb._targets] == [real_tokenizer.bottleneck]

    def test_raw_logging_from_class_attrs(self):
        """VAEBetaAnnealingCallback logs _last_kl -> train/kl_raw by default."""
        cb = self._beta_cb()
        trainer, pl_module = self._make_trainer_and_module()
        cb.on_train_start(trainer, pl_module)
        cb._targets[0][1]._last_kl = 2.5
        cb.on_train_batch_end(trainer, pl_module, None, None, 0)
        pl_module.log.assert_any_call("train/kl_raw", 2.5)

    def test_no_raw_logging_when_raw_attr_unset(self):
        """A subclass without raw_attr does no raw logging."""

        class NoRawCallback(AnnealingCallback):
            target_class = VAEBottleneck
            target_param = "beta"

        cb = NoRawCallback(schedule="linear", value_max=1.0, value_min=0.0)
        trainer, pl_module = self._make_trainer_and_module()
        cb.on_train_start(trainer, pl_module)
        cb._targets[0][1]._last_kl = 2.5
        cb.on_train_batch_end(trainer, pl_module, None, None, 0)
        for call in pl_module.log.call_args_list:
            assert call.args[0] != "train/kl_raw"

    def test_zeta_subclass_targets_bsq(self):
        """BSQZetaAnnealingCallback locates a BSQBottleneck and anneals zeta."""

        class BSQTokenizer(nn.Module):
            def __init__(self):
                super().__init__()
                from ntt.contrib.tokenizer.wanneural.bottleneck import BSQBottleneck

                self.bottleneck = BSQBottleneck(z_dim=8, codebook_dim=8)

        cb = BSQZetaAnnealingCallback(
            schedule="linear", value_max=1.0, value_min=0.0, n_cycles=1, ratio=0.5
        )
        trainer, pl_module = self._make_trainer_and_module()
        pl_module.tokenizer = BSQTokenizer()
        cb.on_train_start(trainer, pl_module)
        trainer.global_step = 25
        cb.on_train_batch_start(trainer, pl_module, None, 0)
        assert pl_module.tokenizer.bottleneck.zeta == pytest.approx(0.5)
        pl_module.log.assert_any_call("train/zeta", pytest.approx(0.5))

    def test_generalizes_to_arbitrary_target(self):
        """A subclass drives any (class, attribute) pair, not just beta."""

        class GammaModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.gamma = 0.0

        class GammaTokenizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = GammaModule()

        class GammaCallback(AnnealingCallback):
            target_class = GammaModule
            target_param = "gamma"

        cb = GammaCallback(schedule="linear", value_max=1.0, value_min=0.0, n_cycles=1, ratio=0.5)
        trainer, pl_module = self._make_trainer_and_module()
        pl_module.tokenizer = GammaTokenizer()
        cb.on_train_start(trainer, pl_module)
        trainer.global_step = 25
        cb.on_train_batch_start(trainer, pl_module, None, 0)
        assert pl_module.tokenizer.inner.gamma == pytest.approx(0.5)

    def test_anneals_all_matching_modules_by_default(self):
        """With no target_name, every matching submodule is annealed in lockstep."""

        class TwoBottleneckTokenizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = VAEBottleneck(z_dim=4, beta=1e-6)
                self.second = VAEBottleneck(z_dim=4, beta=1e-6)

        cb = self._beta_cb()
        trainer, pl_module = self._make_trainer_and_module()
        pl_module.tokenizer = TwoBottleneckTokenizer()
        cb.on_train_start(trainer, pl_module)
        assert {name for name, _ in cb._targets} == {"first", "second"}
        trainer.global_step = 25
        cb.on_train_batch_start(trainer, pl_module, None, 0)
        assert pl_module.tokenizer.first.beta == pytest.approx(0.5)
        assert pl_module.tokenizer.second.beta == pytest.approx(0.5)

    def test_target_name_selects_single_module(self):
        """target_name restricts annealing to the one submodule with that name."""

        class TwoBottleneckTokenizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = VAEBottleneck(z_dim=4, beta=1e-6)
                self.second = VAEBottleneck(z_dim=4, beta=1e-6)

        cb = self._beta_cb(target_name="second")
        trainer, pl_module = self._make_trainer_and_module()
        pl_module.tokenizer = TwoBottleneckTokenizer()
        cb.on_train_start(trainer, pl_module)
        assert [name for name, _ in cb._targets] == ["second"]
        trainer.global_step = 25
        cb.on_train_batch_start(trainer, pl_module, None, 0)
        assert pl_module.tokenizer.first.beta == pytest.approx(1e-6)  # untouched
        assert pl_module.tokenizer.second.beta == pytest.approx(0.5)

    def test_target_name_selects_multiple_modules(self):
        """target_name accepts a list, selecting those modules in requested order."""

        class ThreeBottleneckTokenizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = VAEBottleneck(z_dim=4, beta=1e-6)
                self.second = VAEBottleneck(z_dim=4, beta=1e-6)
                self.third = VAEBottleneck(z_dim=4, beta=1e-6)

        cb = self._beta_cb(target_name=["third", "first"])
        trainer, pl_module = self._make_trainer_and_module()
        pl_module.tokenizer = ThreeBottleneckTokenizer()
        cb.on_train_start(trainer, pl_module)
        assert [name for name, _ in cb._targets] == ["third", "first"]
        trainer.global_step = 25
        cb.on_train_batch_start(trainer, pl_module, None, 0)
        assert pl_module.tokenizer.first.beta == pytest.approx(0.5)
        assert pl_module.tokenizer.second.beta == pytest.approx(1e-6)  # untouched
        assert pl_module.tokenizer.third.beta == pytest.approx(0.5)

    def test_target_name_not_found_raises(self):
        cb = self._beta_cb(target_name="nope")
        trainer, pl_module = self._make_trainer_and_module()
        with pytest.raises(RuntimeError, match=r"named \['nope'\]"):
            cb.on_train_start(trainer, pl_module)

    def test_target_name_list_with_unknown_raises(self):
        """Naming a real module plus a missing one reports only the missing one."""

        class TwoBottleneckTokenizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = VAEBottleneck(z_dim=4, beta=1e-6)
                self.second = VAEBottleneck(z_dim=4, beta=1e-6)

        cb = self._beta_cb(target_name=["first", "ghost"])
        trainer, pl_module = self._make_trainer_and_module()
        pl_module.tokenizer = TwoBottleneckTokenizer()
        with pytest.raises(RuntimeError, match=r"named \['ghost'\]"):
            cb.on_train_start(trainer, pl_module)

    def test_raw_logging_disambiguates_multiple_modules(self):
        """With >1 module, raw keys are suffixed with the module name."""

        class TwoBottleneckTokenizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = VAEBottleneck(z_dim=4, beta=1e-6)
                self.second = VAEBottleneck(z_dim=4, beta=1e-6)

        cb = self._beta_cb()
        trainer, pl_module = self._make_trainer_and_module()
        pl_module.tokenizer = TwoBottleneckTokenizer()
        cb.on_train_start(trainer, pl_module)
        pl_module.tokenizer.first._last_kl = 1.0
        pl_module.tokenizer.second._last_kl = 2.0
        cb.on_train_batch_end(trainer, pl_module, None, None, 0)
        pl_module.log.assert_any_call("train/kl_raw/first", 1.0)
        pl_module.log.assert_any_call("train/kl_raw/second", 2.0)

    def test_missing_attr_on_instance_raises(self):
        """A target_class instance lacking target_param errors at train start."""

        class BetalessCallback(AnnealingCallback):
            target_class = nn.Linear
            target_param = "beta"

        class LinearTokenizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(2, 2)

        cb = BetalessCallback(schedule="linear", value_max=1.0, value_min=0.0)
        trainer, pl_module = self._make_trainer_and_module()
        pl_module.tokenizer = LinearTokenizer()
        with pytest.raises(RuntimeError, match="no 'beta' attribute"):
            cb.on_train_start(trainer, pl_module)
