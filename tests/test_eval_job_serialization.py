"""Tests for eval job JSON serialization safety."""

import json

import numpy as np
import pytest
import torch

from ntt.evaluation.utils.serialization import _json_default


class TestEvalJsonDefault:
    def test_numpy_array_serialized_as_list(self):
        result = _json_default(np.array([1.0, 2.0, 3.0]))
        assert result == [1.0, 2.0, 3.0]

    def test_numpy_scalar_serialized_as_python(self):
        result = _json_default(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_numpy_integer_serialized_as_python(self):
        result = _json_default(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_torch_tensor_serialized_as_list(self):
        result = _json_default(torch.tensor([1.0, 2.0]))
        assert result == [1.0, 2.0]

    def test_torch_scalar_serialized_as_python(self):
        result = _json_default(torch.tensor(3.14))
        assert isinstance(result, float)

    def test_unknown_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_default(object())

    def test_full_json_dump_with_numpy(self):
        """Verify that a dict with numpy values round-trips through JSON correctly."""
        data = {
            "scores": np.array([0.1, 0.2, 0.3]),
            "mean": np.float64(0.2),
            "count": np.int64(10),
        }
        json_str = json.dumps(data, default=_json_default)
        loaded = json.loads(json_str)

        assert loaded["scores"] == [0.1, 0.2, 0.3]
        assert loaded["mean"] == 0.2
        assert loaded["count"] == 10
