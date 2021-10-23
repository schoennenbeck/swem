import pytest

from swem.utils.classes import ClfMetricTracker, KeyDependentDefaultdict


class TestKeyDependentDefaultdict:
    def test_default_factory(self):

        with pytest.raises(ValueError):
            KeyDependentDefaultdict("factory")

        def factory(key):
            return {key: key}

        d = KeyDependentDefaultdict(factory, a=1)
        d["b"]["c"] = 2
        assert "a" in d
        assert d["a"] == 1
        assert "b" in d
        assert d["b"] == {"b": "b", "c": 2}

    def test_metric_tracker(self):
        d = KeyDependentDefaultdict(ClfMetricTracker)

        assert d["a"].name == "a"
        assert d["a"].support == 0
        with pytest.warns(UserWarning):
            assert d["a"].recall == 0
        with pytest.warns(UserWarning):
            assert d["a"].precision == 0
        with pytest.warns(UserWarning):
            assert d["a"].f1_score == 0

        d["a"].tp += 1
        d["a"].fp += 1
        d["a"].fn += 1

        assert d["a"].support == 2
        assert abs(d["a"].recall - 1 / 2) < 1e-5
        assert abs(d["a"].precision - 1 / 2) < 1e-5
        assert abs(d["a"].f1_score - 1 / 2) < 1e-5
