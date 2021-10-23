"""Some useful helper classes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass


class KeyDependentDefaultdict(dict):
    """Defaultdict where the default factory can depend on the key.

    Args:
        default_factory (callable): Default factory called when a missing key is
          encountered. The call will be default_factory(key) so the callable should
          take exactly one argument.

    Examples:
        >>> d = KeyDependentDefaultdict(lambda key: {"name": key})
        >>> d["a"]["b"] = 1
        >>> d
        {'a': {'name': 'a', 'b': 1}}
    """

    def __init__(self, /, *args, **kwargs):
        if not callable(args[0]):
            raise ValueError("First argument must be a callable.")

        self.default_factory = args[0]
        super().__init__(*args[1:], **kwargs)

    def __missing__(self, key):
        res = self.default_factory(key)
        self[key] = res
        return res


@dataclass
class ClfMetricTracker:
    """Class for keeping track of the metrics corresponding to a single label in a
    classification task.

    Args:
        name (str | int | None): The label this instance is keeping track of.
        tp (int): Start value for true positives. Defaults to 0.
        fp (int): Start-value for false positives. Defaults to 0.
        fn (int): Start-value for false negatives. Defaults to 0.

    Attributes:
        support: Number of instances for the label that were encountered.
        recall: Current value for the recall metric.
        precision: Current value for the precision metric.
        f1_score: Current value for the f1_score metric.

    Examples:
        >>> from swem.utils.classes import ClfMetricTracker
        >>> tracker = ClfMetricTracker(name="Label_1", tp=5, fp=1, fn=3)
        >>> tracker
        ClfMetricTracker(name='Label_1', tp=5, fp=1, fn=3)
        >>> tracker.support
        8
        >>> tracker.recall
        0.625
        >>> tracker.precision
        0.8333333333333334
        >>> tracker.f1_score
        0.7142857142857143

    """

    name: str | int | None = None
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __post_init__(self):
        assert self.tp >= 0
        assert self.fp >= 0
        assert self.fn >= 0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def recall(self) -> float:
        if self.support == 0:
            warnings.warn(f"Recall is ill-defined with empty support, defaulting to 0.")
            return 0
        return self.tp / self.support

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            warnings.warn(
                f"Precision is ill-defined without positive predictions, defaulting to 0."
            )
            return 0
        return self.tp / (self.tp + self.fp)

    @property
    def f1_score(self) -> float:
        if self.recall + self.precision == 0:
            warnings.warn("Both recall and precision are 0, defaulting to f1_score 0.")
            return 0
        return 2 * self.recall * self.precision / (self.recall + self.precision)
