"""Implementation of certain useful metrics."""
from typing import Dict, List, Optional, Union

import torch

from swem.utils.classes import ClfMetricTracker, KeyDependentDefaultdict


class ClassificationReport:
    """A class for tracking various metrics in a classification task.

    The class is particularly useful when doing handling a dataset batch-wise where
    metrics should be aggregated over the whole dataset.

    Args:
        target_names (Optional[List[str, int]]): Labels in the classification task in
          the same order as the output of the model.
        binary (bool): Whether or not we are doing binary classification (i.e. the
          model output is the pre-sigmoid logit for the positive class). Defaults to
          False.

    Examples:
        >>> report = ClassificationReport(target_names=["A", "B"])
        >>> logits = torch.tensor([[1,0], [0,1], [1,0]])
        >>> labels = torch.tensor([0, 0, 1])
        >>> report.update(logits, labels)
        >>> report.get()
        {
            "num_samples": 3,
            "accuracy": 0.3333333333333333,
            "class_metrics": {
                "A": {
                "support": 2,
                "recall": 0.5,
                "precision": 0.5,
                "f1_score": 0.5
                },
                "B": {
                "support": 1,
                "recall": 0.0,
                "precision": 0.0,
                "f1_score": null
                }
            }
        }
        >>> mask = torch.tensor([1,1,0])
        >>> report.reset()
        >>> report.update(logits, labels, mask)
        >>> report.get()
        {
            "num_samples": 2,
            "accuracy": 0.5,
            "class_metrics": {
                "A": {
                "support": 2,
                "recall": 0.5,
                "precision": 1.0,
                "f1_score": 0.6666666666666666
                },
                "B": {
                "support": 0,
                "recall": null,
                "precision": 0.0,
                "f1_score": null
                }
            }
        }
    """

    def __init__(
        self, target_names: Optional[List[Union[str, int]]] = None, binary: bool = False
    ):
        self.target_names = target_names
        self.binary = binary
        self.num_samples = 0
        self.num_correct = 0
        self.class_metrics = KeyDependentDefaultdict(ClfMetricTracker)

    def reset(self):
        """Reset all tracked values."""
        self.num_samples = 0
        self.num_correct = 0
        self.class_metrics = KeyDependentDefaultdict(ClfMetricTracker)

    def get(
        self,
    ) -> Optional[
        Dict[str, Union[int, float, Dict[Union[int, str], Dict[str, Optional[float]]]]]
    ]:
        """Get the current state of all tracked metrics."""
        if self.num_samples == 0:
            return None
        return {
            "num_samples": self.num_samples,
            "accuracy": self.num_correct / self.num_samples,
            "class_metrics": {
                name: {
                    "support": met.support,
                    "recall": met.recall,
                    "precision": met.precision,
                    "f1_score": met.f1_score,
                }
                for name, met in self.class_metrics.items()
            },
        }

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Update the tracked metrics with the results from a new batch.

        Args:
            logits (torch.Tensor): Output of the model in the classification task
              (pre-softmax/sigmoid).
            labels (torch.Tensor): The correct labels.
            mask (Optional[torch.Tensor], optional): A 0/1-mask telling us which
              samples to take into account (where mask is 1). Defaults to None.
        """
        if self.binary:
            num_classes = 2
            preds = torch.tensor(logits > 0).to(dtype=torch.int64).view(-1)
        else:
            num_classes = logits.size(-1)
            preds = torch.argmax(torch.tensor(logits), dim=-1).view(-1)

        labels = labels.view(-1)
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool, device=labels.device)
        else:
            mask = torch.tensor(mask, dtype=torch.bool)

        assert labels.size() == mask.size()
        assert preds.size() == mask.size()

        self.num_samples += torch.sum(mask).item()
        self.num_correct += torch.sum((labels == preds) * mask).item()

        for i in range(num_classes):
            if self.target_names is None:
                name = i
            else:
                name = self.target_names[i]

            label_mask = labels == i
            preds_mask = preds == i

            self.class_metrics[name].tp += torch.sum(
                label_mask * preds_mask * mask
            ).item()
            self.class_metrics[name].fp += torch.sum(
                (~label_mask) * preds_mask * mask
            ).item()
            self.class_metrics[name].fn += torch.sum(
                label_mask * (~preds_mask) * mask
            ).item()
