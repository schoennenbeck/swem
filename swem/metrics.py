"""Implementation of certain useful metrics."""
from __future__ import annotations

import json

import torch

from swem.utils.classes import ClfMetricTracker, KeyDependentDefaultdict


class ClassificationReport:
    """A class for tracking various metrics in a classification task.

    The class is particularly useful when doing handling a dataset batch-wise where
    metrics should be aggregated over the whole dataset.

    Args:
        target_names (list[str | int] | None): Labels in the classification task in
          the same order as the output of the model.
        binary (bool): Whether or not we are doing binary classification (i.e. the
          model output is the pre-sigmoid logit for the positive class). Defaults to
          False.
        from_probas (bool): If True we interpret the input as probabilities instead of
          logits. This is only relevant when dealing with binary classification (since
          in the multiclass setting the predicted label is an argmax which is the same
          for logits and probabilities). Defaults to False.

    Examples:
        >>> report = ClassificationReport(target_names=["A", "B"])
        >>> logits = torch.tensor([[1,0], [0,1], [1,0]])
        >>> labels = torch.tensor([0, 0, 1])
        >>> report.update(logits, labels)
        >>> report.get()
        {
            "num_samples": 3,
            "accuracy": 0.3333333333333333,
            "recall_macro_avg": 0.25,
            "recall_weighted_avg": 0.3333333333333333,
            "precision_macro_avg": 0.25,
            "precision_weighted_avg": 0.3333333333333333,
            "f1_score_macro_avg": 0.25,
            "f1_score_weighted_avg": 0.3333333333333333,
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
                "f1_score": 0
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
            "recall_macro_avg": 0.25,
            "recall_weighted_avg": 0.5,
            "precision_macro_avg": 0.5,
            "precision_weighted_avg": 1.0,
            "f1_score_macro_avg": 0.3333333333333333,
            "f1_score_weighted_avg": 0.6666666666666666,
            "class_metrics": {
                "A": {
                "support": 2,
                "recall": 0.5,
                "precision": 1.0,
                "f1_score": 0.6666666666666666
                },
                "B": {
                "support": 0,
                "recall": 0,
                "precision": 0.0,
                "f1_score": 0
                }
            }
        }
    """

    def __init__(
        self,
        target_names: list[str | int] | None = None,
        binary: bool = False,
        from_probas: bool = False,
    ):
        self.target_names = target_names
        self.binary = binary
        self.from_probas = from_probas
        self.num_samples = 0
        self.num_correct = 0
        self.class_metrics = KeyDependentDefaultdict(ClfMetricTracker)

    def reset(self):
        """Reset all tracked values."""
        self.num_samples = 0
        self.num_correct = 0
        self.class_metrics = KeyDependentDefaultdict(ClfMetricTracker)

    def __repr__(self) -> str:
        state = self.get()
        if state is None:
            return ""
        return json.dumps(state, indent=2)

    def get(
        self,
    ) -> dict[str, int | float | dict] | None:
        """Get the current state of all tracked metrics."""
        if self.num_samples == 0:
            return None

        accuracy: float = self.num_correct / self.num_samples
        precision_macro_avg: float = sum(
            met.precision for met in self.class_metrics.values()
        ) / len(self.class_metrics)
        precision_weighted_avg: float = (
            sum(met.precision * met.support for met in self.class_metrics.values())
            / self.num_samples
        )
        recall_macro_avg: float = sum(
            met.recall for met in self.class_metrics.values()
        ) / len(self.class_metrics)
        recall_weighted_avg: float = (
            sum(met.recall * met.support for met in self.class_metrics.values())
            / self.num_samples
        )
        f1_score_macro_avg: float = sum(
            met.f1_score for met in self.class_metrics.values()
        ) / len(self.class_metrics)
        f1_score_weighted_avg: float = (
            sum(met.f1_score * met.support for met in self.class_metrics.values())
            / self.num_samples
        )
        return {
            "num_samples": self.num_samples,
            "accuracy": accuracy,
            "recall_macro_avg": recall_macro_avg,
            "recall_weighted_avg": recall_weighted_avg,
            "precision_macro_avg": precision_macro_avg,
            "precision_weighted_avg": precision_weighted_avg,
            "f1_score_macro_avg": f1_score_macro_avg,
            "f1_score_weighted_avg": f1_score_weighted_avg,
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
        logits: "array_like",
        labels: "array_like",
        mask: "array_like" | None = None,
    ):
        """Update the tracked metrics with the results from a new batch.

        The inputs can be any type that can be turned into a torch.Tensor
        ("array_like").

        Args:
            logits (array_like): Output of the model in the classification task
              (pre-softmax/sigmoid if self.from_probas is False or probabilites if
              self.from_probas is True).
            labels (array_like): The correct labels.
            mask (array_like | None): A 0/1-mask telling us which
              samples to take into account (where mask is 1). Defaults to None.

        Shapes:
            - logits: :math:`(*, \\text{num_classes})` if self.binary is False otherwise
              :math:`(*,)` where * is any number of dimensions.
            - labels: :math:`(*,)` where * is the same as for logits.
            - mask: :math:`(*,)` where * is the same as for logits.
        """
        logits = (
            logits.clone().detach()
            if isinstance(logits, torch.Tensor)
            else torch.tensor(logits)
        )
        labels = (
            labels.clone().detach()
            if isinstance(labels, torch.Tensor)
            else torch.tensor(labels)
        )

        if self.binary:
            num_classes = 2
            if self.from_probas:
                preds = (logits > 0.5).to(dtype=torch.int64)
            else:
                preds = (logits > 0).to(dtype=torch.int64)
        else:
            num_classes = logits.size(-1)
            preds = torch.argmax(logits, dim=-1)

        assert (
            preds.size() == labels.size()
        ), f"Expected predictions and labels to have the same shape but got {preds.shape} and {labels.shape}."

        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool, device=labels.device)
        else:
            mask = (
                mask.clone().detach()
                if isinstance(mask, torch.Tensor)
                else torch.tensor(mask)
            ).to(dtype=torch.bool)

        assert (
            labels.size() == mask.size()
        ), f"Expected mask and labels to have the same shape but got {mask.shape} and {labels.shape}."
        mask = mask.view(-1)
        labels = labels.view(-1)
        preds = preds.view(-1)

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
