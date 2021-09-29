import torch

from swem.metrics import ClassificationReport


class TestClassificationReport:
    def test_multiclass(self):
        report = ClassificationReport(target_names=["a", "b"])
        assert report.get() is None

        logits = torch.tensor([[1, 0], [0, 1], [1, 0]])
        labels = torch.tensor([0, 0, 1])
        report.update(logits, labels)
        state = report.get()
        assert state["num_samples"] == 3
        assert abs(state["accuracy"] - 1 / 3) < 1e-5
        assert state["class_metrics"]["a"]["support"] == 2
        assert abs(state["class_metrics"]["a"]["recall"] - 1 / 2) < 1e-5

        mask = torch.tensor([1, 1, 0])
        report.reset()
        report.update(logits, labels, mask)
        state = report.get()
        assert state["num_samples"] == 2
        assert abs(state["accuracy"] - 1 / 2) < 1e-5
        assert state["class_metrics"]["b"]["support"] == 0
        assert state["class_metrics"]["a"]["precision"] == 1

    def test_binary(self):
        report = ClassificationReport(binary=True)
        assert report.get() is None

        logits = torch.tensor([1, -1, 1])
        labels = torch.tensor([0, 0, 1])
        report.update(logits, labels)
        state = report.get()
        assert state["num_samples"] == 3
        assert abs(state["accuracy"] - 2 / 3) < 1e-5
        assert state["class_metrics"][0]["support"] == 2
        assert abs(state["class_metrics"][0]["recall"] - 1 / 2) < 1e-5

        mask = torch.tensor([1, 1, 0])
        report.reset()
        report.update(logits, labels, mask)
        state = report.get()
        assert state["num_samples"] == 2
        assert abs(state["accuracy"] - 1 / 2) < 1e-5
        assert state["class_metrics"][1]["support"] == 0
        assert state["class_metrics"][1]["precision"] == 0
