import pytest
import torch

from swem.models.pooling import (
    AttentionPooling,
    HierarchicalPooling,
    MaxPooling,
    MeanPooling,
    SwemPoolingLayer,
)


class TestSwemPoolingLayer:
    def test_from_config(self):
        pool = SwemPoolingLayer.from_config(
            {"class": "HierarchicalPooling", "window_size": 5}
        )
        assert isinstance(pool, HierarchicalPooling)

        pool = SwemPoolingLayer.from_config({"class": "MaxPooling"})
        assert isinstance(pool, MaxPooling)

        pool = SwemPoolingLayer.from_config({"class": "MeanPooling"})
        assert isinstance(pool, MeanPooling)

        pool = SwemPoolingLayer.from_config(
            {"class": "AttentionPooling", "input_dim": 5}
        )
        assert isinstance(pool, AttentionPooling)

        with pytest.raises(NotImplementedError):
            pool = SwemPoolingLayer.from_config({"class": "UnknownPooling"})

    def test_forward(self):
        pool = SwemPoolingLayer()
        with pytest.raises(NotImplementedError):
            pool(torch.randn(8, 19, 3))


class TestHierarchicalPooling:
    def test_output_size(self):
        hier = HierarchicalPooling(window_size=5)
        input_1 = torch.randn((8, 19, 3))
        output_1 = hier(input_1)
        assert output_1.size() == (8, 3)

        input_2 = torch.randn((8, 2, 3))
        output_2 = hier(input_2)
        assert output_2.size() == (8, 3)

    def test_correct_output(self):
        hier = HierarchicalPooling(window_size=2)
        input_1 = torch.tensor([1, 2], dtype=torch.float32).view(1, 2, 1)
        output_1 = hier(input_1)
        assert torch.allclose(output_1.view(-1), torch.tensor([1.5]))

        input_2 = torch.tensor([1, 2, 3], dtype=torch.float32).view(1, 3, 1)
        output_2 = hier(input_2)
        assert torch.allclose(output_2.view(-1), torch.tensor([2.5]))

    def test_config(self):
        hier = HierarchicalPooling(window_size=2)
        assert hier.config["window_size"] == 2


class TestMeanPooling:
    def test_output_size(self):
        mean = MeanPooling()
        input = torch.randn((8, 19, 3))
        output = mean(input)
        assert output.size() == (8, 3)

    def test_correct_output(self):
        mean = MeanPooling()
        input = torch.tensor([1, 2, 3], dtype=torch.float32).view(1, 3, 1)
        output_1 = mean(input)
        assert torch.allclose(output_1.view(-1), torch.tensor([2.0]))

        mask = torch.tensor([1, 1, 0], dtype=torch.float32)
        output_2 = mean(input, mask)
        assert torch.allclose(output_2.view(-1), torch.tensor([1.5]))

    def test_config(self):
        mean = MeanPooling()
        assert mean.config["class"] == "MeanPooling"


class TestMaxPooling:
    def test_output_size(self):
        max = MaxPooling()
        input = torch.randn((8, 19, 3))
        output = max(input)
        assert output.size() == (8, 3)

    def test_correct_output(self):
        max = MaxPooling()
        input = torch.tensor([1, 2, 3], dtype=torch.float32).view(1, 3, 1)
        output_1 = max(input)
        assert torch.allclose(output_1.view(-1), torch.tensor([3.0]))

    def test_config(self):
        max = MaxPooling()
        assert max.config["class"] == "MaxPooling"


class TestAttentionPooling:
    def test_output_size(self):
        pool = AttentionPooling(input_dim=2)
        input = torch.rand(8, 19, 2)
        output = pool(input)
        assert output.size() == (8, 2)

    def test_mask(self):
        pool = AttentionPooling(input_dim=2)
        input = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).view(1, 2, 2)
        mask = torch.tensor([1, 0], dtype=torch.float32).view(1, 2)
        output = pool(input, mask)
        assert torch.allclose(
            output.view(-1), torch.tensor([1, 2], dtype=torch.float32)
        )

    def test_config(self):
        pool = AttentionPooling(input_dim=10)
        assert pool.config["class"] == "AttentionPooling"
        assert pool.config["input_dim"] == 10
