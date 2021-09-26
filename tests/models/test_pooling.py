import torch

from swem.models.pooling import HierarchicalPooling


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
