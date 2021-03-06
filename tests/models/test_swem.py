import pytest
import torch
from torch import nn

from swem.models.pooling import HierarchicalPooling
from swem.models.swem import Swem
from swem.models.word_drop_embedding import WordDropEmbedding


class TestSwem:
    embedding = nn.Embedding(10, 3)
    pooling = HierarchicalPooling(window_size=3)

    def test_generic(self):
        swem = Swem(
            embedding=self.embedding,
            pooling_layer=self.pooling,
            pre_pooling_dims=(5, 6, 7),
            post_pooling_dims=(8, 9),
        )

        input = torch.randint(0, 10, (8, 21))
        output = swem(input)
        assert output.size() == (8, 9)

    def test_no_trafos(self):
        swem = Swem(
            embedding=self.embedding,
            pooling_layer=self.pooling,
            pre_pooling_dims=None,
            post_pooling_dims=None,
        )

        input = torch.randint(0, 10, (8, 21))
        output = swem(input)
        assert output.size() == (8, 3)

    def test_short_trafos(self):
        swem = Swem(
            embedding=self.embedding,
            pooling_layer=self.pooling,
            pre_pooling_dims=(5,),
            post_pooling_dims=(11,),
        )

        input = torch.randint(0, 10, (8, 21))
        output = swem(input)
        assert output.size() == (8, 11)

    def test_config(self):
        wemb = WordDropEmbedding(10, 3, p=0.1)
        swem = Swem(
            embedding=wemb,
            pooling_layer=self.pooling,
            pre_pooling_dims=[
                5,
            ],
            post_pooling_dims=(11,),
        )

        assert swem.config.embedding.type == "WordDropEmbedding"
        assert swem.config.pooling.window_size == 3

        new_swem = Swem.from_config(swem.config)
        assert new_swem.pre_pooling_dims == (5,)
        assert new_swem.post_pooling_dims == (11,)
        assert isinstance(new_swem.pooling_layer, HierarchicalPooling)
        assert new_swem.embedding.num_embeddings == 10

    def test_save_and_load(self, tmp_path):
        swem = Swem(
            embedding=self.embedding,
            pooling_layer=self.pooling,
            pre_pooling_dims=(5,),
            post_pooling_dims=(11,),
        )

        with pytest.raises(FileNotFoundError):
            swem.save(tmp_path / "test" / "test")

        (tmp_path / "testdir").mkdir()
        with open(tmp_path / "testdir" / "testfile", "w") as f:
            f.write("test")

        with pytest.raises(NotADirectoryError):
            swem.save(tmp_path / "testdir" / "testfile")

        with pytest.raises(FileExistsError):
            swem.save(tmp_path / "testdir")

        swem.save(tmp_path / "model")

        loaded = Swem.load(tmp_path / "model")

        assert loaded.config == swem.config
        assert all(
            torch.allclose(swem_param, loaded.state_dict()[param_name])
            for param_name, swem_param in swem.state_dict().items()
        )
