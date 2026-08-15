from __future__ import annotations

import ast
import inspect
from pathlib import Path

from hackathon.egoflow.dataset import split_episode_paths
from hackathon.egoflow.models.progress_model import ProgressModel
from hackathon.egoflow.train import HARD_MAX_STEPS


def test_split_is_episode_disjoint_and_keeps_holdouts() -> None:
    split = split_episode_paths([Path(f"episode-{index}.npz") for index in range(10)])
    names = {key: set(value) for key, value in split.items()}
    assert len(names["train"]) == 6
    assert len(names["val"]) == 2
    assert len(names["test"]) == 2
    assert not (names["train"] & names["val"])
    assert not (names["train"] & names["test"])
    assert not (names["val"] & names["test"])


def test_training_limits_and_architecture_contract() -> None:
    assert HARD_MAX_STEPS == 2_500
    assert ProgressModel.ALLOWED_HIDDEN_SIZES == (128, 256)


def test_model_source_has_no_time_input_or_trainable_encoder() -> None:
    source = inspect.getsource(inspect.getmodule(ProgressModel))
    tree = ast.parse(source)
    forwards = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "forward"]
    assert forwards
    argument_names = {argument.arg for argument in forwards[0].args.args}
    assert "timestamps" not in argument_names
    assert "frame_indices" not in argument_names
    assert "commitment_head" not in source
    assert "nn.GRU" in source


def test_negative_stage_id_never_indexes_last_language_embedding(tmp_path: Path) -> None:
    np = __import__("numpy")
    from hackathon.egoflow.dataset import load_episode

    path = tmp_path / "negative-stage.npz"
    np.savez_compressed(
        path,
        visual_embeddings=np.ones((3, 2), dtype=np.float32),
        language_embeddings=np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        stage_ids=np.asarray([-1, 0, 1]),
        timestamps=np.asarray([0.0, 0.25, 0.5]),
        frame_indices=np.asarray([0, 1, 2]),
    )
    episode = load_episode(path)
    assert episode.language_embeddings[0].tolist() == [0.0, 0.0]
    assert episode.language_embeddings[1].tolist() == [2.0, 3.0]
    assert episode.language_embeddings[2].tolist() == [4.0, 5.0]
