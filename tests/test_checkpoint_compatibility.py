import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch

from alphaapta.models import D1SequenceFrequencyRegressor, D2GNNTransformer
from alphaapta.runtime import AlphaAptaD3Runtime


def test_d1_deposited_checkpoint_when_configured():
    checkpoint = os.environ.get("ALPHAAPTA_D1_CHECKPOINT")
    model_directory = os.environ.get("ALPHAAPTA_D1_MODEL_DIRECTORY")
    if not checkpoint or not model_directory:
        pytest.skip("Set D1 checkpoint and model-directory environment variables")
    model, metadata = D1SequenceFrequencyRegressor.from_checkpoint(
        checkpoint, model_directory
    )
    assert metadata.epoch == 3
    assert len(model.encoder.state_dict()) == 115
    assert len(model.head.state_dict()) == 4


def test_d2_deposited_checkpoint_when_configured():
    checkpoint = os.environ.get("ALPHAAPTA_D2_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set ALPHAAPTA_D2_CHECKPOINT")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = D2GNNTransformer().eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    assert len(model.state_dict()) == 72


def test_d2_matches_registered_runtime_when_configured():
    checkpoint = os.environ.get("ALPHAAPTA_D2_CHECKPOINT")
    runtime_source = os.environ.get("ALPHAAPTA_D2_RUNTIME_SOURCE")
    if not checkpoint or not runtime_source:
        pytest.skip("Set D2 checkpoint and runtime-source environment variables")
    spec = importlib.util.spec_from_file_location(
        "registered_d2_runtime", Path(runtime_source)
    )
    runtime = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    source_parent = str(Path(runtime_source).resolve().parent)
    sys.path.insert(0, source_parent)
    try:
        spec.loader.exec_module(runtime)
    finally:
        sys.path.remove(source_parent)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    candidate = D2GNNTransformer().eval()
    registered = runtime.GNNTransformerD2().eval()
    candidate.load_state_dict(payload["state_dict"], strict=True)
    registered.load_state_dict(payload["state_dict"], strict=True)

    torch.manual_seed(7)
    states = torch.randn(3, 41, 256)
    mask = torch.ones(3, 41, dtype=torch.bool)
    mask[1, -4:] = False
    cell_context = torch.randn(3, 3257)
    with torch.no_grad():
        candidate_logits = candidate(states, mask, cell_context)[0]
        registered_logits = registered(states, mask, cell_context)[0]
    torch.testing.assert_close(candidate_logits, registered_logits, rtol=0, atol=0)


def test_d3_deposited_fold_checkpoints_when_configured():
    bundle = os.environ.get("ALPHAAPTA_D3_RUNTIME_BUNDLE")
    if not bundle:
        pytest.skip("Set ALPHAAPTA_D3_RUNTIME_BUNDLE")
    runtime = AlphaAptaD3Runtime(bundle)
    assert set(runtime.models) == {
        "AGS",
        "CRL-3621",
        "GES-1",
        "MKN-1",
        "MKN-28",
        "NCI-N87",
    }
    assert all(
        generator.config.sequence_length == 40
        for _, generator in runtime.models.values()
    )
