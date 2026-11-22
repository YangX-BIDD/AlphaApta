import torch

from alphaapta.models.d3_energy_based_generator import (
    CellConditionedEnergyGenerator,
    D3EnergyConfig,
    decode_tokens,
)


def test_generation_returns_complete_acgt_sequences():
    config = D3EnergyConfig(
        sequence_length=8,
        condition_dimension=4,
        hidden_dimension=8,
        attention_heads=2,
        feedforward_dimension=16,
        transformer_layers=1,
        dropout=0.0,
        langevin_steps=2,
        langevin_gradient_step_size=0.1,
        langevin_noise_standard_deviation=0.0,
    )
    model = CellConditionedEnergyGenerator(config).eval()
    tokens = model.generate(torch.zeros(2, 4))
    sequences = decode_tokens(tokens)
    assert len(sequences) == 2
    assert all(len(sequence) == 8 for sequence in sequences)
    assert all(set(sequence) <= set("ACGT") for sequence in sequences)
