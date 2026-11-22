import torch

from alphaapta.models.d2_gnn_transformer import (
    D2GNNTransformer,
    D2GNNTransformerConfig,
    d2_training_objective,
)


def test_d2_shape_and_objective():
    config = D2GNNTransformerConfig(
        d1_hidden_dimension=8,
        hidden_dimension=8,
        cell_context_dimension=9,
        wnn_dimension=2,
        named_cell_features=7,
        graph_neighbors=2,
        graph_layers=1,
        transformer_layers=1,
        cross_attention_layers=1,
        attention_heads=2,
        transformer_dropout=0.0,
        cross_attention_dropout=0.0,
    )
    model = D2GNNTransformer(config).eval()
    logits, aptamer, cell = model(
        torch.randn(4, 6, 8),
        torch.ones(4, 6, dtype=torch.bool),
        torch.randn(4, 9),
    )
    labels = torch.tensor([1, 0, 1, 0])
    loss, terms = d2_training_objective(
        logits, labels, aptamer, cell, model.logit_scale
    )
    assert logits.shape == (4, 2)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(terms) == {"cross_entropy", "infonce"}
