"""
AlphaApta: a multimodal computational framework for cell-conditioned aptamer design.

Core model classes:
    from alphaapta.models import D1SequenceFrequencyRegressor
    from alphaapta.models import D2GNNTransformer
    from alphaapta.models import CellConditionedEnergyGenerator, D3EnergyConfig, decode_tokens
    from alphaapta.runtime import AlphaAptaD3Runtime
"""

__version__ = "2.0.0"


def __getattr__(name: str):
    if name == "AptamerGenerator":
        from .models.d3_energy_based_generator import CellConditionedEnergyGenerator
        return CellConditionedEnergyGenerator
    if name == "D3EnergyConfig":
        from .models.d3_energy_based_generator import D3EnergyConfig
        return D3EnergyConfig
    if name == "decode_tokens":
        from .models.d3_energy_based_generator import decode_tokens
        return decode_tokens
    if name == "BindingPredictor":
        from .models.d2_gnn_transformer import D2GNNTransformer
        return D2GNNTransformer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
