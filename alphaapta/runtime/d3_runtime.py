"""Runtime for a checksum-verified, fold-specific AlphaApta D3 EBM bundle."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import torch
import torch.nn as nn

from alphaapta.models.d3_energy_based_generator import (
    CellConditionedEnergyGenerator,
    decode_tokens,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class D2CellEncoder(nn.Sequential):
    """Frozen 3,257-value cell-context projection stored with each D3 fold."""

    def __init__(self) -> None:
        super().__init__(
            nn.Linear(3257, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )


class AlphaAptaD3Runtime:
    """Load a six-fold D3 bundle and generate from measured cellular context."""

    def __init__(self, bundle: str | Path, device: str | torch.device = "cpu") -> None:
        self.bundle = Path(bundle).resolve()
        self.device = torch.device(device)
        self.lock = threading.Lock()
        self.manifest = json.loads((self.bundle / "manifest.json").read_text())
        self.fold_order = tuple(self.manifest["folds"])
        self.models: dict[
            str, tuple[D2CellEncoder, CellConditionedEnergyGenerator]
        ] = {}
        self._load()

    def _load(self) -> None:
        if self.manifest.get("architecture") != "cell-conditioned energy-based model":
            raise RuntimeError("Unexpected D3 bundle architecture")
        for fold, record in self.manifest["folds"].items():
            ebm_path = self.bundle / record["ebm_checkpoint"]
            cell_path = self.bundle / record["d2_cell_encoder"]
            if _sha256(ebm_path) != record["ebm_checkpoint_sha256"]:
                raise RuntimeError(f"EBM checksum mismatch for {fold}")
            if _sha256(cell_path) != record["d2_cell_encoder_sha256"]:
                raise RuntimeError(f"D2 cell-encoder checksum mismatch for {fold}")

            ebm_payload = torch.load(ebm_path, map_location="cpu", weights_only=False)
            cell_payload = torch.load(cell_path, map_location="cpu", weights_only=False)
            if (
                ebm_payload.get("arch") != "ebm"
                or ebm_payload.get("cell_condition_mode") != "full"
            ):
                raise RuntimeError(f"Invalid EBM checkpoint for {fold}")
            if (
                cell_payload.get("arch") != "d2_cell_pool"
                or cell_payload.get("holdout") != fold
            ):
                raise RuntimeError(f"Invalid D2 cell encoder for {fold}")

            cell_encoder = D2CellEncoder().to(self.device)
            generator = CellConditionedEnergyGenerator().to(self.device)
            cell_encoder.load_state_dict(cell_payload["state_dict"], strict=True)
            generator.load_state_dict(ebm_payload["state_dict"], strict=True)
            cell_encoder.eval()
            generator.eval()
            self.models[fold] = (cell_encoder, generator)

    def generate(
        self,
        combined_expression,
        n_candidates: int,
        cell_type: str | None = None,
        temperature: float = 1.0,
        seed: int = 20260710,
        batch_size: int = 64,
    ) -> tuple[list[dict], dict]:
        vector = torch.as_tensor(combined_expression, dtype=torch.float32)
        if vector.shape != (3257,):
            raise ValueError(f"Expected 3,257 cellular values; found {tuple(vector.shape)}")
        if not torch.isfinite(vector).all():
            raise ValueError("Cellular context contains non-finite values")
        if not 1 <= int(n_candidates) <= 2000:
            raise ValueError("n_candidates must be between 1 and 2,000")

        selected_fold = cell_type if cell_type in self.models else None
        assignments = (
            [selected_fold] * int(n_candidates)
            if selected_fold
            else [
                self.fold_order[index % len(self.fold_order)]
                for index in range(int(n_candidates))
            ]
        )
        rows: list[dict] = []
        with self.lock:
            for fold_index, fold in enumerate(self.fold_order):
                indices = [
                    index for index, assignment in enumerate(assignments) if assignment == fold
                ]
                if not indices:
                    continue
                cell_encoder, generator = self.models[fold]
                for start in range(0, len(indices), batch_size):
                    chunk_indices = indices[start : start + batch_size]
                    fold_offset = 0 if selected_fold else fold_index * 100_000
                    generation_seed = int(seed) + fold_offset + start
                    torch.manual_seed(generation_seed)
                    if self.device.type == "cuda":
                        torch.cuda.manual_seed_all(generation_seed)
                    batch = vector.to(self.device).unsqueeze(0).repeat(
                        len(chunk_indices), 1
                    )
                    with torch.no_grad():
                        condition = cell_encoder(batch)
                        tokens = generator.generate(condition, temperature=temperature)
                    for output_index, sequence in zip(
                        chunk_indices, decode_tokens(tokens.cpu())
                    ):
                        if len(sequence) != 40 or set(sequence) > set("ACGT"):
                            raise RuntimeError(f"Invalid D3 output for {fold}: {sequence!r}")
                        rows.append(
                            {
                                "generation_index": output_index + 1,
                                "sequence": sequence,
                                "model_fold": fold,
                            }
                        )
        rows.sort(key=lambda row: row["generation_index"])
        metadata = {
            "architecture": self.manifest["architecture"],
            "cell_condition": cell_type or "custom profile",
            "input_layout": "50 WNN + 3,000 RNA + 207 AbSeq",
            "sequence_length": 40,
            "langevin_steps": 20,
            "langevin_gradient_step_size": 10.0,
            "langevin_noise_standard_deviation": 0.005,
            "temperature": float(temperature),
            "seed": int(seed),
            "top_p": None,
            "top_k": None,
        }
        return rows, metadata
