#Author - Dipanwita Thakur

from __future__ import annotations

import argparse
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# IMPORT EXISTING IMPLEMENTATION
# ------------------------------------------------------------

from model import MASTFuse, build_model, count_parameters

from encoders import MultimodalEncoders

from fusion import (
    MASTFuseFusion,
    CrossModalFusion,
    CrossModalTransformerBlock,
    ModalityGate,
    TemporalAttentionPooling,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 21

WEATHER_DIM = 17
SPECTRAL_DIM = 3
DYNAMIC_SOIL_DIM = 4
STATIC_SOIL_DIM = 28

FUSION_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
DROPOUT = 0.10

SEEDS = [42, 123, 456, 789, 1011]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducible CUDA behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# PARAMETER COUNT
# ============================================================

def parameter_count(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# ABLATION FUSION
# ============================================================

class AblationFusion(nn.Module):
    

    def __init__(
        self,
        fusion_dim=128,
        num_heads=4,
        ff_dim=256,
        dropout=0.1,

        use_weather=True,
        use_spectral=True,
        use_dynamic_soil=True,
        use_static_soil=True,

        use_cross_attention=True,
        use_gate=True,
        use_residual=True,
    ):

        super().__init__()

        self.fusion_dim = fusion_dim

        self.use_cross_attention = use_cross_attention
        self.use_gate = use_gate
        self.use_residual = use_residual

        # ----------------------------------------------------
        # Active modalities
        # ----------------------------------------------------

        self.modality_names = []

        if use_weather:
            self.modality_names.append("weather")

        if use_spectral:
            self.modality_names.append("spectral")

        if use_dynamic_soil:
            self.modality_names.append("dynamic_soil")

        if use_static_soil:
            self.modality_names.append("static_soil")

        if len(self.modality_names) == 0:
            raise ValueError(
                "At least one modality must be enabled."
            )

        # ----------------------------------------------------
        # Cross-modal attention
        #
        # Same structure as fusion.py.
        # ----------------------------------------------------

        self.cross_modal_blocks = nn.ModuleDict()

        if self.use_cross_attention:

            for query_name in self.modality_names:

                blocks = nn.ModuleDict()

                for context_name in self.modality_names:

                    if query_name == context_name:
                        continue

                    blocks[context_name] = (
                        CrossModalTransformerBlock(
                            dim=fusion_dim,
                            num_heads=num_heads,
                            ff_dim=ff_dim,
                            dropout=dropout,
                        )
                    )

                self.cross_modal_blocks[
                    query_name
                ] = blocks

        # ----------------------------------------------------
        # Modality gating
        # ----------------------------------------------------

        if self.use_gate:

            self.modality_gate = ModalityGate(
                dim=fusion_dim,
                hidden_dim=fusion_dim // 2,
                dropout=dropout,
            )

        # ----------------------------------------------------
        # Residual fusion projection
        #
        # This corresponds to the projection used by the
        # original fusion.py after:
        #
        #     gated_fusion + modality_mean
        # ----------------------------------------------------

        self.fusion_projection = nn.Sequential(

            nn.Linear(
                fusion_dim,
                fusion_dim,
            ),

            nn.LayerNorm(
                fusion_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),
        )

        self.fusion_norm = nn.LayerNorm(
            fusion_dim
        )

        # ----------------------------------------------------
        # Temporal attention pooling
        # ----------------------------------------------------

        self.temporal_pooling = (
            TemporalAttentionPooling(
                dim=fusion_dim,
                hidden_dim=fusion_dim // 2,
                dropout=dropout,
            )
        )

    # ========================================================
    # CROSS-MODAL ATTENTION
    # ========================================================

    def _cross_modal(self, active):

        if not self.use_cross_attention:
            return active

        outputs = {}

        for query_name in self.modality_names:

            query = active[query_name]

            # Start from the query representation
            x = query

            for context_name in self.modality_names:

                if context_name == query_name:
                    continue

                context = active[context_name]

                x = self.cross_modal_blocks[
                    query_name
                ][context_name](
                    x,
                    context,
                )

            outputs[query_name] = x

        return outputs

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        encoded_modalities,
        return_attention=False,
    ):

        # ----------------------------------------------------
        # Select active modalities
        # ----------------------------------------------------

        active = {}

        for name in self.modality_names:

            if name not in encoded_modalities:
                continue

            x = encoded_modalities[name]

            if x.ndim != 3:
                raise ValueError(
                    f"{name} must have shape (B,T,D), "
                    f"got {tuple(x.shape)}"
                )

            active[name] = x

        if len(active) == 0:
            raise ValueError(
                "No active modality representations."
            )

        # ----------------------------------------------------
        # Cross-modal attention
        # ----------------------------------------------------

        cross_modal_outputs = self._cross_modal(
            active
        )

        # ----------------------------------------------------
        # Modality gating
        # ----------------------------------------------------

        if self.use_gate:

            gated_fusion, modality_weights = (
                self.modality_gate(
                    cross_modal_outputs
                )
            )

        else:

            # No learned modality gate:
            # simple arithmetic mean.
            gated_fusion = torch.stack(
                list(cross_modal_outputs.values()),
                dim=0,
            ).mean(dim=0)

            modality_weights = None

        # ----------------------------------------------------
        # Residual fusion
        # ----------------------------------------------------

        if self.use_residual:

            modality_mean = torch.stack(
                list(active.values()),
                dim=0,
            ).mean(dim=0)

            fused_sequence = (
                gated_fusion +
                modality_mean
            )

        else:

            fused_sequence = gated_fusion

        # ----------------------------------------------------
        # Fusion projection
        # ----------------------------------------------------

        fused_sequence = self.fusion_projection(
            fused_sequence
        )

        # ----------------------------------------------------
        # Temporal attention pooling
        # ----------------------------------------------------

        fused_vector, temporal_weights = (
            self.temporal_pooling(
                fused_sequence
            )
        )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        if return_attention:

            attention_info = {

                "modality_weights":
                    modality_weights,

                "temporal_weights":
                    temporal_weights,

                "modality_names":
                    self.modality_names,

            }

            return (
                fused_vector,
                fused_sequence,
                attention_info,
            )

        return (
            fused_vector,
            fused_sequence,
        )


# ============================================================
# ABLATION MODEL
# ============================================================

class AblationMASTFuse(nn.Module):
    """
    MAST-Fuse using the EXISTING modality encoders and
    temporal refinement from model.py, but replacing only
    the fusion block with AblationFusion.

    Therefore:

        Encoders       = unchanged
        TCN            = unchanged
        Regression     = unchanged

    Only the selected fusion component is modified.
    """

    def __init__(
        self,

        fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,

        use_weather=True,
        use_spectral=True,
        use_dynamic_soil=True,
        use_static_soil=True,

        temporal_encoder="tcn",

        use_cross_attention=True,
        use_gate=True,
        use_residual=True,
    ):

        super().__init__()

        self.use_weather = use_weather
        self.use_spectral = use_spectral
        self.use_dynamic_soil = use_dynamic_soil
        self.use_static_soil = use_static_soil

        self.fusion_dim = fusion_dim

        # ----------------------------------------------------
        # IMPORTANT:
        # Use the EXISTING MASTFuse model first.
        #
        # This gives us the exact encoders, temporal module,
        # and regression head from the attached model.py.
        # ----------------------------------------------------

        self.base_model = MASTFuse(

            weather_dim=WEATHER_DIM,

            spectral_dim=SPECTRAL_DIM,

            dynamic_soil_dim=DYNAMIC_SOIL_DIM,

            static_soil_dim=STATIC_SOIL_DIM,

            sequence_length=SEQUENCE_LENGTH,

            fusion_dim=fusion_dim,

            num_heads=num_heads,

            ff_dim=ff_dim,

            dropout=dropout,

            use_weather=use_weather,

            use_spectral=use_spectral,

            use_dynamic_soil=use_dynamic_soil,

            use_static_soil=use_static_soil,

            temporal_encoder=temporal_encoder,

            # Existing fusion is replaced immediately below.
            fusion_type="mast_fuse",
        )

        # ----------------------------------------------------
        # Replace ONLY the fusion block.
        # ----------------------------------------------------

        self.base_model.fusion = AblationFusion(

            fusion_dim=fusion_dim,

            num_heads=num_heads,

            ff_dim=ff_dim,

            dropout=dropout,

            use_weather=use_weather,

            use_spectral=use_spectral,

            use_dynamic_soil=use_dynamic_soil,

            use_static_soil=use_static_soil,

            use_cross_attention=use_cross_attention,

            use_gate=use_gate,

            use_residual=use_residual,
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        weather=None,
        spectral=None,
        dynamic_soil=None,
        static_soil=None,
        
    ):

        return self.base_model(

            weather=weather,

            spectral=spectral,

            dynamic_soil=dynamic_soil,

            static_soil=static_soil,

            
        )


# ============================================================
# ABLATION CONFIGURATIONS
# ============================================================

ABLATIONS = {

    # --------------------------------------------------------
    # Proposed model
    # --------------------------------------------------------

    "full_model": {

        "use_cross_attention": True,

        "use_gate": True,

        "use_residual": True,
    },

    # --------------------------------------------------------
    # Remove cross-modal attention
    # --------------------------------------------------------

    "without_cross_modal_attention": {

        "use_cross_attention": False,

        "use_gate": True,

        "use_residual": True,
    },

    # --------------------------------------------------------
    # Remove modality gate
    # --------------------------------------------------------

    "without_modality_gate": {

        "use_cross_attention": True,

        "use_gate": False,

        "use_residual": True,
    },

    # --------------------------------------------------------
    # Remove residual fusion
    # --------------------------------------------------------

    "without_residual_fusion": {

        "use_cross_attention": True,

        "use_gate": True,

        "use_residual": False,
    },

    # --------------------------------------------------------
    # Remove cross-modal attention + gate
    # --------------------------------------------------------

    "without_attention_and_gate": {

        "use_cross_attention": False,

        "use_gate": False,

        "use_residual": True,
    },

    # --------------------------------------------------------
    # Remove cross-modal attention + residual
    # --------------------------------------------------------

    "without_attention_and_residual": {

        "use_cross_attention": False,

        "use_gate": True,

        "use_residual": False,
    },

    # --------------------------------------------------------
    # Remove gate + residual
    # --------------------------------------------------------

    "without_gate_and_residual": {

        "use_cross_attention": True,

        "use_gate": False,

        "use_residual": False,
    },
}


# ============================================================
# SMOKE TEST
# ============================================================

def smoke_test():

    print("=" * 70)
    print("MAST-Fuse ARCHITECTURAL ABLATION SMOKE TEST")
    print("=" * 70)

    print("Device:", DEVICE)

    B = 2
    T = SEQUENCE_LENGTH

    # --------------------------------------------------------
    # Dummy inputs
    # --------------------------------------------------------

    weather = torch.randn(
        B,
        T,
        WEATHER_DIM,
        device=DEVICE,
    )

    spectral = torch.randn(
        B,
        T,
        SPECTRAL_DIM,
        device=DEVICE,
    )

    dynamic_soil = torch.randn(
        B,
        T,
        DYNAMIC_SOIL_DIM,
        device=DEVICE,
    )

    static_soil = torch.randn(
        B,
        STATIC_SOIL_DIM,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Test each configuration
    # --------------------------------------------------------

    for name, config in ABLATIONS.items():

        print()
        print("-" * 70)
        print("Testing:", name)

        set_seed(42)

        model = AblationMASTFuse(
            fusion_dim=FUSION_DIM,

            num_heads=NUM_HEADS,

            ff_dim=FF_DIM,

            dropout=DROPOUT,

            use_weather=True,

            use_spectral=True,

            use_dynamic_soil=True,

            use_static_soil=True,

            temporal_encoder="tcn",

            **config,
        ).to(DEVICE)

        model.eval()

        with torch.no_grad():

            output = model(

                weather=weather,

                spectral=spectral,

                dynamic_soil=dynamic_soil,

                static_soil=static_soil,
            )

        print(
            "Output shape:",
            tuple(output.shape)
        )

        print(
            "Output finite:",
            bool(torch.isfinite(output).all())
        )

        total, trainable = (
            parameter_count(model)
        )

        print(
            "Parameters:",
            f"{total:,}"
        )

        assert output.shape == (B,), (
            f"Expected {(B,)}, "
            f"got {tuple(output.shape)}"
        )

        assert torch.isfinite(output).all(), (
            f"{name} produced NaN/Inf"
        )

        print("[PASS]", name)

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 70)


# ============================================================
# BUILD MODEL FOR ONE ABLATION
# ============================================================

def build_ablation_model(
    name,
    seed=42,
):

    if name not in ABLATIONS:
        raise ValueError(
            f"Unknown ablation: {name}"
        )

    set_seed(seed)

    config = ABLATIONS[name]

    model = AblationMASTFuse(

        fusion_dim=FUSION_DIM,

        num_heads=NUM_HEADS,

        ff_dim=FF_DIM,

        dropout=DROPOUT,

        use_weather=True,

        use_spectral=True,

        use_dynamic_soil=True,

        use_static_soil=True,

        temporal_encoder="tcn",

        **config,
    )

    return model


# ============================================================
# PRINT ARCHITECTURE
# ============================================================

def print_ablation_summary():

    print()
    print("=" * 80)
    print("ARCHITECTURAL ABLATION CONFIGURATIONS")
    print("=" * 80)

    for name, config in ABLATIONS.items():

        print()
        print(f"{name}")

        print(
            f"  Cross-modal attention : "
            f"{config['use_cross_attention']}"
        )

        print(
            f"  Modality gating       : "
            f"{config['use_gate']}"
        )

        print(
            f"  Residual fusion       : "
            f"{config['use_residual']}"
        )


# ============================================================
# PARAMETER REPORT
# ============================================================

def parameter_report():

    print()
    print("=" * 80)
    print("PARAMETER COUNTS")
    print("=" * 80)

    for name in ABLATIONS:

        model = build_ablation_model(
            name,
            seed=42,
        )

        total, trainable = (
            parameter_count(model)
        )

        print(
            f"{name:40s} "
            f"total={total:,} "
            f"trainable={trainable:,}"
        )

        del model


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Architectural ablation study for MAST-Fuse"
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only architecture forward-pass tests.",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print ablation configuration summary.",
    )

    parser.add_argument(
        "--params",
        action="store_true",
        help="Print parameter counts.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Build one specific ablation model.",
    )

    args = parser.parse_args()

    if args.smoke_test:

        smoke_test()

        return

    if args.summary:

        print_ablation_summary()

        return

    if args.params:

        parameter_report()

        return

    if args.model is not None:

        if args.model not in ABLATIONS:

            print(
                "Available models:"
            )

            for name in ABLATIONS:
                print("  ", name)

            raise ValueError(
                f"Unknown model: {args.model}"
            )

        model = build_ablation_model(
            args.model
        )

        total, trainable = (
            parameter_count(model)
        )

        print()
        print("=" * 70)
        print("MODEL:", args.model)
        print("=" * 70)

        print(
            "Total parameters     :",
            f"{total:,}"
        )

        print(
            "Trainable parameters :",
            f"{trainable:,}"
        )

        print()
        print(model)

        return

    # Default action
    print_ablation_summary()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
