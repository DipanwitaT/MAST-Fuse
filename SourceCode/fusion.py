#Author - Dipanwita Thakur
from __future__ import annotations

import math
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Modality Self-Normalization
# ============================================================

class ModalityNorm(nn.Module):
    """
    Layer normalization for each modality.

    Input:
        (B, T, D)

    Output:
        (B, T, D)
    """

    def __init__(self, dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        return self.norm(x)


# ============================================================
# 2. Cross-Modal Attention
# ============================================================

class CrossModalAttention(nn.Module):
    """
    Multi-head cross-modal attention.

    Query modality attends to a context modality.

    Example:

        Weather -> Spectral

    means weather representation generates queries,
    while spectral representation provides keys/values.

    Input:
        query   : (B, T, D)
        context : (B, T, D)

    Output:
        (B, T, D)
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if dim % num_heads != 0:

            raise ValueError(
                f"fusion_dim={dim} must be divisible "
                f"by num_heads={num_heads}"
            )

        self.dim = dim

        self.num_heads = num_heads

        self.head_dim = dim // num_heads

        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_projection = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.k_projection = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.v_projection = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.output_projection = nn.Linear(
            dim,
            dim,
        )

        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ):

        if query.ndim != 3:

            raise ValueError(
                "query must have shape (B,T,D), "
                f"got {tuple(query.shape)}"
            )

        if context.ndim != 3:

            raise ValueError(
                "context must have shape (B,T,D), "
                f"got {tuple(context.shape)}"
            )

        if query.shape != context.shape:

            raise ValueError(
                "Query and context must have "
                f"the same shape. "
                f"Got {tuple(query.shape)} and "
                f"{tuple(context.shape)}"
            )

        B, T, D = query.shape

        residual = query

        q = self.q_projection(query)

        k = self.k_projection(context)

        v = self.v_projection(context)

        # ----------------------------------------------------
        # (B,T,D)
        # ->
        # (B,H,T,D_head)
        # ----------------------------------------------------

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # ----------------------------------------------------
        # Attention
        #
        # (B,H,T,T)
        # ----------------------------------------------------

        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) * self.scale

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        attended = torch.matmul(
            attention_weights,
            v,
        )

        # ----------------------------------------------------
        # Back to (B,T,D)
        # ----------------------------------------------------

        attended = attended.transpose(
            1,
            2,
        ).contiguous()

        attended = attended.view(
            B,
            T,
            D,
        )

        attended = self.output_projection(
            attended
        )

        attended = self.dropout(
            attended
        )

        # Residual connection
        output = self.norm(
            residual + attended
        )

        return output


# ============================================================
# 3. Cross-Modal Transformer Block
# ============================================================

class CrossModalTransformerBlock(nn.Module):
    """
    Cross-modal attention + feed-forward network.

    This provides a complete transformer-style
    cross-modal interaction block.

    Input:
        query
        context

    Output:
        fused query
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attention = CrossModalAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim)

        self.feed_forward = nn.Sequential(

            nn.Linear(
                dim,
                ff_dim,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                ff_dim,
                dim,
            ),

            nn.Dropout(dropout),

        )

    def forward(
        self,
        query,
        context,
    ):

        x = self.cross_attention(
            query,
            context,
        )

        residual = x

        x = self.ff_norm(x)

        x = self.feed_forward(x)

        x = residual + x

        return x


# ============================================================
# 4. Modality Gating
# ============================================================

class ModalityGate(nn.Module):
    """
    Learns the contribution of each modality.

    Instead of assuming that every modality is equally
    informative, the gate learns:

        Weather weight
        Spectral weight
        Dynamic soil weight
        Static soil weight

    The weights are normalized with softmax.

    Input:
        modality representations

    Output:
        fused representation
        modality weights
    """

    def __init__(
        self,
        dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.score_network = nn.Sequential(

            nn.Linear(
                dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                1,
            ),

        )

    def forward(
        self,
        modalities: Dict[str, torch.Tensor],
    ):

        if len(modalities) == 0:

            raise ValueError(
                "No modalities were provided "
                "to ModalityGate."
            )

        names = list(modalities.keys())

        scores = []

        for name in names:

            x = modalities[name]

            # Temporal global representation
            pooled = x.mean(dim=1)

            score = self.score_network(
                pooled
            )

            scores.append(score)

        scores = torch.cat(
            scores,
            dim=1,
        )

        weights = F.softmax(
            scores,
            dim=1,
        )

        fused = None

        for i, name in enumerate(names):

            weight = weights[:, i].view(
                -1,
                1,
                1,
            )

            contribution = (
                modalities[name] * weight
            )

            if fused is None:

                fused = contribution

            else:

                fused = fused + contribution

        return fused, weights


# ============================================================
# 5. Temporal Attention Pooling
# ============================================================

class TemporalAttentionPooling(nn.Module):
    """
    Learns which time steps are most important.

    Input:
        (B,T,D)

    Output:
        vector  : (B,D)
        weights : (B,T)
    """

    def __init__(
        self,
        dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(
                dim,
                hidden_dim,
            ),

            nn.Tanh(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                1,
            ),

        )

    def forward(self, x):

        scores = self.attention(x)

        scores = scores.squeeze(-1)

        weights = F.softmax(
            scores,
            dim=1,
        )

        pooled = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        return pooled, weights


# ============================================================
# 6. Main MAST-Fuse Fusion Module
# ============================================================

class MASTFuseFusion(nn.Module):
    """
    Main multimodal fusion module.

    Architecture:

             Weather ─────┐
                          │
          Spectral ──────┤
                          │
       Dynamic Soil ─────┤
                          │
        Static Soil ─────┘
                 │
                 ▼
        Cross-modal attention
                 │
                 ▼
        Modality gating
                 │
                 ▼
         Residual fusion
                 │
                 ▼
       Temporal attention
                 │
                 ▼
         Fused vector
    """

    def __init__(
        self,
        fusion_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,

        use_weather: bool = True,
        use_spectral: bool = True,
        use_dynamic_soil: bool = True,
        use_static_soil: bool = True,
    ):
        super().__init__()

        self.fusion_dim = fusion_dim

        self.use_weather = use_weather
        self.use_spectral = use_spectral
        self.use_dynamic_soil = use_dynamic_soil
        self.use_static_soil = use_static_soil

        # ----------------------------------------------------
        # Determine active modalities
        # ----------------------------------------------------

        self.modality_names = []

        if use_weather:
            self.modality_names.append(
                "weather"
            )

        if use_spectral:
            self.modality_names.append(
                "spectral"
            )

        if use_dynamic_soil:
            self.modality_names.append(
                "dynamic_soil"
            )

        if use_static_soil:
            self.modality_names.append(
                "static_soil"
            )

        if len(self.modality_names) == 0:

            raise ValueError(
                "At least one modality must be enabled."
            )

        # ----------------------------------------------------
        # Cross-modal blocks
        #
        # Each modality attends to the other modalities.
        # ----------------------------------------------------

        self.cross_modal_blocks = nn.ModuleDict()

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
        # Modality gate
        # ----------------------------------------------------

        self.modality_gate = ModalityGate(
            dim=fusion_dim,
            hidden_dim=fusion_dim // 2,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Residual fusion
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
        # Temporal pooling
        # ----------------------------------------------------

        self.temporal_pooling = (
            TemporalAttentionPooling(
                dim=fusion_dim,
                hidden_dim=fusion_dim // 2,
                dropout=dropout,
            )
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        modalities: Dict[str, torch.Tensor],
        return_attention: bool = False,
    ):
        """
        Parameters
        ----------
        modalities:
            Dictionary containing tensors:

                {
                    "weather":      (B,T,D),
                    "spectral":     (B,T,D),
                    "dynamic_soil": (B,T,D),
                    "static_soil":  (B,T,D)
                }

        return_attention:
            If True, return modality and temporal
            attention weights.

        Returns
        -------
        fused_vector:
            (B,D)

        fused_sequence:
            (B,T,D)

        attention_info:
            dictionary if requested.
        """

        # ----------------------------------------------------
        # Validate modalities
        # ----------------------------------------------------

        if not isinstance(
            modalities,
            dict,
        ):

            raise TypeError(
                "modalities must be a dictionary."
            )

        active = {}

        for name in self.modality_names:

            if name not in modalities:

                raise ValueError(
                    f"Required modality '{name}' "
                    f"is missing."
                )

            x = modalities[name]

            if x.ndim != 3:

                raise ValueError(
                    f"{name} must have shape "
                    f"(B,T,D), got {tuple(x.shape)}"
                )

            if x.size(-1) != self.fusion_dim:

                raise ValueError(
                    f"{name} has feature dimension "
                    f"{x.size(-1)} but expected "
                    f"{self.fusion_dim}."
                )

            if not torch.isfinite(x).all():

                raise ValueError(
                    f"{name} contains NaN or Inf."
                )

            active[name] = x

        # ----------------------------------------------------
        # Cross-modal interaction
        # ----------------------------------------------------

        cross_modal_outputs = {}

        for query_name in self.modality_names:

            query = active[query_name]

            contexts = []

            for context_name in self.modality_names:

                if query_name == context_name:
                    continue

                context = active[
                    context_name
                ]

                block = (
                    self.cross_modal_blocks[
                        query_name
                    ][
                        context_name
                    ]
                )

                attended = block(
                    query,
                    context,
                )

                contexts.append(
                    attended
                )

            # ------------------------------------------------
            # Average all cross-modal interactions
            # ------------------------------------------------

            if len(contexts) == 0:

                output = query

            else:

                output = torch.stack(
                    contexts,
                    dim=0,
                ).mean(dim=0)

                # Residual preservation
                output = output + query

                output = self.fusion_norm(
                    output
                )

            cross_modal_outputs[
                query_name
            ] = output

        # ----------------------------------------------------
        # Modality gating
        # ----------------------------------------------------

        gated_fusion, modality_weights = (
            self.modality_gate(
                cross_modal_outputs
            )
        )

        # ----------------------------------------------------
        # Residual fusion
        # ----------------------------------------------------

        # Mean representation of active modalities
        modality_mean = torch.stack(
            list(active.values()),
            dim=0,
        ).mean(dim=0)

        fused_sequence = (
            gated_fusion + modality_mean
        )

        fused_sequence = (
            self.fusion_projection(
                fused_sequence
            )
        )

        # ----------------------------------------------------
        # Temporal pooling
        # ----------------------------------------------------

        fused_vector, temporal_weights = (
            self.temporal_pooling(
                fused_sequence
            )
        )

        # ----------------------------------------------------
        # Attention information
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
# 7. Convenience Wrapper
# ============================================================

class CrossModalFusion(nn.Module):
    """
    Convenience wrapper matching the output of
    MultimodalEncoders from encoders.py.

    This allows:

        encoded = encoders(...)

        fused = fusion(encoded)

    """

    def __init__(
        self,
        fusion_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,

        use_weather: bool = True,
        use_spectral: bool = True,
        use_dynamic_soil: bool = True,
        use_static_soil: bool = True,
    ):
        super().__init__()

        self.fusion = MASTFuseFusion(

            fusion_dim=fusion_dim,

            num_heads=num_heads,

            ff_dim=ff_dim,

            dropout=dropout,

            use_weather=use_weather,

            use_spectral=use_spectral,

            use_dynamic_soil=use_dynamic_soil,

            use_static_soil=use_static_soil,

        )

    def forward(
        self,
        encoded_modalities,
        return_attention=False,
    ):

        return self.fusion(
            encoded_modalities,
            return_attention=return_attention,
        )


# ============================================================
# 8. Test
# ============================================================

def test_fusion():

    print("=" * 80)
    print("TESTING MAST-Fuse CROSS-MODAL FUSION")
    print("=" * 80)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Dummy encoder outputs
    # --------------------------------------------------------

    B = 4
    T = 21
    D = 128

    modalities = {

        "weather": torch.randn(
            B,
            T,
            D,
            device=device,
        ),

        "spectral": torch.randn(
            B,
            T,
            D,
            device=device,
        ),

        "dynamic_soil": torch.randn(
            B,
            T,
            D,
            device=device,
        ),

        "static_soil": torch.randn(
            B,
            T,
            D,
            device=device,
        ),

    }

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MASTFuseFusion(

        fusion_dim=128,

        num_heads=4,

        ff_dim=256,

        dropout=0.1,

        use_weather=True,

        use_spectral=True,

        use_dynamic_soil=True,

        use_static_soil=True,

    ).to(device)

    model.eval()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        (
            fused_vector,
            fused_sequence,
            attention,
        ) = model(
            modalities,
            return_attention=True,
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\nInput encoded modalities:")

    for name, x in modalities.items():

        print(
            f"  {name:<15}: "
            f"{tuple(x.shape)}"
        )

    print("\nFusion outputs:")

    print(
        f"  Fused sequence : "
        f"{tuple(fused_sequence.shape)}"
    )

    print(
        f"  Fused vector   : "
        f"{tuple(fused_vector.shape)}"
    )

    # --------------------------------------------------------
    # Expected shapes
    # --------------------------------------------------------

    assert tuple(
        fused_sequence.shape
    ) == (
        B,
        T,
        D,
    )

    assert tuple(
        fused_vector.shape
    ) == (
        B,
        D,
    )

    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    assert torch.isfinite(
        fused_sequence
    ).all()

    assert torch.isfinite(
        fused_vector
    ).all()

    # --------------------------------------------------------
    # Modality attention
    # --------------------------------------------------------

    modality_weights = (
        attention[
            "modality_weights"
        ]
    )

    temporal_weights = (
        attention[
            "temporal_weights"
        ]
    )

    print(
        "\nModality attention shape: "
        f"{tuple(modality_weights.shape)}"
    )

    print(
        "Temporal attention shape: "
        f"{tuple(temporal_weights.shape)}"
    )

    print(
        "\nMean modality weights:"
    )

    mean_weights = (
        modality_weights
        .mean(dim=0)
        .detach()
        .cpu()
    )

    names = attention[
        "modality_names"
    ]

    for name, weight in zip(
        names,
        mean_weights,
    ):

        print(
            f"  {name:<15}: "
            f"{weight.item():.4f}"
        )

    # --------------------------------------------------------
    # Check weights sum to one
    # --------------------------------------------------------

    modality_sum = (
        modality_weights.sum(
            dim=1
        )
    )

    temporal_sum = (
        temporal_weights.sum(
            dim=1
        )
    )

    assert torch.allclose(
        modality_sum,
        torch.ones_like(
            modality_sum
        ),
        atol=1e-5,
    )

    assert torch.allclose(
        temporal_sum,
        torch.ones_like(
            temporal_sum
        ),
        atol=1e-5,
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nParameter count:"
    )

    print(
        f"  Total parameters     : "
        f"{total_params:,}"
    )

    print(
        f"  Trainable parameters : "
        f"{trainable_params:,}"
    )

    print("\n" + "=" * 80)
    print("FUSION TEST PASSED")
    print("=" * 80)


# ============================================================
# 9. Modality Ablation Test
# ============================================================

def test_modality_ablation():

    print("\n")
    print("=" * 80)
    print("TESTING MODALITY ABLATIONS")
    print("=" * 80)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    B = 2
    T = 21
    D = 128

    all_modalities = {

        "weather": torch.randn(
            B, T, D, device=device
        ),

        "spectral": torch.randn(
            B, T, D, device=device
        ),

        "dynamic_soil": torch.randn(
            B, T, D, device=device
        ),

        "static_soil": torch.randn(
            B, T, D, device=device
        ),

    }

    experiments = {

        "full":
        dict(
            use_weather=True,
            use_spectral=True,
            use_dynamic_soil=True,
            use_static_soil=True,
        ),

        "no_weather":
        dict(
            use_weather=False,
            use_spectral=True,
            use_dynamic_soil=True,
            use_static_soil=True,
        ),

        "no_spectral":
        dict(
            use_weather=True,
            use_spectral=False,
            use_dynamic_soil=True,
            use_static_soil=True,
        ),

        "no_dynamic_soil":
        dict(
            use_weather=True,
            use_spectral=True,
            use_dynamic_soil=False,
            use_static_soil=True,
        ),

        "no_static_soil":
        dict(
            use_weather=True,
            use_spectral=True,
            use_dynamic_soil=True,
            use_static_soil=False,
        ),

    }

    for name, settings in experiments.items():

        print(
            f"\nTesting: {name}"
        )

        model = MASTFuseFusion(
            fusion_dim=D,
            num_heads=4,
            ff_dim=256,
            dropout=0.1,
            **settings,
        ).to(device)

        active_modalities = {

            key: value

            for key, value
            in all_modalities.items()

            if settings[
                f"use_{key}"
            ]

        }

        model.eval()

        with torch.no_grad():

            fused_vector, fused_sequence = (
                model(
                    active_modalities
                )
            )

        assert torch.isfinite(
            fused_vector
        ).all()

        print(
            f"  Output: "
            f"{tuple(fused_vector.shape)}"
        )

    print("\n" + "=" * 80)
    print("MODALITY ABLATION TEST PASSED")
    print("=" * 80)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    test_fusion()

    test_modality_ablation()
