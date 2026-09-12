#Author - Dipanwita 

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import MultimodalEncoders
from fusion import CrossModalFusion


# ============================================================
# Configuration
# ============================================================

DEFAULT_WEATHER_DIM = 17
DEFAULT_SPECTRAL_DIM = 3
DEFAULT_DYNAMIC_SOIL_DIM = 4
DEFAULT_STATIC_SOIL_DIM = 28

DEFAULT_SEQUENCE_LENGTH = 21
DEFAULT_FUSION_DIM = 128

DEFAULT_NUM_HEADS = 4
DEFAULT_FF_DIM = 256
DEFAULT_DROPOUT = 0.10

DEFAULT_TEMPORAL_ENCODER = "tcn"
DEFAULT_FUSION_TYPE = "cross_attention"


SUPPORTED_TEMPORAL_ENCODERS = {
    "tcn",
    "gru",
    "bilstm",
    "transformer",
    "none",
}

SUPPORTED_FUSION_TYPES = {
    "cross_attention",
    "mast_fuse",
}


# ============================================================
# Temporal Refinement Modules
# ============================================================

class TemporalTCN(nn.Module):
    """
    Additional temporal refinement using dilated Conv1D.

    Input:
        (B,T,D)

    Output:
        (B,T,D)
    """

    def __init__(
        self,
        dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            dilation=1,
        )

        self.conv2 = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=2,
            dilation=2,
        )

        self.conv3 = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=4,
            dilation=4,
        )

        self.norm1 = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.norm3 = nn.BatchNorm1d(dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x):

        residual = x

        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)

        x = self.output_norm(
            x + residual
        )

        return x


# ============================================================

class TemporalGRU(nn.Module):
    """
    Bidirectional GRU temporal encoder.

    Input:
        (B,T,D)

    Output:
        (B,T,D)
    """

    def __init__(
        self,
        dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = dim // 2

        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.projection = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                dim,
            ),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        residual = x

        x, _ = self.gru(x)

        x = self.projection(x)

        x = self.norm(
            x + residual
        )

        return x


# ============================================================

class TemporalBiLSTM(nn.Module):
    """
    Bidirectional LSTM temporal encoder.

    Input:
        (B,T,D)

    Output:
        (B,T,D)
    """

    def __init__(
        self,
        dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = dim // 2

        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.projection = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                dim,
            ),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        residual = x

        x, _ = self.lstm(x)

        x = self.projection(x)

        x = self.norm(
            x + residual
        )

        return x


# ============================================================

class TemporalTransformer(nn.Module):
    """
    Transformer temporal encoder.

    Input:
        (B,T,D)

    Output:
        (B,T,D)
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        num_layers: int = 2,
        max_len: int = 21,
    ):
        super().__init__()

        self.position = nn.Parameter(
            torch.zeros(
                1,
                max_len,
                dim,
            )
        )

        nn.init.normal_(
            self.position,
            mean=0.0,
            std=0.02,
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        T = x.size(1)

        if T > self.position.size(1):
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"maximum length "
                f"{self.position.size(1)}."
            )

        residual = x

        x = x + self.position[:, :T, :]

        x = self.encoder(x)

        x = self.norm(
            x + residual
        )

        return x


# ============================================================
# Temporal Encoder Factory
# ============================================================

def build_temporal_refiner(
    temporal_encoder: str,
    dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    dropout: float = 0.1,
    sequence_length: int = 21,
):
    """
    Build the selected temporal refinement module.

    Supported:
        tcn
        gru
        bilstm
        transformer
        none
    """

    temporal_encoder = (
        temporal_encoder.lower()
        if temporal_encoder is not None
        else "none"
    )

    if temporal_encoder not in SUPPORTED_TEMPORAL_ENCODERS:

        raise ValueError(
            f"Unsupported temporal_encoder="
            f"'{temporal_encoder}'. "
            f"Supported options: "
            f"{sorted(SUPPORTED_TEMPORAL_ENCODERS)}"
        )

    if temporal_encoder == "none":

        return nn.Identity()

    if temporal_encoder == "tcn":

        return TemporalTCN(
            dim=dim,
            dropout=dropout,
        )

    if temporal_encoder == "gru":

        return TemporalGRU(
            dim=dim,
            dropout=dropout,
        )

    if temporal_encoder == "bilstm":

        return TemporalBiLSTM(
            dim=dim,
            dropout=dropout,
        )

    if temporal_encoder == "transformer":

        return TemporalTransformer(
            dim=dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=2,
            max_len=sequence_length,
        )

    raise RuntimeError(
        "Unexpected temporal encoder."
    )


# ============================================================
# Regression Head
# ============================================================

class RegressionHead(nn.Module):
    """
    Regression head for continuous aphid/pest prediction.

    Input:
        (B,D)

    Output:
        (B,)
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_FUSION_DIM,
        hidden_dims=(128, 64),
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        layers = []

        previous_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.append(
                nn.Linear(
                    previous_dim,
                    hidden_dim,
                )
            )

            layers.append(
                nn.LayerNorm(
                    hidden_dim
                )
            )

            layers.append(
                nn.GELU()
            )

            layers.append(
                nn.Dropout(
                    dropout
                )
            )

            previous_dim = hidden_dim

        layers.append(
            nn.Linear(
                previous_dim,
                1,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):

        if x.ndim != 2:

            raise ValueError(
                "RegressionHead expects "
                f"(B,D), got {tuple(x.shape)}"
            )

        return self.network(
            x
        ).squeeze(-1)


# ============================================================
# MAST-Fuse
# ============================================================

class MASTFuse(nn.Module):
    """
    Complete MAST-Fuse model.

    The modality encoders from encoders.py produce:

        weather       -> (B,T,128)
        spectral      -> (B,T,128)
        dynamic soil  -> (B,T,128)
        static soil   -> (B,T,128)

    A selectable temporal refinement module is then applied
    to each encoded modality.

    Finally CrossModalFusion performs:

        Cross-modal attention
        Modality gating
        Residual fusion
        Temporal attention pooling

    Parameters
    ----------
    temporal_encoder:
        "tcn"
        "gru"
        "bilstm"
        "transformer"
        "none"

    fusion_type:
        "cross_attention"
        "mast_fuse"
    """

    def __init__(
        self,

        # ----------------------------------------------------
        # Input dimensions
        # ----------------------------------------------------

        weather_dim: int = DEFAULT_WEATHER_DIM,
        spectral_dim: int = DEFAULT_SPECTRAL_DIM,
        dynamic_soil_dim: int = DEFAULT_DYNAMIC_SOIL_DIM,
        static_soil_dim: int = DEFAULT_STATIC_SOIL_DIM,

        # ----------------------------------------------------
        # Sequence
        # ----------------------------------------------------

        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        fusion_dim: int = DEFAULT_FUSION_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        ff_dim: int = DEFAULT_FF_DIM,

        dropout: float = DEFAULT_DROPOUT,

        # ----------------------------------------------------
        # Modalities
        # ----------------------------------------------------

        use_weather: bool = True,
        use_spectral: bool = True,
        use_dynamic_soil: bool = True,
        use_static_soil: bool = True,

        # ----------------------------------------------------
        # Architectural selection
        # ----------------------------------------------------

        temporal_encoder: str = DEFAULT_TEMPORAL_ENCODER,

        fusion_type: str = DEFAULT_FUSION_TYPE,
    ):
        super().__init__()

        # ====================================================
        # Configuration
        # ====================================================

        self.weather_dim = weather_dim
        self.spectral_dim = spectral_dim
        self.dynamic_soil_dim = dynamic_soil_dim
        self.static_soil_dim = static_soil_dim

        self.sequence_length = sequence_length

        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.use_weather = use_weather
        self.use_spectral = use_spectral
        self.use_dynamic_soil = use_dynamic_soil
        self.use_static_soil = use_static_soil

        self.temporal_encoder = temporal_encoder
        self.fusion_type = fusion_type

        # ====================================================
        # Validation
        # ====================================================

        if not any(
            [
                use_weather,
                use_spectral,
                use_dynamic_soil,
                use_static_soil,
            ]
        ):
            raise ValueError(
                "At least one modality "
                "must be enabled."
            )

        if temporal_encoder not in (
            SUPPORTED_TEMPORAL_ENCODERS
        ):
            raise ValueError(
                f"Unsupported temporal_encoder="
                f"'{temporal_encoder}'. "
                f"Supported: "
                f"{sorted(SUPPORTED_TEMPORAL_ENCODERS)}"
            )

        if fusion_type not in (
            SUPPORTED_FUSION_TYPES
        ):
            raise ValueError(
                f"Unsupported fusion_type="
                f"'{fusion_type}'. "
                f"Supported: "
                f"{sorted(SUPPORTED_FUSION_TYPES)}"
            )

        # ====================================================
        # Modality-specific encoders
        # ====================================================

        self.encoders = MultimodalEncoders(

            weather_dim=weather_dim,

            spectral_dim=spectral_dim,

            dynamic_soil_dim=dynamic_soil_dim,

            static_soil_dim=static_soil_dim,

            fusion_dim=fusion_dim,

            dropout=dropout,

            max_len=sequence_length,

            use_weather=use_weather,

            use_spectral=use_spectral,

            use_dynamic_soil=use_dynamic_soil,

            use_static_soil=use_static_soil,
        )

        # ====================================================
        # Temporal refinement
        #
        # IMPORTANT:
        # This is shared across modalities.
        #
        # Therefore architectural comparison changes the
        # temporal modeling mechanism while keeping the
        # modality encoders and fusion mechanism fixed.
        # ====================================================

        self.temporal_refiner = (
            build_temporal_refiner(
                temporal_encoder=temporal_encoder,
                dim=fusion_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                sequence_length=sequence_length,
            )
        )

        # ====================================================
        # MAST-Fuse cross-modal fusion
        # ====================================================

        if fusion_type in {
            "cross_attention",
            "mast_fuse",
        }:

            self.fusion = CrossModalFusion(

                fusion_dim=fusion_dim,

                num_heads=num_heads,

                ff_dim=ff_dim,

                dropout=dropout,

                use_weather=use_weather,

                use_spectral=use_spectral,

                use_dynamic_soil=use_dynamic_soil,

                use_static_soil=use_static_soil,
            )

        # ====================================================
        # Regression head
        # ====================================================

        self.regression_head = RegressionHead(

            input_dim=fusion_dim,

            hidden_dims=(128, 64),

            dropout=dropout,
        )

    # ========================================================
    # Input validation
    # ========================================================

    @staticmethod
    def _check_temporal_input(
        x,
        expected_features,
        expected_length,
        name,
    ):

        if not isinstance(
            x,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if x.ndim != 3:

            raise ValueError(
                f"{name} must have shape "
                f"(B,T,F), got "
                f"{tuple(x.shape)}"
            )

        if x.shape[1] != expected_length:

            raise ValueError(
                f"{name} has sequence length "
                f"{x.shape[1]}, expected "
                f"{expected_length}."
            )

        if x.shape[2] != expected_features:

            raise ValueError(
                f"{name} has "
                f"{x.shape[2]} features, "
                f"expected "
                f"{expected_features}."
            )

    # ========================================================

    @staticmethod
    def _prepare_static_soil(
        x,
        expected_features,
        expected_length,
        name="static_soil",
    ):
        """
        Accept:

            (B,28)

        or:

            (B,21,28)

        Static soil is repeated over the temporal
        dimension when supplied as (B,28).
        """

        if not isinstance(
            x,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        # --------------------------------------------
        # Static representation
        # --------------------------------------------

        if x.ndim == 2:

            if x.shape[1] != expected_features:

                raise ValueError(
                    f"{name} has "
                    f"{x.shape[1]} features, "
                    f"expected "
                    f"{expected_features}."
                )

            return (
                x.unsqueeze(1)
                .expand(
                    -1,
                    expected_length,
                    -1,
                )
            )

        # --------------------------------------------
        # Temporal representation
        # --------------------------------------------

        if x.ndim == 3:

            if x.shape[1] != expected_length:

                raise ValueError(
                    f"{name} has sequence "
                    f"length {x.shape[1]}, "
                    f"expected "
                    f"{expected_length}."
                )

            if x.shape[2] != expected_features:

                raise ValueError(
                    f"{name} has "
                    f"{x.shape[2]} features, "
                    f"expected "
                    f"{expected_features}."
                )

            return x

        raise ValueError(
            f"{name} must have shape "
            f"(B,28) or (B,21,28), "
            f"got {tuple(x.shape)}"
        )

    # ========================================================

    @staticmethod
    def _check_finite(
        x,
        name,
    ):

        if not torch.isfinite(x).all():

            n_nan = torch.isnan(
                x
            ).sum().item()

            n_inf = torch.isinf(
                x
            ).sum().item()

            raise ValueError(
                f"{name} contains "
                f"non-finite values: "
                f"NaN={n_nan}, "
                f"Inf={n_inf}"
            )

    # ========================================================
    # Temporal refinement
    # ========================================================

    def _apply_temporal_refinement(
        self,
        embeddings,
    ):
        """
        Apply the selected temporal encoder to
        every active modality.

        Input:
            dictionary of (B,T,D)

        Output:
            dictionary of (B,T,D)
        """

        refined = {}

        for name, x in embeddings.items():

            x = self.temporal_refiner(x)

            self._check_finite(
                x,
                f"temporal_{name}",
            )

            refined[name] = x

        return refined

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,

        weather: Optional[torch.Tensor] = None,

        spectral: Optional[torch.Tensor] = None,

        dynamic_soil: Optional[torch.Tensor] = None,

        static_soil: Optional[torch.Tensor] = None,

        return_aux: bool = False,
    ):
        """
        Forward pass.

        Parameters
        ----------

        weather:
            (B,21,17)

        spectral:
            (B,21,3)

        dynamic_soil:
            (B,21,4)

        static_soil:
            (B,28)
            or
            (B,21,28)

        return_aux:
            False:
                prediction

            True:
                dictionary containing prediction,
                embeddings, fused representations,
                and attention.
        """

        # ====================================================
        # Prepare static soil
        # ====================================================

        prepared_static_soil = None

        # ====================================================
        # Weather
        # ====================================================

        if self.use_weather:

            if weather is None:

                raise ValueError(
                    "Weather modality is enabled "
                    "but weather input is None."
                )

            self._check_temporal_input(
                weather,
                self.weather_dim,
                self.sequence_length,
                "weather",
            )

            self._check_finite(
                weather,
                "weather",
            )

        # ====================================================
        # Spectral
        # ====================================================

        if self.use_spectral:

            if spectral is None:

                raise ValueError(
                    "Spectral modality is enabled "
                    "but spectral input is None."
                )

            self._check_temporal_input(
                spectral,
                self.spectral_dim,
                self.sequence_length,
                "spectral",
            )

            self._check_finite(
                spectral,
                "spectral",
            )

        # ====================================================
        # Dynamic soil
        # ====================================================

        if self.use_dynamic_soil:

            if dynamic_soil is None:

                raise ValueError(
                    "Dynamic soil modality is enabled "
                    "but dynamic_soil input is None."
                )

            self._check_temporal_input(
                dynamic_soil,
                self.dynamic_soil_dim,
                self.sequence_length,
                "dynamic_soil",
            )

            self._check_finite(
                dynamic_soil,
                "dynamic_soil",
            )

        # ====================================================
        # Static soil
        # ====================================================

        if self.use_static_soil:

            if static_soil is None:

                raise ValueError(
                    "Static soil modality is enabled "
                    "but static_soil input is None."
                )

            prepared_static_soil = (
                self._prepare_static_soil(
                    static_soil,
                    self.static_soil_dim,
                    self.sequence_length,
                )
            )

            self._check_finite(
                prepared_static_soil,
                "static_soil",
            )

        # ====================================================
        # Modality-specific encoding
        # ====================================================

        embeddings = self.encoders(

            weather=(
                weather
                if self.use_weather
                else None
            ),

            spectral=(
                spectral
                if self.use_spectral
                else None
            ),

            dynamic_soil=(
                dynamic_soil
                if self.use_dynamic_soil
                else None
            ),

            static_soil=(
                prepared_static_soil
                if self.use_static_soil
                else None
            ),
        )

        # ====================================================
        # Validate encoder outputs
        # ====================================================

        for name, embedding in (
            embeddings.items()
        ):

            if embedding.ndim != 3:

                raise RuntimeError(
                    f"Encoder output for "
                    f"{name} must have shape "
                    f"(B,T,D), got "
                    f"{tuple(embedding.shape)}"
                )

            if embedding.shape[1] != (
                self.sequence_length
            ):

                raise RuntimeError(
                    f"Encoder output for "
                    f"{name} has T="
                    f"{embedding.shape[1]}, "
                    f"expected "
                    f"{self.sequence_length}"
                )

            if embedding.shape[2] != (
                self.fusion_dim
            ):

                raise RuntimeError(
                    f"Encoder output for "
                    f"{name} has D="
                    f"{embedding.shape[2]}, "
                    f"expected "
                    f"{self.fusion_dim}"
                )

            self._check_finite(
                embedding,
                f"encoded_{name}",
            )

        # ====================================================
        # Temporal architectural module
        # ====================================================

        temporal_embeddings = (
            self._apply_temporal_refinement(
                embeddings
            )
        )

        # ====================================================
        # Cross-modal fusion
        # ====================================================

        if return_aux:

            (
                fused_vector,
                fused_sequence,
                attention_info,
            ) = self.fusion(
                temporal_embeddings,
                return_attention=True,
            )

        else:

            (
                fused_vector,
                fused_sequence,
            ) = self.fusion(
                temporal_embeddings,
                return_attention=False,
            )

            attention_info = None

        # ====================================================
        # Validate fusion
        # ====================================================

        if fused_vector.ndim != 2:

            raise RuntimeError(
                "Fusion vector must have shape "
                f"(B,D), got "
                f"{tuple(fused_vector.shape)}"
            )

        if fused_sequence.ndim != 3:

            raise RuntimeError(
                "Fusion sequence must have shape "
                f"(B,T,D), got "
                f"{tuple(fused_sequence.shape)}"
            )

        self._check_finite(
            fused_vector,
            "fused_vector",
        )

        self._check_finite(
            fused_sequence,
            "fused_sequence",
        )

        # ====================================================
        # Regression
        # ====================================================

        prediction = (
            self.regression_head(
                fused_vector
            )
        )

        self._check_finite(
            prediction,
            "prediction",
        )

        # ====================================================
        # Return
        # ====================================================

        if not return_aux:

            return prediction

        return {

            "prediction":
                prediction,

            "embeddings":
                embeddings,

            "temporal_embeddings":
                temporal_embeddings,

            "fused_vector":
                fused_vector,

            "fused_sequence":
                fused_sequence,

            "attention":
                attention_info,

            "modality_weights":
                (
                    attention_info[
                        "modality_weights"
                    ]
                    if attention_info is not None
                    else None
                ),

            "temporal_weights":
                (
                    attention_info[
                        "temporal_weights"
                    ]
                    if attention_info is not None
                    else None
                ),

            "modality_names":
                (
                    attention_info[
                        "modality_names"
                    ]
                    if attention_info is not None
                    else list(
                        embeddings.keys()
                    )
                ),

            "active_modalities":
                list(
                    embeddings.keys()
                ),

            "temporal_encoder":
                self.temporal_encoder,

            "fusion_type":
                self.fusion_type,
        }


# ============================================================
# Model Builder
# ============================================================

def build_model(
    use_weather: bool = True,
    use_spectral: bool = True,
    use_dynamic_soil: bool = True,
    use_static_soil: bool = True,

    temporal_encoder: str = DEFAULT_TEMPORAL_ENCODER,

    fusion_type: str = DEFAULT_FUSION_TYPE,

    fusion_dim: int = DEFAULT_FUSION_DIM,

    num_heads: int = DEFAULT_NUM_HEADS,

    ff_dim: int = DEFAULT_FF_DIM,

    dropout: float = DEFAULT_DROPOUT,

    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
):
    """
    Standard model factory used by train.py.
    """

    return MASTFuse(

        weather_dim=DEFAULT_WEATHER_DIM,

        spectral_dim=DEFAULT_SPECTRAL_DIM,

        dynamic_soil_dim=DEFAULT_DYNAMIC_SOIL_DIM,

        static_soil_dim=DEFAULT_STATIC_SOIL_DIM,

        sequence_length=sequence_length,

        fusion_dim=fusion_dim,

        num_heads=num_heads,

        ff_dim=ff_dim,

        dropout=dropout,

        use_weather=use_weather,

        use_spectral=use_spectral,

        use_dynamic_soil=use_dynamic_soil,

        use_static_soil=use_static_soil,

        temporal_encoder=temporal_encoder,

        fusion_type=fusion_type,
    )


# ============================================================
# Parameter Counting
# ============================================================

def count_parameters(model):

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
# Model Summary
# ============================================================

def print_model_summary(model):

    total, trainable = (
        count_parameters(model)
    )

    print()
    print("=" * 80)
    print("MAST-FUSE MODEL SUMMARY")
    print("=" * 80)

    print(
        f"Weather            : "
        f"{model.use_weather}"
    )

    print(
        f"Spectral           : "
        f"{model.use_spectral}"
    )

    print(
        f"Dynamic Soil       : "
        f"{model.use_dynamic_soil}"
    )

    print(
        f"Static Soil        : "
        f"{model.use_static_soil}"
    )

    print(
        f"Sequence            : "
        f"{model.sequence_length}"
    )

    print(
        f"Fusion dimension    : "
        f"{model.fusion_dim}"
    )

    print(
        f"Attention heads     : "
        f"{model.num_heads}"
    )

    print(
        f"FF dimension        : "
        f"{model.ff_dim}"
    )

    print(
        f"Temporal encoder    : "
        f"{model.temporal_encoder}"
    )

    print(
        f"Fusion type         : "
        f"{model.fusion_type}"
    )

    print(
        f"Total parameters    : "
        f"{total:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print("=" * 80)


# ============================================================
# Tensor Statistics
# ============================================================

def tensor_statistics(
    tensor,
    name,
):

    tensor = tensor.detach()

    print(
        f"{name:<22}"
        f"shape={tuple(tensor.shape)} "
        f"min={tensor.min().item():.6f} "
        f"max={tensor.max().item():.6f} "
        f"mean={tensor.mean().item():.6f} "
        f"std={tensor.std().item():.6f}"
    )


# ============================================================
# Standalone Model Test
# ============================================================

def test_model():

    print("=" * 80)
    print("TESTING MAST-FUSE COMPLETE MODEL")
    print("=" * 80)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    B = 4
    T = 21

    weather = torch.randn(
        B,
        T,
        17,
        device=device,
    )

    spectral = torch.randn(
        B,
        T,
        3,
        device=device,
    )

    dynamic_soil = torch.randn(
        B,
        T,
        4,
        device=device,
    )

    static_soil = torch.randn(
        B,
        T,
        28,
        device=device,
    )

    print()
    print("Input shapes:")

    print(
        f"  Weather      : "
        f"{tuple(weather.shape)}"
    )

    print(
        f"  Spectral     : "
        f"{tuple(spectral.shape)}"
    )

    print(
        f"  Dynamic soil : "
        f"{tuple(dynamic_soil.shape)}"
    )

    print(
        f"  Static soil  : "
        f"{tuple(static_soil.shape)}"
    )

    # ========================================================
    # TEST 1
    # ========================================================

    print()
    print("-" * 80)
    print("TEST 1 — FULL MODEL / TCN")
    print("-" * 80)

    model = build_model(
        use_weather=True,
        use_spectral=True,
        use_dynamic_soil=True,
        use_static_soil=True,

        temporal_encoder="tcn",

        fusion_type="cross_attention",
    ).to(device)

    model.eval()

    print_model_summary(model)

    with torch.no_grad():

        prediction = model(
            weather=weather,
            spectral=spectral,
            dynamic_soil=dynamic_soil,
            static_soil=static_soil,
        )

    print(
        f"Prediction shape: "
        f"{tuple(prediction.shape)}"
    )

    tensor_statistics(
        prediction,
        "Prediction",
    )

    assert tuple(
        prediction.shape
    ) == (B,)

    assert torch.isfinite(
        prediction
    ).all()

    print("PASS")

    # ========================================================
    # TEST 2
    # ========================================================

    print()
    print("-" * 80)
    print("TEST 2 — AUXILIARY OUTPUTS")
    print("-" * 80)

    with torch.no_grad():

        outputs = model(
            weather=weather,
            spectral=spectral,
            dynamic_soil=dynamic_soil,
            static_soil=static_soil,
            return_aux=True,
        )

    print(
        "Prediction     :",
        tuple(
            outputs[
                "prediction"
            ].shape
        ),
    )

    print(
        "Fused vector   :",
        tuple(
            outputs[
                "fused_vector"
            ].shape
        ),
    )

    print(
        "Fused sequence :",
        tuple(
            outputs[
                "fused_sequence"
            ].shape
        ),
    )

    print(
        "Temporal encoder:",
        outputs[
            "temporal_encoder"
        ],
    )

    print(
        "Modalities:",
        outputs[
            "modality_names"
        ],
    )

    if outputs[
        "modality_weights"
    ] is not None:

        print(
            "Modality weights:",
            tuple(
                outputs[
                    "modality_weights"
                ].shape
            ),
        )

    if outputs[
        "temporal_weights"
    ] is not None:

        print(
            "Temporal weights:",
            tuple(
                outputs[
                    "temporal_weights"
                ].shape
            ),
        )

    print("PASS")

    # ========================================================
    # TEST 3
    # ========================================================

    print()
    print("-" * 80)
    print("TEST 3 — STATIC SOIL (B,28)")
    print("-" * 80)

    static_profile = torch.randn(
        B,
        28,
        device=device,
    )

    with torch.no_grad():

        prediction = model(
            weather=weather,
            spectral=spectral,
            dynamic_soil=dynamic_soil,
            static_soil=static_profile,
        )

    assert tuple(
        prediction.shape
    ) == (B,)

    print(
        "Output:",
        tuple(
            prediction.shape
        ),
    )

    print("PASS")

    # ========================================================
    # TEST 4-7
    # Modality ablations
    # ========================================================

    ablations = [

        (
            "NO WEATHER",
            False,
            True,
            True,
            True,
        ),

        (
            "NO SPECTRAL",
            True,
            False,
            True,
            True,
        ),

        (
            "NO DYNAMIC SOIL",
            True,
            True,
            False,
            True,
        ),

        (
            "NO STATIC SOIL",
            True,
            True,
            True,
            False,
        ),
    ]

    for (
        label,
        use_weather,
        use_spectral,
        use_dynamic_soil,
        use_static_soil,
    ) in ablations:

        print()
        print("-" * 80)

        print(
            f"TEST — MODALITY ABLATION: "
            f"{label}"
        )

        print("-" * 80)

        ablation_model = build_model(

            use_weather=use_weather,

            use_spectral=use_spectral,

            use_dynamic_soil=
                use_dynamic_soil,

            use_static_soil=
                use_static_soil,

            temporal_encoder="tcn",

            fusion_type="cross_attention",
        ).to(device)

        ablation_model.eval()

        with torch.no_grad():

            prediction = ablation_model(

                weather=(
                    weather
                    if use_weather
                    else None
                ),

                spectral=(
                    spectral
                    if use_spectral
                    else None
                ),

                dynamic_soil=(
                    dynamic_soil
                    if use_dynamic_soil
                    else None
                ),

                static_soil=(
                    static_soil
                    if use_static_soil
                    else None
                ),
            )

        assert tuple(
            prediction.shape
        ) == (B,)

        assert torch.isfinite(
            prediction
        ).all()

        print(
            "Output:",
            tuple(
                prediction.shape
            ),
        )

        print("PASS")

    # ========================================================
    # TEST 8
    # Temporal architecture ablations
    # ========================================================

    print()
    print("=" * 80)
    print("TESTING TEMPORAL ARCHITECTURAL ABLATIONS")
    print("=" * 80)

    temporal_models = [
        "tcn",
        "gru",
        "bilstm",
        "transformer",
        "none",
    ]

    for temporal_type in temporal_models:

        print()
        print(
            f"Testing temporal encoder: "
            f"{temporal_type}"
        )

        temporal_model = build_model(

            use_weather=True,

            use_spectral=True,

            use_dynamic_soil=True,

            use_static_soil=True,

            temporal_encoder=
                temporal_type,

            fusion_type=
                "cross_attention",
        ).to(device)

        temporal_model.eval()

        with torch.no_grad():

            prediction = temporal_model(

                weather=weather,

                spectral=spectral,

                dynamic_soil=dynamic_soil,

                static_soil=static_soil,
            )

        assert tuple(
            prediction.shape
        ) == (B,)

        assert torch.isfinite(
            prediction
        ).all()

        print(
            f"  Output: "
            f"{tuple(prediction.shape)}"
        )

        print("  PASS")

    # ========================================================
    # TEST 9
    # Gradient test
    # ========================================================

    print()
    print("-" * 80)
    print("TEST — GRADIENT FLOW")
    print("-" * 80)

    gradient_model = build_model(
        temporal_encoder="tcn",
        fusion_type="cross_attention",
    ).to(device)

    gradient_model.train()

    prediction = gradient_model(
        weather=weather,
        spectral=spectral,
        dynamic_soil=dynamic_soil,
        static_soil=static_soil,
    )

    loss = prediction.mean()

    loss.backward()

    gradient_count = 0

    for parameter in (
        gradient_model.parameters()
    ):

        if parameter.requires_grad:

            if parameter.grad is not None:

                assert torch.isfinite(
                    parameter.grad
                ).all()

                gradient_count += 1

    print(
        f"Parameters receiving gradients: "
        f"{gradient_count}"
    )

    assert gradient_count > 0

    print("PASS")

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print("ALL MAST-FUSE MODEL TESTS PASSED")
    print("=" * 80)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    test_model()
