# Author - Dipanwita Thakur
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding.

    Input:
        x : (B, T, D)

    Output:
        x : (B, T, D)
    """

    def __init__(
        self,
        max_len: int = 21,
        dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.position = nn.Parameter(
            torch.zeros(1, max_len, dim)
        )

        nn.init.normal_(
            self.position,
            mean=0.0,
            std=0.02
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        t = x.size(1)

        if t > self.position.size(1):
            raise ValueError(
                f"Sequence length {t} exceeds "
                f"max_len={self.position.size(1)}"
            )

        x = x + self.position[:, :t, :]

        return self.dropout(x)


# ============================================================
# Temporal Convolution Block
# ============================================================

class TemporalConvBlock(nn.Module):
    """
    Residual dilated temporal convolution block.

    Input:
        (B, C, T)

    Output:
        (B, C, T)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (
            (kernel_size - 1) * dilation
        ) // 2

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.norm1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.norm2 = nn.BatchNorm1d(channels)

        self.dropout = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual

        x = self.activation(x)

        return x


# ============================================================
# TCN Encoder
# ============================================================

class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Encoder.

    Input:
        (B, T, input_dim)

    Output:
        (B, T, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        dilations=(1, 2, 4),
        dropout: float = 0.1,
        max_len: int = 21,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.positional_encoding = PositionalEncoding(
            max_len=max_len,
            dim=hidden_dim,
            dropout=dropout,
        )

        blocks = []

        for dilation in dilations:

            blocks.append(
                TemporalConvBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.temporal_blocks = nn.Sequential(*blocks)

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                "TCNEncoder expects input "
                f"(B,T,F), got {tuple(x.shape)}"
            )

        # (B,T,F) -> (B,T,H)
        x = self.input_projection(x)

        x = self.positional_encoding(x)

        # Conv1D expects (B,C,T)
        x = x.transpose(1, 2)

        x = self.temporal_blocks(x)

        # Back to (B,T,C)
        x = x.transpose(1, 2)

        x = self.output_projection(x)

        return x


# ============================================================
# Weather Encoder
# ============================================================

class WeatherEncoder(nn.Module):
    """
    Encoder for weather modality.

    Input:
        (B, T, 17)

    Output:
        (B, T, fusion_dim)
    """

    def __init__(
        self,
        input_dim: int = 17,
        fusion_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 21,
    ):
        super().__init__()

        self.encoder = TCNEncoder(
            input_dim=input_dim,
            output_dim=fusion_dim,
            hidden_dim=fusion_dim,
            kernel_size=3,
            dilations=(1, 2, 4),
            dropout=dropout,
            max_len=max_len,
        )

    def forward(self, x):

        return self.encoder(x)


# ============================================================
# Spectral Encoder
# ============================================================

class SpectralEncoder(nn.Module):
    """
    Encoder for vegetation spectral indices.

    Input:
        (B, T, 3)

    Output:
        (B, T, fusion_dim)

    Features:
        NDVI
        NDWI
        EVI
    """

    def __init__(
        self,
        input_dim: int = 3,
        fusion_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 21,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        self.temporal_encoder = TCNEncoder(
            input_dim=64,
            output_dim=fusion_dim,
            hidden_dim=fusion_dim,
            kernel_size=3,
            dilations=(1, 2, 4),
            dropout=dropout,
            max_len=max_len,
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.temporal_encoder(x)

        return x


# ============================================================
# Dynamic Soil Encoder
# ============================================================

class DynamicSoilEncoder(nn.Module):
    """
    Encoder for dynamic soil indices.

    Input:
        (B, T, 4)

    Output:
        (B, T, fusion_dim)

    Features:
        BSI
        SAVI
        NDTI
        RI
    """

    def __init__(
        self,
        input_dim: int = 4,
        fusion_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 21,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        self.temporal_encoder = TCNEncoder(
            input_dim=64,
            output_dim=fusion_dim,
            hidden_dim=fusion_dim,
            kernel_size=3,
            dilations=(1, 2, 4),
            dropout=dropout,
            max_len=max_len,
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.temporal_encoder(x)

        return x


# ============================================================
# Static Soil Encoder
# ============================================================

class StaticSoilEncoder(nn.Module):
    """
    Encoder for static soil properties.

    Current dataset representation:
        (B, T, 28)

    Since soil properties are static, the encoder first
    represents the 28-dimensional soil profile and then
    projects it into the temporal representation.

    Output:
        (B, T, fusion_dim)
    """

    def __init__(
        self,
        input_dim: int = 28,
        fusion_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 21,
    ):
        super().__init__()

        self.profile_encoder = nn.Sequential(

            nn.Linear(input_dim, 96),

            nn.LayerNorm(96),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(96, fusion_dim),

            nn.LayerNorm(fusion_dim),

            nn.GELU(),

        )

        self.positional_encoding = PositionalEncoding(
            max_len=max_len,
            dim=fusion_dim,
            dropout=dropout,
        )

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                "StaticSoilEncoder expects "
                f"(B,T,28), got {tuple(x.shape)}"
            )

        # Encode each temporal copy of the static profile.
        x = self.profile_encoder(x)

        # Positional encoding allows the static modality
        # to participate in temporal cross-modal fusion.
        x = self.positional_encoding(x)

        return x


# ============================================================
# Modality Encoder Container
# ============================================================

class MultimodalEncoders(nn.Module):
    """
    Complete encoder bank.

    Modalities:

        weather
        spectral
        dynamic_soil
        static_soil

    Input:
        weather       : (B,T,17)
        spectral      : (B,T,3)
        dynamic_soil  : (B,T,4)
        static_soil   : (B,T,28)

    Output:
        dictionary containing:

        weather
        spectral
        dynamic_soil
        static_soil

    Each:
        (B,T,fusion_dim)
    """

    def __init__(
        self,
        weather_dim: int = 17,
        spectral_dim: int = 3,
        dynamic_soil_dim: int = 4,
        static_soil_dim: int = 28,

        fusion_dim: int = 128,

        dropout: float = 0.1,

        max_len: int = 21,

        use_weather: bool = True,
        use_spectral: bool = True,
        use_dynamic_soil: bool = True,
        use_static_soil: bool = True,
    ):
        super().__init__()

        self.use_weather = use_weather
        self.use_spectral = use_spectral
        self.use_dynamic_soil = use_dynamic_soil
        self.use_static_soil = use_static_soil

        if use_weather:

            self.weather = WeatherEncoder(
                input_dim=weather_dim,
                fusion_dim=fusion_dim,
                dropout=dropout,
                max_len=max_len,
            )

        if use_spectral:

            self.spectral = SpectralEncoder(
                input_dim=spectral_dim,
                fusion_dim=fusion_dim,
                dropout=dropout,
                max_len=max_len,
            )

        if use_dynamic_soil:

            self.dynamic_soil = DynamicSoilEncoder(
                input_dim=dynamic_soil_dim,
                fusion_dim=fusion_dim,
                dropout=dropout,
                max_len=max_len,
            )

        if use_static_soil:

            self.static_soil = StaticSoilEncoder(
                input_dim=static_soil_dim,
                fusion_dim=fusion_dim,
                dropout=dropout,
                max_len=max_len,
            )

    def forward(
        self,
        weather=None,
        spectral=None,
        dynamic_soil=None,
        static_soil=None,
    ):

        outputs = {}

        if self.use_weather:

            if weather is None:
                raise ValueError(
                    "Weather encoder enabled but "
                    "weather input is None."
                )

            outputs["weather"] = self.weather(weather)

        if self.use_spectral:

            if spectral is None:
                raise ValueError(
                    "Spectral encoder enabled but "
                    "spectral input is None."
                )

            outputs["spectral"] = self.spectral(spectral)

        if self.use_dynamic_soil:

            if dynamic_soil is None:
                raise ValueError(
                    "Dynamic soil encoder enabled but "
                    "dynamic_soil input is None."
                )

            outputs["dynamic_soil"] = self.dynamic_soil(
                dynamic_soil
            )

        if self.use_static_soil:

            if static_soil is None:
                raise ValueError(
                    "Static soil encoder enabled but "
                    "static_soil input is None."
                )

            outputs["static_soil"] = self.static_soil(
                static_soil
            )

        return outputs


# ============================================================
# Standalone Test
# ============================================================

def test_encoders():

    print("=" * 80)
    print("TESTING MuSTIPest-V3 ENCODERS")
    print("=" * 80)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    batch_size = 4
    seq_len = 21

    weather = torch.randn(
        batch_size,
        seq_len,
        17,
        device=device,
    )

    spectral = torch.randn(
        batch_size,
        seq_len,
        3,
        device=device,
    )

    dynamic_soil = torch.randn(
        batch_size,
        seq_len,
        4,
        device=device,
    )

    static_soil = torch.randn(
        batch_size,
        seq_len,
        28,
        device=device,
    )

    model = MultimodalEncoders(
        weather_dim=17,
        spectral_dim=3,
        dynamic_soil_dim=4,
        static_soil_dim=28,

        fusion_dim=128,

        dropout=0.1,

        max_len=21,

        use_weather=True,
        use_spectral=True,
        use_dynamic_soil=True,
        use_static_soil=True,
    ).to(device)

    model.eval()

    with torch.no_grad():

        outputs = model(
            weather=weather,
            spectral=spectral,
            dynamic_soil=dynamic_soil,
            static_soil=static_soil,
        )

    print("\nInput shapes:")

    print(
        f"  Weather      : {tuple(weather.shape)}"
    )

    print(
        f"  Spectral     : {tuple(spectral.shape)}"
    )

    print(
        f"  Dynamic soil : {tuple(dynamic_soil.shape)}"
    )

    print(
        f"  Static soil  : {tuple(static_soil.shape)}"
    )

    print("\nEncoded shapes:")

    for name, tensor in outputs.items():

        print(
            f"  {name:<15}: "
            f"{tuple(tensor.shape)}"
        )

        expected = (
            batch_size,
            seq_len,
            128,
        )

        assert tuple(tensor.shape) == expected, (
            f"{name} has incorrect shape: "
            f"{tuple(tensor.shape)}; "
            f"expected {expected}"
        )

        assert torch.isfinite(tensor).all(), (
            f"{name} contains NaN/Inf"
        )

    print("\nParameter count:")

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
        f"  Total parameters     : {total_params:,}"
    )

    print(
        f"  Trainable parameters : {trainable_params:,}"
    )

    print("\n" + "=" * 80)
    print("ENCODER TEST PASSED")
    print("=" * 80)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    test_encoders()
