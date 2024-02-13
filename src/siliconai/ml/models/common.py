"""Common base models."""
from torch import Tensor, nn


class SequentialMLP(nn.Module):
    """Sequential MLP module."""

    def __init__(
        self,
        layers_dim: list[int],
        activation: str,
        activation_parameters: list[float],
        batch_norm: bool,
        dropout: float,
        output_activation: bool = True,
        output_batch_norm: bool = True,
    ) -> None:
        """Initialize the module."""
        super().__init__()
        activation_function = getattr(nn, activation)

        layers = []
        for i in range(1, len(layers_dim)):
            in_dim, out_dim = layers_dim[i - 1], layers_dim[i]
            layers += self.generate_layers(
                in_dim,
                out_dim,
                activation_function
                if output_activation or i < len(layers_dim) - 1
                else None,
                activation_parameters,
                batch_norm if output_batch_norm or i < len(layers_dim) - 1 else False,
                dropout,
            )

        self.model = nn.Sequential(*layers)

    @staticmethod
    def generate_layers(
        input_dim: int,
        output_dim: int,
        activation_function: type[nn.Module] | None,
        activation_parameters: list[float],
        batch_norm: bool,
        dropout: float,
    ) -> list[nn.Module]:
        """Initialize the module."""
        layers: list[nn.Module] = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(input_dim, output_dim))

        if batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))

        if activation_function is not None:
            layers.append(activation_function(*activation_parameters))

        return layers

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        z: Tensor = self.model(x)
        return z


class SequentialConv2d(nn.Module):
    """Sequential convolutional module in 2 dimensions."""

    def __init__(
        self,
        layers_dim: list[int],
        activation: str,
        activation_parameters: list[float],
        batch_norm: bool,
        dropout: float,
        output_activation: bool = True,
        output_batch_norm: bool = True,
        transpose: bool = False,
    ) -> None:
        """Initialize the module."""
        super().__init__()
        activation_function = getattr(nn, activation)

        layers = []
        for i in range(1, len(layers_dim)):
            in_dim, out_dim = layers_dim[i - 1], layers_dim[i]
            layers += self.generate_layers(
                in_dim,
                out_dim,
                activation_function
                if output_activation or i < len(layers_dim) - 1
                else None,
                activation_parameters,
                batch_norm if output_batch_norm or i < len(layers_dim) - 1 else False,
                dropout,
                transpose,
            )

        self.model = nn.Sequential(*layers)

    @staticmethod
    def generate_layers(
        input_dim: int,
        output_dim: int,
        activation_function: type[nn.Module] | None,
        activation_parameters: list[float],
        batch_norm: bool,
        dropout: float,
        transpose: bool = False,
    ) -> list[nn.Module]:
        """Initialize the module."""
        layers: list[nn.Module] = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        if transpose:
            layers.append(
                nn.ConvTranspose2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            )
        else:
            layers.append(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1),
            )

        if batch_norm:
            layers.append(nn.BatchNorm2d(output_dim))

        if activation_function is not None:
            layers.append(activation_function(*activation_parameters))

        return layers

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        z: Tensor = self.model(x)
        return z
