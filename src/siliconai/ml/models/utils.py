"""Model utilities."""


def conv2d_calculator(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> int:
    """Calculate Conv2d output size."""
    return 1 + (input_size - kernel_size + 2 * padding) // stride


def conv2d_transpose_calculator(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> int:
    """Calculate Conv2d output size."""
    return (input_size - 1) * stride - 2 * padding + kernel_size


def conv2d_sizes(input_dim: int, layers: list[tuple[int, int, int, int]]) -> list[int]:
    """Calculate Conv2d output sizes."""
    sizes = [input_dim]
    for layer_spec in layers:
        sizes.append(
            conv2d_calculator(
                sizes[-1],
                kernel_size=layer_spec[1],
                stride=layer_spec[2],
                padding=layer_spec[3],
            ),
        )
    return sizes


def conv2d_transpose_sizes(
    input_dim: int,
    layers: list[tuple[int, int, int, int]],
    encoder_sizes: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Calculate ConvTranspose2d output sizes."""
    sizes = [input_dim]
    paddings = [0]
    for i in range(len(layers)):
        layer_spec = layers[i]
        sizes.append(
            conv2d_transpose_calculator(
                sizes[-1],
                kernel_size=layer_spec[1],
                stride=layer_spec[2],
                padding=layer_spec[3],
            ),
        )
        padding = 0
        if (
            encoder_sizes is not None
            and sizes[-1] != encoder_sizes[len(encoder_sizes) - i - 2]
        ):
            sizes[-1] += 1
            padding = 1
        paddings.append(padding)
    return sizes, paddings
