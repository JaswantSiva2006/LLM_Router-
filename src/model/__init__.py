"""Model adaptations for explicit decoder-layer execution routes."""

from src.model.routed_qwen import (
    LayerAction,
    RoutedQwen2Model,
    enable_qwen2_routing,
    generate_with_route,
    labels_to_path,
    path_to_labels,
    validate_path,
)

__all__ = [
    "LayerAction",
    "RoutedQwen2Model",
    "enable_qwen2_routing",
    "generate_with_route",
    "labels_to_path",
    "path_to_labels",
    "validate_path",
]

