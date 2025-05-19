import collections
import itertools
from typing import Optional

import numpy as np
import torch
from torch import nn

import corpora
import mdlrnn
import network
import utils

_TORCH_FLOAT_TYPE = torch.float64


def _get_forward_mesh_layers(net) -> dict[int, frozenset[int]]:
    # Returns dict: {layer depth: set of units in that depth}
    # Places input and output units in separate layers (needed for standard feed-forward using Torch).

    forward_mesh_layers = network.bfs_layers(
        forward_connections=net.forward_connections,
        reverse_forward_connections=net.reverse_forward_connections,
        units=network.get_units(net),
        input_units_range=net.input_units_range,
        output_units_range=net.output_units_range,
    )

    # Move inputs to own layer.
    input_units = frozenset(net.input_units_range)
    for depth in range(2):  # Inputs are at most depth 1.
        forward_mesh_layers[depth] = forward_mesh_layers[depth] - input_units
    if len(forward_mesh_layers[0]) == 0:
        input_level = 0
    else:
        input_level = -1
    forward_mesh_layers[input_level] = input_units

    # Move outputs to own layer.
    output_units = frozenset(net.output_units_range)
    for depth in forward_mesh_layers:
        forward_mesh_layers[depth] = forward_mesh_layers[depth] - output_units
    forward_mesh_layers = {k: v for k, v in forward_mesh_layers.items() if len(v)}
    max_depth = max(forward_mesh_layers)
    forward_mesh_layers[max_depth + 1] = output_units

    if -1 in forward_mesh_layers:
        # Ensure that layer numbers start at 0.
        forward_mesh_layers = {
            n + 1: forward_mesh_layers[n] for n in forward_mesh_layers
        }

    return forward_mesh_layers


def _make_linear_weights(weights: torch.Tensor, bias: Optional[torch.Tensor] = None):
    linear = nn.Linear(
        in_features=weights.shape[1],
        out_features=weights.shape[0],
        bias=bias is not None,
        dtype=_TORCH_FLOAT_TYPE,
    )
    with torch.no_grad():
        linear.weight.copy_(weights)

        if bias is not None:
            linear.bias.copy_(bias)

    return linear


def _get_unit_to_bfs_layer(
    bfs_layer_to_units: dict[int, frozenset[int]],
) -> dict[int, int]:
    unit_to_layer = {}
    for source_layer, source_layer_units in bfs_layer_to_units.items():
        for unit in source_layer_units:
            unit_to_layer[unit] = source_layer
    return unit_to_layer


def _build_memory_layers(
    bfs_layer_to_units: dict[int, frozenset[int]],
    net: network.Network,
):
    memory_units = frozenset(
        unit
        for unit in net.recurrent_connections
        if len(net.recurrent_connections[unit])
    )
    memory_size = len(memory_units)

    unit_to_memory_idx = {unit: i for i, unit in enumerate(sorted(memory_units))}

    layer_to_memory_weights = {}
    memory_to_layer_weights = {}

    for layer, layer_units in bfs_layer_to_units.items():
        to_memory_weights = torch.zeros((memory_size, len(layer_units)))
        from_memory_weight = torch.zeros((len(layer_units), memory_size))

        for i, unit in enumerate(sorted(layer_units)):
            if unit in memory_units:
                to_memory_weights[unit_to_memory_idx[unit], i] = 1

            if unit in net.reverse_recurrent_connections:
                for source_memory_unit in net.reverse_recurrent_connections[unit]:
                    incoming_weight = network.weight_to_float(
                        net.recurrent_weights[(source_memory_unit, unit)]
                    )
                    incoming_memory_idx = unit_to_memory_idx[source_memory_unit]
                    from_memory_weight[i, incoming_memory_idx] = incoming_weight

        if not torch.all(torch.eq(to_memory_weights, 0)):
            layer_to_memory_weights[layer] = _make_linear_weights(
                to_memory_weights, bias=torch.zeros((memory_size,))
            )

        if not torch.all(torch.eq(from_memory_weight, 0)):
            memory_to_layer_weights[layer] = _make_linear_weights(
                from_memory_weight, bias=torch.zeros((from_memory_weight.shape[0],))
            )

    return layer_to_memory_weights, memory_to_layer_weights


def _freeze_defaultdict(dd) -> dict:
    if "lambda" in str(dd.default_factory):
        return {key: _freeze_defaultdict(val) for key, val in dd.items()}
    frozen_dict = {}
    for key, val in dd.items():
        if type(val) == set:
            val = frozenset(val)
        elif type(val) == list:
            val = tuple(val)
        frozen_dict[key] = val
    return frozen_dict


def _build_computation_graph(
    net: network.Network, bfs_layer_to_units
) -> mdlrnn.ComputationGraph:
    # Returns dict: {source_layer: ((target_layer, weights from source to target), ...)}
    unit_to_layer = _get_unit_to_bfs_layer(bfs_layer_to_units)

    layer_to_outgoing_layers = collections.defaultdict(set)
    for layer in bfs_layer_to_units:
        for unit in bfs_layer_to_units[layer]:
            for target_unit in net.forward_connections.get(unit, set()):
                layer_to_outgoing_layers[layer].add(unit_to_layer[target_unit])
    layer_to_outgoing_layers = _freeze_defaultdict(layer_to_outgoing_layers)

    units_with_biases = set()  # Makes sure biases are only added once.

    computation_graph = collections.defaultdict(list)

    # Create weights between connected layers.
    for source_layer in sorted(bfs_layer_to_units):
        source_layer_units = bfs_layer_to_units[source_layer]
        source_layer_size = len(source_layer_units)
        source_to_idx = dict((x, i) for i, x in enumerate(sorted(source_layer_units)))

        for target_layer in layer_to_outgoing_layers.get(source_layer, frozenset()):
            target_layer_units = bfs_layer_to_units[target_layer]
            target_to_idx = dict(
                (x, i) for i, x in enumerate(sorted(target_layer_units))
            )
            target_layer_size = len(target_layer_units)

            weights = torch.zeros(
                (target_layer_size, source_layer_size), dtype=_TORCH_FLOAT_TYPE
            )

            for source in source_layer_units:
                source_idx = source_to_idx[source]
                source_unit_targets = (
                    net.forward_connections.get(source, frozenset())
                    & target_layer_units
                )
                for target in source_unit_targets:
                    target_idx = target_to_idx[target]

                    weight = network.weight_to_float(
                        net.forward_weights[(source, target)]
                    )
                    weights[target_idx, source_idx] = weight

            bias = torch.zeros((target_layer_size,))

            for target in target_layer_units:
                if target in net.biases and target not in units_with_biases:
                    units_with_biases.add(target)
                    bias[target_to_idx[target]] = network.weight_to_float(
                        net.biases[target]
                    )

            weights = _make_linear_weights(weights=weights, bias=bias)

            computation_graph[source_layer].append((target_layer, weights))

    # Connect inputs to 'floating' layers that have no inputs. Use weights = 0.
    floating_layers = (
        set(bfs_layer_to_units)
        - set([x[0] for x in list(itertools.chain(*computation_graph.values()))])
        - {0}  # Input level.
    )

    # TODO: consider making floating unit weights `requires_grad=False`.
    input_size = len(bfs_layer_to_units[0])
    for floating_layer in floating_layers:
        layer_size = len(bfs_layer_to_units[floating_layer])

        floating_layer_bias = torch.zeros((layer_size,))
        for floating_unit_idx, floating_unit in enumerate(
            sorted(bfs_layer_to_units[floating_layer])
        ):
            if floating_unit in net.biases and floating_unit not in units_with_biases:
                units_with_biases.add(floating_unit)
                floating_layer_bias[floating_unit_idx] = network.weight_to_float(
                    net.biases[floating_unit]
                )

        floating_layer_weights = _make_linear_weights(
            weights=torch.zeros((layer_size, input_size)), bias=floating_layer_bias
        )
        computation_graph[0].append((floating_layer, floating_layer_weights))

    return _freeze_defaultdict(computation_graph)


def _build_activation_function_maps(
    net: network.Network,
    bfs_layer_to_units: dict[int, frozenset[int]],
) -> dict[int, dict[int, tuple[int, ...]]]:
    layer_to_activation_to_units = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for source_layer, source_layer_units in bfs_layer_to_units.items():
        for unit_idx, unit in enumerate(sorted(source_layer_units)):
            activation = net.activations[unit]
            layer_to_activation_to_units[source_layer][activation].append(unit_idx)

    return _freeze_defaultdict(layer_to_activation_to_units)


def mdlnn_to_torch(net: network.Network) -> mdlrnn.MDLRNN:
    bfs_layer_to_units = _get_forward_mesh_layers(net)

    computation_graph = _build_computation_graph(net, bfs_layer_to_units)
    layer_to_activation_to_units = _build_activation_function_maps(
        net, bfs_layer_to_units
    )
    layer_to_memory_weights, memory_to_layer_weights = _build_memory_layers(
        bfs_layer_to_units=bfs_layer_to_units, net=net
    )

    return mdlrnn.MDLRNN(
        computation_graph=computation_graph,
        layer_to_memory_weights=layer_to_memory_weights,
        memory_to_layer_weights=memory_to_layer_weights,
        layer_to_activation_to_units=layer_to_activation_to_units,
    )


def eval(net, corpus, accuracy_type, output_layer="normalize") -> float:
    inputs = torch.Tensor(corpus.input_sequence).to(_TORCH_FLOAT_TYPE)
    with torch.no_grad():
        outputs, _ = net(inputs, output_layer=output_layer)

    outputs = outputs.numpy()

    target_classes = corpus.target_sequence[corpus.deterministic_steps_mask].argmax(
        axis=-1
    )

    if accuracy_type == "deterministic":
        predicted_classes = outputs[corpus.deterministic_steps_mask].argmax(axis=-1)
        num_correct = np.sum(predicted_classes == target_classes).item()
        accuracy = num_correct / len(target_classes)
    elif accuracy_type == "categorical":
        valid_targets_mask = corpora.get_dyck_valid_targets(corpus, n=1)
        accuracy, _ = utils.calculate_symbolic_accuracy(
            predicted_probabs=outputs,
            valid_targets_mask=valid_targets_mask,
            input_mask=corpus.input_mask,
            sample_weights=corpus.sample_weights,
            plots=False,
            epsilon=0.005,
        )

    return accuracy
