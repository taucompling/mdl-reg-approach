import network


def make_differentiable_an_bn_net():
    return network.make_custom_network(
        input_size=3,
        output_size=3,
        num_units=7,
        forward_weights={
            1: ((6, 1, 1, 2),),
            2: ((6, -1, 1, 2), (3, 1, 1, 1), (4, -1, 3, 1)),
            6: ((5, 1, 3, 1),),
        },
        recurrent_weights={6: ((3, -1, 1, 1), (6, 1, 1, 1))},
        activations={
            3: network.TANH,
            4: network.RELU,
            5: network.TANH,
            6: network.RELU,
        },
        biases={4: 3},
    )


def make_emmanuel_dyck_2_network(nesting_probab: float):
    opening_bracket_output_bias = nesting_probab / (2 * (1 - nesting_probab))
    return network.make_custom_network(
        input_size=5,
        output_size=5,
        num_units=17,
        forward_weights={
            1: ((3, 1, 2, 1),),
            2: ((12, -1, 1, 1), (13, 1, 1, 1)),
            3: ((10, 1, 1, 1),),
            4: ((2, 1, 1, 1),),
            5: ((9, -1, 1, 1),),
            6: ((8, 1, 1, 1),),
            7: ((9, -1, 1, 1),),
            10: ((11, 1, 1, 1),),
            11: ((15, 1, 1, 1),),
            12: ((11, 1, 1, 1),),
            13: ((15, 1, 1, 1),),
            14: ((13, 1, 1, 1),),
            15: ((16, 1, 1, 1),),
            16: ((5, -1, 1, 1), (7, 1, 1, 1)),
        },
        recurrent_weights={15: ((10, 1, 3, 1), (14, 1, 1, 3))},
        unit_types={
            11: network.MULTIPLICATION_UNIT,
            13: network.MULTIPLICATION_UNIT,
        },
        biases={5: 1, 6: opening_bracket_output_bias, 7: -1, 9: 1, 12: 1},
        activations={
            5: network.UNSIGNED_STEP,
            7: network.UNSIGNED_STEP,
            14: network.FLOOR,
            16: network.MODULO_3,
        },
    )


def make_emmanuel_dyck_2_network_io_protection(nesting_probab: float):
    opening_bracket_output_bias = nesting_probab / (2 * (1 - nesting_probab))
    return network.make_custom_network(
        input_size=5,
        output_size=5,
        num_units=23,
        forward_weights={
            1: ((18, 1, 2, 1),),
            2: ((17, 1, 1, 1),),
            3: ((18, 1, 1, 1),),
            4: ((17, 1, 1, 1),),
            10: ((11, 1, 1, 1),),
            11: ((15, 1, 1, 1),),
            12: ((11, 1, 1, 1),),
            13: ((15, 1, 1, 1),),
            14: ((13, 1, 1, 1),),
            15: ((16, 1, 1, 1),),
            16: ((19, -1, 1, 1), (20, 1, 1, 1)),
            17: ((12, -1, 1, 1), (13, 1, 1, 1)),
            18: ((10, 1, 1, 1),),
            19: ((21, -1, 1, 1), (20, 1, 1, 1), (5, 1, 1, 1)),
            20: ((7, 1, 1, 1), (21, -1, 1, 1)),
            21: ((9, 1, 1, 1),),
            22: ((6, 1, 1, 1), (8, 1, 1, 1)),
        },
        recurrent_weights={15: ((10, 1, 3, 1), (14, 1, 1, 3))},
        unit_types={
            11: network.MULTIPLICATION_UNIT,
            13: network.MULTIPLICATION_UNIT,
        },
        biases={12: 1, 19: 1, 20: -1, 21: 1, 22: opening_bracket_output_bias},
        activations={
            14: network.FLOOR,
            16: network.MODULO_3,
            19: network.UNSIGNED_STEP,
            20: network.UNSIGNED_STEP,
        },
    )


def make_emmanuel_triplet_xor_network():
    net = network.make_custom_network(
        input_size=1,
        output_size=1,
        num_units=7,
        forward_weights={
            0: ((2, 1, 1, 1),),
            2: (
                (5, -1, 1, 1),
                (6, 1, 1, 1),
            ),
            3: ((6, 1, 1, 1), (5, 1, 1, 1)),
            4: ((3, 1, 1, 1),),
            5: ((1, -1, 1, 2),),
            6: ((1, 1, 1, 2),),
        },
        recurrent_weights={
            0: ((2, -1, 1, 1),),
            3: ((4, -1, 3, 1),),
            4: ((4, 1, 1, 1),),
        },
        biases={1: 0.5, 3: -1, 4: 1, 6: -1},
        activations={
            2: network.SQUARE,
            3: network.RELU,
            5: network.RELU,
            6: network.RELU,
        },
    )
    return net


def make_tacl_paper_an_bn_net():
    return network.make_custom_network(
        input_size=3,
        output_size=3,
        num_units=7,
        forward_weights={
            1: ((6, 1, 2, 1),),
            2: ((4, -1, 3, 1),),
            6: ((5, 1, 1, 1),),
        },
        recurrent_weights={6: ((6, 1, 1, 1),)},
        biases={3: -15, 4: (7 / 3), 6: -1},
        activations={3: network.SIGMOID, 5: network.UNSIGNED_STEP, 6: network.RELU},
    )


def make_tacl_paper_an_bn_cn_net():
    return network.make_custom_network(
        input_size=4,
        output_size=4,
        num_units=10,
        forward_weights={
            0: ((5, 1, 1, 1),),
            1: ((5, 1, 7, 3), (8, -1, 2, 1)),
            3: ((4, 1, 1, 1), (9, 1, 1, 1)),
            8: ((6, -1, 1, 1),),
            9: ((4, 1, 1, 1),),
        },
        recurrent_weights={8: ((8, 1, 1, 1),), 9: ((9, 1, 1, 1),)},
        biases={6: 1, 7: -15, 8: 1, 9: -(1 / 3)},
        activations={
            6: network.UNSIGNED_STEP,
            7: network.SIGMOID,
        },
    )


def make_tacl_paper_dyck_1_net():
    return network.make_custom_network(
        input_size=3,
        output_size=3,
        num_units=7,
        forward_weights={1: ((6, -1, 2, 1),), 6: ((3, 1, 1, 1), (5, -1, 1, 3))},
        recurrent_weights={6: ((6, 1, 1, 1),)},
        biases={5: 1, 6: 1},
        activations={
            4: network.SIGMOID,
            5: network.FLOOR,
        },
    )


def make_tacl_paper_dyck_2_net(nesting_probab: float):
    opening_bracket_output_bias = nesting_probab / (2 * (1 - nesting_probab))
    return network.make_custom_network(
        input_size=5,
        output_size=5,
        num_units=23,
        forward_weights={
            1: ((10, 1, 2, 1),),
            2: ((14, 1, 1, 1),),
            3: ((10, 1, 1, 1),),
            4: ((14, 1, 1, 1),),
            10: ((11, 1, 1, 1),),
            11: ((12, 1, 1, 1),),
            12: ((13, 1, 1, 1),),
            13: ((18, 1, 1, 1),),
            14: ((15, -1, 1, 1), (16, 1, 1, 1)),
            15: ((12, 1, 1, 1),),
            16: ((13, 1, 1, 1),),
            17: ((16, 1, 1, 1),),
            18: ((19, 1, 1, 1), (20, -1, 1, 1)),
            19: ((7, 1, 1, 1), (21, -1, 1, 1)),
            20: ((5, 1, 1, 1), (19, 1, 1, 1), (21, -1, 1, 1)),
            21: ((9, 1, 1, 1),),
            22: ((6, 1, 1, 1), (8, 1, 1, 1)),
        },
        recurrent_weights={13: ((11, 1, 3, 1), (17, 1, 1, 3))},
        unit_types={
            12: network.MULTIPLICATION_UNIT,
            16: network.MULTIPLICATION_UNIT,
        },
        biases={15: 1, 19: -1, 20: 1, 21: 1, 22: opening_bracket_output_bias},
        activations={
            17: network.FLOOR,
            18: network.MODULO_3,
            19: network.UNSIGNED_STEP,
            20: network.UNSIGNED_STEP,
        },
    )


def make_golden_arithmetic_net():
    return network.make_custom_network(
        input_size=5,
        output_size=5,
        num_units=28,
        forward_weights={
            0: ((6, 1, 1, 1),),
            1: (
                (6, 1, 1, 3),
                (9, 1, 2, 3),
                (10, 1, 1, 1),
                (11, 1, 1, 1),
                (16, 1, 1, 1),
            ),
            2: (
                (12, 1, 1, 1),
                (19, 1, 1, 1),
                (21, 1, 1, 1),
                (26, 1, 1, 1),
            ),
            3: ((6, 1, 1, 3), (9, 1, 2, 3), (23, 1, 1, 1)),
            4: ((21, 1, 1, 1), (25, 1, 1, 1), (26, 1, 1, 1)),
            10: ((11, 1, 1, 1),),
            11: ((27, 1, 1, 1),),
            12: ((13, -1, 1, 1),),
            13: ((10, 1, 3, 1), (17, 1, 1, 1), (27, 1, 1, 1)),
            14: ((15, 1, 1, 1),),
            15: ((16, -1, 1, 1),),
            16: ((13, 1, 1, 1),),
            17: ((18, -1, 1, 1), (20, 1, 1, 1), (22, 1, 1, 1)),
            18: ((19, 1, 1, 1), (22, 1, 1, 1), (24, -1, 1, 1)),
            19: ((5, 1, 1, 1),),
            20: ((21, 1, 1, 1), (22, -1, 2, 1), (24, -1, 1, 1)),
            21: ((7, 1, 1, 1),),
            22: ((23, 1, 1, 1), (24, -1, 1, 1), (26, 1, 1, 1)),
            23: ((27, 1, 1, 1),),
            24: ((25, 1, 1, 1), (26, 1, 1, 1)),
            25: ((27, 1, 1, 1),),
            26: ((8, 1, 1, 1),),
        },
        recurrent_weights={
            27: ((12, 1, 3, 4), (13, 1, 1, 1), (14, 1, 1, 1)),
        },
        unit_types={
            10: network.MULTIPLICATION_UNIT,
            12: network.MULTIPLICATION_UNIT,
        },
        biases={
            15: -1,
            18: 1,
            19: -1,
            20: -2,
            21: -1,
            22: -1,
            23: -1,
            24: 1,
            25: -1,
            26: -1,
        },
        activations={
            13: network.FLOOR,
            14: network.MODULO_4,
            15: network.ABS,
            16: network.UNSIGNED_STEP,
            17: network.MODULO_4,
            18: network.UNSIGNED_STEP,
            19: network.UNSIGNED_STEP,
            20: network.UNSIGNED_STEP,
            21: network.UNSIGNED_STEP,
            22: network.UNSIGNED_STEP,
            23: network.UNSIGNED_STEP,
            25: network.UNSIGNED_STEP,
            26: network.UNSIGNED_STEP,
        },
    )


def make_golden_toy_english_net():
    return network.make_custom_network(
        input_size=5,
        output_size=5,
        num_units=25,
        forward_weights={
            1: (
                (12, -1, 1, 1),
                (15, 1, 1, 1),
                (18, 1, 1, 1),
                (19, 1, 1, 1),
            ),
            2: (
                (13, 1, 1, 1),
                (14, 1, 1, 1),
                (16, 1, 1, 1),
            ),
            3: (
                (5, 1, 1, 1),
                (7, -1, 1, 1),
            ),
            4: ((10, -1, 1, 1),),
            10: ((11, 1, 1, 1),),
            12: (
                (11, 1, 1, 1),
                (13, -1, 1, 1),
                (15, -1, 1, 1),
                (16, -1, 1, 1),
                (23, 1, 1, 1),
                (24, 1, 1, 1),
            ),
            13: ((6, 1, 1, 1), (8, 1, 1, 1), (9, 1, 1, 1)),
            14: ((6, 1, 3, 1),),
            15: ((17, 1, 1, 1),),
            16: ((5, 1, 3, 1),),
            17: ((13, -1, 1, 1), (16, 1, 1, 1), (21, 1, 1, 1), (22, -1, 1, 1)),
            18: ((20, 1, 1, 1),),
            19: (
                (6, 1, 1, 1),
                (7, -1, 1, 1),
                (20, -1, 1, 1),
            ),
            20: ((21, 1, 1, 1), (22, 1, 1, 1)),
            21: ((5, 1, 1, 1), (7, -1, 1, 1)),
            22: ((6, 1, 1, 1), (7, -1, 1, 1), (8, 1, 1, 1), (9, 1, 1, 1)),
            23: ((14, 1, 1, 1), (18, 1, 1, 1)),
            24: ((19, 1, 1, 1),),
        },
        recurrent_weights={
            2: ((12, 1, 1, 1),),
            11: ((12, 1, 1, 1),),
            17: ((17, 1, 1, 1),),
        },
        unit_types={11: network.MULTIPLICATION_UNIT},
        biases={7: 1, 10: 1, 14: -1, 16: -1, 18: -1, 19: -1, 21: -1, 24: -1},
        activations={
            13: network.UNSIGNED_STEP,
            14: network.UNSIGNED_STEP,
            15: network.UNSIGNED_STEP,
            16: network.UNSIGNED_STEP,
            18: network.UNSIGNED_STEP,
            19: network.UNSIGNED_STEP,
            21: network.UNSIGNED_STEP,
            22: network.UNSIGNED_STEP,
            23: network.UNSIGNED_STEP,
            24: network.UNSIGNED_STEP,
        },
    )


def make_found_differentiable_dyck1_net():
    return network.make_custom_network(
        input_size=3,
        output_size=3,
        num_units=7,
        forward_weights={
            1: ((6, 1, 2, 1),),
            6: (
                (5, 1, 3, 1),
                (3, -1, 1, 1),
            ),
        },
        recurrent_weights={
            6: ((6, 1, 1, 1),),
        },
        biases={3: 1, 6: -1},
        activations={
            3: network.RELU,
            4: network.SIGMOID,
            5: network.TANH,
            6: network.RELU,
        },
    )


def make_found_differentiable_an_bn_cn_net():
    return network.make_custom_network(
        input_size=4,
        output_size=4,
        num_units=10,
        forward_weights={
            0: ((5, 1, 1, 1),),
            1: ((5, 1, 7, 3), (9, 1, 2, 1)),
            3: ((8, -1, 2, 1),),
            8: ((4, -1, 1, 1),),
            9: ((6, 1, 3, 1),),
        },
        recurrent_weights={8: ((8, 1, 1, 1),), 3: ((8, 1, 1, 1),), 9: ((9, 1, 1, 1),)},
        biases={7: -15, 8: 1 / 3, 9: -1},
        activations={
            5: network.RELU,
            6: network.TANH,
            7: network.SIGMOID,
            9: network.RELU,
        },
    )


def make_found_diff_softmax_an_bn_net():
    return network.make_custom_network(
        input_size=3,
        output_size=3,
        num_units=7,
        forward_weights={
            0: ((4, 1, 15, 1),),
            1: ((6, 1, 2, 1),),
            2: ((4, -1, 15, 1),),
            6: ((3, -1, 15, 1),),
        },
        recurrent_weights={6: ((6, 1, 1, 1),)},
        # The network found by the simulations had -1 bias for unit 5, but we changed it manually
        # for -0.85 to have probabilities closer to the underlying distribution
        biases={3: 6, 5: -0.85, 6: -1},
        activations={6: network.RELU},
    )


def make_found_diff_softmax_dyck_1_net():
    return network.make_custom_network(
        input_size=3,
        output_size=3,
        num_units=8,
        forward_weights={
            1: ((6, -1, 1, 1),),
            2: ((6, 1, 1, 1),),
            6: ((3, 1, 7, 1), (7, 1, 1, 1)),
            7: ((5, -1, 7, 1),),
        },
        recurrent_weights={6: ((6, 1, 1, 1),)},
        # The network found by the simulations had -1 bias for unit 4, but we changed it manually
        # for -0.75 to have probabilities closer to the underlying distribution
        biases={4: -0.75, 7: 1},
        activations={4: network.TANH, 7: network.RELU},
    )


def make_found_diff_softmax_an_bn_cn_net():
    return network.make_custom_network(
        input_size=4,
        output_size=4,
        num_units=11,
        forward_weights={
            0: ((5, 1, 15, 1),),
            1: ((8, -1, 2, 1),),
            2: ((5, -1, 15, 1), (8, 1, 1, 1)),
            3: ((8, 1, 1, 1),),
            8: ((4, 1, 31, 1), (9, 1, 2, 1)),
            9: ((7, 1, 15, 1),),
        },
        recurrent_weights={
            0: ((8, 1, 1, 1),),
            8: ((8, 1, 1, 1),),
            10: (
                (9, -1, 1, 1),
                (10, 1, 1, 1),
            ),
        },
        biases={5: 4 / 3, 10: -1},
        activations={
            6: network.SIGMOID,
            9: network.TANH,
        },
    )


GOLDEN_NETWORKS = {
    "an_bn": make_tacl_paper_an_bn_net(),
    "an_bn_cn": make_tacl_paper_an_bn_cn_net(),
    "dyck_1": make_found_differentiable_dyck1_net(),
    "dyck_2": make_tacl_paper_dyck_2_net(nesting_probab=0.33333),
    "arithmetic": make_golden_arithmetic_net(),
    "toy_english": make_golden_toy_english_net(),
}
