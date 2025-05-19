import corpora
import network

_DIFFERENTIABLE_ACTIVATIONS = (
    network.LINEAR,
    network.RELU,
    network.SIGMOID,
    network.TANH,
)

DEFAULT_ACTIVATIONS = (
    network.LINEAR,
    network.RELU,
    network.TANH,
)

EXTENDED_ACTIVATIONS = DEFAULT_ACTIVATIONS + (
    network.MODULO_2,
    network.MODULO_3,
    network.SIGMOID,
    network.SQUARE,
    network.FLOOR,
    network.UNSIGNED_STEP,
    network.ABS,
    network.MODULO_4,
)

DEFAULT_UNIT_TYPES = (network.SUMMATION_UNIT,)
EXTENDED_UNIT_TYPES = (network.SUMMATION_UNIT, network.MULTIPLICATION_UNIT)

DEFAULT_CONFIG = {
    "migration_ratio": 0.01,
    "migration_interval_seconds": 1800,
    "migration_interval_generations": 1000,
    "num_generations": 25_000,
    "population_size": 500,
    "elite_ratio": 0.001,
    "allowed_activations": DEFAULT_ACTIVATIONS,
    "allowed_unit_types": DEFAULT_UNIT_TYPES,
    "start_smooth": False,
    "compress_grammar_encoding": False,
    "tournament_size": 2,
    "mutation_probab": 1.0,
    "crossover_probab": 0.0,
    "mini_batch_size": None,
    "grammar_multiplier": 1,
    "data_given_grammar_multiplier": 1,
    "max_grammar_size": None,
    "allow_test_overlap": True,
    "max_network_units": 1024,
    "softmax_outputs": False,
    "truncate_large_values": True,
    "bias_connections": True,
    "recurrent_connections": True,
    "seed": 100,
    "corpus_seed": 100,
    "parallelize": True,
    "resumed_from_simulation_id": None,
    "comment": None,
    "log_to_cloud": False,
    "migration_channel": "file",
    "generation_dump_interval": 250,
    "regularization_method": "none",
    "regularization_multiplier": 1.0,
    "golden_networks": [],
    "num_golden_copies_in_initialization": 1,
    "no_improvement_time": 120,
    "try_net_early_stop": True,
    "data_given_grammar_smoothing_epsilon": None,
    "allow_architecture_changing_mutations": True,
}


SIMULATIONS = {
    "identity": {
        "corpus": {
            "factory": corpora.make_identity_binary,
            "args": {"sequence_length": 100, "batch_size": 10},
        }
    },
    "repeat_last_char": {
        "corpus": {
            "factory": corpora.make_prev_char_repetition_binary,
            "args": {
                "sequence_length": 100,
                "batch_size": 10,
                "repetition_offset": 1,
            },
        }
    },
    "binary_addition": {
        "corpus": {
            "factory": corpora.make_binary_addition,
            "args": {"min_n": 0, "max_n": 20},
        },
    },
    "dyck_1": {
        "corpus": {
            "factory": corpora.make_dyck_n,
            "args": {
                "n": 1,
                "batch_size": 500,
                "nesting_probab": 0.33333,
                "max_sequence_length": 200,  # TODO: Why?
            },
        },
        "config": {
            "migration_channel": "mpi",
            "allowed_activations": DEFAULT_ACTIVATIONS
            + (network.SIGMOID,),  # Used in the golden network
        },
    },
    "dyck_1_train_weights_from_golden": {
        "corpus": {
            "factory": corpora.make_dyck_n,
            "args": {
                "n": 1,
                "batch_size": 500,
                "nesting_probab": 0.33333,
                "max_sequence_length": 200,  # prevent lengths from exploding
            },
        },
        "config": {
            "migration_channel": "mpi",
            "no_improvement_time": 0,
            "golden_networks": ["dyck_1"],
            "num_golden_copies_in_initialization": 500,
            "allowed_activations": DEFAULT_ACTIVATIONS
            + (network.SIGMOID,),  # Used in the golden network
            "allow_architecture_changing_mutations": False,
        },
    },
    "differentiable_dyck_1": {
        "corpus": {
            "factory": corpora.make_dyck_n,
            "args": {
                "n": 1,
                "batch_size": 500,
                "nesting_probab": 0.33333,
                "max_sequence_length": 200,
            },
        },
        "config": {
            "allowed_activations": _DIFFERENTIABLE_ACTIVATIONS,
            "migration_channel": "mpi",
        },
    },
    "dyck_1_diff_softmax": {
        "corpus": {
            "factory": corpora.make_dyck_n,
            "args": {
                "n": 1,
                "batch_size": 500,
                "nesting_probab": 0.33333,
                "max_sequence_length": 200,
            },
        },
        "config": {
            "allowed_activations": _DIFFERENTIABLE_ACTIVATIONS,
            "migration_channel": "mpi",
            "softmax_outputs": True,
        },
    },
    "dyck_2": {
        "corpus": {
            "factory": corpora.make_dyck_n,
            "args": {
                "batch_size": 500,
                "nesting_probab": 0.33333,
                "n": 2,
                "max_sequence_length": 200,
            },
        },
        "config": {
            "allowed_activations": DEFAULT_ACTIVATIONS
            + (
                network.FLOOR,
                network.MODULO_3,
                network.UNSIGNED_STEP,
            ),  # We use those in the golden network
            "allowed_unit_types": EXTENDED_UNIT_TYPES,  # We use multiplication in the golden network
            "migration_channel": "mpi",
        },
    },
    "dyck_2_train_weights_from_golden": {
        "corpus": {
            "factory": corpora.make_dyck_n,
            "args": {
                "batch_size": 500,
                "nesting_probab": 0.33333,
                "n": 2,
                "max_sequence_length": 200,
            },
        },
        "config": {
            "migration_channel": "mpi",
            "no_improvement_time": 0,
            "golden_networks": ["dyck_2"],
            "num_golden_copies_in_initialization": 500,
            "allowed_activations": DEFAULT_ACTIVATIONS
            + (
                network.FLOOR,
                network.MODULO_3,
                network.UNSIGNED_STEP,
            ),  # We use those in the golden network
            "allowed_unit_types": EXTENDED_UNIT_TYPES,  # We use multiplication in the golden network
            "allow_architecture_changing_mutations": False,
        },
    },
    "differentiable_dyck_2": {
        "corpus": {
            "factory": corpora.make_dyck_n,
            "args": {
                "batch_size": 500,
                "nesting_probab": 0.33333,
                "n": 2,
                "max_sequence_length": 200,  # TODO: Why?
            },
        },
        "config": {
            "allowed_activations": _DIFFERENTIABLE_ACTIVATIONS,
            "migration_channel": "mpi",
        },
    },
    "dyck_1_range": {
        "corpus": {
            "factory": corpora.make_dyck_n,
            "args": {
                "n": 1,
                "batch_size": 100,
                "exhaustive": True,
                "nesting_probab": 0.5,
                "max_sequence_length": 200,
            },
        },
    },
    "dyck_2_range": {
        "corpus": {
            "factory": corpora.make_dyck_n,
            "args": {
                "n": 2,
                "exhaustive": True,
                "batch_size": 100,
                "nesting_probab": 0.5,
                "max_sequence_length": 200,
            },
        },
    },
    "an_bn_range": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn_range,
            "args": {
                "training_n_start": 0,
                "training_n_stop": 100,
                "test_n_start": 101,
                "test_n_stop": 1000,
                "sort_by_length": False,
                "multipliers": (1, 1, 0, 0),
            },
        }
    },
    "an_bn_cn_range": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn_range,
            "args": {
                "training_n_start": 0,
                "training_n_stop": 100,
                "test_n_start": 101,
                "test_n_stop": 1000,
                "sort_by_length": False,
                "multipliers": (1, 1, 1, 0),
            },
        }
    },
    "an_bn_cn_dn_range": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn_range,
            "args": {
                "training_n_start": 0,
                "training_n_stop": 50,
                "test_n_start": 51,
                "test_n_stop": 1000,
                "sort_by_length": False,
                "multipliers": (1, 1, 1, 1),
            },
        }
    },
    "an_bn": {
        "corpus": {
            "factory": corpora.make_an_bn,
            "args": {"batch_size": 500, "prior": 0.3, "sort_by_length": False},
        },
        "config": {
            "migration_channel": "mpi",
            "allowed_activations": DEFAULT_ACTIVATIONS
            + (
                network.SIGMOID,
                network.UNSIGNED_STEP,
            ),  # We use those in the golden network
        },
    },
    "an_bn_train_weights_from_golden": {
        "corpus": {
            "factory": corpora.make_an_bn,
            "args": {"batch_size": 500, "prior": 0.3, "sort_by_length": False},
        },
        "config": {
            "migration_channel": "mpi",
            "no_improvement_time": 0,
            "golden_networks": ["an_bn"],
            "num_golden_copies_in_initialization": 500,
            "allowed_activations": DEFAULT_ACTIVATIONS
            + (
                network.SIGMOID,
                network.UNSIGNED_STEP,
            ),  # We use those in the golden network
            "allow_architecture_changing_mutations": False,
        },
    },
    "an_bn_diff_softmax": {
        "corpus": {
            "factory": corpora.make_an_bn,
            "args": {"batch_size": 500, "prior": 0.3, "sort_by_length": False},
        },
        "config": {
            "migration_channel": "mpi",
            "allowed_activations": _DIFFERENTIABLE_ACTIVATIONS,
            "softmax_outputs": True,
        },
    },
    "an_bn_cn": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn,
            "args": {"batch_size": 500, "prior": 0.3, "multipliers": (1, 1, 1, 0)},
        },
        "config": {
            "migration_channel": "mpi",
            "allowed_activations": DEFAULT_ACTIVATIONS
            + (
                network.SIGMOID,
                network.UNSIGNED_STEP,
            ),  # We use those in the golden network
        },
    },
    "an_bn_cn_train_weights_from_golden": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn,
            "args": {"batch_size": 500, "prior": 0.3, "multipliers": (1, 1, 1, 0)},
        },
        "config": {
            "migration_channel": "mpi",
            "no_improvement_time": 0,
            "golden_networks": ["an_bn_cn"],
            "num_golden_copies_in_initialization": 500,
            "allowed_activations": DEFAULT_ACTIVATIONS
            + (
                network.SIGMOID,
                network.UNSIGNED_STEP,
            ),  # We use those in the golden network
            "allow_architecture_changing_mutations": False,
        },
    },
    "differentiable_an_bn_cn": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn,
            "args": {"batch_size": 500, "prior": 0.3, "multipliers": (1, 1, 1, 0)},
        },
        "config": {
            "migration_channel": "mpi",
            "allowed_activations": _DIFFERENTIABLE_ACTIVATIONS,
        },
    },
    "an_bn_cn_diff_softmax": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn,
            "args": {"batch_size": 500, "prior": 0.3, "multipliers": (1, 1, 1, 0)},
        },
        "config": {
            "migration_channel": "mpi",
            "allowed_activations": _DIFFERENTIABLE_ACTIVATIONS,
            "softmax_outputs": True,
        },
    },
    "an_bn_cn_dn": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn,
            "args": {
                "batch_size": 500,
                "prior": 0.3,
                "multipliers": (1, 1, 1, 1),
            },
        },
    },
    "an_b2n": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn,
            "args": {
                "batch_size": 500,
                "prior": 0.3,
                "multipliers": (1, 2, 0, 0),
            },
        }
    },
    "an_bn_square": {
        "corpus": {
            "factory": corpora.make_an_bn_square,
            "args": {"batch_size": 1000, "prior": 0.5},
        }
    },
    "an_bm_cn_dm": {
        "corpus": {
            "factory": corpora.make_an_bm_cn_dm,
            "args": {
                "batch_size": 500,
                "prior": 0.3,
                "limit_vocabulary": False,
            },
        },
    },
    "an_bm_an_bm": {
        "corpus": {
            "factory": corpora.make_an_bm_cn_dm,
            "args": {
                "batch_size": 500,
                "prior": 0.3,
                "limit_vocabulary": True,
            },
        },
    },
    "palindrome_fixed_length": {
        "corpus": {
            "factory": corpora.make_binary_palindrome_fixed_length,
            "args": {
                "batch_size": 1000,
                "sequence_length": 50,
                "train_set_ratio": 0.7,
            },
        }
    },
    "mnist": {
        "corpus": {
            "factory": corpora.make_mnist_corpus,
            "args": {"samples_limit": 1000, "width_height": 16},
        },
        "config": {
            "softmax_outputs": True,
            "recurrent_connections": False,
            "data_given_grammar_multiplier": 50,
        },
    },
    "an_bm_cn_plus_m": {
        "corpus": {
            "factory": corpora.make_an_bm_cn_plus_m_from_cfg,
            "args": {
                "batch_size": 500,
                "prior": 0.3,
            },
        }
    },
    "an_bm_cn_plus_m_range": {
        "corpus": {
            "factory": corpora.make_an_bm_cn_plus_m_from_range,
            "args": {
                "start_n": 1,
                "num_strings": 100,
                "sort_by_length": False,
            },
        }
    },
    "center_embedding": {
        "corpus": {
            "factory": corpora.make_center_embedding,
            "args": {
                "batch_size": 20_000,
                "embedding_depth_probab": 0.3,
                "dependency_distance_probab": 0.0,
            },
        },
        "config": {
            "allowed_activations": DEFAULT_ACTIVATIONS + (network.MODULO_2,),
            "allowed_unit_types": (network.SUMMATION_UNIT, network.MULTIPLICATION_UNIT),
        },
    },
    "arithmetic": {
        "config": {
            "allowed_activations": DEFAULT_ACTIVATIONS
            + (
                network.FLOOR,
                network.MODULO_4,
                network.ABS,
                network.UNSIGNED_STEP,
            ),  # We use those in the golden network
            "allowed_unit_types": EXTENDED_UNIT_TYPES,  # We use multiplication in the golden network
            "migration_channel": "mpi",
            "truncate_large_values": False,  # Because the stack very quickly reaches high values
        },
        "corpus": {
            "factory": corpora.make_golden_arithmetic_syntax_pcfg,
            "args": {"sequence_length": None, "batch_size": 500},
        },
    },
    "arithmetic_train_weights_from_golden": {
        "config": {
            "migration_channel": "mpi",
            "no_improvement_time": 0,
            "golden_networks": ["arithmetic"],
            "num_golden_copies_in_initialization": 500,
            "allowed_activations": DEFAULT_ACTIVATIONS
            + (
                network.FLOOR,
                network.MODULO_4,
                network.ABS,
                network.UNSIGNED_STEP,
            ),  # We use those in the golden network
            "allowed_unit_types": EXTENDED_UNIT_TYPES,  # We use multiplication in the golden network
            "truncate_large_values": False,  # Because the stack very quickly reaches high values
            "allow_architecture_changing_mutations": False,
        },
        "corpus": {
            "factory": corpora.make_golden_arithmetic_syntax_pcfg,
            "args": {"sequence_length": None, "batch_size": 500},
        },
    },
    "toy_english": {
        "config": {
            "allowed_activations": DEFAULT_ACTIVATIONS + (network.UNSIGNED_STEP,),
            "allowed_unit_types": EXTENDED_UNIT_TYPES,  # We use multiplication in the golden network
            "migration_channel": "mpi",
            "truncate_large_values": False,  # Because the stack very quickly reaches high values
        },
        "corpus": {
            "factory": corpora.make_golden_toy_english_pcfg,
            "args": {"sequence_length": None, "batch_size": 500},
        },
    },
    "toy_english_train_weights_from_golden": {
        "config": {
            "migration_channel": "mpi",
            "no_improvement_time": 0,
            "golden_networks": ["toy_english"],
            "num_golden_copies_in_initialization": 500,
            "allowed_activations": DEFAULT_ACTIVATIONS + (network.UNSIGNED_STEP,),
            "allowed_unit_types": EXTENDED_UNIT_TYPES,  # We use multiplication in the golden network
            "allow_architecture_changing_mutations": False,
            "truncate_large_values": False,  # Because the stack very quickly reaches high values
        },
        "corpus": {
            "factory": corpora.make_golden_toy_english_pcfg,
            "args": {"sequence_length": None, "batch_size": 500},
        },
    },
    "0_1_pattern_binary": {
        "corpus": {
            "factory": corpora.make_0_1_pattern_binary,
            "args": {"sequence_length": 20, "batch_size": 1},
        }
    },
    "0_1_pattern_one_hot_no_eos": {
        "corpus": {
            "factory": corpora.make_0_1_pattern_one_hot,
            "args": {
                "add_end_of_sequence": False,
                "sequence_length": 50,
                "batch_size": 1,
            },
        }
    },
    "0_1_pattern_one_hot_with_eos": {
        "corpus": {
            "factory": corpora.make_0_1_pattern_one_hot,
            "args": {
                "add_end_of_sequence": True,
                "sequence_length": 50,
                "batch_size": 1,
            },
        }
    },
}
