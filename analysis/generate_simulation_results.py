import dataclasses
import itertools
import os
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

SMOOTHING_D_GIVEN_G_EPSILON = 1e-10

import configuration
import corpora
import manual_nets
import network
import simulations
import test_network
import utils

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
NETWORKS_PATH = BASE_DIR / "networks"


@dataclasses.dataclass
class SimulationResult:
    name: str
    task: str

    optimal_train_d_given_g: float
    optimal_test_d_given_g: float
    optimal_test_d_given_g_no_overlap: float

    grammar_encoding_length: float
    l1_reg_value: float
    l2_reg_value: float

    train_data_encoding_length: float
    is_inf_train_data_encoding_length: bool

    test_data_encoding_length: float
    is_inf_test_data_encoding_length: bool

    test_data_encoding_length_no_overlap: float
    is_inf_test_data_encoding_length_no_overlap: bool


def load_net(simulation_id: str) -> network.Network:
    net_files = []
    # First check for global best networks as produced by the result collector process in the simulation
    priority_files = [
        f"{simulation_id}__best_network.pickle",
        f"{simulation_id}__current_best.pickle",
    ]
    for filename in priority_files:
        net_files.extend(NETWORKS_PATH.glob(filename))

    # Then add any remaining matching files - since sometimes the simulation is interrupted (e.g. due to time limit)
    # and the result collector process is not run, so we need to check for any network files that were saved
    net_files.extend(
        f
        for f in NETWORKS_PATH.glob(f"{simulation_id}*.pickle")
        if f.name not in priority_files
    )
    if not net_files:
        raise ValueError(f"No network files found for simulation {simulation_id}")

    best_net = None
    best_fitness = float("inf")
    best_file = None

    for net_path in net_files:
        with open(net_path, "rb") as file:
            net = pickle.load(file)
            if net.fitness.loss < best_fitness:
                best_fitness = net.fitness.loss
                best_net = net
                best_file = net_path

    if best_net is None:
        raise ValueError(
            f"No networks with valid fitness found for simulation {simulation_id}"
        )

    logger.info(f"Best network found in {best_file.name} with fitness {best_fitness}")
    return best_net


def verify_network_fitness_consistency(
    original_net: network.Network,
    corpus: corpora.Corpus,
    config: configuration.SimulationConfig,
) -> None:
    if original_net.fitness is None:
        logger.warning(
            "Original network fitness is None, should not happen since it is a network from a simulation"
        )
        return

    invalidated_net = network.invalidate_fitness(original_net)
    # Replacing the config as in the original simulation
    config = dataclasses.replace(
        config,
        try_net_early_stop=True,
        data_given_grammar_smoothing_epsilon=None,
    )
    train_net = network.calculate_fitness(invalidated_net, corpus, config=config)

    if not np.isclose(
        train_net.fitness.grammar_encoding_length,
        original_net.fitness.grammar_encoding_length,
    ).all():
        logger.error(
            f"Network |G| mismatch: {train_net.fitness.grammar_encoding_length} vs "
            f"{original_net.fitness.grammar_encoding_length}"
        )

    if not np.isclose(
        train_net.fitness.data_encoding_length,
        original_net.fitness.data_encoding_length,
    ).all():
        logger.error(
            f"Network |D:G| mismatch: {train_net.fitness.data_encoding_length} vs "
            f"{original_net.fitness.data_encoding_length}"
        )


def check_if_inf_d_given_g(
    net: network.Network, corpus: corpora.Corpus, config: configuration.SimulationConfig
) -> bool:
    invalidated_net = network.invalidate_fitness(net)
    early_stop_no_epsilon_config = dataclasses.replace(
        config,
        try_net_early_stop=True,
        data_given_grammar_smoothing_epsilon=None,
    )
    early_stop_no_epsilon_net = network.calculate_fitness(
        invalidated_net, corpus, early_stop_no_epsilon_config
    )
    return early_stop_no_epsilon_net.fitness.data_encoding_length == network._INF


def calculate_regularization(
    reg_method: str,
    net: network.Network,
) -> float:
    reg = 0
    weight_values = [
        weight.numerator / weight.denominator
        for edge, weight in itertools.chain(
            net.forward_weights.items(), net.recurrent_weights.items()
        )
    ]
    if reg_method == "l1":
        reg = sum([abs(weight_value) for weight_value in weight_values])
    elif reg_method == "l2":
        reg = sum([weight_value**2 for weight_value in weight_values])

    return reg


def analyze_simulation(
    net: network.Network,
    train_corpus: corpora.Corpus,
    test_corpus: corpora.Corpus,
    simulation_name: str,
    task_name: str,
    config: configuration.SimulationConfig,
) -> SimulationResult:
    invalidated_net = network.invalidate_fitness(net)
    train_net = network.calculate_fitness(invalidated_net, train_corpus, config=config)
    is_inf_train_data_encoding_length = check_if_inf_d_given_g(
        train_net, train_corpus, config
    )
    if simulation_name != "Golden":
        verify_network_fitness_consistency(net, train_corpus, config)

    test_net = network.calculate_fitness(invalidated_net, test_corpus, config=config)
    is_inf_test_data_encoding_length = check_if_inf_d_given_g(
        test_net, test_corpus, config
    )

    no_overlap_config = dataclasses.replace(config, allow_test_overlap=False)
    test_net_no_overlap = network.calculate_fitness(
        invalidated_net, test_corpus, config=no_overlap_config
    )
    is_inf_test_data_encoding_length_no_overlap = check_if_inf_d_given_g(
        test_net_no_overlap, test_corpus, no_overlap_config
    )

    l1_reg_value = calculate_regularization("l1", train_net)
    l2_reg_value = calculate_regularization("l2", train_net)

    return SimulationResult(
        name=simulation_name,
        task=task_name,
        optimal_train_d_given_g=train_corpus.optimal_d_given_g,
        optimal_test_d_given_g=test_corpus.optimal_d_given_g,
        optimal_test_d_given_g_no_overlap=test_corpus.optimal_d_given_g_no_overlap,
        grammar_encoding_length=train_net.fitness.grammar_encoding_length,
        l1_reg_value=l1_reg_value,
        l2_reg_value=l2_reg_value,
        train_data_encoding_length=train_net.fitness.data_encoding_length,
        is_inf_train_data_encoding_length=is_inf_train_data_encoding_length,
        test_data_encoding_length=test_net.fitness.data_encoding_length,
        is_inf_test_data_encoding_length=is_inf_test_data_encoding_length,
        test_data_encoding_length_no_overlap=test_net_no_overlap.fitness.data_encoding_length,
        is_inf_test_data_encoding_length_no_overlap=is_inf_test_data_encoding_length_no_overlap,
    )


def get_task_simulations_from_csv(
    simulations_csv_path: Path,
) -> dict[str, dict[str, str]]:
    if not simulations_csv_path.exists():
        raise FileNotFoundError(
            f"Simulations CSV file {simulations_csv_path} does not exist."
        )

    df = pd.read_csv(simulations_csv_path)

    # Filter only finished or time limit simulations
    df = df[(df["state"] == "Finished") | (df["state"] == "Time Limit")]

    # Log how many simulations we're analyzing
    logger.info(
        f"Analyzing {len(df)} finished simulations out of {len(pd.read_csv(simulations_csv_path))} total"
    )

    task_to_simulations = {}
    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        task_to_simulations[task] = dict(
            zip(task_df["objective"], task_df["simulation_id"])
        )

    return task_to_simulations


def run_analysis(
    task_name: str,
    simulation_ids_to_names: dict[str, str],
    golden_net: network.Network,
    corpus_generator: Callable[[], corpora.Corpus],
    config_override: dict = None,
) -> list[SimulationResult]:
    names_to_nets = {"Golden": golden_net}

    # Load networks
    for name, simulation_id in simulation_ids_to_names.items():
        net = load_net(simulation_id)
        names_to_nets[name] = net

    # Prepare config and corpus
    config = test_network._TEST_CONFIG
    # Make sure:
    # 1. Not to try early stop while feeding the networks
    # 2. Use small epsilon to smooth zero probabilities and allow D:G comparisons
    config = dataclasses.replace(
        config,
        try_net_early_stop=False,
        data_given_grammar_smoothing_epsilon=SMOOTHING_D_GIVEN_G_EPSILON,
    )

    # Apply config overrides if provided
    if config_override:
        config = dataclasses.replace(
            config,
            **config_override,
        )

    utils._set_seed_to_corpus_seed(config)

    corpus = corpus_generator()
    train_corpus = corpora.optimize_for_feeding(corpus)
    test_corpus = corpora.optimize_for_feeding(corpus.test_corpus)

    # Analyze each network
    simulation_results = []
    for name, net in names_to_nets.items():
        logger.info(f"Analyzing network {name}")

        simulation_result = analyze_simulation(
            net, train_corpus, test_corpus, name, task_name, config
        )
        simulation_results.append(simulation_result)

    return simulation_results


def analyze_task(
    task_name: str, simulations_dict: dict[str, str]
) -> list[SimulationResult]:
    """
    Analyze simulations for a specific task

    Args:
        task_name: Name of the task to analyze
        simulations_dict: Dictionary mapping objective names to simulation IDs

    Returns:
        List of simulation results
    """
    task_mapping = {
        "an_bn": (
            manual_nets.make_tacl_paper_an_bn_net,
            lambda: corpora.compute_train_test_overlap_mask(
                corpora.make_an_bn(batch_size=500, prior=0.3, sort_by_length=False),
                is_exhaustive_test_corpus=True,
            ),
            {
                "allowed_activations": simulations.DEFAULT_ACTIVATIONS
                + (
                    network.SIGMOID,
                    network.UNSIGNED_STEP,
                ),  # We use those in the golden network
            },
        ),
        "an_bn_cn": (
            manual_nets.make_tacl_paper_an_bn_cn_net,
            lambda: corpora.compute_train_test_overlap_mask(
                corpora.make_ain_bjn_ckn_dtn(
                    batch_size=500, prior=0.3, multipliers=(1, 1, 1, 0)
                ),
                is_exhaustive_test_corpus=True,
            ),
            {
                "allowed_activations": simulations.DEFAULT_ACTIVATIONS
                + (
                    network.SIGMOID,
                    network.UNSIGNED_STEP,
                ),  # We use those in the golden network
            },
        ),
        "dyck_1": (
            manual_nets.make_found_differentiable_dyck1_net,
            lambda: corpora.compute_train_test_overlap_mask(
                corpora.make_dyck_n(
                    n=1, batch_size=500, nesting_probab=0.33333, max_sequence_length=200
                ),
                is_exhaustive_test_corpus=True,
            ),
            {
                "allowed_activations": simulations.DEFAULT_ACTIVATIONS
                + (network.SIGMOID,),  # Used in the golden network
            },
        ),
        "dyck_2": (
            lambda: manual_nets.make_tacl_paper_dyck_2_net(nesting_probab=0.33333),
            lambda: corpora.compute_train_test_overlap_mask(
                corpora.make_dyck_n(
                    n=2, batch_size=500, nesting_probab=0.33333, max_sequence_length=200
                ),
                is_exhaustive_test_corpus=True,
            ),
            {
                "allowed_activations": simulations.DEFAULT_ACTIVATIONS
                + (
                    network.FLOOR,
                    network.MODULO_3,
                    network.UNSIGNED_STEP,
                ),  # We use those in the golden network
                "allowed_unit_types": simulations.EXTENDED_UNIT_TYPES,  # We use multiplication in the golden network
            },
        ),
        "arithmetic": (
            manual_nets.make_golden_arithmetic_net,
            lambda: corpora.compute_train_test_overlap_mask(
                corpora.make_golden_arithmetic_syntax_pcfg(
                    batch_size=500, sequence_length=None
                ),
                is_exhaustive_test_corpus=True,
            ),
            {
                "allowed_activations": simulations.DEFAULT_ACTIVATIONS
                + (
                    network.FLOOR,
                    network.MODULO_4,
                    network.ABS,
                    network.UNSIGNED_STEP,
                ),  # We use those in the golden network
                "allowed_unit_types": simulations.EXTENDED_UNIT_TYPES,  # We use multiplication in the golden network
                "truncate_large_values": False,  # Because the stack very quickly reaches high values
            },
        ),
        "toy_english": (
            manual_nets.make_golden_toy_english_net,
            lambda: corpora.compute_train_test_overlap_mask(
                corpora.make_golden_toy_english_pcfg(
                    batch_size=500, sequence_length=None
                ),
                is_exhaustive_test_corpus=True,
            ),
            {
                "allowed_activations": simulations.DEFAULT_ACTIVATIONS
                + (network.UNSIGNED_STEP,),  # We use those in the golden network
                "allowed_unit_types": simulations.EXTENDED_UNIT_TYPES,  # We use multiplication in the golden network
                "truncate_large_values": False,  # Because the stack very quickly reaches high values
            },
        ),
    }

    if task_name not in task_mapping:
        logger.error(f"Unknown task: {task_name}")
        return []

    golden_net_func, corpus_generator, config_override = task_mapping[task_name]
    golden_net = golden_net_func()

    return run_analysis(
        task_name, simulations_dict, golden_net, corpus_generator, config_override
    )


def create_combined_dataframe(all_results: list[SimulationResult]) -> pd.DataFrame:
    data = []
    for result in all_results:
        data.append(
            {
                "task": result.task,
                "objective": result.name,
                "optimal_train_d_given_g": result.optimal_train_d_given_g,
                "optimal_test_d_given_g": result.optimal_test_d_given_g,
                "optimal_test_d_given_g_no_overlap": result.optimal_test_d_given_g_no_overlap,
                "g": result.grammar_encoding_length,
                "l1_reg": result.l1_reg_value,
                "l2_reg": result.l2_reg_value,
                "train_d_given_g": result.train_data_encoding_length,
                "test_d_given_g": result.test_data_encoding_length,
                "test_d_given_g_no_overlap": result.test_data_encoding_length_no_overlap,
                "is_inf_train_d_given_g": result.is_inf_train_data_encoding_length,
                "is_inf_test_d_given_g": result.is_inf_test_data_encoding_length,
                "is_inf_test_d_given_g_no_overlap": result.is_inf_test_data_encoding_length_no_overlap,
            }
        )
    return pd.DataFrame(data)


def main():
    analysis_configurations = [
        {
            "input_csv_name": "new_obj_function_simulations.csv",
            "output_dir_name": "obj_function_analysis_results",
            "output_csv_name": "new_obj_function_simulation_results.csv",
        },
        {
            "input_csv_name": "weights_only_simulations.csv",
            "output_dir_name": "obj_function_analysis_results",
            "output_csv_name": "weights_only_simulation_results.csv",
        },
    ]

    for config in analysis_configurations:
        input_csv_path = BASE_DIR / config["input_csv_name"]
        output_dir_path = BASE_DIR / config["output_dir_name"]
        output_result_file_path = output_dir_path / config["output_csv_name"]

        logger.info(
            f"Processing {config['input_csv_name']} -> {output_result_file_path}"
        )

        if not input_csv_path.exists():
            logger.warning(
                f"Input CSV {input_csv_path} not found. Skipping this configuration."
            )
            continue

        task_to_simulations = get_task_simulations_from_csv(input_csv_path)
        all_results = []

        os.makedirs(output_dir_path, exist_ok=True)
        if not output_result_file_path.exists():
            for task_name, simulations_dict in task_to_simulations.items():
                logger.info(f"Analyzing task: {task_name}")
                results = analyze_task(task_name, simulations_dict)
                all_results.extend(results)

            df = create_combined_dataframe(all_results)
            df.to_csv(output_result_file_path, index=False)
            logger.info(f"Results saved to {output_result_file_path}")
        else:
            logger.info(f"Results already exist at {output_result_file_path}")


if __name__ == "__main__":
    main()
