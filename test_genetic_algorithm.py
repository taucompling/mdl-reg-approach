import dataclasses
import random
import unittest

import numpy as np

import configuration
import corpora
import genetic_algorithm
import network

_TEST_CONFIG = configuration.SimulationConfig(
    simulation_id="test",
    num_islands=1,
    migration_ratio=0.1,
    migration_interval_seconds=20,
    migration_interval_generations=1000,
    num_generations=1000,
    population_size=20,
    elite_ratio=0.05,
    allowed_activations=(
        network.SIGMOID,
        network.LINEAR,
        network.RELU,
        network.SQUARE,
    ),
    start_smooth=False,
    allowed_unit_types=(network.SUMMATION_UNIT, network.MULTIPLICATION_UNIT),
    tournament_size=4,
    mutation_probab=0.9,
    crossover_probab=0.1,
    grammar_multiplier=1,
    data_given_grammar_multiplier=1,
    max_grammar_size=None,
    allow_test_overlap=True,
    compress_grammar_encoding=False,
    max_network_units=1024,
    softmax_outputs=False,
    truncate_large_values=False,
    bias_connections=True,
    recurrent_connections=True,
    seed=1,
    corpus_seed=100,
    generation_dump_interval=1,
    comment=None,
    golden_networks=[],
    num_golden_copies_in_initialization=1,
    log_to_cloud=False,
    migration_channel="MPI",
    mini_batch_size=None,
    no_improvement_time=0,
    parallelize=True,
    regularization_method="none",
    regularization_multiplier=1,
    resumed_from_simulation_id=None,
    try_net_early_stop=True,
    data_given_grammar_smoothing_epsilon=None,
    allow_architecture_changing_mutations=True,
)


def _num_non_masked_inputs(x):
    return np.sum(~np.all(corpora.is_masked(x), axis=-1))


def _make_random_population(size):
    population = []
    for _ in range(size):
        net = network.make_random_net(
            input_size=3,
            output_size=3,
            allowed_activations=_TEST_CONFIG.allowed_activations,
            start_smooth=False,
        )
        p = random.random()
        if p < 0.3:
            grammar_encoding_length = np.inf
        else:
            grammar_encoding_length = np.random.randint(100)
        data_encoding_length = np.random.randint(100)
        net = dataclasses.replace(
            net,
            fitness=network.Fitness(
                loss=grammar_encoding_length + data_encoding_length,
                grammar_encoding_length=grammar_encoding_length,
                data_encoding_length=data_encoding_length,
                regularization_term=0.0,
                accuracy=1.0,
            ),
        )
        population.append(net)
    return population


def _test_elite(population, elite):
    population_fitnesses = tuple(map(genetic_algorithm._GET_NET_LOSS, population))
    elite_fitnesses = tuple(map(genetic_algorithm._GET_NET_LOSS, elite))

    population_fitness_without_elite = frozenset(population_fitnesses) - frozenset(
        elite_fitnesses
    )

    worst_elite_fitness = max(elite_fitnesses)
    best_non_elite_fitness = min(population_fitness_without_elite)

    assert worst_elite_fitness <= best_non_elite_fitness


class TestGeneticAlgorithm(unittest.TestCase):
    def test_get_elite_idxs(self):
        population = _make_random_population(size=1000)
        best_idxs = genetic_algorithm._get_elite_idxs(population, elite_ratio=0.01)
        assert len(best_idxs) == 10, len(best_idxs)

        elite = [population[i] for i in best_idxs]
        _test_elite(population, elite)

    def test_get_elite(self):
        population = _make_random_population(size=1000)
        elite = genetic_algorithm._get_elite(elite_ratio=0.01, population=population)
        assert len(elite) == 10
        _test_elite(population, elite)

    def test_get_migration_target_island(self):
        island_num = 3
        migration_interval = 20
        total_islands = 16
        targets = []
        generator = genetic_algorithm._make_migration_target_island_generator(
            island_num, total_islands
        )

        for generation in range(
            1, (total_islands + 1) * migration_interval, migration_interval
        ):
            targets.append(next(generator))

        assert island_num not in targets
        assert tuple(targets) == (
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            0,
            1,
            2,
            4,
            5,
        )

    def test_tournament_selection_break_infinity_ties(self):
        population = _make_random_population(size=10000)
        inf_population = [x for x in population if np.isinf(x.fitness.loss)]
        for _ in range(100):
            tournament = genetic_algorithm._tournament_selection(
                population=inf_population, tournament_size=2
            )
            winner_idx = tournament.winner_idx
            loser_idx = tournament.loser_idx
            assert (
                inf_population[winner_idx].fitness.grammar_encoding_length
                <= inf_population[loser_idx].fitness.grammar_encoding_length
            )

    def test_tournament_selection(self):
        population = _make_random_population(size=1000)
        tournament = genetic_algorithm._tournament_selection(
            population=population, tournament_size=2
        )
        winner_idx = tournament.winner_idx
        loser_idx = tournament.loser_idx
        assert population[winner_idx].fitness.loss < population[loser_idx].fitness.loss

        tournament = genetic_algorithm._tournament_selection(
            population=population,
            tournament_size=len(population),
        )
        absolute_best_offspring_idx = tournament.winner_idx
        absolute_worst_offspring_idx = tournament.loser_idx

        assert (
            population[absolute_best_offspring_idx].fitness.loss
            == min(population, key=genetic_algorithm._GET_NET_LOSS).fitness.loss
        )

        assert (
            population[absolute_worst_offspring_idx].fitness.loss
            == max(population, key=genetic_algorithm._GET_NET_LOSS).fitness.loss
        )

        identical_fitness_population = [population[0], population[0]]
        tournament = genetic_algorithm._tournament_selection(
            identical_fitness_population, tournament_size=2
        )
        winner_idx = tournament.winner_idx
        loser_idx = tournament.loser_idx
        assert winner_idx != loser_idx
