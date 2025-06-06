import dataclasses
import datetime
import itertools
import math
import multiprocessing
import operator
import os
import pathlib
import pickle
import queue
import random
import shutil
import threading
import uuid
from typing import Iterator, Optional

import cachetools
import numpy as np
from loguru import logger

import configuration
import corpora
import manual_nets
import network
import utils

_DEBUG_MODE = False

_ISLAND_MIGRATIONS_PATH = pathlib.Path("/tmp/mdlnn/island_migrations/")

_GET_NET_LOSS = operator.attrgetter("fitness.loss")

_Population = list[network.Network]

_NETWORKS_CACHE = cachetools.LRUCache(maxsize=1_000_000)

try:
    from mpi4py import MPI

    _MPI_COMMUNICATOR = MPI.COMM_WORLD
    _MPI_RANK = _MPI_COMMUNICATOR.Get_rank()

except ModuleNotFoundError:
    _MPI_COMMUNICATOR = None
    _MPI_RANK = None

_MPI_MIGRANTS_BUFFER_SIZE = 10_000_000


@dataclasses.dataclass(frozen=True)
class _Tournament:
    winner_idx: int
    loser_idx: int


def _s3_upload_worker(simulation_id: str, cloud_upload_queue: queue.Queue):
    from botocore import exceptions as boto_exceptions

    while True:
        obj, key = cloud_upload_queue.get()
        try:
            utils.upload_object_to_simulation_s3(simulation_id, obj, key)
            logger.info(f"Uploaded {key} to storage.")
        except boto_exceptions.HTTPClientError:
            logger.exception(f"Error while uploading file '{key}' to storage")


def _start_s3_upload_worker(cloud_upload_queue: queue.Queue, simulation_id: str):
    t = threading.Thread(
        target=_s3_upload_worker, args=(simulation_id, cloud_upload_queue)
    )
    t.daemon = True
    t.start()


@cachetools.cached(_NETWORKS_CACHE, key=lambda net, corpus, config: hash(net))
def _evaluate_net_cached(
    net: network.Network,
    corpus: corpora.Corpus,
    config: configuration.SimulationConfig,
) -> network.Fitness:
    return network.calculate_fitness(net, corpus, config).fitness


def get_migration_path(simulation_id) -> pathlib.Path:
    return _ISLAND_MIGRATIONS_PATH.joinpath(simulation_id)


def _make_migration_target_island_generator(
    island_num: int,
    total_islands: int,
):
    yield from itertools.cycle(
        itertools.chain(range(island_num + 1, total_islands), range(island_num))
    )


def _get_migrants_from_cloud(simulation_id, island_num: int) -> Optional[_Population]:
    from botocore import exceptions as boto_exceptions

    migrants_filename = get_target_island_filename(island_num)
    try:
        migrants = utils.load_object_from_simulation_s3(
            simulation_id, migrants_filename
        )
        utils.delete_simulation_s3_object(simulation_id, migrants_filename)
        return migrants
    except boto_exceptions.ClientError:
        return None


def get_migrants_through_mpi() -> Optional[_Population]:
    migrant_batches = []

    while True:
        has_awaiting_migrants = _MPI_COMMUNICATOR.iprobe(tag=utils.MPI_MIGRANTS_TAG)
        if not has_awaiting_migrants:
            break
        migrants = _MPI_COMMUNICATOR.recv(bytearray(_MPI_MIGRANTS_BUFFER_SIZE))
        migrant_batches.append(migrants)

    if not migrant_batches:
        return None

    return min(migrant_batches, key=_mean_population_fitness)


def _get_migrants_from_file(
    simulation_id: str, island_num: int, file_lock: multiprocessing.Lock
) -> Optional[_Population]:
    migrants_filename = get_target_island_filename(island_num)
    incoming_migrants_path = get_migration_path(simulation_id).joinpath(
        migrants_filename
    )

    lock_start = datetime.datetime.now()

    with file_lock:
        if not incoming_migrants_path.exists():
            return None

        with incoming_migrants_path.open("rb") as f:
            incoming_migrants = pickle.load(f)
        incoming_migrants_path.unlink()

    lock_end = datetime.datetime.now()
    lock_delta = lock_end - lock_start
    logger.info(
        f"Incoming lock took {lock_delta.seconds}.{str(lock_delta.microseconds)[:2]} seconds"
    )
    return incoming_migrants


def get_target_island_filename(target_island: int) -> str:
    return f"island_{target_island}_incoming_migrants"


def _make_random_population(
    config, input_size, output_size, golden_nets
) -> _Population:
    nets_to_generate = config.population_size - (
        len(golden_nets) * config.num_golden_copies_in_initialization
    )
    return [
        network.make_random_net(
            input_size=input_size,
            output_size=output_size,
            allowed_activations=config.allowed_activations,
            start_smooth=config.start_smooth,
        )
        for _ in range(nets_to_generate)
    ] + golden_nets * config.num_golden_copies_in_initialization


def _mean_population_fitness(population: _Population) -> float:
    return np.mean([x.fitness.loss for x in population]).item()


def _should_migrate(
    outgoing_migrants: _Population, awaiting_migrants_at_target: _Population
) -> bool:
    mean_awaiting_fitness = _mean_population_fitness(awaiting_migrants_at_target)
    mean_outgoing_fitness = _mean_population_fitness(outgoing_migrants)
    logger.info(f"Awaiting mean fitness: {mean_awaiting_fitness:.2f}")
    logger.info(f"Outgoing mean fitness: {mean_outgoing_fitness:.2f}")
    return mean_outgoing_fitness < mean_awaiting_fitness


def _send_migrants_through_cloud(
    migrants: _Population,
    target_island: int,
    simulation_id: str,
    cloud_upload_queue: queue.Queue,
) -> bool:
    from botocore import exceptions as boto_exceptions

    target_island_filename = get_target_island_filename(target_island)
    try:
        awaiting_migrants_at_target: _Population = utils.load_object_from_simulation_s3(
            simulation_id, target_island_filename
        )
        should_migrate = _should_migrate(
            outgoing_migrants=migrants,
            awaiting_migrants_at_target=awaiting_migrants_at_target,
        )
    except boto_exceptions.ClientError:
        # Key doesn't exist.
        should_migrate = True

    if should_migrate:
        cloud_upload_queue.put((migrants, target_island_filename))

    return should_migrate


def _send_migrants_through_mpi(migrants: _Population, target_island: int) -> bool:
    # We can't use _should_migrate() here because in MPI we can't override the target's buffer, so filtering the best migrants is done at the receiving side.
    _MPI_COMMUNICATOR.send(migrants, dest=target_island, tag=utils.MPI_MIGRANTS_TAG)
    return True


def _send_migrants_through_file(
    migrants: _Population,
    target_island: int,
    file_lock: multiprocessing.Lock,
    simulation_id: str,
) -> bool:
    lock_start = datetime.datetime.now()
    target_island_filename = get_target_island_filename(target_island)
    with file_lock:
        outgoing_path = get_migration_path(simulation_id).joinpath(
            target_island_filename
        )

        if outgoing_path.exists():
            with outgoing_path.open("rb") as f:
                awaiting_migrants_at_target: _Population = pickle.load(f)
            should_migrate = _should_migrate(
                outgoing_migrants=migrants,
                awaiting_migrants_at_target=awaiting_migrants_at_target,
            )
        else:
            should_migrate = True

        if should_migrate:
            with outgoing_path.open("wb") as f:
                pickle.dump(migrants, f)

    lock_end = datetime.datetime.now()
    lock_delta = lock_end - lock_start
    logger.info(
        f"Outgoing file lock took {lock_delta.seconds}.{str(lock_delta.microseconds)[:2]} seconds"
    )

    return should_migrate


def _simulation_exists(simulation_id: str, use_cloud: bool) -> bool:
    if use_cloud:
        key = f"{simulation_id}/{simulation_id}__{get_target_island_filename(0)}.pickle"
        return utils.s3_key_exists(key)
    else:
        return get_migration_path(simulation_id).exists()


def remove_simulation_directory(simulation_id: str, use_cloud: bool):
    if use_cloud:
        utils.get_s3_simulations_bucket().objects.filter(
            Prefix=f"{simulation_id}/"
        ).delete()
    else:
        shutil.rmtree(get_migration_path(simulation_id), ignore_errors=True)
    logger.info(
        f"Removed directory {simulation_id} from {'S3' if use_cloud else 'local storage'}."
    )


def verify_existing_simulation_override(simulation_id: str, use_cloud: bool):
    if _simulation_exists(simulation_id, use_cloud):
        logger.info(
            f"{'S3' if use_cloud else 'local'} directory for simulation {simulation_id} already exists. Re-run using `--override` flag to delete the previous run.\n"
        )
        exit()


def _select_best_no_repetition(population: _Population, k: int) -> _Population:
    population_fitness = set()
    individuals_no_repetition = []
    for net in population:
        fitness = net.fitness.loss
        if fitness not in population_fitness:
            individuals_no_repetition.append(net)
            population_fitness.add(fitness)
    return sorted(individuals_no_repetition, key=_GET_NET_LOSS)[:k]


def _get_elite_idxs(population: _Population, elite_ratio: float) -> frozenset[int]:
    elite_size = math.ceil(len(population) * elite_ratio)
    fitness = [x.fitness.loss for x in population]
    argsort = np.argsort(fitness)
    seen = set()
    best_idxs = set()
    for i in argsort:
        if len(best_idxs) == elite_size:
            break
        net = population[i]
        if net.fitness.loss in seen or np.isinf(net.fitness.loss):
            continue
        seen.add(net.fitness.loss)
        best_idxs.add(i)
    return frozenset(best_idxs)


def _get_elite(elite_ratio: float, population: _Population) -> _Population:
    elite_size = math.ceil(len(population) * elite_ratio)
    return _select_best_no_repetition(population, elite_size)


def _tournament_selection(population: _Population, tournament_size: int) -> _Tournament:
    tournament_idxs = random.sample(range(len(population)), tournament_size)
    tournament_nets = tuple(population[i] for i in tournament_idxs)
    nets_and_idxs = tuple(zip(tournament_nets, tournament_idxs))

    if len(set(x.fitness.loss for x in tournament_nets)) == 1:
        # MDL Tie.
        if np.isinf(tournament_nets[0].fitness.loss):
            # Break |D:G| infinity ties using |G|.
            argsort_by_g = tuple(
                np.argsort([x.fitness.grammar_encoding_length for x in tournament_nets])
            )
            return _Tournament(winner_idx=argsort_by_g[0], loser_idx=argsort_by_g[-1])
        return _Tournament(*tuple(random.sample(tournament_idxs, k=2)))

    sorted_tournament = sorted(nets_and_idxs, key=lambda x: x[0].fitness.loss)
    return _Tournament(
        winner_idx=sorted_tournament[0][1], loser_idx=sorted_tournament[-1][1]
    )


def _get_population_incoming_degrees(
    population: _Population, edge_type: int
) -> np.ndarray:
    degrees = []
    for net in population:
        (
            _,
            reverse_connections,
            _,
        ) = network.get_connections_and_weights_by_edge_type(net, edge_type)
        degrees += list(map(len, reverse_connections.values()))
    return np.array(degrees)


def _initialize_population_and_generation_from_existing_simulation(
    config, island_num
) -> tuple[_Population, int]:
    if config.migration_channel in {"mpi", "file"}:
        with open(
            f"./generations/{config.resumed_from_simulation_id}_latest_generation_island_{island_num}.pickle",
            "rb",
        ) as f:
            latest_generation_data = pickle.load(f)
    elif config.migration_channel == "cloud" or config.log_to_cloud:
        latest_generation_data = utils.load_object_from_simulation_s3(
            config.resumed_from_simulation_id,
            f"latest_generation_island_{island_num}",
        )
    else:
        raise ValueError(config.migration_channel)
    generation = latest_generation_data["generation"]
    population = latest_generation_data["population"]
    logger.info(f"Loaded population island {island_num} from generation {generation}")
    return population, generation


def _initialize_population_and_generation(
    config: configuration.SimulationConfig,
    golden_nets: list[network.Network],
    island_num: int,
    input_size: int,
    output_size: int,
) -> tuple[_Population, int]:
    if config.resumed_from_simulation_id is not None:
        return _initialize_population_and_generation_from_existing_simulation(
            config, island_num
        )

    generation = 1
    population = _make_random_population(
        config=config,
        input_size=input_size,
        output_size=output_size,
        golden_nets=golden_nets,
    )
    logger.debug(
        f"Initialized random population size {config.population_size}, with {len(golden_nets) * config.num_golden_copies_in_initialization} golden networks"
    )

    if _DEBUG_MODE:
        [
            network.visualize(
                population[i],
                f"random_initial_net_{i}__island_{island_num}",
            )
            for i in random.sample(range(len(population)), 10)
        ]

    return population, generation


def _evaluate_population(
    population: list[network.Network],
    corpus: corpora.Corpus,
    config: configuration.SimulationConfig,
) -> list[network.Network]:
    return [
        dataclasses.replace(
            net,
            fitness=_evaluate_net_cached(net=net, corpus=corpus, config=config),
        )
        for net in population
    ]


def _make_single_reproduction(
    population: _Population,
    elite_idxs: frozenset[int],
    corpus: corpora.Corpus,
    config: configuration.SimulationConfig,
) -> _Population:
    # Select parent(s) using tournament selection, create an offspring, replace tournament loser with offspring.
    p = random.random()
    if p < config.mutation_probab:
        tournament = _tournament_selection(population, config.tournament_size)
        parent_idx = tournament.winner_idx
        killed_idx = tournament.loser_idx
        offspring = network.mutate(population[parent_idx], config=config)
    elif p < config.mutation_probab + config.crossover_probab:
        tournament1 = _tournament_selection(population, config.tournament_size)
        tournament2 = _tournament_selection(population, config.tournament_size)
        offspring = network.crossover(
            population[tournament1.winner_idx],
            population[tournament2.winner_idx],
            max_units=config.max_network_units,
        )
        killed_idx = max(
            (tournament1.loser_idx, tournament2.loser_idx),
            key=lambda i: population[i].fitness.loss,
        )
    else:
        tournament = _tournament_selection(population, config.tournament_size)
        offspring = population[tournament.winner_idx]
        killed_idx = tournament.loser_idx

    offspring_fitness = _evaluate_net_cached(offspring, corpus, config)
    offspring = dataclasses.replace(offspring, fitness=offspring_fitness)

    if (
        killed_idx in elite_idxs
        and offspring.fitness.loss >= population[killed_idx].fitness.loss
    ):
        # Only kill a losing elite if the offspring is better.
        return population

    population[killed_idx] = offspring
    return population


def _make_generation(
    population: _Population,
    corpus: corpora.Corpus,
    config: configuration.SimulationConfig,
) -> _Population:
    # Calculate elite once per generation for performance.
    elite_idxs = _get_elite_idxs(population, config.elite_ratio)
    for _ in range(len(population)):
        population = _make_single_reproduction(
            population=population, elite_idxs=elite_idxs, corpus=corpus, config=config
        )
    return population


def _save_generation(
    generation: int,
    population: _Population,
    island_num: int,
    config: configuration.SimulationConfig,
    cloud_upload_queue: queue.Queue,
):
    data = {
        "generation": generation,
        "population": population,
        "island": island_num,
    }
    if config.migration_channel in {"mpi", "file"}:
        path = pathlib.Path(
            f"./generations/{config.simulation_id}_latest_generation_island_{island_num}.pickle"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    if config.migration_channel == "cloud" or config.log_to_cloud:
        cloud_upload_queue.put((data, f"latest_generation_island_{island_num}"))


def _golden_networks_survived(
    population_fitnesses: list[float],
    golden_nets: list[network.Network],
):
    for golden_net in golden_nets:
        if golden_net.fitness.loss not in population_fitnesses:
            return False
    return True


def _log_generation_to_logging_process(
    island_num: int,
    generation: int,
    best_net: network.Network,
    corpus: corpora.Corpus,
    config: configuration.SimulationConfig,
    logging_queue: multiprocessing.Queue,
):
    if config.migration_channel == "mpi":
        _MPI_COMMUNICATOR.Send(
            np.array(
                [
                    island_num,
                    generation,
                    best_net.fitness.loss,
                    best_net.fitness.grammar_encoding_length,
                    best_net.fitness.data_encoding_length,
                    network.get_num_units(best_net),
                    network.get_total_connections(best_net, include_biases=True),
                ]
            ),
            dest=config.num_islands,
            tag=utils.MPI_LOGGING_TAG,
        )
        return

    stats = {
        "island": island_num,
        "generation": generation,
        "loss": best_net.fitness.loss,
        "|g|": best_net.fitness.grammar_encoding_length,
        "|d:g|": best_net.fitness.data_encoding_length,
        "units": network.get_num_units(best_net),
        "connections": network.get_total_connections(best_net, include_biases=True),
        "accuracy": best_net.fitness.accuracy,
        "dot": utils.get_graphviz_url(
            network.get_dot_string(best_net, class_to_label=corpus.vocabulary)
        ),
    }

    logging_queue.put({"best_net": best_net, "stats": stats})


def _log_generation(
    population: _Population,
    golden_nets: list[network.Network],
    corpus: corpora.Corpus,
    config: configuration.SimulationConfig,
    generation: int,
    island_num: int,
    generation_time_delta: datetime.timedelta,
    logging_queue: multiprocessing.Queue,
    cloud_upload_queue: queue.Queue,
):
    all_fitness = [x.fitness.loss for x in population]
    valid_population = [
        population[i] for i in range(len(population)) if not np.isinf(all_fitness[i])
    ]

    if valid_population:
        best_net_idx = int(np.argmin(all_fitness))
    else:
        best_net_idx = 0
        valid_population = [population[0]]

    best_net = population[best_net_idx]
    best_fitness = all_fitness[best_net_idx]

    valid_fitnesses = [x.fitness.loss for x in valid_population]
    mean_fitness = np.mean(valid_fitnesses)
    fitness_std = np.std(valid_fitnesses)

    num_connections = [
        network.get_total_connections(x, include_biases=True) for x in population
    ]
    num_connections_mean = np.mean(num_connections)
    num_connections_std = np.std(num_connections)
    num_connections_max = np.max(num_connections)

    num_units = [network.get_num_units(x) for x in population]
    num_units_mean = np.mean(num_units)
    num_units_std = np.std(num_units)
    num_units_max = np.max(num_units)

    incoming_forward_degrees = _get_population_incoming_degrees(
        population=population, edge_type=network.FORWARD_CONNECTION
    )
    multiple_inputs_forward_degrees = incoming_forward_degrees[
        incoming_forward_degrees > 1
    ]
    incoming_recurrent_degrees = _get_population_incoming_degrees(
        population=population, edge_type=network.RECURRENT_CONNECTION
    )
    multiple_inputs_recurrent_degrees = incoming_recurrent_degrees[
        incoming_recurrent_degrees > 1
    ]

    g_s = [x.fitness.grammar_encoding_length for x in population]
    mean_g = np.mean(g_s)
    std_g = np.std(g_s)
    max_g = np.max(g_s)
    d_g_s = [x.fitness.data_encoding_length for x in valid_population]
    mean_d_g = np.mean(d_g_s)
    std_d_g = np.std(d_g_s)
    max_d_g = np.max(d_g_s)
    mean_accuracy = np.mean([x.fitness.accuracy for x in valid_population])

    all_weights = []
    for x in population:
        all_weights.append(network.get_forward_weights(x))
        all_weights.append(network.get_recurrent_weights(x))

    all_weights = np.concatenate(all_weights)
    mean_weight = np.mean(all_weights)
    max_weight = np.max(all_weights)

    num_invalid = len(
        list(
            filter(
                np.isinf,
                [x.fitness.loss for x in population],
            )
        )
    )
    invalid_ratio = num_invalid / len(population)

    unique_ratio = len(set(population)) / len(population)

    if len(golden_nets) > 0:
        golden_nets_survived = _golden_networks_survived(valid_fitnesses, golden_nets)
        best_is_golden = best_net.fitness.loss in [x.fitness.loss for x in golden_nets]
    else:
        golden_nets_survived = None
        best_is_golden = None

    logger.info(
        f"\nIsland {island_num} (pid {os.getpid()}) Generation {generation}"
        f"\n\tGeneration took {generation_time_delta.seconds}.{str(generation_time_delta.microseconds)[:2]} seconds"
        f"\n\tMean fitness: {mean_fitness:.2f} (±{fitness_std:.2f}, worst valid {np.max(valid_fitnesses):.2f}) \tBest fitness: {best_fitness:.2f}"
        f"\n\tMean num nodes: {num_units_mean:.2f} (±{num_units_std:.2f}, max {num_units_max}) \tMean num connections: {num_connections_mean:.2f} (±{num_connections_std:.2f}, max {num_connections_max}) \tMean G: {mean_g:.2f} (±{std_g:.2f}, max {max_g:.2f})\tMean D:G: {mean_d_g:.2f} (±{std_d_g:.2f}, max {max_d_g:.2f})"
        f"\n\tMean forward in degree: {np.mean(incoming_forward_degrees):.2f} (±{np.std(incoming_forward_degrees):.2f}, max {np.max(incoming_forward_degrees)}) \tMean recurrent in degree: {np.mean(incoming_recurrent_degrees):.2f} (±{np.std(incoming_recurrent_degrees):.2f}, max {np.max(incoming_recurrent_degrees) if incoming_recurrent_degrees.size else '-'})"
        f"\n\tMean forward in degree>1: {np.mean(multiple_inputs_forward_degrees):.2f} (±{np.std(multiple_inputs_forward_degrees):.2f}) \tMean recurrent in degree>1: {np.mean(multiple_inputs_recurrent_degrees):.2f} (±{np.std(multiple_inputs_recurrent_degrees):.2f})"
        f"\n\tMean weight: {mean_weight:.2f} (max {max_weight})\tMean accuracy: {mean_accuracy:.2f}\tInvalid: {invalid_ratio * 100:.1f}%\tUnique: {unique_ratio * 100:.1f}%"
        f"\n\tBest network:\n\t{network.to_string(best_net)}"
        f"\n\tGolden networks survived: {golden_nets_survived}"
        f"\n\tBest network is golden: {best_is_golden}\n\n"
    )

    if generation == 1 or generation % 100 == 0:
        network_filename = f"{config.simulation_id}__island_{island_num}__best_network"
        network.visualize(best_net, network_filename, class_to_label=corpus.vocabulary)
        network.save(best_net, network_filename)

        _log_generation_to_logging_process(
            island_num=island_num,
            generation=generation,
            best_net=best_net,
            corpus=corpus,
            config=config,
            logging_queue=logging_queue,
        )

    if generation == 1 or generation % config.generation_dump_interval == 0:
        _save_generation(
            generation=generation,
            population=population,
            island_num=island_num,
            config=config,
            cloud_upload_queue=cloud_upload_queue,
        )

    if _DEBUG_MODE and generation > 0 and generation % 5 == 0:
        [
            network.visualize(
                population[x],
                f"random_gen_{generation}__island_{island_num}__{str(uuid.uuid1())}",
            )
            for x in random.sample(range(len(population)), 5)
        ]


def _send_migrants(
    population: _Population,
    island_num: int,
    config: configuration.SimulationConfig,
    target_island: int,
    target_process_lock: multiprocessing.Lock,
    cloud_upload_queue: queue.Queue,
):
    num_migrants = math.floor(config.migration_ratio * config.population_size)
    migrants = list(
        set(
            population[
                _tournament_selection(
                    population, tournament_size=config.tournament_size
                ).winner_idx
            ]
            for _ in range(num_migrants)
        )
    )
    if config.migration_channel == "cloud":
        did_send = _send_migrants_through_cloud(
            migrants=migrants,
            target_island=target_island,
            simulation_id=config.simulation_id,
            cloud_upload_queue=cloud_upload_queue,
        )
    elif config.migration_channel == "file":
        did_send = _send_migrants_through_file(
            migrants=migrants,
            target_island=target_island,
            simulation_id=config.simulation_id,
            file_lock=target_process_lock,
        )
    elif config.migration_channel == "mpi":
        did_send = _send_migrants_through_mpi(
            migrants=migrants,
            target_island=target_island,
        )
    else:
        raise ValueError(config.migration_channel)

    if did_send:
        best_sent = min(migrants, key=_GET_NET_LOSS)
        logger.info(
            f"Island {island_num} sent {len(migrants)} migrants to island {target_island} through {config.migration_channel}. Best sent: {best_sent.fitness.loss:,.2f}"
        )


def _integrate_migrants(
    incoming_migrants: _Population,
    population: _Population,
    config: configuration.SimulationConfig,
    island_num: int,
) -> _Population:
    losing_idxs = tuple(
        _tournament_selection(population, config.tournament_size).loser_idx
        for _ in range(len(incoming_migrants))
    )
    prev_best_fitness = min([x.fitness.loss for x in population])
    for migrant_idx, local_idx in enumerate(losing_idxs):
        population[local_idx] = incoming_migrants[migrant_idx]
    new_best_fitness = min([x.fitness.loss for x in population])
    logger.info(
        f"Island {island_num} got {len(incoming_migrants)} incoming migrants. Previous best fitness: {prev_best_fitness:.2f}, new best: {new_best_fitness:.2f}"
    )
    return population


def _receive_and_integrate_migrants(
    population, config, island_num, process_lock
) -> _Population:
    if config.migration_channel == "cloud":
        incoming_migrants = _get_migrants_from_cloud(
            simulation_id=config.simulation_id,
            island_num=island_num,
        )
    elif config.migration_channel == "file":
        incoming_migrants = _get_migrants_from_file(
            simulation_id=config.simulation_id,
            island_num=island_num,
            file_lock=process_lock,
        )
    elif config.migration_channel == "mpi":
        incoming_migrants = get_migrants_through_mpi()
    else:
        raise ValueError(config.migration_channel)

    if incoming_migrants is None:
        logger.info(f"Island {island_num} has no incoming migrants waiting.")
        return population

    return _integrate_migrants(
        incoming_migrants=incoming_migrants,
        population=population,
        config=config,
        island_num=island_num,
    )


def _make_migration(
    population: _Population,
    island_num: int,
    config: configuration.SimulationConfig,
    migration_target_generator: Iterator[int],
    process_locks: tuple[multiprocessing.Lock, ...],
    cloud_upload_queue: queue.Queue,
) -> _Population:
    target_island = next(migration_target_generator)
    _send_migrants(
        population=population,
        island_num=island_num,
        config=config,
        target_island=target_island,
        target_process_lock=process_locks[target_island] if process_locks else None,
        cloud_upload_queue=cloud_upload_queue,
    )
    return _receive_and_integrate_migrants(
        population,
        config,
        island_num,
        process_locks[island_num] if process_locks else None,
    )


def run(
    island_num: int,
    corpus: corpora.Corpus,
    config: configuration.SimulationConfig,
    process_locks: tuple[multiprocessing.Lock, ...],
    logging_queue: multiprocessing.Queue,
):
    seed = config.seed + island_num
    utils.seed(seed)
    logger.info(f"Island {island_num}, seed {seed}")

    golden_nets = [manual_nets.GOLDEN_NETWORKS[task] for task in config.golden_networks]
    population, generation = _initialize_population_and_generation(
        config,
        golden_nets,
        island_num,
        input_size=corpus.input_sequence.shape[-1],
        output_size=corpus.target_sequence.shape[-1],
    )

    population = _evaluate_population(population, corpus, config)
    golden_nets = _evaluate_population(golden_nets, corpus, config)

    migration_target_generator = _make_migration_target_island_generator(
        island_num=island_num, total_islands=config.num_islands
    )

    cloud_upload_queue = queue.Queue()
    if config.log_to_cloud:
        _start_s3_upload_worker(cloud_upload_queue, config.simulation_id)

    stopwatch_start = datetime.datetime.now()
    while generation <= config.num_generations:
        generation_start_time = datetime.datetime.now()
        population = _make_generation(population, corpus, config)
        generation_time_delta = datetime.datetime.now() - generation_start_time

        time_delta = datetime.datetime.now() - stopwatch_start
        if config.num_islands > 1 and (
            time_delta.total_seconds() >= config.migration_interval_seconds
            or generation % config.migration_interval_generations == 0
        ):
            logger.info(
                f"Island {island_num} performing migration, time passed {time_delta.total_seconds()} seconds."
            )
            population = _make_migration(
                population=population,
                island_num=island_num,
                config=config,
                migration_target_generator=migration_target_generator,
                process_locks=process_locks,
                cloud_upload_queue=cloud_upload_queue,
            )
            stopwatch_start = datetime.datetime.now()

        _log_generation(
            population=population,
            golden_nets=golden_nets,
            corpus=corpus,
            config=config,
            generation=generation,
            island_num=island_num,
            generation_time_delta=generation_time_delta,
            logging_queue=logging_queue,
            cloud_upload_queue=cloud_upload_queue,
        )
        generation += 1

    population = _make_migration(
        population=population,
        island_num=island_num,
        config=config,
        migration_target_generator=migration_target_generator,
        process_locks=process_locks,
        cloud_upload_queue=cloud_upload_queue,
    )

    best_network = min(population, key=_GET_NET_LOSS)
    return best_network
