import dataclasses
import json
import sys

from loguru import logger

import configuration
import genetic_algorithm
import island
import simulations
import utils

logger.remove()
logger.add(sys.stdout, level="INFO")


def _make_corpus(factory, args, corpus_seed):
    utils.seed(corpus_seed)
    return factory(**args)


def run():
    arg_parser = utils.make_cli_arguments()
    arguments = arg_parser.parse_args()

    simulation = simulations.SIMULATIONS[arguments.simulation_name]

    corpus_args = simulation["corpus"]["args"]
    if arguments.corpus_args is not None:
        corpus_args.update(json.loads(arguments.corpus_args))

    override_config = {}
    if arguments.config is not None:
        override_config.update(json.loads(arguments.config))

    config = configuration.SimulationConfig(
        simulation_id=arguments.simulation_name,
        num_islands=arguments.total_islands,
        **{
            **simulations.DEFAULT_CONFIG,
            **simulation.get("config", {}),
            **override_config,
        },
    )

    corpus = _make_corpus(
        factory=simulation["corpus"]["factory"],
        args=corpus_args,
        corpus_seed=config.corpus_seed,
    )

    config = dataclasses.replace(
        config,
        comment=f"Corpus params: {json.dumps(simulation['corpus']['args'])}, input shape: {corpus.input_sequence.shape}. Output shape: {corpus.target_sequence.shape}",
        simulation_id=corpus.name,
        resumed_from_simulation_id=arguments.resumed_simulation_id,
    )
    config = utils.add_hash_to_simulation_id(config, corpus)

    if arguments.override_existing:
        genetic_algorithm.remove_simulation_directory(
            simulation_id=config.simulation_id,
            use_cloud=config.migration_channel == "cloud",
        )

    utils.seed(config.seed)

    island.run(
        corpus=corpus,
        config=config,
        first_island=(
            arguments.first_island if arguments.first_island is not None else 0
        ),
        last_island=(
            arguments.last_island
            if arguments.last_island is not None
            else config.num_islands - 1
        ),
    )


if __name__ == "__main__":
    run()
