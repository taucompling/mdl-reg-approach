import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class SimulationConfig:
    simulation_id: str
    num_islands: int
    migration_interval_seconds: int
    migration_interval_generations: int
    migration_ratio: float

    num_generations: int
    population_size: int
    elite_ratio: float
    mutation_probab: float
    crossover_probab: float
    allowed_activations: tuple[int, ...]
    allowed_unit_types: tuple[int, ...]
    start_smooth: bool

    max_network_units: int
    tournament_size: int

    grammar_multiplier: int
    data_given_grammar_multiplier: int
    max_grammar_size: Optional[int]
    allow_test_overlap: bool

    compress_grammar_encoding: bool
    softmax_outputs: bool
    truncate_large_values: bool
    bias_connections: bool
    recurrent_connections: bool
    generation_dump_interval: int

    seed: int
    corpus_seed: int

    mini_batch_size: Optional[int]
    resumed_from_simulation_id: Optional[str]
    comment: Optional[str]
    parallelize: bool
    log_to_cloud: bool
    migration_channel: str  # {'file', 'cloud', 'mpi'}

    regularization_method: str  # {"none", "l1", "l2"}
    regularization_multiplier: float

    golden_networks: list[
        str
    ]  # {"an_bn", "an_bn_cn", "dyck_1", "dyck_2", "arithmetic", "toy_english"}
    num_golden_copies_in_initialization: int

    no_improvement_time: float  # 0 to avoid early stop
    try_net_early_stop: bool  # True to stop feeding the network if it assigns 0 probability to a target class
    data_given_grammar_smoothing_epsilon: Optional[
        float
    ]  # Whether to smooth zero probabilities when computing data given grammar

    allow_architecture_changing_mutations: bool
