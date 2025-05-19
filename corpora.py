import collections
import dataclasses
import itertools
import math
import random
from typing import List, Optional, Union

import numpy as np
from loguru import logger

import configuration
import dfa
import utils

_DEFAULT_CONFIG = configuration.SimulationConfig(
    simulation_id="test",
    num_islands=1,
    migration_ratio=0.1,
    migration_interval_seconds=20,
    migration_interval_generations=1000,
    num_generations=1000,
    population_size=20,
    elite_ratio=0.05,
    allowed_activations=(
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ),
    start_smooth=False,
    allowed_unit_types=(
        0,
        1,
    ),
    tournament_size=4,
    mutation_probab=1.0,
    crossover_probab=0.0,
    grammar_multiplier=1,
    data_given_grammar_multiplier=1,
    max_grammar_size=None,
    allow_test_overlap=True,
    compress_grammar_encoding=False,
    max_network_units=1024,
    softmax_outputs=False,
    truncate_large_values=True,
    bias_connections=True,
    recurrent_connections=True,
    seed=1,
    corpus_seed=100,
    generation_dump_interval=1,
    parallelize=False,
    log_to_cloud=False,
    migration_channel="file",
    mini_batch_size=None,
    resumed_from_simulation_id=None,
    comment=None,
    regularization_method="none",
    regularization_multiplier=1.0,
    golden_networks=[],
    num_golden_copies_in_initialization=1,
    no_improvement_time=120,
    try_net_early_stop=True,
    data_given_grammar_smoothing_epsilon=None,
    allow_architecture_changing_mutations=True,
)


MASK_VALUE = np.nan

_SEQ_START_OR_END = 0
_A = 1
_B = 2
_C = 3
_D = 4


_DYCK_BRACKET_PAIRS = (
    ("[", "]"),
    ("(", ")"),
    ("{", "}"),
    ("<", ">"),
    ("⟦", "⟧"),
    ("〔", " 〕"),
)


_Vocabulary = dict[int, str]


def is_masked(x: Union[np.ndarray, float]) -> Union[np.ndarray, bool]:
    return np.isnan(x)


@dataclasses.dataclass(frozen=True)
class Corpus:
    name: str
    input_sequence: np.ndarray
    target_sequence: np.ndarray

    optimal_d_given_g: Optional[float] = None
    vocabulary: Optional[_Vocabulary] = None
    deterministic_steps_mask: Optional[np.ndarray] = None

    # Precomputed values for feeding efficiency.
    input_mask: Optional[np.ndarray] = None
    targets_mask: Optional[np.ndarray] = None
    input_values_per_time_step: Optional[dict[int, list[np.ndarray]]] = None
    sample_weights: Optional[tuple[int, ...]] = None

    test_corpus: Optional["Corpus"] = None
    train_test_overlap_mask: Optional[np.ndarray] = None
    optimal_d_given_g_no_overlap: Optional[float] = None


def precompute_mask_idxs(corpus: Corpus) -> Corpus:
    masked = is_masked(corpus.input_sequence)
    input_mask = np.array(
        [
            ~np.all(masked[i, j])
            for (i, j) in np.ndindex(corpus.input_sequence.shape[:2])
        ],
        dtype=bool,
    ).reshape(corpus.input_sequence.shape[:2])
    return dataclasses.replace(corpus, input_mask=input_mask)


def _precompute_input_unit_values(corpus: Corpus) -> Corpus:
    unit_to_timestep_val = {}
    for unit in range(corpus.input_sequence.shape[-1]):
        unit_to_timestep_val[unit] = []
        for time_step in range(corpus.input_sequence.shape[1]):
            time_step_input = np.ascontiguousarray(
                corpus.input_sequence[:, time_step, unit]
            )
            time_step_input.flags.writeable = False
            unit_to_timestep_val[unit].append(time_step_input)
        unit_to_timestep_val[unit] = tuple(unit_to_timestep_val[unit])
    return dataclasses.replace(corpus, input_values_per_time_step=unit_to_timestep_val)


def _precompute_targets_mask(corpus: Corpus) -> Corpus:
    if corpus.target_sequence.shape[-1] == 1:
        targets_mask = corpus.target_sequence == 1
    else:
        targets_mask = np.zeros_like(corpus.target_sequence, dtype=bool)
        target_classes = corpus.target_sequence.argmax(axis=-1).flatten()
        batch_idxs, time_idxs = tuple(
            zip(*np.ndindex(corpus.target_sequence.shape[:2]))
        )
        targets_mask[batch_idxs, time_idxs, target_classes] = True
    return dataclasses.replace(corpus, targets_mask=targets_mask)


def _make_inputs_read_only(corpus: Corpus) -> Corpus:
    corpus.input_sequence.flags.writeable = False
    return corpus


def compute_train_test_overlap_mask(
    corpus: Corpus, is_exhaustive_test_corpus: bool
) -> Corpus:
    train_sequences = corpus.input_sequence
    test_sequences = corpus.test_corpus.input_sequence

    # Compute mask of shape test_sequences, which states if the sequence is in train_sequences
    overlap_mask = np.zeros(test_sequences.shape[0], dtype=bool)
    for i, test_sequence in enumerate(test_sequences):
        for train_sequence in train_sequences:
            # Strip NaN values
            stripped_train_sequence = train_sequence[~np.isnan(train_sequence)]
            stripped_test_sequence = test_sequence[~np.isnan(test_sequence)]
            if np.array_equal(stripped_train_sequence, stripped_test_sequence):
                overlap_mask[i] = True
                break

    test_corpus_with_overlap = dataclasses.replace(
        corpus.test_corpus, train_test_overlap_mask=overlap_mask
    )

    # This is a hack to generically compute optimal test D:G without overlap
    # Since we are dealing with exhaustive tets corpora, the probabilities of the strings are given in the
    # sample weights, so we can compute the optimal without overlap using them
    if is_exhaustive_test_corpus:
        # Remove unique_n_values if they are present in corpus.train_test_overlap_mask
        no_overlap_sample_weights = []
        for i, w in enumerate(test_corpus_with_overlap.sample_weights):
            if not test_corpus_with_overlap.train_test_overlap_mask[i]:
                no_overlap_sample_weights.append(w)
        optimal_d_given_g_no_overlap = -np.sum(
            [prob * np.log2(prob) for prob in no_overlap_sample_weights]
        )
        test_corpus_with_overlap = dataclasses.replace(
            test_corpus_with_overlap,
            optimal_d_given_g_no_overlap=optimal_d_given_g_no_overlap,
        )
        logger.info(f"Optimal test D:G without overlap: {optimal_d_given_g_no_overlap}")
    else:
        logger.warning(
            f"Overlap mask computed for {corpus.name} but test corpus is not exhaustive, so cannot compute optimal D:G without overlap."
        )

    return dataclasses.replace(corpus, test_corpus=test_corpus_with_overlap)


def optimize_for_feeding(corpus: Corpus) -> Corpus:
    logger.info("Optimizing corpus for feeding...")
    corpus = _make_inputs_read_only(corpus)
    corpus = _precompute_targets_mask(corpus)
    corpus = precompute_mask_idxs(corpus)
    corpus = _precompute_input_unit_values(corpus)
    return corpus


def _sample_geometric(prior: float, batch_size: int) -> tuple[int, ...]:
    return tuple(np.random.geometric(p=prior, size=batch_size))


def make_random_binary(
    sequence_length: int = 100,
    batch_size: int = 1,
) -> Corpus:
    return Corpus(
        "random_binary",
        input_sequence=np.random.randint(0, 2, size=(batch_size, sequence_length, 1)),
        target_sequence=np.random.randint(0, 2, size=(batch_size, sequence_length, 1)),
    )


def make_random_one_hot(
    num_input_classes: int,
    num_target_classes: int,
    sequence_length: int = 100,
    batch_size: int = 1,
) -> Corpus:
    input_classes = np.random.randint(
        0, num_input_classes, size=(batch_size, sequence_length)
    )
    target_classes = np.random.randint(
        0, num_target_classes, size=(batch_size, sequence_length)
    )
    return make_one_hot_corpus(
        "random_one_hot",
        input_classes=input_classes,
        target_classes=target_classes,
        num_input_classes=num_input_classes,
        num_target_classes=num_target_classes,
    )


def make_one_hot_corpus(
    name: str,
    input_classes: Union[list, np.ndarray],
    target_classes: Union[list, np.ndarray],
    num_input_classes: int,
    num_target_classes: int,
    weights: Optional[tuple[int, ...]] = None,
    vocabulary: Optional[_Vocabulary] = None,
) -> Corpus:
    return Corpus(
        name,
        input_sequence=_make_one_hot_sequence(
            np.array(input_classes), num_input_classes
        ),
        target_sequence=_make_one_hot_sequence(
            np.array(target_classes), num_target_classes
        ),
        sample_weights=weights,
        vocabulary=vocabulary,
        input_mask=~is_masked(input_classes),
    )


def _force_batch_dimension(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.expand_dims(arr, axis=0)
    return arr


def _make_one_hot_sequence(classes: np.ndarray, num_classes: int) -> np.ndarray:
    # Assumes masked sequences are masked from first mask index onwards.
    classes = _force_batch_dimension(classes)
    batch_size = classes.shape[0]
    sequence_length = classes.shape[1]

    one_hot = np.zeros(
        (batch_size, sequence_length, num_classes), dtype=utils.FLOAT_DTYPE, order="C"
    )

    for b in range(batch_size):
        for s in range(sequence_length):
            c = classes[b, s]
            if is_masked(c):
                one_hot[b, s:] = MASK_VALUE
                break
            else:
                one_hot[b, s, int(c)] = 1.0
    return one_hot


def make_between_dfa(start: int, end: int) -> dfa.DFA:
    final_state = end + 1
    transitions = {}
    for i in range(start):
        transitions[i] = {"0": i, "1": i + 1}
    for i in range(start, end):
        transitions[i] = {"0": i, "1": i + 1, dfa.END_OF_SEQUENCE: final_state}
    transitions[end] = {"0": end, dfa.END_OF_SEQUENCE: final_state}
    return dfa.DFA(transitions=transitions, accepting_states={final_state})


def make_at_least_dfa(n: int) -> dfa.DFA:
    transitions = {}
    for i in range(n):
        transitions[i] = {"0": i, "1": i + 1}
    transitions[n] = {"0": n, "1": n, dfa.END_OF_SEQUENCE: n + 1}
    return dfa.DFA(transitions=transitions, accepting_states={n + 1})


def _make_at_most_dfa(n: int) -> dfa.DFA:
    transitions = {}
    accepting_state = n + 1
    for i in range(n):
        transitions[i] = {"0": i, "1": i + 1, dfa.END_OF_SEQUENCE: accepting_state}
    transitions[n] = {"0": n, dfa.END_OF_SEQUENCE: accepting_state}
    return dfa.DFA(transitions=transitions, accepting_states={accepting_state})


def _make_simple_arithmetic_syntax_dfa() -> dfa.DFA:
    """
    Corresponds to the regex: `1+(\+1+)*=1+(\+1+)*#`
    https://dreampuf.github.io/GraphvizOnline/#digraph%20G%20%7B%20%0A%20%20%20%20s0%20-%3E%20s1%20%5Blabel%3D%221%22%5D%0A%20%20%20%20s1%20-%3E%20s1%20%5Blabel%3D%221%22%5D%0A%20%20%20%20s1%20-%3E%20s2%20%5Blabel%3D%22%2B%22%5D%0A%20%20%20%20s2%20-%3E%20s1%20%5Blabel%3D%221%22%5D%0A%20%20%20%20s1%20-%3E%20s3%20%5Blabel%3D%22%3D%22%5D%0A%20%20%20%20s3%20-%3E%20s4%20%5Blabel%3D%221%22%5D%0A%20%20%20%20s4%20-%3E%20s4%20%5Blabel%3D%221%22%5D%0A%20%20%20%20s4%20-%3E%20s5%20%5Blabel%3D%22%2B%22%5D%0A%20%20%20%20s5%20-%3E%20s4%20%5Blabel%3D%221%22%5D%0A%20%20%20%20s4%20-%3E%20s6%20%5Blabel%3D%22%23%22%5D%0A%20%20%20%20s6%20%5Bperipheries%3D2%5D%0A%7D
    """
    transitions = {
        0: {"1": 1},
        1: {"1": 1, "+": 2, "=": 3},
        2: {"1": 1},
        3: {"1": 4},
        4: {"1": 4, "+": 5, dfa.END_OF_SEQUENCE: 6},
        5: {"1": 4},
    }
    accepting_states = {6}
    return dfa.DFA(transitions=transitions, accepting_states=accepting_states)


def _dfa_to_inputs(
    dfa_: dfa.DFA,
    batch_size: int,
    end_of_sequence_char: int,
    max_sequence_length: int,
) -> np.ndarray:
    batch = np.empty((batch_size, max_sequence_length))
    batch.fill(MASK_VALUE)

    for b in range(batch_size):
        input_idx = 0
        while True:
            string = dfa_.generate_string()
            if len(string) > max_sequence_length:
                continue
            if input_idx + len(string) > max_sequence_length:
                break
            for i, char in enumerate(string):
                if char == dfa.END_OF_SEQUENCE:
                    char_int = end_of_sequence_char
                else:
                    char_int = int(char)
                batch[b, input_idx + i] = char_int
            input_idx += len(string)

    return batch


def make_identity(
    sequence_length: int = 1000, batch_size: int = 10, num_classes: int = 2
) -> Corpus:
    sequence = np.random.randint(
        num_classes, size=(batch_size, sequence_length)
    ).astype(utils.FLOAT_DTYPE)
    return make_one_hot_corpus("identity", sequence, sequence, num_classes, num_classes)


def make_identity_binary(sequence_length: int, batch_size: int) -> Corpus:
    sequence = np.random.randint(2, size=(batch_size, sequence_length, 1)).astype(
        utils.FLOAT_DTYPE
    )
    input_sequence = np.copy(sequence)
    input_sequence[sequence == 0] = -1
    return Corpus("identity", input_sequence, sequence)


def _make_repetition_sequence(
    input_sequence: np.ndarray, offset: int, padding: float = 0.0
) -> np.ndarray:
    """[a,b,c,d, ..., y,z] -> [<padding>,a,b,c, ..., y]"""
    assert input_sequence.ndim == 2
    batch_size = input_sequence.shape[0]
    padded_arr = np.empty((batch_size, offset), dtype=utils.FLOAT_DTYPE)
    padded_arr.fill(padding)
    return np.concatenate(
        (
            padded_arr,
            input_sequence[:, :-offset],
        ),
        axis=-1,
    )


def _make_prediction_sequence(
    input_sequence: np.ndarray, lookahead: int = 1, padding: float = 0.0
) -> np.ndarray:
    """[a,b,c,d, ..., y,z] -> [b,c,d,e, ..., z,<padding>]"""
    input_sequence = _force_batch_dimension(input_sequence)
    assert input_sequence.ndim == 2
    batch_size = input_sequence.shape[0]
    padded_arr = np.empty((batch_size, lookahead), dtype=utils.FLOAT_DTYPE)
    padded_arr.fill(padding)
    return np.concatenate((input_sequence[:, lookahead:], padded_arr), axis=-1)


def make_prev_char_repetition(
    sequence_length: int = 1000,
    batch_size: int = 10,
    repetition_offset: int = 1,
    num_classes: int = 2,
) -> Corpus:
    input_sequence = np.random.randint(num_classes, size=(batch_size, sequence_length))
    target_sequence = _make_repetition_sequence(input_sequence, repetition_offset)

    return make_one_hot_corpus(
        f"repeat_prev_{repetition_offset}_char",
        input_sequence,
        target_sequence,
        num_classes,
        num_classes,
    )


def make_prev_char_repetition_binary(
    sequence_length: int,
    batch_size: int,
    repetition_offset: int,
) -> Corpus:
    input_sequence = np.random.randint(2, size=(batch_size, sequence_length)).astype(
        utils.FLOAT_DTYPE
    )
    target_sequence = _make_repetition_sequence(input_sequence, repetition_offset)
    return Corpus(
        f"repeat_prev_{repetition_offset}_char_binary",
        np.expand_dims(input_sequence, -1),
        np.expand_dims(target_sequence, -1),
    )


def make_elman_xor_binary(sequence_length: int = 3000, batch_size: int = 1) -> Corpus:
    assert sequence_length % 3 == 0

    input_batch = []
    target_batch = []
    for b in range(batch_size):
        sequence = []
        for pair_idx in range(sequence_length // 3):
            a, b = random.choice([0, 1]), random.choice([0, 1])
            sequence += [a, b, a ^ b]
        input_batch.append(sequence)
        # Target output is the next character of input
        target_sequence = sequence[1:] + [0]
        target_batch.append(target_sequence)
    input_batch = np.expand_dims(
        np.array(input_batch, dtype=utils.FLOAT_DTYPE), axis=-1
    )
    target_batch = np.expand_dims(
        np.array(target_batch, dtype=utils.FLOAT_DTYPE), axis=-1
    )
    return Corpus("elman_xor_binary", input_batch, target_batch)


def make_elman_xor_one_hot(sequence_length: int = 3000, batch_size: int = 1) -> Corpus:
    binary_corpus = make_elman_xor_binary(sequence_length, batch_size)

    return make_one_hot_corpus(
        "elman_xor_one_hot",
        binary_corpus.input_sequence,
        binary_corpus.target_sequence,
        num_input_classes=2,
        num_target_classes=2,
    )


def make_semi_random_corpus(sequence_length: int = 100, batch_size: int = 10) -> Corpus:
    """One random bit, one identical bit, e.g.: [0,0,0,0,1,1,0,0,1,1,0,0, ...]"""
    assert sequence_length % 2 == 0
    input_batch = []
    target_batch = []
    for _ in range(batch_size):
        sequence = []
        for s in range(sequence_length // 2):
            sequence += [random.randrange(2)] * 2
        input_batch.append(sequence)
        target_sequence = sequence[1:] + [0]
        target_batch.append(target_sequence)

    input_batch = np.expand_dims(np.array(input_batch), axis=-1)
    target_batch = np.expand_dims(np.array(target_batch), axis=-1)
    return Corpus("semi_random_pairs", input_batch, target_batch)


def make_elman_badigu(num_consonants: int = 1000) -> Corpus:
    # ba, dii, guuu
    feature_table = {
        # cons, vowel, int, high, back, voiced
        "b": [1, 0, 1, 0, 0, 1],  # b
        "d": [1, 0, 1, 1, 0, 1],  # d
        "g": [1, 0, 1, 0, 1, 1],  # g
        "a": [0, 1, 0, 0, 1, 1],  # a
        "i": [0, 1, 0, 1, 0, 1],  # i
        "u": [0, 1, 0, 1, 1, 1],  # u
    }
    segments = list("bdgaiu")
    num_classes = len(segments)
    segment_to_idx = {x: i for i, x in enumerate(segments)}

    consonant_to_sequence = {"b": list("ba"), "d": list("dii"), "g": list("guuu")}
    consonant_sequence = np.random.choice(["b", "d", "g"], size=num_consonants)

    letters_sequence = list(
        itertools.chain(*(consonant_to_sequence[c] for c in consonant_sequence))
    )
    input_sequence = [segment_to_idx[x] for x in letters_sequence]
    target_sequence = input_sequence[1:] + [0]

    logger.info(f"Elman badigu sequence: {letters_sequence}")
    consonants = tuple("bdg")

    consonant_percentage = len([x for x in letters_sequence if x in consonants]) / len(
        letters_sequence
    )

    logger.info(f"Max accuracy for task: {1 - consonant_percentage:.2f}")

    return make_one_hot_corpus(
        "elman_badigu", input_sequence, target_sequence, num_classes, num_classes
    )


def _make_0_1_pattern_binary(sequence_length: int, batch_size: int) -> Corpus:
    assert sequence_length % 2 == 0

    input_seq = np.array([[0, 1] * (sequence_length // 2)], dtype=utils.FLOAT_DTYPE)
    target_seq = _make_prediction_sequence(input_seq, lookahead=1, padding=0.0)

    input_seq = np.expand_dims(input_seq, axis=2)
    target_seq = np.expand_dims(target_seq, axis=2)

    return Corpus(
        name=f"0_1_pattern_binary_length_{sequence_length}_batch_{batch_size}",
        input_sequence=input_seq,
        target_sequence=target_seq,
        optimal_d_given_g=0.0,
        vocabulary={0: "1", 1: "1"},
        sample_weights=(batch_size,) if batch_size > 1 else None,
    )


def make_0_1_pattern_binary(sequence_length: int, batch_size: int) -> Corpus:
    train_corpus = _make_0_1_pattern_binary(sequence_length, batch_size)
    test_corpus = _make_0_1_pattern_binary(
        sequence_length=sequence_length * 50_000, batch_size=1
    )
    return dataclasses.replace(train_corpus, test_corpus=test_corpus)


def _make_0_1_pattern_one_hot(
    sequence_length: int, add_end_of_sequence: bool, batch_size: int
) -> Corpus:
    assert sequence_length % 2 == 0

    input_classes = [0, 1] * (sequence_length // 2)
    num_classes = 2
    vocabulary = {0: "0", 1: "1"}

    if add_end_of_sequence:
        num_classes = 3
        input_classes += [2]
        vocabulary[2] = "#"

    vocabulary.update({x + len(vocabulary): vocabulary[x] for x in vocabulary})

    input_classes_arr = np.array([input_classes])
    target_classes_arr = _make_prediction_sequence(
        input_classes_arr, lookahead=1, padding=0.0
    )

    corpus = make_one_hot_corpus(
        name=f"0_1_pattern_one_hot_length_{sequence_length}_batch_{batch_size}{'_eos' if add_end_of_sequence else ''}",
        input_classes=input_classes_arr,
        target_classes=target_classes_arr,
        num_input_classes=num_classes,
        num_target_classes=num_classes,
        weights=(batch_size,) if batch_size > 1 else None,
        vocabulary=vocabulary,
    )
    return dataclasses.replace(
        # TODO: calculate optimal D
        corpus,
        optimal_d_given_g=0.0,
    )


def make_0_1_pattern_one_hot(
    sequence_length: int, add_end_of_sequence: bool, batch_size: int
) -> Corpus:
    train_corpus = _make_0_1_pattern_one_hot(
        sequence_length, add_end_of_sequence, batch_size
    )
    test_corpus = _make_0_1_pattern_one_hot(
        sequence_length=sequence_length * 50_000,
        add_end_of_sequence=add_end_of_sequence,
        batch_size=1,
    )
    return dataclasses.replace(train_corpus, test_corpus=test_corpus)


def make_123_n_pattern_corpus(
    base_sequence_length: int = 3, sequence_length: int = 100
):
    # [0,1,2, ..., n-1] repeated
    assert sequence_length % base_sequence_length == 0
    input_sequence = np.array(
        [list(range(base_sequence_length)) * (sequence_length // base_sequence_length)]
    )
    target_sequence = _make_prediction_sequence(input_sequence, lookahead=1)
    return make_one_hot_corpus(
        f"1_to_{base_sequence_length}_pattern",
        input_sequence,
        target_sequence,
        num_input_classes=base_sequence_length,
        num_target_classes=base_sequence_length,
    )


def make_between_quantifier(
    start: int, end: int, sequence_length: int = 100, batch_size: int = 1
) -> Corpus:
    assert end <= sequence_length
    between_dfa = make_between_dfa(start, end)
    between_dfa.visualize(f"between_{start}_{end}_dfa")
    input_batch = _dfa_to_inputs(
        between_dfa,
        batch_size=batch_size,
        end_of_sequence_char=2,
        max_sequence_length=sequence_length,
    )
    target_batch = _make_prediction_sequence(input_batch, lookahead=1)
    if start == end:
        name = f"exactly_{start}"
    else:
        name = f"between_{start}_{end}"
    num_classes = 3
    return make_one_hot_corpus(
        name, input_batch, target_batch, num_classes, num_classes
    )


def make_exactly_n_quantifier(
    n: int = 1, sequence_length: int = 100, batch_size: int = 1
) -> Corpus:
    return make_between_quantifier(
        start=n, end=n, sequence_length=sequence_length, batch_size=batch_size
    )


def make_at_least_quantifier(
    n: int = 1, sequence_length: int = 100, batch_size: int = 1
) -> Corpus:
    name = f"at_least_{n}"
    at_least_dfa = make_at_least_dfa(n)
    at_least_dfa.visualize(name)
    input_batch = _dfa_to_inputs(
        at_least_dfa,
        batch_size=batch_size,
        end_of_sequence_char=2,
        max_sequence_length=sequence_length,
    )
    target_batch = _make_prediction_sequence(input_batch, lookahead=1)
    num_classes = 3
    return make_one_hot_corpus(
        name, input_batch, target_batch, num_classes, num_classes
    )


def make_at_most_quantifier(
    n: int = 1, sequence_length: int = 100, batch_size: int = 1
) -> Corpus:
    name = f"at_most_{n}"
    at_most_dfa = _make_at_most_dfa(n)
    at_most_dfa.visualize(name)
    input_batch = _dfa_to_inputs(
        at_most_dfa,
        batch_size=batch_size,
        end_of_sequence_char=2,
        max_sequence_length=sequence_length,
    )
    target_batch = _make_prediction_sequence(input_batch, lookahead=1)
    num_classes = 3
    return make_one_hot_corpus(
        name, input_batch, target_batch, num_classes, num_classes
    )


def make_every_quantifier(sequence_length: int = 100, batch_size: int = 1) -> Corpus:
    input_batch = np.ones((batch_size, sequence_length))
    return make_one_hot_corpus("every_quantifier", input_batch, input_batch, 2, 2)


def _int_to_classes(n: int) -> list[int]:
    return list(reversed(list(map(int, list(str(n))))))


def make_count_corpus(max_int: 100, batch_size: int = 100) -> Corpus:
    # Predict n+1 from n in a language-model setting.
    sequence_length = int(np.floor(np.log10(max_int))) + 1
    input_classes = np.zeros((batch_size, sequence_length))
    target_classes = np.zeros((batch_size, sequence_length))

    for b in range(batch_size):
        n = random.randrange(max_int)
        input_ = _int_to_classes(n)
        target = _int_to_classes(n + 1)
        input_classes[b, : len(input_)] = input_
        target_classes[b, : len(target)] = target

    return make_one_hot_corpus(
        f"count_to_{max_int}",
        input_classes,
        target_classes,
        num_input_classes=10,
        num_target_classes=10,
    )


def base10_to_binary_vector(n: int, sequence_length=None) -> np.ndarray:
    """8 -> [0,0,0,1], 7 -> [1,1,1]"""
    if n == 0:
        return np.zeros(sequence_length)

    powers = []
    while n:
        power = int(np.floor(np.log2(n)))
        powers.append(power)
        n -= 2**power

    rightmost_one_position = int(max(powers))
    if sequence_length is None:
        sequence_length = rightmost_one_position + 1
    binary = np.zeros(sequence_length)
    binary[powers] = 1.0
    # TODO: mask redundant positions?
    return binary


def _make_binary_addition_corpus(min_n: int, max_n: int):
    all_summands = tuple(itertools.product(range(min_n, max_n), repeat=2))
    summands = []
    for b, (n1, n2) in enumerate(all_summands):
        summands.append([n1, n2])
    summands = np.array(summands)
    sums = np.sum(summands, axis=1)
    sequence_length = int(np.ceil(np.log2(np.max(sums)))) + 1

    summand_binaries = []
    sum_binaries = []
    for (n1, n2), sum_ in zip(summands, sums):
        summand_binaries.append(
            [
                base10_to_binary_vector(n1, sequence_length),
                base10_to_binary_vector(n2, sequence_length),
            ]
        )
        sum_binaries.append(base10_to_binary_vector(sum_, sequence_length))

    summand_inputs = np.array(
        [
            np.stack(
                summands,
                axis=1,
            )
            for summands in summand_binaries
        ]
    )

    sum_outputs = np.expand_dims(np.stack(sum_binaries), axis=-1)

    return dataclasses.replace(
        Corpus(
            name=f"binary_addition_{min_n}_to_{max_n}",
            input_sequence=summand_inputs,
            target_sequence=sum_outputs,
        ),
        optimal_d_given_g=0.0,
    )


def make_binary_addition(min_n: int, max_n: int) -> Corpus:
    training_corpus = _make_binary_addition_corpus(min_n=min_n, max_n=max_n)
    test_corpus = _make_binary_addition_corpus(min_n=max_n + 1, max_n=max_n + 251)
    training_corpus = dataclasses.replace(training_corpus, test_corpus=test_corpus)
    return training_corpus


def an_bn_handmade_net(input_seq: np.ndarray, prior: float):
    # Optimal network according to Schimdhuber (2001).
    outputs = np.zeros_like(input_seq)
    for b in range(input_seq.shape[0]):
        num_seen_a = 0
        for t in range(input_seq.shape[1]):
            input_vec = input_seq[b, t]
            input_class = input_vec.argmax()
            if input_class == 0:
                # Start of sequence symbol, always predict "a" (no empty string in current corpus).
                outputs[b, t] = [0.0, 1.0, 0.0]
            elif input_class == 1:
                # "a".
                num_seen_a += 1
                outputs[b, t] = [0.0, 1 - prior, prior]
            elif input_class == 2:
                # "b".
                num_seen_a -= 1
                if num_seen_a > 0:
                    outputs[b, t] = [0.0, 0.0, 1.0]
                else:
                    outputs[b, t] = [1.0, 0.0, 0.0]

    return outputs


class _DerivationTooLong(Exception):
    pass


def _generate_string_from_pcfg(
    pcfg: dict, max_length: Optional[int] = None, with_prob: bool = False
) -> Union[tuple[str, ...], tuple[tuple[str, ...], float]]:
    """
    PCFG structure:
    ```
    palindrome_pcfg = {
        "S": (
            (("0", "S", "0"), 0.25),
            (("1", "S", "1"), 0.25),
            (("@",), 0.5),
        )
    }
    ```
    Stops when all generated characters are terminals.
    As epsilon, use the empty string ''.
    """
    stack = ["S"]
    terminals = []
    string_prob = 1.0

    while stack:
        node = stack[0]
        stack = stack[1:]

        if node not in pcfg:
            if node != "":
                terminals.append(node)
            if max_length is not None and len(terminals) > max_length:
                raise _DerivationTooLong
            continue

        rules, probabs = list(zip(*pcfg[node]))
        rule_idx = np.random.choice(len(rules), p=probabs)
        rule = rules[rule_idx]

        string_prob *= probabs[rule_idx]

        stack = list(rule) + stack

    return (tuple(terminals), string_prob) if with_prob else tuple(terminals)


def _generate_all_strings_from_pcfg(
    pcfg: dict[str, tuple[tuple[tuple[str, ...], float], ...]],
    start_symbol: str,
    max_length: int,
    with_probs: bool = False,
) -> Union[set[tuple[str, ...]], dict[tuple[str, ...], float]]:
    """
    Generate all possible terminal strings from the given PCFG up to `max_length`.
    Returns:
        - If with_probs=False: a set of tuples representing terminal strings.
        - If with_probs=True: a dict {terminal_string (tuple): probability (float)}.
    """
    if with_probs:
        final_strings: dict[tuple[str, ...], float] = {}
        queue = collections.deque([((start_symbol,), 1.0)])
    else:
        final_strings: set[tuple[str, ...]] = set()
        queue = collections.deque([(start_symbol,)])

    while queue:
        if with_probs:
            current_seq, current_prob = queue.popleft()
        else:
            current_seq = queue.popleft()

        # Find the first non-terminal, if any
        nt_index = None
        for i, sym in enumerate(current_seq):
            if sym in pcfg:  # if the symbol is a non-terminal in the grammar
                nt_index = i
                break

        # If no non-terminals, we have a fully expanded terminal string
        if nt_index is None:
            if len(current_seq) <= max_length:
                if with_probs:
                    # Accumulate probability in case the same sequence can be derived via multiple paths
                    final_strings[current_seq] = (
                        final_strings.get(current_seq, 0) + current_prob
                    )
                else:
                    final_strings.add(current_seq)
            continue

        # Otherwise, expand that non-terminal by each of its right-hand sides
        non_terminal = current_seq[nt_index]
        for rhs, prob in pcfg[non_terminal]:
            # Replace the non-terminal in current_seq with this rhs
            if rhs == ("",) or rhs == "":
                # Epsilon: remove the non-terminal
                new_seq = current_seq[:nt_index] + current_seq[nt_index + 1 :]
            else:
                new_seq = current_seq[:nt_index] + rhs + current_seq[nt_index + 1 :]

            # Count the number of terminals in new_seq
            # (Terminals are symbols not present as keys in pcfg)
            num_terminals = sum(1 for s in new_seq if s not in pcfg)
            if num_terminals <= max_length:
                if with_probs:
                    new_prob = current_prob * prob
                    queue.append((new_seq, new_prob))
                else:
                    queue.append(new_seq)

    return final_strings


def _generate_string_from_dfa(
    dfa: dfa.DFA, max_length: Optional[int] = None
) -> tuple[str, ...]:
    while True:
        string = dfa.generate_string()
        if max_length is not None and len(string) > max_length:
            continue
        string = string.strip("#")
        return tuple(string)


def _corpus_from_strings(
    name: str,
    strings: list[tuple[str, ...]],
    sort_by_length: bool,
    exhaustive: bool = False,
    strings_probs: Optional[tuple[float, ...]] = None,
) -> Corpus:
    if sort_by_length:
        strings = sorted(strings, key=len, reverse=True)

    lengths = list(map(len, strings))

    sequence_counts = collections.Counter(strings)
    unique_sequences, weights = tuple(zip(*sequence_counts.items()))
    optimal_d_given_g = None

    if strings_probs is not None:
        # We can retrieve the optimal D:G based on the probabilities
        # In the case of exhaustive generation, there is one string of each length, so we multiply p * log(p) for each
        # string. In the case of non-exhaustive, we sampled a lot of strings and we use the frequency in the
        # calculation. No need to multiply by anything, as we iterate over all strings and not over unique ones.
        # TODO: Iterate only on unique sentence to save time.
        if exhaustive:
            # If we generated all strings once, the weights are the probabilities of those strings
            weights = tuple(strings_probs)
            optimal_d_given_g = -np.sum([prob * np.log2(prob) for prob in weights])
        else:
            optimal_d_given_g = -np.sum([np.log2(prob) for prob in strings_probs])

        logger.info(f"Optimal D:G: {optimal_d_given_g}")

    logger.info(f"Sum of sequence lengths: {sum(lengths)}")
    logger.info(f"Max sequence length: {max(lengths)}")
    logger.info(f"Mean sequence length: {np.mean(lengths)}")
    logger.info(
        f"Unique sequences: {len(unique_sequences)}/{len(strings)} ({len(unique_sequences) / len(strings):.2f})"
    )

    alphabet = set()
    for seq in strings:
        alphabet |= set(seq)
    alphabet = ("#",) + tuple(sorted(alphabet))

    symbol_to_idx = {x: i for i, x in enumerate(alphabet)}

    max_seq_length = max(map(len, unique_sequences))
    input_classes = np.empty((len(unique_sequences), max_seq_length + 1))
    target_classes = np.empty_like(input_classes)
    input_classes.fill(MASK_VALUE)
    target_classes.fill(MASK_VALUE)

    for i, sequence in enumerate(unique_sequences):
        sequence_classes = [symbol_to_idx[symbol] for symbol in sequence]
        input_row = [symbol_to_idx["#"]] + sequence_classes
        target_row = sequence_classes + [symbol_to_idx["#"]]
        input_classes[i, : len(sequence_classes) + 1] = input_row
        target_classes[i, : len(sequence_classes) + 1] = target_row

    inputs = _make_one_hot_sequence(input_classes, num_classes=len(alphabet))
    targets = _make_one_hot_sequence(target_classes, num_classes=len(alphabet))

    vocabulary = _make_identical_input_output_vocabulary(alphabet)

    return Corpus(
        name=name,
        input_sequence=inputs,
        target_sequence=targets,
        sample_weights=weights,
        vocabulary=vocabulary,
        input_mask=~is_masked(input_classes),
        optimal_d_given_g=optimal_d_given_g,
    )


def _make_corpus_from_pcfg(
    name: str,
    pcfg: dict,
    batch_size: int,
    max_derivation_length: Optional[int] = None,
    sort_by_length: bool = False,
    exhaustive: bool = False,
    start_symbol: str = "S",
) -> Corpus:
    if exhaustive:
        sequences_and_probs = _generate_all_strings_from_pcfg(
            pcfg, start_symbol, max_derivation_length, with_probs=True
        )
        sequences, probs = zip(*sequences_and_probs.items())
    else:
        sequences: list[tuple[str, ...]] = []
        probs = []

        while len(sequences) < batch_size:
            try:
                sequence, prob = _generate_string_from_pcfg(
                    pcfg, max_length=max_derivation_length, with_prob=True
                )
            except _DerivationTooLong:
                continue

            sequences.append(sequence)
            probs.append(prob)

    return _corpus_from_strings(
        name=name,
        strings=sequences,
        sort_by_length=sort_by_length,
        exhaustive=exhaustive,
        strings_probs=probs,
    )


def _make_corpus_from_dfa(
    name: str,
    dfa_inst: dfa.DFA,
    batch_size: int,
    max_derivation_length: Optional[int] = None,
    sort_by_length: bool = False,
) -> Corpus:
    sequences = []
    while len(sequences) < batch_size:
        sequence = _generate_string_from_dfa(dfa_inst, max_length=max_derivation_length)
        sequences.append(sequence)

    lengths = list(map(len, sequences))

    sequence_counts = collections.Counter(sequences)
    unique_sequences, weights = tuple(zip(*sequence_counts.items()))

    corpus = _corpus_from_strings(
        name=name,
        strings=sequences,
        sort_by_length=sort_by_length,
    )
    optimal_d_given_g = dfa_inst.get_optimal_data_given_grammar_for_dfa(corpus)
    corpus = dataclasses.replace(corpus, optimal_d_given_g=optimal_d_given_g)

    logger.info(f"DFA sum of sequence lengths: {sum(lengths)}")
    logger.info(f"DFA max sequence length: {max(lengths)}")
    logger.info(f"DFA mean sequence length: {np.mean(lengths)}")
    logger.info(
        f"DFA unique sequences: {len(unique_sequences)}/{len(sequences)} ({len(unique_sequences) / len(sequences):.2f})"
    )
    logger.info(f"Optimal D:G based on DFA: {corpus.optimal_d_given_g:.2f}")

    return corpus


def make_simple_arithmetic_syntax_pcfg(sequence_length: int, batch_size: int) -> Corpus:
    # This is a simple arithmetic syntax that only allows for addition of numbers that contain only 1s.
    pcfg = {
        "S": ((("E", "=", "E"), 1.0),),
        "E": (
            (("N",), 0.33),
            (("E", "+", "E"), 0.67),
        ),
        "N": (
            (("D", "N"), 0.5),
            (("D",), 0.5),
        ),
        "D": ((("1",), 1),),
    }
    return _make_corpus_from_pcfg(
        f"simple_arithmetic_syntax_pcfg_up_to_{sequence_length}_chars",
        pcfg=pcfg,
        batch_size=batch_size,
        max_derivation_length=sequence_length,
    )


def make_simple_arithmetic_syntax_dfa(sequence_length: int, batch_size: int) -> Corpus:
    dfa_inst = _make_simple_arithmetic_syntax_dfa()
    return _make_corpus_from_dfa(
        f"simple_arithmetic_syntax_dfa_up_to_{sequence_length}_chars",
        dfa_inst=dfa_inst,
        batch_size=batch_size,
        max_derivation_length=sequence_length,
    )


def make_new_arithmetic_syntax_pcfg(sequence_length: int, batch_size: int) -> Corpus:
    pcfg = {
        "S": ((("E", "=", "E"), 1.0),),
        "E": (
            (("N",), 0.6),
            (("(", "E", "OP", "E", ")"), 0.3),
            (("-", "E"), 0.1),
        ),
        "N": (
            (("0",), 0.1),
            (("1", "M"), 0.9),
        ),
        "M": (
            (("",), 0.5),  # Epsilon as the empty string
            (("0", "M"), 0.25),
            (("1", "M"), 0.25),
        ),
        "OP": ((("+",), 1),),  # Only addition for simplicity
    }
    return _make_corpus_from_pcfg(
        f"new_arithmetic_syntax_pcfg_up_to_{sequence_length}_chars",
        pcfg=pcfg,
        batch_size=batch_size,
        max_derivation_length=sequence_length,
    )


def make_new_minified_arithmetic_syntax_pcfg(
    sequence_length: int, batch_size: int
) -> Corpus:
    pcfg = {
        "S": ((("E", "=", "E"), 1.0),),
        "E": (
            (("N",), 0.7),
            (("(", "E", "OP", "E", ")"), 0.3),
        ),
        "N": ((("1", "M"), 1),),
        "M": (
            (("",), 0.5),  # Epsilon as the empty string
            (("1", "M"), 0.5),
        ),
        "OP": ((("+",), 1),),  # Only addition for simplicity
    }
    return _make_corpus_from_pcfg(
        f"new_minified_arithmetic_syntax_pcfg_up_to_{sequence_length}_chars",
        pcfg=pcfg,
        batch_size=batch_size,
        max_derivation_length=sequence_length,
    )


def _make_arithmetic_syntax_corpus(
    batch_size: int, exhaustive: bool, max_sequence_length: Optional[int]
):
    pcfg = {
        "S": ((("(", "E", "+", "E", ")"), 1.0),),
        "E": (
            (("1",), 0.67),
            (("(", "E", "+", "E", ")"), 0.33),
        ),
    }
    corpus = _make_corpus_from_pcfg(
        f"golden_arithmetic_syntax_pcfg_up_to_{max_sequence_length}_chars",
        pcfg=pcfg,
        batch_size=batch_size,
        max_derivation_length=max_sequence_length,
        exhaustive=exhaustive,
    )
    input_classes = np.argmax(corpus.input_sequence, axis=-1)
    deterministic_steps_mask = corpus.input_mask & (
        #  not '(' or '+'
        (input_classes != 1) & (input_classes != 3)
    )
    return dataclasses.replace(
        corpus, deterministic_steps_mask=deterministic_steps_mask
    )


def make_golden_arithmetic_syntax_pcfg(
    sequence_length: Optional[int],
    batch_size: int,
    max_test_sequence_length: int = 40,  # max length of 40 already generates ~7k sequences
) -> Corpus:
    corpus = _make_arithmetic_syntax_corpus(
        batch_size=batch_size, exhaustive=False, max_sequence_length=sequence_length
    )

    # Create exhaustive test corpus
    test_corpus = _make_arithmetic_syntax_corpus(
        batch_size=1,  # Batch size is irrelevant for exhaustive=True
        exhaustive=True,
        max_sequence_length=max_test_sequence_length,
    )

    return dataclasses.replace(corpus, test_corpus=test_corpus)


def make_new_logic_syntax_pcfg(sequence_length: int, batch_size: int) -> Corpus:
    pcfg = {
        "S": ((("Eseq", "⊢", "E"), 1.0),),
        "Eseq": (
            (("E", "Eseq"), 0.7),
            (("",), 0.3),  # Epsilon as the empty string
        ),
        "E": (
            (("N",), 0.6),
            (("(", "E", "OP", "E", ")"), 0.3),
            (("~", "E"), 0.1),
        ),
        "N": (
            (("0",), 0.1),
            (("1", "M"), 0.9),
        ),
        "M": (
            (("",), 0.5),  # Epsilon as the empty string
            (("0", "M"), 0.25),
            (("1", "M"), 0.25),
        ),
        "OP": ((("&",), 1),),  # Only conjunction for simplicity
    }
    return _make_corpus_from_pcfg(
        f"new_logic_syntax_pcfg_up_to_{sequence_length}_chars",
        pcfg=pcfg,
        batch_size=batch_size,
        max_derivation_length=sequence_length,
    )


def make_new_toy_english_syntax_pcfg(sequence_length: int, batch_size: int) -> Corpus:
    pcfg = {
        "S": ((("NP", "VP"), 1.0),),
        "NP": (
            (("N",), 0.7),
            (("NP", "RC"), 0.3),
        ),
        "RC": ((("NP", "Vtr"), 1.0),),
        "VP": (
            (("Vtr", "NP"), 0.4),
            (("Vi",), 0.4),
            (("Vcp", "that", "S"), 0.2),
        ),
        "N": ((("dogs",), 1),),  # Only dogs for simplicity
        "Vtr": ((("chase",), 1),),  # Only chase for simplicity
        "Vi": ((("sleep",), 1),),  # Only sleep for simplicity
        "Vcp": ((("think",), 1),),  # Only think for simplicity
    }
    return _make_corpus_from_pcfg(
        f"new_toy_english_syntax_pcfg_up_to_{sequence_length}_chars",
        pcfg=pcfg,
        batch_size=batch_size,
        max_derivation_length=sequence_length,
    )


def _make_toy_english_corpus(
    batch_size: int, exhaustive: bool, max_sequence_length: Optional[int]
) -> Corpus:
    pcfg = {
        "S": ((("NP", "VP"), 1.0),),
        "NP": (
            (("N",), 0.75),
            (("N", "RC"), 0.25),
        ),
        "RC": ((("NP", "Vtr"), 1.0),),
        "VP": (
            (("Vtr", "NP"), 0.34),
            (("Vi",), 0.33),
            (("Vcp", "S"), 0.33),
        ),
        "N": ((("dogs",), 1),),  # Only dogs for simplicity
        "Vtr": ((("chase",), 1),),  # Only chase for simplicity
        "Vi": ((("sleep",), 1),),  # Only sleep for simplicity
        "Vcp": ((("think that",), 1),),  # Only think for simplicity
    }
    corpus = _make_corpus_from_pcfg(
        f"golden_toy_english_pcfg_up_to_{max_sequence_length}_chars",
        pcfg=pcfg,
        batch_size=batch_size,
        max_derivation_length=max_sequence_length,
        exhaustive=exhaustive,
    )
    input_classes = np.argmax(corpus.input_sequence, axis=-1)
    deterministic_steps_mask = corpus.input_mask & (
        #  not 'chase' or 'dogs' - partial test, sometimes what comes after 'chase' is deterministic
        (input_classes != 1) & (input_classes != 2)
    )
    return dataclasses.replace(
        corpus, deterministic_steps_mask=deterministic_steps_mask
    )


def make_golden_toy_english_pcfg(
    sequence_length: Optional[int],
    batch_size: int,
    max_test_sequence_length: int = 20,  # max length of 20 already generates ~2k sequences
) -> Corpus:
    corpus = _make_toy_english_corpus(
        batch_size=batch_size, exhaustive=False, max_sequence_length=sequence_length
    )

    # Create exhaustive test corpus
    test_corpus = _make_toy_english_corpus(
        batch_size=1,  # Batch size is irrelevant for exhaustive=True
        exhaustive=True,
        max_sequence_length=max_test_sequence_length,
    )

    return dataclasses.replace(corpus, test_corpus=test_corpus)


def make_toy_english_syntax_no_sentencial_complement(
    sequence_length: int, batch_size: int
) -> Corpus:
    pcfg = {
        "S": ((("NP", "VP"), 1.0),),
        "NP": (
            (("N",), 0.7),
            (("NP", "RC"), 0.3),
        ),
        "RC": ((("NP", "Vtr"), 1.0),),
        "VP": ((("Vtr", "NP"), 0.5), (("Vi",), 0.5)),
        "N": ((("dogs",), 1),),  # Only dogs for simplicity
        "Vtr": ((("chase",), 1),),  # Only chase for simplicity
        "Vi": ((("sleep",), 1),),  # Only sleep for simplicity
    }
    return _make_corpus_from_pcfg(
        "toy_english_syntax_no_sentencial_complement",
        pcfg=pcfg,
        batch_size=batch_size,
        max_derivation_length=sequence_length,
    )


def make_toy_english_syntax_no_rc(sequence_length: int, batch_size: int) -> Corpus:
    pcfg = {
        "S": ((("NP", "VP"), 1.0),),
        "VP": (
            (("Vtr", "NP"), 0.4),
            (("Vi",), 0.4),
            (("Vcp", "that", "S"), 0.2),
        ),
        "NP": ((("dogs",), 1),),  # Only dogs for simplicity
        "Vtr": ((("chase",), 1),),  # Only chase for simplicity
        "Vi": ((("sleep",), 1),),  # Only sleep for simplicity
        "Vcp": ((("think",), 1),),  # Only think for simplicity
    }
    return _make_corpus_from_pcfg(
        "toy_english_syntax_no_rc",
        pcfg=pcfg,
        batch_size=batch_size,
        max_derivation_length=sequence_length,
    )


def make_center_embedding(
    batch_size: int, embedding_depth_probab: float, dependency_distance_probab: float
) -> Corpus:
    pcfg = {
        "S": (
            (("NP_s", "VP_s"), (1 - embedding_depth_probab) / 2),
            (("NP_p", "VP_p"), (1 - embedding_depth_probab) / 2),
            (("NP_s", "S", "VP_s"), embedding_depth_probab / 2),
            (("NP_p", "S", "VP_p"), embedding_depth_probab / 2),
        ),
        "NP_s": ((("N_s",), 1 - dependency_distance_probab),),
        "NP_p": ((("N_p",), 1 - dependency_distance_probab),),
        "VP_s": ((("V_s",), 1 - dependency_distance_probab),),
        "VP_p": ((("V_p",), 1 - dependency_distance_probab),),
        "N_s": (
            (("cat",), 1.0),
            (("dog",), 1.0),
        ),
        "N_p": (
            (("cats",), 1.0),
            (("dogs",), 1.0),
        ),
        "V_s": (
            (("runs",), 1.0),
            (("talks",), 1.0),
        ),
        "V_p": (
            (("run",), 1.0),
            (("talk",), 1.0),
        ),
    }
    corpus = _make_corpus_from_pcfg(
        f"center_embedding_pcfg_embedding_{embedding_depth_probab}_distance_{dependency_distance_probab}",
        pcfg=pcfg,
        batch_size=batch_size,
    )
    input_classes = np.argmax(corpus.input_sequence, axis=-1)
    deterministic_steps_mask = (~np.all(is_masked(corpus.input_sequence), axis=-1)) & (
        # "cat/s"
        (input_classes == 3) | (input_classes == 4)
    )
    return dataclasses.replace(
        corpus, deterministic_steps_mask=deterministic_steps_mask
    )


def make_palindrome_with_middle_marker_distinct(batch_size: int, nesting_probab: float):
    pcfg = {
        "S": (
            (("0", "S", "0"), nesting_probab / 2),
            (("1", "S", "1"), nesting_probab / 2),
            (("@",), 1 - nesting_probab),
        )
    }
    return _make_corpus_from_pcfg(
        name=f"palindrome_middle_marker__batch_{batch_size}__p_{nesting_probab}",
        pcfg=pcfg,
        batch_size=batch_size,
    )


def _optimal_d_g_for_fixed_palindrome(corpus) -> float:
    sequence_length = corpus.input_sequence.shape[1]
    batch_size = sum(corpus.sample_weights)
    deterministic_length = sequence_length // 2
    return batch_size * deterministic_length


def make_binary_palindrome_fixed_length(
    batch_size: int, sequence_length: int, train_set_ratio: float
) -> Corpus:
    assert sequence_length % 2 == 0
    prefixes_non_unique = np.random.randint(
        2, size=(batch_size, sequence_length // 2)
    ).astype(utils.FLOAT_DTYPE)

    sequence_counts = collections.Counter(list(map(tuple, prefixes_non_unique)))
    unique_prefixes, weights = list(zip(*sequence_counts.items()))

    prefixes = np.array(unique_prefixes)
    suffixes = np.flip(prefixes, axis=1)
    sequences = np.concatenate([prefixes, suffixes], axis=1)
    targets = _make_prediction_sequence(input_sequence=sequences, lookahead=1)

    input_sequences = np.expand_dims(sequences, axis=2)
    target_sequences = np.expand_dims(targets, axis=2)

    logger.info(
        f"Fixed palindrome: {len(unique_prefixes)}/{len(prefixes_non_unique)} unique sequences"
    )

    full_corpus = optimize_for_feeding(
        Corpus(
            name=f"palindrome_binary_fixed_length_batch_{batch_size}_length_{sequence_length}",
            input_sequence=input_sequences,
            target_sequence=target_sequences,
            sample_weights=weights,
        )
    )

    train, test = split_train_test(full_corpus, train_ratio=train_set_ratio)
    logger.info(
        f"Train size: {train.input_sequence.shape[0]}, test size: {test.input_sequence.shape[0]}"
    )
    test = dataclasses.replace(
        test, optimal_d_given_g=_optimal_d_g_for_fixed_palindrome(test)
    )
    return dataclasses.replace(
        train,
        test_corpus=test,
        optimal_d_given_g=_optimal_d_g_for_fixed_palindrome(train),
    )


def _make_an_bn_square_corpus(n_values: tuple[int, ...], prior: float):
    max_n = max(n_values)
    max_sequence_length = max_n + (max_n**2) + 1

    n_values_counts = collections.Counter(n_values)
    unique_n_values, n_values_weights = tuple(zip(*n_values_counts.items()))

    inputs = np.empty((len(unique_n_values), max_sequence_length))
    targets = np.empty_like(inputs)
    inputs.fill(MASK_VALUE)
    targets.fill(MASK_VALUE)

    for b, n in enumerate(unique_n_values):
        input_seq = [_SEQ_START_OR_END] + ([_A] * n) + ([_B] * n**2)
        target_seq = input_seq[1:] + [_SEQ_START_OR_END]
        inputs[b, : len(input_seq)] = input_seq
        targets[b, : len(input_seq)] = target_seq

    corpus = make_one_hot_corpus(
        f"an_bn_square_batch_{len(n_values)}_p_{prior}",
        inputs,
        targets,
        num_input_classes=3,
        num_target_classes=3,
        vocabulary=_make_identical_input_output_vocabulary(alphabet=("#", "a", "b")),
        weights=n_values_weights,
    )
    return dataclasses.replace(
        corpus,
        optimal_d_given_g=_get_ain_bjn_ckn_dtn_optimal_d_given_g(
            prior, n_values, corpus.sample_weights
        ),
    )


def make_an_bn_square(batch_size: int, prior: float) -> Corpus:
    training_n_values = _sample_geometric(prior, batch_size)
    training_corpus = _make_an_bn_square_corpus(training_n_values, prior)

    max_training_n = max(training_n_values)
    test_n_values = tuple(range(max_training_n + 1, max_training_n + 11))
    test_corpus = _make_an_bn_square_corpus(test_n_values, prior)
    return dataclasses.replace(training_corpus, test_corpus=test_corpus)


def _make_identical_input_output_vocabulary(alphabet: tuple[str, ...]) -> _Vocabulary:
    # Create class index to symbol mapping, assuming inputs and outputs are identical and ordered identically.
    class_to_symbol = {idx: alphabet[idx] for idx in range(len(alphabet))}
    class_to_symbol.update(
        {idx + len(alphabet): symbol for idx, symbol in class_to_symbol.items()}
    )
    return class_to_symbol


def _get_ain_bjn_ckn_dtn_optimal_d_given_g(
    prior: float, n_values: tuple[int, ...], sample_weights: tuple[float, ...]
) -> float:
    return -np.sum(
        [
            weight * ((n - 1) * np.log2(1 - prior) + np.log2(prior))
            for n, weight in zip(n_values, sample_weights)
        ]
    ).item()


def get_num_chars_in_corpus(corpus: Corpus) -> int:
    non_masked = ~np.all(is_masked(corpus.input_sequence), axis=-1)
    num_chars_per_row = np.sum(non_masked, axis=1)
    if corpus.sample_weights:
        total_chars = np.dot(num_chars_per_row, corpus.sample_weights)
    else:
        total_chars = np.sum(num_chars_per_row)
    return total_chars.item()


def _make_an_bm_cn_dm(
    n_values: tuple[int, ...],
    m_values: tuple[int, ...],
    prior: float,
    limit_vocabulary: bool,
    sort_by_length: bool,
) -> Corpus:
    max_n = max(n_values)
    max_m = max(m_values)

    max_sequence_length = (2 * max_n) + (2 * max_m) + 1

    n_m_values_counts = collections.Counter(zip(n_values, m_values))
    n_m_values_counts_items = tuple(n_m_values_counts.items())

    if sort_by_length:
        n_m_values_counts_items = sorted(
            n_m_values_counts_items, key=lambda x: sum(x[0]), reverse=True
        )

    unique_n_m_values, n_m_values_weights = tuple(zip(*n_m_values_counts_items))

    inputs = np.empty((len(unique_n_m_values), max_sequence_length))
    targets = np.empty_like(inputs)
    inputs.fill(MASK_VALUE)
    targets.fill(MASK_VALUE)

    if limit_vocabulary:
        c_symbol = _A
        d_symbol = _B
        alphabet = (
            "#",
            "a",
            "b",
        )
        corpus_name = "an_bm_an_bm"
    else:
        alphabet = (
            "#",
            "a",
            "b",
            "c",
            "d",
        )
        c_symbol = _C
        d_symbol = _D
        corpus_name = "an_bm_cn_dm"

    for i in range(len(unique_n_m_values)):
        n, m = unique_n_m_values[i]
        input_seq = (
            [_SEQ_START_OR_END]
            + ([_A] * n)
            + ([_B] * m)
            + ([c_symbol] * n)
            + ([d_symbol] * m)
        )
        target_seq = input_seq[1:] + [_SEQ_START_OR_END]
        inputs[i, : len(input_seq)] = input_seq
        targets[i, : len(input_seq)] = target_seq

    corpus = make_one_hot_corpus(
        f"{corpus_name}__batch_{len(n_values)}_p_{prior}",
        inputs,
        targets,
        num_input_classes=len(alphabet),
        num_target_classes=len(alphabet),
        weights=n_m_values_weights,
        vocabulary=_make_identical_input_output_vocabulary(alphabet=alphabet),
    )

    unique_n_plus_m_values = tuple(map(sum, unique_n_m_values))
    deterministic_steps_mask = np.zeros_like(inputs, dtype=bool)
    for i, n_plus_m in enumerate(unique_n_plus_m_values):
        deterministic_steps_mask[i, n_plus_m + 1 :] = True
    deterministic_steps_mask &= ~is_masked(inputs)

    return dataclasses.replace(
        corpus,
        optimal_d_given_g=_get_an_bm_cn_plus_m_corpus_optimal_d_given_g(
            n_plus_m_values=tuple(np.add(n_values, m_values)), prior=prior
        ),
        deterministic_steps_mask=deterministic_steps_mask,
    )


def make_an_bm_cn_dm(
    batch_size: int, prior: float, limit_vocabulary: bool, sort_by_length: bool = False
) -> Corpus:
    training_n_values = _sample_geometric(prior, batch_size)
    training_m_values = _sample_geometric(prior, batch_size)

    training_corpus = _make_an_bm_cn_dm(
        training_n_values, training_m_values, prior, limit_vocabulary, sort_by_length
    )
    max_n = max(training_n_values)
    max_m = max(training_m_values)
    max_training_n_or_m = max(max_n, max_m)

    test_n_values, test_m_values = zip(
        *itertools.product(
            range(max_training_n_or_m + 1, max_training_n_or_m + 51), repeat=2
        )
    )

    test_corpus = _make_an_bm_cn_dm(
        test_n_values, test_m_values, prior, limit_vocabulary, sort_by_length
    )
    test_corpus = dataclasses.replace(test_corpus, name=f"{test_corpus.name}_test")

    logger.info(f"Created corpus {training_corpus.name}")
    logger.info(f"Max n,m in training: {max_n}, {max_m}")
    logger.info(f"Optimal training set D:G: {training_corpus.optimal_d_given_g}")
    logger.info(f"Optimal test set D:G: {test_corpus.optimal_d_given_g}")
    logger.info(f"Training set dimensions: {training_corpus.input_sequence.shape}")
    logger.info(f"Test set dimensions: {test_corpus.input_sequence.shape}")

    return dataclasses.replace(training_corpus, test_corpus=test_corpus)


def _get_ain_bjn_ckn_dtn_sequence_probability(n: int, prior: float) -> float:
    """
    The probability is like choosing n times "a" and then "b". From there the probability is 1.
    """
    return (1 - prior) ** n * prior


def _make_ain_bjn_ckn_dtn_corpus(
    n_values: tuple[int, ...],
    multipliers: tuple[int, int, int, int],
    prior: float,
    sort_by_length: bool,
    apply_sequence_probability_to_weights: bool = True,
) -> Corpus:
    """
    Create a corpus of a^in, b^jn, c^kn, d^tn, multipliers = [i,j,k,n].
    Sample weights will be set to the probability of each sequence by default, unless
    apply_sequence_probability_to_weights is False (for example in cases where the values in n_values are sampled).
    """
    #
    max_n = max(n_values)
    max_sequence_length = (max_n * sum(multipliers)) + 1

    n_values_counts = collections.Counter(n_values)
    n_value_counts_items = tuple(n_values_counts.items())
    if sort_by_length:
        n_value_counts_items = sorted(n_value_counts_items, reverse=True)

    unique_n_values, n_values_weights = tuple(zip(*n_value_counts_items))

    # Multiply the n_values_weights by the probability of the sequence
    n_values_weights = tuple(
        (
            weight * _get_ain_bjn_ckn_dtn_sequence_probability(n - 1, prior)
            if apply_sequence_probability_to_weights
            else weight
        )
        for n, weight in zip(unique_n_values, n_values_weights)
    )

    inputs = np.empty((len(unique_n_values), max_sequence_length))
    targets = np.empty_like(inputs)
    inputs.fill(MASK_VALUE)
    targets.fill(MASK_VALUE)

    for b, n in enumerate(unique_n_values):
        input_seq = (
            [_SEQ_START_OR_END]
            + ([_A] * n * multipliers[0])
            + ([_B] * n * multipliers[1])
            + ([_C] * n * multipliers[2])
            + ([_D] * n * multipliers[3])
        )
        target_seq = input_seq[1:] + [_SEQ_START_OR_END]
        inputs[b, : len(input_seq)] = input_seq
        targets[b, : len(input_seq)] = target_seq

    name = f"a{multipliers[0]}n_b{multipliers[1]}n_c{multipliers[2]}n_d{multipliers[3]}n__p_{prior}__batch_{len(n_values)}"
    num_input_classes = sum([1 for x in multipliers if x != 0]) + 1

    alphabet = ("#", "a", "b", "c", "d")[:num_input_classes]
    vocabulary = _make_identical_input_output_vocabulary(alphabet)

    corpus = make_one_hot_corpus(
        name,
        inputs,
        targets,
        num_input_classes=num_input_classes,
        num_target_classes=num_input_classes,
        weights=n_values_weights,
        vocabulary=vocabulary,
    )
    return dataclasses.replace(
        corpus,
        optimal_d_given_g=_get_ain_bjn_ckn_dtn_optimal_d_given_g(
            prior, unique_n_values, corpus.sample_weights
        ),
        deterministic_steps_mask=(~is_masked(inputs))
        & (inputs != _SEQ_START_OR_END)
        & (inputs != _A),
    )


def make_ain_bjn_ckn_dtn_range(
    training_n_start: int,
    training_n_stop: int,
    test_n_start: int,
    test_n_stop: int,
    multipliers: tuple[int, int, int, int],
    sort_by_length: bool,
) -> Corpus:
    # TODO: figure out correct prior.
    training_n_values = tuple(range(training_n_start, training_n_stop + 1))
    training_corpus = _make_ain_bjn_ckn_dtn_corpus(
        training_n_values, multipliers, prior=0.5, sort_by_length=sort_by_length
    )
    training_corpus = dataclasses.replace(
        training_corpus,
        name=f"a{multipliers[0]}n_b{multipliers[1]}n_c{multipliers[2]}n_d{multipliers[3]}n__range_{training_n_start}_{training_n_stop}",
    )

    test_n_values = tuple(range(test_n_start, test_n_stop + 1))
    test_corpus = _make_ain_bjn_ckn_dtn_corpus(
        test_n_values, multipliers, prior=0.5, sort_by_length=sort_by_length
    )
    test_corpus = dataclasses.replace(test_corpus, name=f"{test_corpus.name}_test")

    logger.info(f"Created corpus {training_corpus.name}")
    logger.info(f"Training n values: {training_n_start}-{training_n_stop}")
    logger.info(f"Test n values: {test_n_start}-{test_n_stop}")

    return dataclasses.replace(training_corpus, test_corpus=test_corpus)


def _get_sequence_lengths(corpus) -> np.ndarray:
    # Assuming all sequence end with end-of-sequence symbol.
    where_end_of_sequence = np.where(
        corpus.target_sequence[:, :, _SEQ_START_OR_END] == 1
    )
    sequence_lengths = where_end_of_sequence[1].astype(int)
    return sequence_lengths


def _make_an_bn_pcfg(prior: float):
    return {
        # n > 0
        "S": ((("a", "X", "b"), 1),),
        "X": (
            (("a", "X", "b"), 1 - prior),
            (("",), prior),
        ),
    }


def _make_an_bn_from_cfg(batch_size: int, prior: float, sort_by_length: bool):
    an_bn_pcfg = _make_an_bn_pcfg(prior)
    corpus = _make_corpus_from_pcfg(
        name=f"an_bn_prior_{prior}_batch_{batch_size}",
        pcfg=an_bn_pcfg,
        batch_size=batch_size,
        sort_by_length=sort_by_length,
    )
    n_values = (_get_sequence_lengths(corpus) / 2).tolist()
    input_classes = np.argmax(corpus.input_sequence, axis=-1)

    return dataclasses.replace(
        corpus,
        optimal_d_given_g=_get_ain_bjn_ckn_dtn_optimal_d_given_g(
            prior, n_values, corpus.sample_weights
        ),
        deterministic_steps_mask=corpus.input_mask
        & (input_classes != _SEQ_START_OR_END)
        & (input_classes != _A),
    )


def make_an_bn(
    batch_size: int,
    prior: float,
    sort_by_length: bool,
):
    training_corpus = _make_an_bn_from_cfg(
        batch_size=batch_size, prior=prior, sort_by_length=sort_by_length
    )
    training_n_values = _get_sequence_lengths(training_corpus) / 2
    max_training_n = int(training_n_values.max())

    test_n_values = tuple(range(1, max_training_n + 1001))
    test_corpus = _make_ain_bjn_ckn_dtn_corpus(
        test_n_values, (1, 1, 0, 0), prior, sort_by_length
    )
    test_corpus = dataclasses.replace(test_corpus, name=f"{test_corpus.name}_test")

    logger.info(f"Created corpus {training_corpus.name}")
    logger.info(f"Max n in training set: {max_training_n}")
    logger.info(f"Max n in test set: {max(test_n_values)}")
    logger.info(f"Optimal training set D:G: {training_corpus.optimal_d_given_g}")
    logger.info(f"Optimal test set D:G: {test_corpus.optimal_d_given_g}")

    return dataclasses.replace(training_corpus, test_corpus=test_corpus)


def make_ain_bjn_ckn_dtn(
    batch_size: int,
    prior: float,
    multipliers: tuple[int, int, int, int],
    sort_by_length: bool = False,
) -> Corpus:
    training_n_values = _sample_geometric(prior, batch_size)
    training_corpus = _make_ain_bjn_ckn_dtn_corpus(
        training_n_values,
        multipliers,
        prior,
        sort_by_length,
        apply_sequence_probability_to_weights=False,
    )

    max_training_n = max(training_n_values)
    test_n_values = tuple(range(1, max_training_n + 1001))
    test_corpus = _make_ain_bjn_ckn_dtn_corpus(
        test_n_values, multipliers, prior, sort_by_length
    )
    test_corpus = dataclasses.replace(test_corpus, name=f"{test_corpus.name}_test")

    logger.info(f"Created corpus {training_corpus.name}")
    logger.info(f"Max n in training set: {max_training_n}")
    logger.info(f"Max n in test set: {max(test_n_values)}")
    logger.info(f"Optimal training set D:G: {training_corpus.optimal_d_given_g}")
    logger.info(f"Optimal test set D:G: {test_corpus.optimal_d_given_g}")

    return dataclasses.replace(training_corpus, test_corpus=test_corpus)


def _get_an_bm_cn_plus_m_corpus_optimal_d_given_g(
    n_plus_m_values: tuple[int, ...], prior: float
) -> float:
    return -np.sum(
        [
            ((n_plus_m - 2) * np.log2(1 - prior)) + (2 * np.log2(prior))
            for n_plus_m in n_plus_m_values
        ]
    ).item()


def _make_an_bm_cn_plus_m_corpus(
    n_values: tuple[int, ...],
    m_values: tuple[int, ...],
    prior: float,
    sort_by_length: bool,
) -> Corpus:
    sum_values = tuple(np.add(n_values, m_values))
    max_sequence_length = 2 * max(sum_values) + 1

    n_m_values_counts = collections.Counter(zip(n_values, m_values))
    n_m_values_counts_items = tuple(n_m_values_counts.items())
    if sort_by_length:
        n_m_values_counts_items = sorted(
            n_m_values_counts_items, key=lambda x: sum(x[0]), reverse=True
        )

    unique_n_m_values, n_m_values_weights = tuple(zip(*n_m_values_counts_items))

    inputs = np.empty((len(unique_n_m_values), max_sequence_length))
    targets = np.empty_like(inputs)
    inputs.fill(MASK_VALUE)
    targets.fill(MASK_VALUE)

    for b in range(len(unique_n_m_values)):
        n, m = unique_n_m_values[b]
        input_seq = [_SEQ_START_OR_END] + ([_A] * n) + ([_B] * m) + ([_C] * (n + m))
        target_seq = input_seq[1:] + [_SEQ_START_OR_END]
        inputs[b, : len(input_seq)] = input_seq
        targets[b, : len(input_seq)] = target_seq

    vocabulary = _make_identical_input_output_vocabulary(alphabet=("#", "a", "b", "c"))

    corpus = make_one_hot_corpus(
        f"an_bm_cn_plus_m__batch_{len(n_values)}_p_{prior}",
        inputs,
        targets,
        num_input_classes=4,
        num_target_classes=4,
        weights=n_m_values_weights,
        vocabulary=vocabulary,
    )

    return _add_meta_data_to_n_plus_m_corpus(
        corpus, prior=prior, unweighted_n_plus_m_values=sum_values
    )


def _add_meta_data_to_n_plus_m_corpus(
    corpus, prior, unweighted_n_plus_m_values
) -> Corpus:
    input_classes = np.argmax(corpus.input_sequence, axis=-1)
    return dataclasses.replace(
        corpus,
        optimal_d_given_g=_get_an_bm_cn_plus_m_corpus_optimal_d_given_g(
            unweighted_n_plus_m_values, prior
        ),
        deterministic_steps_mask=corpus.input_mask
        & (input_classes != _SEQ_START_OR_END)
        & (input_classes != _A)
        & (input_classes != _B),
    )


def make_an_bm_cn_plus_m_from_range(
    start_n: int,
    num_strings: int,
    sort_by_length: bool,
):
    sqrt = math.ceil(math.sqrt(num_strings))
    n_m_values = sorted(
        itertools.product(range(start_n, start_n + sqrt), repeat=2),
        key=sum,
    )[:num_strings]
    n_values, m_values = zip(*n_m_values)
    logger.info(f"N training range: {n_values[0]}-{n_values[-1]}")
    logger.info(f"M training range: {m_values[0]}-{m_values[-1]}")
    # TODO figure out correct prior.
    train_corpus = _make_an_bm_cn_plus_m_corpus(
        n_values, m_values, prior=0.5, sort_by_length=sort_by_length
    )

    max_training_n_or_m = get_max_n_m_values_for_n_plus_m(train_corpus)
    test_n_values, test_m_values = zip(
        *itertools.product(
            range(max_training_n_or_m + 1, max_training_n_or_m + 51), repeat=2
        )
    )
    test_corpus = _make_an_bm_cn_plus_m_corpus(
        test_n_values, test_m_values, prior=0.5, sort_by_length=sort_by_length
    )

    logger.info(f"N test range: {test_n_values[0]}-{test_n_values[-1]}")
    logger.info(f"M training range: {test_m_values[0]}-{test_m_values[-1]}")

    logger.info(f"Created a^nb^mc^n+m corpus. Num strings: {num_strings}.")

    # TODO fix corpus names.
    test_corpus = dataclasses.replace(test_corpus, name=f"{test_corpus.name}_test")
    return dataclasses.replace(
        train_corpus,
        test_corpus=test_corpus,
        name=f"an_bm_cn_plus_m_range_start_{start_n}_batch_{num_strings}",
    )


def _get_n_m_values(corpus):
    n_values = (corpus.input_sequence[:, :, 1] == 1).sum(axis=1)
    m_values = (corpus.input_sequence[:, :, 2] == 1).sum(axis=1)
    return n_values, m_values


def get_max_n_m_values_for_n_plus_m(corpus):
    n_values, m_values = _get_n_m_values(corpus)

    max_n = max(n_values)
    max_m = max(m_values)
    max_n_or_m = max(max_n, max_m)
    max_pair = sorted(list(zip(n_values, m_values)))[-1]
    logger.info(f"Max n={max_n}, Max m={max_m}, max={max_n_or_m}, max pair={max_pair}")
    return max_n_or_m


def get_n_values_for_an_bn_cn_dn(corpus):
    training_n_values = (corpus.input_sequence[:, :, _A] == 1).sum(axis=1)
    return tuple(training_n_values)


def make_an_bm_cn_plus_m_from_cfg(
    batch_size: int,
    prior: float,
    sort_by_length: bool = False,
):
    pcfg = {
        # n, m > 1
        "S": ((("a", "X", "c"), 1),),
        "X": (
            (("a", "X", "c"), 1 - prior),
            (("Y",), prior),
        ),
        "Y": ((("b", "Z", "c"), 1),),
        "Z": (
            (("b", "Z", "c"), 1 - prior),
            (("",), prior),
        ),
    }
    training_corpus = _make_corpus_from_pcfg(
        name=f"an_bm_c_n_plus_m_p__{prior}_batch_{batch_size}",
        pcfg=pcfg,
        batch_size=batch_size,
        sort_by_length=sort_by_length,
    )
    n_values, m_values = _get_n_m_values(training_corpus)
    n_plus_m_values = np.sum((n_values, m_values), axis=0).tolist()
    assert len(n_plus_m_values) == len(training_corpus.sample_weights)
    unweighted_n_plus_m_values = [
        [n_plus_m_values[i]] * training_corpus.sample_weights[i]
        for i in range(len(n_plus_m_values))
    ]
    unweighted_n_plus_m_values_flattened = tuple(
        itertools.chain(*unweighted_n_plus_m_values)
    )

    training_corpus = _add_meta_data_to_n_plus_m_corpus(
        training_corpus,
        prior=prior,
        unweighted_n_plus_m_values=unweighted_n_plus_m_values_flattened,
    )

    max_training_n_or_m = get_max_n_m_values_for_n_plus_m(training_corpus)
    test_n_values, test_m_values = zip(
        *itertools.product(
            range(max_training_n_or_m + 1, max_training_n_or_m + 51), repeat=2
        )
    )

    test_corpus = _make_an_bm_cn_plus_m_corpus(
        test_n_values, test_m_values, prior, sort_by_length
    )
    test_corpus = dataclasses.replace(test_corpus, name=f"{test_corpus.name}_test")
    logger.info(
        f"Test n,m values: {min(test_n_values)}<=n<={max(test_n_values)}, {min(test_m_values)}<=m<={max(test_m_values)}, max sum: {max(test_n_values) + max(test_m_values)}"
    )
    return dataclasses.replace(training_corpus, test_corpus=test_corpus)


def make_an_bm_cn_plus_m_from_geometric_sample(
    batch_size: int,
    prior: float,
    sort_by_length: bool = False,
) -> Corpus:
    training_n_values = _sample_geometric(prior, batch_size)
    training_m_values = _sample_geometric(prior, batch_size)

    training_corpus = _make_an_bm_cn_plus_m_corpus(
        training_n_values, training_m_values, prior, sort_by_length
    )
    max_n = max(training_n_values)
    max_m = max(training_m_values)
    max_training_n_or_m = max(max_n, max_m)

    test_n_values, test_m_values = zip(
        *itertools.product(
            range(max_training_n_or_m + 1, max_training_n_or_m + 51), repeat=2
        )
    )

    test_corpus = _make_an_bm_cn_plus_m_corpus(
        test_n_values, test_m_values, prior, sort_by_length
    )
    test_corpus = dataclasses.replace(test_corpus, name=f"{test_corpus.name}_test")

    logger.info(f"Created corpus {training_corpus.name}")
    logger.info(f"Max n in training: {max_n}")
    logger.info(f"Max m in training: {max_m}")
    logger.info(f"Optimal training set D:G: {training_corpus.optimal_d_given_g}")
    logger.info(f"Optimal test set D:G: {test_corpus.optimal_d_given_g}")
    logger.info(f"Training set dimensions: {training_corpus.input_sequence.shape}")
    logger.info(f"Test set dimensions: {test_corpus.input_sequence.shape}")

    return dataclasses.replace(training_corpus, test_corpus=test_corpus)


def _calculate_dyck_nesting_depths(corpus):
    input_classes = corpus.input_sequence.argmax(axis=-1)
    opening_classes = {1, 3}
    closing_classes = {2, 4}
    depths = []
    for b in range(input_classes.shape[0]):
        depth = 0
        max_depth = 0
        for i in range(input_classes[b].shape[0]):
            if np.all(is_masked(corpus.input_sequence[b, i])):
                break
            if input_classes[b, i] in opening_classes:
                depth += 1
                max_depth = max(max_depth, depth)
            elif input_classes[b, i] in closing_classes:
                depths.append(depth)
                depth -= 1

    depth_counts = dict(collections.Counter(depths).most_common())
    max_depth = max(depths)
    logger.info(f"Max depth in corpus: {max_depth}")
    logger.info(f"Depth counts: {depth_counts}")


def _get_dyck_1_symbol_counts(corpus) -> tuple[int, int, int]:
    # Masks (nans) become 0 here after argmax, but we ignore it since we count only 1's and 2's.
    input_classes = corpus.input_sequence.argmax(axis=-1)

    num_ends_of_sequence = np.sum(corpus.sample_weights).item()
    num_opening_brackets = np.dot(
        corpus.sample_weights, np.sum(input_classes == 1, axis=1)
    ).item()
    num_closing_brackets = np.dot(
        corpus.sample_weights, np.sum(input_classes == 2, axis=1)
    ).item()

    return num_ends_of_sequence, num_opening_brackets, num_closing_brackets


def get_dyck_valid_targets(corpus, n) -> np.ndarray:
    input_classes = np.argmax(corpus.input_sequence, axis=-1).astype(utils.FLOAT_DTYPE)
    input_classes[~corpus.input_mask] = np.nan

    valid_targets = np.zeros_like(corpus.target_sequence)
    valid_targets[~corpus.input_mask] = np.nan

    opening_bracket_idxs = [2 * i + 1 for i in range(n)]  # ( = 1, < = 3, [ = 5, { = 7
    stack = []

    for b in range(input_classes.shape[0]):
        for i in range(input_classes.shape[1]):
            current_class = input_classes[b, i]
            if np.isnan(current_class):
                break
            current_class = int(current_class)
            valid_targets[b, i, opening_bracket_idxs] = 1  # Always true.
            if current_class == _SEQ_START_OR_END:
                valid_targets[b, i, 0] = 1
            elif current_class in opening_bracket_idxs:
                stack = [current_class] + stack
            else:
                stack = stack[1:]

            if len(stack) > 0:
                valid_targets[b, i, stack[0] + 1] = 1
            else:
                valid_targets[b, i, 0] = 1

    return valid_targets


def _load_dyck_strings_in_range(dyck_n: int, range_start_idx: int, batch_size: int):
    with open(f"dyck_{dyck_n}.txt", "r") as f:
        lines = f.readlines()[range_start_idx : range_start_idx + batch_size]
    return [x.strip() for x in lines]


def _load_dyck_strings_in_length(dyck_n: int, min_length: int, batch_size: int):
    with open(f"dyck_{dyck_n}.txt", "r") as f:
        lines = [x.strip() for x in f.readlines()]
    start_idx = 0
    for i, line in enumerate(lines):
        if len(line) == min_length:
            start_idx = i
    return lines[start_idx : start_idx + batch_size]


def _make_dyck_n_strings_for_length(
    target_length: int, n: int, batch_size: int
) -> List[str]:
    if target_length == 0:
        return [""]  # Include the empty string as a valid Dyck string
    if target_length % 2 != 0:
        return []  # Dyck strings must have even length

    bracket_pairs = _DYCK_BRACKET_PAIRS[:n]
    results = []
    stack = []

    def backtrack(current: List[str], open_count: int, close_count: int) -> None:
        if len(results) >= batch_size:
            # Stop further processing if batch_size is reached
            return

        if len(current) == target_length:
            if open_count == close_count:
                results.append("".join(current))
            return

        # If we can add an opening bracket, try all types
        if open_count < target_length // 2:
            for idx, (open_br, _) in enumerate(bracket_pairs):
                current.append(open_br)
                stack.append(idx)  # Keep track of which bracket type was opened
                backtrack(current, open_count + 1, close_count)
                stack.pop()
                current.pop()

        # If we can add a closing bracket, close the most recent open bracket
        if close_count < open_count and stack:
            last_bracket_idx = stack[-1]
            _, close_br = bracket_pairs[last_bracket_idx]
            current.append(close_br)
            stack.pop()
            backtrack(current, open_count, close_count + 1)
            stack.append(last_bracket_idx)
            current.pop()

    backtrack([], 0, 0)
    return results


def _make_dyck_n_strings_from_range(
    min_length: int, max_length: int, n: int, batch_size: int
) -> List[str]:
    results = []
    for length in range(min_length, max_length + 1, 2):
        dyck_strings = _make_dyck_n_strings_for_length(
            length, n, batch_size - len(results)
        )
        results.extend(dyck_strings)
    return results


def _make_dyck_n_from_range(
    n: int,
    batch_size: int,
    sort_by_length: bool,
    min_length: int,
    max_length: Optional[int],
):
    strings = _make_dyck_n_strings_from_range(
        min_length=min_length, max_length=max_length, n=n, batch_size=batch_size
    )
    return _corpus_from_strings(
        name=f"dyck_{n}",
        strings=strings,
        sort_by_length=sort_by_length,
    )


def _make_dyck_n_from_pcfg(
    batch_size: int,
    nesting_probab: float,
    n: int,
    exhaustive: bool = False,
    max_sequence_length: Optional[int] = None,
    sort_by_length: bool = False,
):
    single_nesting_probab = nesting_probab / n

    bracket_derivations = []
    for i in range(n):
        bracket_derivations.append(
            # e.g. `S -> ("[", S, "]", S)`.
            (
                (_DYCK_BRACKET_PAIRS[i][0], "S", _DYCK_BRACKET_PAIRS[i][1], "S"),
                single_nesting_probab,
            )
        )

    pcfg = {"S": tuple(bracket_derivations) + (("", 1 - nesting_probab),)}
    return _make_corpus_from_pcfg(
        name=f"dyck_{n}__batch_{batch_size}__p_{nesting_probab}",
        pcfg=pcfg,
        batch_size=batch_size,
        exhaustive=exhaustive,
        max_derivation_length=max_sequence_length,
        sort_by_length=sort_by_length,
    )


def make_dyck_n_corpus(
    batch_size: int,
    nesting_probab: float,
    n: int,
    exhaustive: bool = False,
    max_sequence_length: Optional[int] = None,
    sort_by_length: bool = False,
):
    corpus = _make_dyck_n_from_pcfg(
        batch_size=batch_size,
        nesting_probab=nesting_probab,
        n=n,
        exhaustive=exhaustive,
        max_sequence_length=max_sequence_length,
        sort_by_length=sort_by_length,
    )

    _calculate_dyck_nesting_depths(corpus)
    return corpus


def _get_sequence_strings(corpus) -> frozenset[str]:
    unique_sequences = set()
    for b in range(corpus.input_sequence.shape[0]):
        seq = corpus.input_sequence[b]
        seq = seq[~np.isnan(seq).all(axis=1)]
        seq_str = str(np.argmax(seq, axis=-1).tolist())
        unique_sequences.add(seq_str)
    return frozenset(unique_sequences)


def make_dyck_n(
    batch_size: int,
    nesting_probab: float,
    n: int,
    max_sequence_length: Optional[int] = None,
    sort_by_length: bool = False,
    max_test_sequence_length: int = 10,
) -> Corpus:
    training_corpus = make_dyck_n_corpus(
        batch_size=batch_size,
        nesting_probab=nesting_probab,
        n=n,
        exhaustive=False,
        max_sequence_length=max_sequence_length,
        sort_by_length=sort_by_length,
    )

    # Create exhaustive test corpus
    test_corpus = make_dyck_n_corpus(
        batch_size=1,  # Batch size is irrelevant for exhaustive=True
        nesting_probab=nesting_probab,
        n=n,
        exhaustive=True,
        max_sequence_length=max_test_sequence_length,
        sort_by_length=sort_by_length,
    )

    training_sequences = _get_sequence_strings(training_corpus)
    test_sequences = _get_sequence_strings(test_corpus)
    shared = training_sequences & test_sequences

    logger.info(
        f"Dyck-{n} Sequences shared between train and test: {len(shared)} ({len(shared) / len(test_sequences):.2f} of test)"
    )

    return dataclasses.replace(training_corpus, test_corpus=test_corpus)


def make_mnist_corpus(samples_limit=None, width_height: int = 16) -> Corpus:
    import mnist

    train_images = np.expand_dims(
        utils.preprocess_mnist_images(mnist.train_images(), width_height=width_height),
        1,
    )
    test_images = np.expand_dims(utils.preprocess_mnist_images(mnist.test_images()), 1)
    train_targets = _make_one_hot_sequence(
        mnist.train_labels().reshape(-1, 1), num_classes=10
    )
    test_targets = _make_one_hot_sequence(
        mnist.test_labels().reshape(-1, 1), num_classes=10
    )

    if samples_limit is not None:
        # Samples are already shuffled.
        train_images = train_images[:samples_limit]
        train_targets = train_targets[:samples_limit]

    test_corpus = Corpus(
        f"mnist_test_{test_images.shape[0]}_samples",
        input_sequence=test_images,
        target_sequence=test_targets,
    )

    return Corpus(
        f"mnist_{train_images.shape[0]}_samples",
        input_sequence=train_images,
        target_sequence=train_targets,
        test_corpus=test_corpus,
    )


def make_circle(num_samples: int = 100, degrees_of_freedom: int = 8) -> Corpus:
    input_batch = []
    target_batch = []

    num_buckets = 2**degrees_of_freedom

    assert num_samples <= num_buckets, (num_samples, num_buckets)

    linspace = np.linspace(-1, 1, num=num_buckets)

    for bucket, n in enumerate(linspace):
        if n != 0:
            x = 1 / n
        else:
            x = 0

        rad = x * np.pi

        x = np.cos(rad)
        y = np.sin(rad)

        input_binary = base10_to_binary_vector(
            bucket, sequence_length=degrees_of_freedom
        )
        input_batch.append(input_binary)
        target_batch.append([x, y])

    random.shuffle(target_batch)

    sample_idxs = np.random.choice(range(len(input_batch)), num_samples)

    input_sequence = np.expand_dims(np.array(input_batch), axis=1)[sample_idxs]
    target_sequence = np.expand_dims(np.array(target_batch), axis=1)[sample_idxs]

    return Corpus(
        "circle", input_sequence=input_sequence, target_sequence=target_sequence
    )


def make_circle_through_time(num_samples: int = 250, batch_size: int = 100) -> Corpus:
    input_sequence = []

    # TODO Random points on the circle.
    for i in np.linspace(0, 2, num=num_samples):
        x = np.cos(i * np.pi)
        y = np.sin(i * np.pi)
        input_sequence.append([x, y])

    target_sequence = input_sequence[1:] + [input_sequence[0]]

    input_batch = np.expand_dims(np.array(input_sequence), axis=0)
    target_batch = np.expand_dims(np.array(target_sequence), axis=0)

    return Corpus(
        "circle_through_time", input_sequence=input_batch, target_sequence=target_batch
    )


def make_circle_through_time_bitmap(
    num_samples: int = 250,
    grid_size: int = 25,
    visualize: bool = False,
) -> Corpus:
    input_coordinates = []

    for i in np.linspace(0, 2, num=num_samples):
        # Caretsian to 0-corner coordinates.
        x = np.cos(i * np.pi) * 0.5 + 0.5
        y = np.sin(i * np.pi) * 0.5 + 0.5
        input_coordinates.append([x, y])

    bitmap_inputs = []
    square_size = 1 / grid_size

    full_circle_bitmap = np.zeros((grid_size, grid_size))

    for x, y in input_coordinates:
        x_square_idx = min(grid_size - 1, int(x // square_size))
        y_square_idx = min(grid_size - 1, int(y // square_size))
        bitmap = np.zeros((grid_size, grid_size), dtype=np.uint8)
        bitmap[y_square_idx, x_square_idx] = 1.0

        bitmap_inputs.append(bitmap.flatten())

        full_circle_bitmap[y_square_idx, x_square_idx] = 1.0

    inputs = np.expand_dims(np.array(bitmap_inputs), axis=1)
    targets = np.concatenate([inputs[1:], np.expand_dims(inputs[1], axis=0)], axis=0)

    if visualize:
        import matplotlib.pyplot as plt

        for i in range(num_samples):
            batch_data = np.zeros((grid_size, grid_size))

            input_data = inputs[i, 0].reshape((grid_size, grid_size))
            target_data = targets[i, 0].reshape((grid_size, grid_size)) * -1

            print("input", np.where(input_data))
            print("target", np.where(target_data))

            batch_data += input_data
            batch_data += target_data

            fig, ax = plt.subplots()
            ax.imshow(batch_data)
            plt.show()

    return Corpus(
        "circle_through_time_bitmap", input_sequence=inputs, target_sequence=targets
    )


def split_train_test(corpus: Corpus, train_ratio: float) -> tuple[Corpus, Corpus]:
    batch_size = corpus.input_sequence.shape[0]
    train_size = math.floor(train_ratio * batch_size)
    shuffled_idxs = np.random.permutation(batch_size)
    train_idxs = sorted(shuffled_idxs[:train_size])
    test_idxs = sorted(shuffled_idxs[train_size:])
    return _split_corpus(corpus, batch_idxs_per_corpus=(train_idxs, test_idxs))


def _split_corpus(
    corpus: Corpus, batch_idxs_per_corpus: tuple[np.ndarray, ...]
) -> tuple[Corpus, ...]:
    new_corpora = []
    for batch_idxs in batch_idxs_per_corpus:
        max_sample_length = max(np.where(corpus.input_mask[batch_idxs])[-1]) + 1

        inputs = corpus.input_sequence[batch_idxs, :, :max_sample_length]
        targets = corpus.target_sequence[batch_idxs, :, :max_sample_length]
        target_mask = corpus.targets_mask[batch_idxs, :, :max_sample_length]
        input_mask = corpus.input_mask[batch_idxs, :max_sample_length]

        # TODO: recalculating indices for this is hard, need to change the data format.
        inputs_per_time_step = None

        if corpus.sample_weights:
            sample_weights = tuple(corpus.sample_weights[i] for i in batch_idxs)
        else:
            sample_weights = None

        new_corpora.append(
            Corpus(
                name=corpus.name,
                input_sequence=inputs,
                target_sequence=targets,
                input_values_per_time_step=inputs_per_time_step,
                input_mask=input_mask,
                targets_mask=target_mask,
                sample_weights=sample_weights,
            )
        )

    return tuple(new_corpora)


def split_to_mini_batches(
    corpus: Corpus, mini_batch_size: Optional[int]
) -> tuple[Corpus, ...]:
    if mini_batch_size is None:
        return (corpus,)

    num_samples = corpus.input_sequence.shape[0]
    mini_batch_idxs = tuple(
        np.split(np.arange(num_samples), np.arange(0, num_samples, mini_batch_size)[1:])
    )

    return _split_corpus(corpus, mini_batch_idxs)
