import unittest

import matplotlib.pyplot as plt
import numpy as np

import configuration
import corpora
import manual_nets
import network
import simulations
import utils


def _test_xor_correctness(input_sequence, target_sequence):
    memory = {"step_t_minus_1": 0, "step_t_minus_2": 0}

    def predictor(x):
        memory["step_t_minus_2"] = memory["step_t_minus_1"]
        memory["step_t_minus_1"] = x
        return memory["step_t_minus_1"] ^ memory["step_t_minus_2"]

    correct = 0
    for in_, target in zip(input_sequence[0], target_sequence[0]):
        prediction = predictor(in_)
        if np.all(prediction == target):
            correct += 1

    accuracy = correct / input_sequence.shape[1]
    print(f"Accuracy: {accuracy:.2f}")

    # A perfect network predicts 33% of the values deterministically, and another 66% x 0.5 by chance.
    assert accuracy >= 0.66


class TestCorpus(unittest.TestCase):
    def test_train_test_shuffle(self):
        train_ratio = 0.7
        batch_size = 1000
        train_corpus = corpora.make_binary_palindrome_fixed_length(
            batch_size=batch_size, sequence_length=10, train_set_ratio=train_ratio
        )
        test_corpus = train_corpus.test_corpus
        assert (
            sum(train_corpus.sample_weights) + sum(test_corpus.sample_weights)
            == batch_size
        )

        assert train_corpus.input_sequence.shape[0] == 22
        assert test_corpus.input_sequence.shape[0] == 10

    def test_mini_batches(self):
        mini_batch_size = 100

        net = manual_nets.make_emmanuel_triplet_xor_network()
        full_corpus = corpora.optimize_for_feeding(
            corpora.make_elman_xor_binary(sequence_length=300, batch_size=1000)
        )

        mini_batches = corpora.split_to_mini_batches(
            full_corpus, mini_batch_size=mini_batch_size
        )
        assert len(mini_batches) == 10

        config = configuration.SimulationConfig(
            **{
                **simulations.DEFAULT_CONFIG,
                "mini_batch_size": mini_batch_size,
                "simulation_id": "",
                "num_islands": 1,
                "seed": 1,
            },
        )
        net = network.calculate_fitness(net, full_corpus, config)
        print(net)
        assert net.fitness.data_encoding_length == 200000

    def test_0_1_pattern_binary(self):
        seq_length = 10
        batch_size = 10
        corpus = corpora.make_0_1_pattern_binary(
            sequence_length=seq_length, batch_size=batch_size
        )
        assert corpus.sample_weights == (batch_size,)
        assert corpus.test_corpus.input_sequence.shape == (1, seq_length * 50_000, 1)

        seq_length = 10
        batch_size = 1
        corpus = corpora.make_0_1_pattern_binary(
            sequence_length=seq_length, batch_size=batch_size
        )
        assert corpus.sample_weights is None

    def test_xor_corpus_correctness(self):
        sequence_length = 9999
        xor_corpus_binary = corpora.make_elman_xor_binary(sequence_length, batch_size=1)
        xor_corpus_one_hot = corpora.make_elman_xor_one_hot(
            sequence_length, batch_size=1
        )
        _test_xor_correctness(
            xor_corpus_binary.input_sequence.astype(int),
            xor_corpus_binary.target_sequence.astype(int),
        )
        _test_xor_correctness(
            xor_corpus_one_hot.input_sequence.argmax(axis=-1),
            xor_corpus_one_hot.target_sequence.argmax(axis=-1),
        )

    def test_binary_addition_corpus_correctness(self):
        addition_corpus = corpora.make_binary_addition(min_n=0, max_n=100)

        for input_sequence, target_sequence in zip(
            addition_corpus.input_sequence, addition_corpus.target_sequence
        ):
            n1 = 0
            n2 = 0
            sum_ = 0
            for i, bin_digit in enumerate(input_sequence):
                n1_binary_digit = bin_digit[0]
                n2_binary_digit = bin_digit[1]
                current_exp = 2**i
                if n1_binary_digit == 1:
                    n1 += current_exp
                if n2_binary_digit == 1:
                    n2 += current_exp
                target_binary_digit = target_sequence[i]
                if target_binary_digit == 1:
                    sum_ += current_exp

            assert n1 + n2 == sum_, (n1, n2, sum_)

    def test_cross_serial_corpus(self):
        corpus = corpora.make_an_bm_cn_dm(
            batch_size=100, prior=0.3, limit_vocabulary=False, sort_by_length=False
        )
        classes = corpus.input_sequence.argmax(axis=-1)

        assert ((classes == 1).sum(axis=1) == (classes == 3).sum(axis=1)).all()
        assert ((classes == 2).sum(axis=1) == (classes == 4).sum(axis=1)).all()

    def test_an_bn_corpus(self):
        n_values = tuple(range(50))
        an_bn_corpus = corpora.optimize_for_feeding(
            corpora._make_ain_bjn_ckn_dtn_corpus(
                n_values, multipliers=(1, 1, 0, 0), prior=0.1, sort_by_length=True
            )
        )

        for n in n_values:
            row = 49 - n  # Sequences are sorted by decreasing length.
            input_seq = an_bn_corpus.input_sequence[row]
            target_seq = an_bn_corpus.target_sequence[row]
            seq_len = 1 + (2 * n)

            zeros_start = 1
            ones_start = n + 1

            assert not np.all(corpora.is_masked(input_seq[:seq_len]))
            assert np.all(corpora.is_masked(input_seq[seq_len:]))

            input_classes = np.argmax(input_seq, axis=-1)[:seq_len]
            target_classes = np.argmax(target_seq, axis=-1)[:seq_len]

            assert np.sum(input_classes == 1) == n
            assert np.sum(input_classes == 2) == n

            assert input_classes[0] == 0  # Start of sequence.
            assert np.all(input_classes[zeros_start:ones_start] == 1)
            assert np.all(input_classes[ones_start:seq_len] == 2)

            assert target_classes[seq_len - 1] == 0  # End of sequence.
            assert np.all(target_classes[zeros_start - 1 : ones_start - 1] == 1)
            assert np.all(target_classes[ones_start - 1 : seq_len - 1] == 2)

    def test_simple_arithmetic_optimal_d_g(self):
        utils.seed(3)
        corpus = corpora.make_simple_arithmetic_syntax_dfa(
            sequence_length=20, batch_size=1
        )
        print(corpus.input_sequence.argmax(axis=-1))
        inputs = "".join(
            [corpus.vocabulary[x] for x in corpus.input_sequence[0].argmax(axis=-1)]
        )
        targets = "".join(
            [corpus.vocabulary[x] for x in corpus.target_sequence[0].argmax(axis=-1)]
        )
        print(f"Inputs:  {inputs}\nTargets: {targets}")
        # For the string "1+1=1", we have three visits in states with three choices.
        target_optimal_d_g = 3 * np.log2(3)
        assert np.round(corpus.optimal_d_given_g, 4) == np.round(target_optimal_d_g, 4)

    @unittest.skip(
        "Qunatifiers corpora should implement vocabulary and fix this test"
    )  # TODO: Fix this test after changes for DFA optimal D|G calculations
    def test_dfa_baseline_d_g(self):
        dfa_ = corpora.make_between_dfa(start=4, end=4)

        corpus = corpora.make_exactly_n_quantifier(4, sequence_length=50, batch_size=10)

        optimal_d_g = dfa_.get_optimal_data_given_grammar_for_dfa(corpus.input_sequence)

        num_non_masked_steps = np.sum(np.all(~np.isnan(corpus.input_sequence), axis=-1))
        assert optimal_d_g == num_non_masked_steps

    @unittest.skip("Only for visualization")
    def test_circle(self):
        circle_corpus = corpora.make_circle(degrees_of_freedom=8, num_samples=250)
        X, Y = zip(*circle_corpus.target_sequence.reshape(-1, 2))

        plt.rc("grid", color="w", linestyle="solid")
        fig, ax = plt.subplots(figsize=(20, 20))

        ax.scatter(X, Y)

        ax.grid()

        ax.grid(b=True, color="#bcbcbc")
        plt.show()

    @unittest.skip("Only for visualization")
    def test_circle_through_time(self):
        circle_corpus = corpora.make_circle_through_time(num_samples=100)
        X, Y = zip(*circle_corpus.target_sequence.reshape(-1, 2))

        plt.rc("grid", color="w", linestyle="solid")
        fig, ax = plt.subplots(figsize=(20, 20))

        ax.scatter(X, Y)

        ax.grid()

        ax.grid(b=True, color="#bcbcbc")
        plt.show()

    @staticmethod
    def _assert_exhaustive_an_bn_generation(sequences, with_probs: bool, prior: float):
        for seq in sequences:
            # Check if the sequence is a^n b^n
            assert len(seq) % 2 == 0, f"Invalid length for {seq}"
            assert seq.count("a") == seq.count("b"), f"Invalid counts for {seq}"

            if with_probs:
                n = len(seq) // 2
                expected_prob = corpora._get_ain_bjn_ckn_dtn_sequence_probability(
                    n - 1,
                    prior,  # We use n-1 since the PCFG is for n > 0
                )
                assert np.isclose(sequences[seq], expected_prob), (
                    f"Invalid probability for {seq}, {sequences[seq]} != {expected_prob}"
                )

    def test_generate_all_strings_from_pcfg(
        self, prior: float = 0.3, max_length: int = 10
    ):
        pcfg = corpora._make_an_bn_pcfg(prior=prior)
        results = corpora._generate_all_strings_from_pcfg(
            pcfg=pcfg, start_symbol="S", max_length=max_length, with_probs=False
        )
        results_with_probs = corpora._generate_all_strings_from_pcfg(
            pcfg=pcfg,
            start_symbol="S",
            max_length=max_length,
            with_probs=True,
        )

        self._assert_exhaustive_an_bn_generation(results, with_probs=False, prior=prior)
        self._assert_exhaustive_an_bn_generation(
            results_with_probs, with_probs=True, prior=prior
        )

    def test_generate_corpus_with_test_probs(self):
        corpus = corpora.make_golden_arithmetic_syntax_pcfg(
            sequence_length=None, batch_size=1, max_test_sequence_length=10
        )

        # Assert test probabilities exist and are floats (TODO: Verify actual values)
        for weight in corpus.test_corpus.sample_weights:
            assert isinstance(weight, float)
