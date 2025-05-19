import unittest

import configuration
import corpora
import manual_nets
import network
import simulations
import torch_conversion
import utils


class TestTorch(unittest.TestCase):
    def test_torch_conversion(self):
        net = manual_nets.make_differentiable_an_bn_net()
        torch_net = torch_conversion.mdlnn_to_torch(net)

        utils.seed(100)
        corpus = corpora.make_ain_bjn_ckn_dtn(
            batch_size=500, prior=0.3, multipliers=(1, 1, 0, 0)
        )

        accuracy = torch_conversion.eval(
            torch_net, corpus, accuracy_type="deterministic"
        )
        print(f"Accuracy: {accuracy:.5f}")
        assert accuracy == 1.0

    def test_an_bn_cn_net(self):
        mdlrnn = manual_nets.make_found_differentiable_an_bn_cn_net()

        net_torch = torch_conversion.mdlnn_to_torch(net=mdlrnn)
        print(net_torch)

        utils.seed(100)

        test_corpus = corpora.optimize_for_feeding(
            corpora._make_ain_bjn_ckn_dtn_corpus(
                n_values=tuple(range(100)),
                multipliers=(1, 1, 1, 0),
                prior=0.3,
                sort_by_length=True,
            )
        )
        config = configuration.SimulationConfig(
            **{
                **simulations.DEFAULT_CONFIG,
                "try_net_early_stop": False,
                "simulation_id": "",
                "num_islands": 1,
                "seed": 1,
            },
        )

        numpy_accuracy = network.calculate_deterministic_accuracy(
            mdlrnn, test_corpus, config
        )
        assert numpy_accuracy == 1.0
        print(f"Numpy det. accuracy: {numpy_accuracy:.5f}")

        torch_accuracy = torch_conversion.eval(
            net_torch, test_corpus, accuracy_type="deterministic"
        )
        assert torch_accuracy == 1.0
        print(f"Torch det. accuracy: {torch_accuracy:.5f}")

    def test_floating_layer_conversion(self):
        numpy_net = manual_nets.make_found_diff_softmax_an_bn_cn_net()
        torch_net = torch_conversion.mdlnn_to_torch(net=numpy_net)

        corpus = corpora.make_ain_bjn_ckn_dtn(
            batch_size=500, prior=0.3, multipliers=(1, 1, 1, 0)
        ).test_corpus

        torch_accuracy = torch_conversion.eval(
            torch_net, corpus, accuracy_type="deterministic"
        )
        assert torch_accuracy == 1.0
