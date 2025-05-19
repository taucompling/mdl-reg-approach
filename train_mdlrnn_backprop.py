import dataclasses
import fractions
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

import corpora
import manual_nets
import mdlrnn
import network
import torch_conversion
import utils

EXPERIMENTS_RESULTS_CSV = "backprop_experiments_results.csv"

CORPUS_SEED = 100
EXPERIMENTS_SEED = 100
TORCH_DTYPE = torch.float64

TASK_NAME_TO_ACCURACY_TYPE = {
    "an_bn": "deterministic",
    "an_bn_cn": "deterministic",
    "dyck1": "categorical",
}


@dataclasses.dataclass
class ExperimentConfig:
    seed: int
    name: str
    task_name: str
    corpus: corpora.Corpus
    network_fn: callable
    reset_weights: bool
    reg_method: Optional[str] = None
    reg_lambda: float = 0.0


@dataclasses.dataclass
class ExperimentResult:
    stage: str
    task_name: str

    approx_g_enc_len: float
    l1_value: float
    l2_value: float

    train_loss: float
    reg_loss: float
    train_encoding_length: float
    train_accuracy: float

    test_loss: float
    test_encoding_length: float
    test_accuracy: float

    test_loss_no_overlap: float
    test_encoding_length_no_overlap: float
    test_accuracy_no_overlap: float


def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callable called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


def filter_overlap_sequences(
    corpus: corpora.Corpus, allow_overlap: bool = True
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if not allow_overlap and corpus.train_test_overlap_mask is not None:
        overlap_mask = corpus.train_test_overlap_mask
        input_sequence = corpus.input_sequence[~overlap_mask]
        target_sequence = corpus.target_sequence[~overlap_mask]
        if corpus.sample_weights is not None:
            sample_weights = []
            for i, weight in enumerate(corpus.sample_weights):
                if overlap_mask[i]:
                    continue
                sample_weights.append(weight)
            sample_weights = tuple(sample_weights)
        else:
            sample_weights = None
    else:
        input_sequence = corpus.input_sequence
        target_sequence = corpus.target_sequence
        sample_weights = corpus.sample_weights

    return input_sequence, target_sequence, sample_weights


def convert_corpus_to_torch(
    corpus: corpora.Corpus,
    ignore_idx: int = -100,
    allow_overlap: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    input_sequence, target_sequence, sample_weights = filter_overlap_sequences(
        corpus, allow_overlap=allow_overlap
    )

    inputs = torch.tensor(input_sequence, dtype=TORCH_DTYPE)
    inputs = torch.nan_to_num(inputs, nan=0.0)

    targets = torch.tensor(target_sequence, dtype=TORCH_DTYPE)
    targets_no_nan = torch.nan_to_num(targets, nan=0.0)
    indices = torch.argmax(targets_no_nan, dim=-1)
    nan_mask = torch.isnan(targets.sum(dim=-1))
    indices[nan_mask] = ignore_idx

    weights = (
        torch.tensor(sample_weights, dtype=TORCH_DTYPE)
        if sample_weights is not None
        else None
    )

    return inputs, indices, weights, ignore_idx


def calculate_grammar_encoding_length_approx(model: mdlrnn.MDLRNN) -> float:
    # Iterate over all weights and convert each weight to numerator and denominator and then call network._get_weight_encoding
    encoding = ""
    for _, param in model.named_parameters():
        for weight in param.flatten():
            weight_value = weight.item()
            sign = 1 if weight_value >= 0 else -1
            fraction = fractions.Fraction(sign * abs(weight_value))
            frac_weight = network._Weight(
                sign=sign,
                numerator=fraction.numerator,
                denominator=fraction.denominator,
            )
            encoding += network._get_weight_encoding(frac_weight)

    return len(encoding)


def calculate_loss(
    model: mdlrnn.MDLRNN,
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: Optional[torch.Tensor],
    ignore_index: int,
    reg_type: Optional[str],
    reg_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, sequence_length, num_classes = logits.shape

    # Flatten batch x time dimensions
    logits_flat = logits.flatten(0, 1)
    labels_flat = labels.flatten()

    # Cross‑entropy per token
    ce_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction="none",
    )

    # Keep only valid positions
    mask = labels_flat != ignore_index
    ce_loss = ce_loss[mask]

    if sample_weights is not None:
        # Expand each sequence weight over its time steps
        weights = sample_weights.repeat_interleave(sequence_length)[mask]
        ce_loss = ce_loss * weights

    total_loss = ce_loss.sum()
    avg_loss = ce_loss.mean()
    reg_loss = calculate_regularization(model, reg_lambda, reg_type)

    return total_loss, avg_loss, reg_loss


def calculate_regularization(
    model: mdlrnn.MDLRNN, reg_lambda: float, reg_type: str
) -> torch.Tensor:
    reg_loss = torch.tensor(0.0, requires_grad=True)
    if reg_type is not None and reg_lambda > 0:
        for param in model.parameters():
            if reg_type == "l1":
                reg_loss = reg_loss + torch.sum(torch.abs(param))
            elif reg_type == "l2":
                reg_loss = reg_loss + torch.sum(torch.square(param))
        reg_loss = reg_lambda * reg_loss
    return reg_loss


def evaluate(
    model: mdlrnn.MDLRNN,
    corpus: corpora.Corpus,
    task_name: str,
    reg_type: Optional[str] = None,
    reg_lambda: float = 0.0,
    allow_overlap: bool = True,
) -> tuple[float, float, float, float]:
    model_inputs, target_labels, sample_weights, ignore_index = convert_corpus_to_torch(
        corpus, allow_overlap=allow_overlap
    )

    model.eval()
    with torch.no_grad():
        logits, _ = model(model_inputs, output_layer=None)
        sum_loss, avg_loss, reg_loss = calculate_loss(
            model,
            logits,
            target_labels,
            sample_weights,
            ignore_index,
            reg_type,
            reg_lambda,
        )
        # Since the loss is in the natural log, we need to convert it to bits for encoding length
        encoding_length = sum_loss / torch.log(torch.tensor(2.0))

    accuracy_type = TASK_NAME_TO_ACCURACY_TYPE[task_name]
    accuracy = torch_conversion.eval(
        model, corpus, accuracy_type, output_layer="softmax"
    )
    return avg_loss.item(), reg_loss.item(), encoding_length.item(), accuracy


def evaluate_train_and_test(
    model: mdlrnn.MDLRNN,
    corpus: corpora.Corpus,
    stage: str,
    task_name: str,
    reg_type: Optional[str] = None,
    reg_lambda: float = 0.0,
) -> ExperimentResult:
    logger.info(f"Evaluating model at stage '{stage}' on training data...")
    train_loss, train_reg_loss, train_enc_length, train_accuracy = evaluate(
        model, corpus, task_name, reg_type, reg_lambda
    )
    logger.info(
        f"Train Loss: {train_loss:.4f}, Reg Loss: {train_reg_loss:.4f}, "
        f"Encoding Length: {train_enc_length:.4f}, Accuracy: {train_accuracy:.4f}"
    )

    logger.info(f"Evaluating model at stage '{stage}' on test data with overlap...")
    test_loss, test_reg_loss, test_enc_length, test_accuracy = evaluate(
        model,
        corpus.test_corpus,
        task_name,
        reg_type,
        reg_lambda,
        allow_overlap=True,
    )
    logger.info(
        f"Test Loss (with overlap): {test_loss:.4f}, Reg Loss: {test_reg_loss:.4f}, "
        f"Encoding Length: {test_enc_length:.4f}, Accuracy: {test_accuracy:.4f}"
    )

    logger.info(f"Evaluating model at stage '{stage}' on test data without overlap...")
    (
        test_loss_no_overlap,
        test_reg_loss_no_overlap,
        test_enc_length_no_overlap,
        test_accuracy_no_overlap,
    ) = evaluate(
        model,
        corpus.test_corpus,
        task_name,
        reg_type,
        reg_lambda,
        allow_overlap=False,
    )
    logger.info(
        f"Test Loss (no overlap): {test_loss_no_overlap:.4f}, Reg Loss: {test_reg_loss_no_overlap:.4f}, "
        f"Encoding Length: {test_enc_length_no_overlap:.4f}, Accuracy: {test_accuracy_no_overlap:.4f}"
    )

    approx_g_enc_len = calculate_grammar_encoding_length_approx(model)
    l1_value = calculate_regularization(model, 1, "l1").item()
    l2_value = calculate_regularization(model, 1, "l2").item()

    return ExperimentResult(
        stage=stage,
        task_name=task_name,
        approx_g_enc_len=approx_g_enc_len,
        l1_value=l1_value,
        l2_value=l2_value,
        train_loss=train_loss,
        reg_loss=train_reg_loss,
        train_encoding_length=train_enc_length,
        train_accuracy=train_accuracy,
        test_loss=test_loss,
        test_encoding_length=test_enc_length,
        test_accuracy=test_accuracy,
        test_loss_no_overlap=test_loss_no_overlap,
        test_encoding_length_no_overlap=test_enc_length_no_overlap,
        test_accuracy_no_overlap=test_accuracy_no_overlap,
    )


def train(
    model: mdlrnn.MDLRNN,
    corpus: corpora.Corpus,
    epochs: int = 1000,
    learning_rate: float = 1e-4,
    log_interval: int = 100,
    reg_type: Optional[str] = None,
    reg_lambda: float = 0.0,
) -> None:
    model_inputs, target_labels, sample_weights, ignore_index = convert_corpus_to_torch(
        corpus
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs, _ = model(model_inputs, output_layer=None)
        sum_loss, avg_loss, reg_loss = calculate_loss(
            model,
            outputs,
            target_labels,
            sample_weights,
            ignore_index,
            reg_type,
            reg_lambda,
        )
        total_loss = (
            avg_loss + reg_loss
            if (reg_type is not None and reg_lambda != 0.0)
            else avg_loss
        )
        total_loss.backward()
        optimizer.step()
        if epoch % log_interval == 0:
            logger.debug(
                f"Epoch {epoch}/{epochs}, Loss: {avg_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}"
            )


def run_experiment(
    config: ExperimentConfig,
) -> dict[str, ExperimentResult]:
    logger.info(
        f"Running experiment with {config.name} on task {config.task_name} and seed {config.seed}..."
    )
    utils.seed(config.seed, set_torch_seed=True)

    corpus = config.corpus
    converted_net = config.network_fn()

    results = {}
    # Golden evaluation (includes both overlap and no-overlap metrics)
    golden_result = evaluate_train_and_test(
        converted_net,
        corpus,
        stage="Golden",
        task_name=config.task_name,
        reg_type=config.reg_method,
        reg_lambda=config.reg_lambda,
    )
    results["Golden"] = golden_result

    if config.reset_weights:
        logger.info("Resetting model weights...")
        reset_all_weights(converted_net)
    else:
        logger.info("Keeping original weights as per configuration...")

    logger.info(f"Training model with {config.name}...")
    train(
        converted_net,
        corpus,
        reg_type=config.reg_method,
        reg_lambda=config.reg_lambda,
    )

    # After training evaluation (includes both overlap and no-overlap metrics)
    after_train_result = evaluate_train_and_test(
        converted_net,
        corpus,
        stage="After Training",
        task_name=config.task_name,
        reg_type=config.reg_method,
        reg_lambda=config.reg_lambda,
    )
    results["After Training"] = after_train_result

    return results


def save_results_to_csv(
    configs_and_results: list[Tuple[ExperimentConfig, ExperimentResult]],
    corpora_dict: dict,
    csv_filename: str,
) -> None:
    df_data = []
    for config, result in configs_and_results:
        task_corpus = corpora_dict[result.task_name]
        optimal_dg_train = task_corpus.optimal_d_given_g
        optimal_dg_test = task_corpus.test_corpus.optimal_d_given_g
        optimal_dg_test_no_overlap = (
            task_corpus.test_corpus.optimal_d_given_g_no_overlap
        )

        df_row = {
            "experiment": config.name,
            "task": result.task_name,
            "stage": result.stage,
            "approx_g_enc_len": result.approx_g_enc_len,
            "l1_value": result.l1_value,
            "l2_value": result.l2_value,
            "reset_weights": config.reset_weights,
            "avg_train_loss": result.train_loss,
            "avg_reg_loss": result.reg_loss,
            "train_enc_len": result.train_encoding_length,
            "train_accuracy": result.train_accuracy,
            "avg_test_loss": result.test_loss,
            "test_enc_len": result.test_encoding_length,
            "test_accuracy": result.test_accuracy,
            "avg_test_loss_no_overlap": result.test_loss_no_overlap,
            "test_enc_len_no_overlap": result.test_encoding_length_no_overlap,
            "test_accuracy_no_overlap": result.test_accuracy_no_overlap,
            "optimal_train_enc_len": optimal_dg_train,
            "optimal_test_enc_len": optimal_dg_test,
            "optimal_test_enc_len_no_overlap": optimal_dg_test_no_overlap,
            "reg_method": config.reg_method,
            "reg_lambda": config.reg_lambda,
            "seed": config.seed,
        }
        df_data.append(df_row)

    df = pd.DataFrame(df_data)
    df.to_csv(csv_filename, index=False)
    logger.info(f"Results exported to {csv_filename}")


def main():
    corpora_dict = {}

    # Set seed before each corpus generation to ensure consistency
    utils.seed(CORPUS_SEED, set_torch_seed=True)
    an_bn_corpus = corpora.make_an_bn(batch_size=500, prior=0.3, sort_by_length=False)
    corpora_dict["an_bn"] = corpora.compute_train_test_overlap_mask(
        an_bn_corpus, is_exhaustive_test_corpus=True
    )

    utils.seed(CORPUS_SEED, set_torch_seed=True)
    an_bn_cn_corpus = corpora.make_ain_bjn_ckn_dtn(
        batch_size=500, prior=0.3, multipliers=(1, 1, 1, 0)
    )
    corpora_dict["an_bn_cn"] = corpora.compute_train_test_overlap_mask(
        an_bn_cn_corpus, is_exhaustive_test_corpus=True
    )

    utils.seed(CORPUS_SEED, set_torch_seed=True)
    dyck_1_corpus = corpora.make_dyck_n(
        n=1, batch_size=500, nesting_probab=0.33333, max_sequence_length=200
    )
    corpora_dict["dyck1"] = corpora.compute_train_test_overlap_mask(
        dyck_1_corpus, is_exhaustive_test_corpus=True
    )

    network_fns = {
        "an_bn": lambda: torch_conversion.mdlnn_to_torch(
            net=manual_nets.make_found_diff_softmax_an_bn_net()
        ),
        "an_bn_cn": lambda: torch_conversion.mdlnn_to_torch(
            net=manual_nets.make_found_diff_softmax_an_bn_cn_net()
        ),
        "dyck1": lambda: torch_conversion.mdlnn_to_torch(
            net=manual_nets.make_found_diff_softmax_dyck_1_net()
        ),
    }

    reg_methods = [
        {"name": "No Regularization", "method": None, "lambda": 0.0},
        {"name": "L1 Regularization", "method": "l1", "lambda": 1},
        {"name": "L2 Regularization", "method": "l2", "lambda": 1},
    ]

    # Create experiment configs using same corpus for each task
    experiment_configs = []
    for task_name, corpus in corpora_dict.items():
        for reg in reg_methods:
            experiment_configs.append(
                ExperimentConfig(
                    seed=EXPERIMENTS_SEED,
                    name=f"{reg['name']}",
                    task_name=task_name,
                    corpus=corpus,
                    network_fn=network_fns[task_name],
                    reg_method=reg["method"],
                    reg_lambda=reg["lambda"],
                    reset_weights=False,
                )
            )

    all_results_tuples = []
    for config in experiment_configs:
        results_dict = run_experiment(config)
        for stage, result in results_dict.items():
            all_results_tuples.append((config, result))

    save_results_to_csv(
        all_results_tuples, corpora_dict, csv_filename=EXPERIMENTS_RESULTS_CSV
    )


if __name__ == "__main__":
    main()
