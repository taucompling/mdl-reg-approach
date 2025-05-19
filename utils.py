import argparse
import dataclasses
import hashlib
import io
import itertools
import os
import pathlib
import pickle
import random
import subprocess
import zlib
from typing import Any, Iterable, Optional
from urllib import parse

import numpy as np
import torch
from loguru import logger
from numba import types
from torch import backends, cuda

import configuration
import corpora

BASE_SEED = 100

MPI_LOGGING_TAG = 1
MPI_RESULTS_TAG = 2
MPI_MIGRANTS_TAG = 3

FLOAT_DTYPE = np.float64
NUMBA_FLOAT_DTYPE = types.float64

_ENABLE_APPLE_GPU = False


def _get_free_gpu():
    max_free = 0
    max_idx = 0

    rows = (
        subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.free"]
        )
        .decode("utf-8")
        .split("\n")
    )
    for i, row in enumerate(rows[1:-1]):
        mb = float(row.rstrip(" [MiB]"))

        if mb > max_free:
            max_idx = i
            max_free = mb

    return max_idx


def get_device():
    if _ENABLE_APPLE_GPU and backends.mps.is_available():
        device = "mps"
    elif cuda.is_available():
        device = f"cuda:{_get_free_gpu()}"
    else:
        device = "cpu"
    logger.info(f"Using device {device}")
    return torch.device(device)


DEVICE = get_device()


def kwargs_from_param_grid(param_grid: dict[str, Iterable[Any]]) -> Iterable[dict]:
    arg_names = list(param_grid.keys())
    arg_products = list(itertools.product(*param_grid.values()))
    for arg_product in arg_products:
        yield {arg: val for arg, val in zip(arg_names, arg_product)}


def get_s3_simulations_bucket():
    import boto3

    bucket_name = os.environ.get("S3_BUCKET", "YOUR_S3_BUCKET_NAME_HERE")
    if bucket_name == "YOUR_S3_BUCKET_NAME_HERE":
        logger.warning(
            "S3_BUCKET environment variable not set. Cloud functionality will use placeholder bucket name and likely fail."
        )
    return boto3.Session().resource("s3").Bucket(bucket_name)


def load_network_from_zoo(name, subdir=None):
    print(name)
    path = pathlib.Path("./network_zoo/")
    if subdir:
        path = path.joinpath(subdir.strip("/"))
    path = path.joinpath(f"{name}.pickle")
    print(path)
    with path.open("rb") as f:
        return pickle.load(f)


class CPUUnpickler(pickle.Unpickler):
    # Needed when loading GPU-trained RNNs.
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location=DEVICE)
        else:
            return super().find_class(module, name)


def load_object_from_s3(key):
    cache_path = pathlib.Path.home() / ".mdlnn/" / key
    cache_path.parent.mkdir(exist_ok=True, parents=True)

    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    bucket = get_s3_simulations_bucket()
    bytes = io.BytesIO()
    bucket.download_fileobj(key, bytes)
    bytes.seek(0)
    try:
        obj = pickle.load(bytes)
    except RuntimeError:
        bytes.seek(0)
        obj = CPUUnpickler(bytes).load()

    # Don't cache live-updating nets.
    if "current" not in str(cache_path):
        with cache_path.open("wb") as f:
            pickle.dump(obj, f)

    return obj


def upload_object_to_s3(obj, key):
    bucket = get_s3_simulations_bucket()
    bucket.put_object(
        Key=key,
        Body=pickle.dumps(obj),
    )


def _s3_simulation_suffix_key(simulation_id, suffix):
    return f"{simulation_id}/{simulation_id}__{suffix}.pickle"


def upload_object_to_simulation_s3(simulation_id, obj, suffix):
    upload_object_to_s3(obj, key=_s3_simulation_suffix_key(simulation_id, suffix))


def load_object_from_simulation_s3(simulation_id, suffix):
    return load_object_from_s3(key=_s3_simulation_suffix_key(simulation_id, suffix))


def get_current_best_net(simulation_id):
    current_best_data = load_object_from_simulation_s3(simulation_id, "current_best")
    best_island_data = load_object_from_simulation_s3(
        simulation_id, f"latest_generation_island_{current_best_data['island']}"
    )
    return min(best_island_data["population"], key=lambda x: x.fitness.mdl)


def delete_simulation_s3_object(simulation_id, suffix):
    get_s3_simulations_bucket().Object(
        _s3_simulation_suffix_key(simulation_id, suffix)
    ).delete()


def s3_key_exists(key) -> bool:
    from botocore import exceptions as boto_exceptions

    try:
        get_s3_simulations_bucket().Object(key).load()
        return True
    except boto_exceptions.ClientError:
        return False


def seed(n, set_torch_seed: bool = False):
    random.seed(n)
    np.random.seed(n)
    if set_torch_seed:
        torch.manual_seed(n)


def get_graphviz_url(network_dot_string: str) -> str:
    dot_string = network_dot_string.replace("\n", " ")
    return f"https://dreampuf.github.io/GraphvizOnline/#{parse.quote(dot_string)}"


def _deskew(image, image_shape, negated=True):
    import cv2

    # negate the image
    if not negated:
        image = 255 - image
    # calculate the moments of the image
    m = cv2.moments(image)
    if abs(m["mu02"]) < 1e-2:
        return image.copy()
    # caclulating the skew
    skew = m["mu11"] / m["mu02"]
    M = np.float32([[1, skew, -0.5 * image_shape[0] * skew], [0, 1, 0]])
    img = cv2.warpAffine(
        image, M, image_shape, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )
    return img


def _preprocess(img, size, patchCorner=(0, 0), patchDim=None, unskew=True):
    import cv2

    if patchDim == None:
        patchDim = size
    nImg = np.shape(img)[0]
    procImg = np.empty((nImg, size[0], size[1]))

    # Unskew and Resize
    if unskew == True:
        for i in range(nImg):
            procImg[i, :, :] = _deskew(cv2.resize(img[i, :, :], size), size)

    # Crop
    cropImg = np.empty((nImg, patchDim[0], patchDim[1]))
    for i in range(nImg):
        cropImg[i, :, :] = procImg[
            i,
            patchCorner[0] : patchCorner[0] + patchDim[0],
            patchCorner[1] : patchCorner[1] + patchDim[1],
        ]
    procImg = cropImg

    return procImg


def preprocess_mnist_images(images, width_height: int = 16):
    """Copied from Gaier & Ha: https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease"""
    images = images / 255
    images = _preprocess(images, (width_height, width_height))
    return images.reshape(-1, width_height**2)


def dict_and_corpus_hash(dict_, corpus) -> str:
    s = corpus.name
    for key in sorted(dict_.keys()):
        s += f"{key} {dict_[key]}"
    # TODO: ugly but works.
    s += corpus.name + " "
    s += str(corpus.input_sequence) + " "
    s += str(corpus.target_sequence)
    hash = hashlib.sha1()
    hash.update(s.encode())
    return hash.hexdigest()


def add_hash_to_simulation_id(simulation_config, corpus):
    config_hash = dict_and_corpus_hash(simulation_config.__dict__, corpus)
    simulation_id = f"{corpus.name}_{config_hash}"
    return dataclasses.replace(simulation_config, simulation_id=simulation_id)


def make_cli_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-s",
        "--simulation",
        dest="simulation_name",
        required=True,
        help="Simulation name.",
    )

    arg_parser.add_argument(
        "-n",
        "--total-islands",
        type=int,
        dest="total_islands",
        default=os.cpu_count(),
        help=f"Total number of islands in entire simulation (including other machines). Default: number of local cores ({os.cpu_count()}).",
    )

    arg_parser.add_argument(
        "--first-island",
        type=int,
        default=None,
        dest="first_island",
        help="First island index on this machine. Default: 0.",
    )

    arg_parser.add_argument(
        "--last-island",
        type=int,
        default=None,
        dest="last_island",
        help="Last island index on this machine. Default: number of islands minus 1.",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        dest="base_seed",
        help=f"Base seed value. Default: {BASE_SEED}. For the i-th reproduction (0-based), the seed will be {BASE_SEED} + i.",
    )

    arg_parser.add_argument(
        "--override",
        action="store_true",
        dest="override_existing",
        help="Override an existing simulation that has the same hash.",
    )

    arg_parser.add_argument(
        "--resume",
        dest="resumed_simulation_id",
        default=None,
        help="Resume simulation <simulation id> from latest generations.",
    )

    arg_parser.add_argument(
        "--corpus-args",
        default=None,
        dest="corpus_args",
        help="json to override default corpus arguments.",
    )

    arg_parser.add_argument(
        "--config",
        default=None,
        dest="config",
        help="json to override default and simulation-specific config.",
    )

    return arg_parser


def compress_string(s) -> bytes:
    return zlib.compress(s.encode("utf-8"))


def calculate_symbolic_accuracy(
    predicted_probabs: np.ndarray,
    valid_targets_mask: np.ndarray,
    input_mask: Optional[np.ndarray],
    sample_weights: tuple[int],
    plots: bool,
    epsilon: float = 0.0,
) -> tuple[float, tuple[int, ...]]:
    zero_target_probabs = valid_targets_mask == 0.0

    zero_predicted_probabs = predicted_probabs <= epsilon

    prediction_matches = np.all(
        np.equal(zero_predicted_probabs, zero_target_probabs), axis=-1
    )

    prediction_matches[~input_mask] = True

    sequence_idxs_with_errors = tuple(np.where(np.any(~prediction_matches, axis=1))[0])
    logger.debug(f"Sequence idxs with mismatches: {sequence_idxs_with_errors}")

    incorrect_predictions_per_time_step = np.sum(~prediction_matches, axis=0)

    if plots:
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        ax.set_title("Num prediction mismatches by time step")
        ax.bar(
            np.arange(len(incorrect_predictions_per_time_step)),
            incorrect_predictions_per_time_step,
        )
        plt.show()

    prediction_matches_without_masked = prediction_matches[input_mask]

    w = np.array(sample_weights).reshape((-1, 1))
    weights_repeated = np.matmul(w, np.ones((1, predicted_probabs.shape[1])))
    weights_masked = weights_repeated[input_mask]

    prediction_matches_weighted = np.multiply(
        prediction_matches_without_masked, weights_masked
    )

    symbolic_accuracy = np.sum(prediction_matches_weighted) / np.sum(weights_masked)
    return symbolic_accuracy, sequence_idxs_with_errors


def plot_probabs(probabs: np.ndarray, input_classes, class_to_label=None):
    from matplotlib import _color_data as matploit_color_data
    from matplotlib import pyplot as plt

    if probabs.shape[-1] == 1:
        # Binary outputs, output is P(1).
        probabs_ = np.zeros((probabs.shape[0], 2))
        probabs_[:, 0] = (1 - probabs).squeeze()
        probabs_[:, 1] = probabs.squeeze()
        probabs = probabs_

    masked_timesteps = np.where(corpora.is_masked(probabs))[0]
    if len(masked_timesteps):
        first_mask_step = masked_timesteps[0]
        probabs = probabs[:first_mask_step]

    plt.rc("grid", color="w", linestyle="solid")
    if class_to_label is None:
        class_to_label = {i: str(i) for i in range(len(input_classes))}
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150, facecolor="white")
    x = np.arange(probabs.shape[0])
    num_classes = probabs.shape[1]
    width = 0.8
    colors = (
        list(matploit_color_data.TABLEAU_COLORS) + list(matploit_color_data.XKCD_COLORS)
    )[:num_classes]
    for c in range(num_classes):
        ax.bar(
            x,
            probabs[:, c],
            label=f"P({class_to_label[c]})" if num_classes > 1 else "P(1)",
            color=colors[c],
            width=width,
            bottom=np.sum(probabs[:, :c], axis=-1),
        )
    ax.set_facecolor("white")
    ax.set_xticks(x)
    ax.set_xticklabels([class_to_label[x] for x in input_classes], fontsize=13)

    ax.set_xlabel("Input characters", fontsize=15)
    ax.set_ylabel("Next character probability", fontsize=15)

    ax.grid(b=True, color="#bcbcbc")
    # plt.title("Next step prediction probabilities", fontsize=22)
    plt.legend(loc="upper left", fontsize=15)

    # fig.savefig("test.png")
    fig.subplots_adjust(bottom=0.1)

    plt.show()
    fig.savefig(
        f"./figures/net_probabs_{random.randint(0, 10_000)}.pdf",
        dpi=300,
        facecolor="white",
    )


def _set_seed_to_corpus_seed(cfg: configuration.SimulationConfig):
    seed(cfg.corpus_seed)
