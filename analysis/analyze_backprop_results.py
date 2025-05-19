import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
EXPERIMENTS_RESULTS_CSV = BASE_DIR / "backprop_experiments_results.csv"
PLOT_SAVE_DIR = BASE_DIR / "backprop_plots"

# Define maps for better presentation
TASK_NAME_MAP = {
    "an_bn": "a$^n$b$^n$",
    "an_bn_cn": "a$^n$b$^n$c$^n$",
    "dyck1": "Dyck-1",
}

EXPERIMENT_NAME_MAP = {
    "No Regularization": "None",
    "L1 Regularization": "$L_1$",
    "L2 Regularization": "$L_2$",
}

COLUMN_NAME_MAP = {
    "task": "Task",
    "experiment": "Regularizer",
    "approx_g_enc_len": "$\\approx |H|$",
    "l1_value": "$L_1$",
    "l2_value": "$L_2$",
    "train_enc_len": "Train $|D:H|$",
    "test_enc_len": "Test $|D:H|$",
    "optimal_train_distance_norm": "$\\Delta^{Train}_{Optim}$ (\\%)",
    "optimal_test_distance_norm": "$\\Delta^{Test}_{Optim}$ (\\%)",
}

REGULARIZER_TO_COLOR = {
    "No Regularization": (
        *sns.color_palette("colorblind")[4],
        0.8,
    ),  # |D:G| + Limit |G| color
    "L1 Regularization": (*sns.color_palette("colorblind")[0], 0.8),  # L1 color
    "L2 Regularization": (*sns.color_palette("colorblind")[3], 0.8),  # L2 color
    "Golden": (*sns.color_palette("colorblind")[1], 0.8),  # Golden color
}


def set_neurips_style():
    plt.style.use("default")
    sns.set_style("white")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.sans-serif"] = "Helvetica"
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"] = 1.25
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams["savefig.dpi"] = 400
    plt.rcParams["savefig.bbox"] = "tight"
    # Set colorblind palette
    sns.set_palette("colorblind")


def create_target_plot(
    df: pd.DataFrame, compare_to_optimal: bool = False, show_golden_axes: bool = False
) -> None:
    set_neurips_style()

    # Create a 1x3 grid for the three tasks
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    axes = axes[0]  # Flatten the axes array

    if compare_to_optimal:
        x_column = "optimal_train_distance_norm"
        y_column = "optimal_test_distance_norm"
        reference_point = "Optimal"
    else:
        x_column = "golden_train_distance_norm"
        y_column = "golden_test_distance_norm"
        reference_point = "Golden"

    tasks = df[df["stage"] == "After Training"]["task"].unique()
    regularizers = df[df["stage"] == "After Training"]["experiment"].unique()

    for task_idx, task in enumerate(tasks):
        ax = axes[task_idx]
        task_df = df[df["task"] == task]

        ax.set_title(f"{TASK_NAME_MAP.get(task, task)}", fontweight="bold")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("0.2")

        ax.set_facecolor("#fafafa")
        ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

        max_abs_val = 0.0
        points_to_draw = []

        # Plot regularizer points
        for regularizer in regularizers:
            reg_df = task_df[
                (task_df["stage"] == "After Training")
                & (task_df["experiment"] == regularizer)
            ]
            if not reg_df.empty:
                x_val = reg_df[x_column].iloc[0] * 100.0
                y_val = reg_df[y_column].iloc[0] * 100.0
                if not (
                    pd.isna(x_val)
                    or pd.isna(y_val)
                    or not np.isfinite(x_val)
                    or not np.isfinite(y_val)
                ):
                    points_to_draw.append(
                        {"x": x_val, "y": y_val, "regularizer": regularizer}
                    )
                    max_abs_val = max(max_abs_val, abs(x_val), abs(y_val))

        # Plot golden point if needed
        golden_x_val, golden_y_val = None, None
        if compare_to_optimal and show_golden_axes:
            golden_df = task_df[task_df["stage"] == "Golden"]
            if not golden_df.empty:
                _gx = golden_df[x_column].iloc[0] * 100.0
                _gy = golden_df[y_column].iloc[0] * 100.0
                if not (
                    pd.isna(_gx)
                    or pd.isna(_gy)
                    or not np.isfinite(_gx)
                    or not np.isfinite(_gy)
                ):
                    golden_x_val, golden_y_val = _gx, _gy
                    max_abs_val = max(max_abs_val, abs(golden_x_val), abs(golden_y_val))

        limit = max_abs_val * 1.2 if max_abs_val > 0 else 10
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")

        if compare_to_optimal:
            # Add overfit arrow
            arrow_origin_x = 0.0
            arrow_origin_y = 0.0
            arrow_length_factor = 0.45
            arrow_dx = -arrow_length_factor * limit
            arrow_dy = arrow_length_factor * limit
            head_w = 0.035 * limit
            head_l = 0.055 * limit

            ax.arrow(
                arrow_origin_x,
                arrow_origin_y,
                arrow_dx,
                arrow_dy,
                head_width=head_w,
                head_length=head_l,
                fc="dimgray",
                ec="dimgray",
                lw=1.2,
                alpha=0.6,
                zorder=3,
                length_includes_head=True,
            )

            # Add "Overfit" text
            mid_arrow_x = arrow_origin_x + arrow_dx * 0.5
            mid_arrow_y = arrow_origin_y + arrow_dy * 0.5
            text_offset_val = 0.04 * limit
            text_x = mid_arrow_x + text_offset_val
            text_y = mid_arrow_y + text_offset_val

            ax.text(
                text_x,
                text_y,
                "Overfit",
                fontsize=9,
                color="black",
                alpha=0.75,
                ha="center",
                va="center",
                rotation=-45,
                zorder=4,
            )

        # Add concentric circles
        circle_radii_proportions = [0.25, 0.5, 0.75, 1.0]
        for prop in circle_radii_proportions:
            radius = limit * prop
            circle = plt.Circle(
                (0, 0),
                radius,
                color="gray",
                fill=False,
                linestyle="--",
                alpha=0.6,
                linewidth=1.0,
                zorder=5,
            )
            ax.add_artist(circle)

        # Plot points
        for point_data in points_to_draw:
            x, y = point_data["x"], point_data["y"]
            regularizer = point_data["regularizer"]

            ax.plot(
                x,
                y,
                marker="o",
                markersize=10,
                color=REGULARIZER_TO_COLOR[regularizer],
                markeredgewidth=0.5,
                markeredgecolor="black",
                linestyle="",
                label=regularizer if task_idx == 0 else "",
                zorder=10,
                alpha=0.6,
            )

        if (
            compare_to_optimal
            and show_golden_axes
            and golden_x_val is not None
            and golden_y_val is not None
        ):
            ax.axvline(
                x=golden_x_val,
                color=REGULARIZER_TO_COLOR["Golden"],
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
                zorder=5,
            )
            ax.axhline(
                y=golden_y_val,
                color=REGULARIZER_TO_COLOR["Golden"],
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
                zorder=5,
            )

        ax.axhline(0, color="black", linestyle="-", alpha=0.0, linewidth=1.5, zorder=1)
        ax.axvline(0, color="black", linestyle="-", alpha=0.0, linewidth=1.5, zorder=1)
        ax.grid(False)

    # Create legend
    handles = []
    labels = []
    for regularizer in regularizers:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=REGULARIZER_TO_COLOR[regularizer],
                markeredgewidth=1.0,
                markeredgecolor="black",
                markersize=8,
                linestyle="",
            )
        )
        labels.append(EXPERIMENT_NAME_MAP.get(regularizer, regularizer))

    if compare_to_optimal and show_golden_axes:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=REGULARIZER_TO_COLOR["Golden"],
                linestyle="--",
                linewidth=2.5,
            )
        )
        labels.append('Manual "Golden" Network')

    common_xlabel = f"Δ Train |D:H| from {reference_point} (%)"
    common_ylabel = f"Δ Test |D:H| from {reference_point} (%)"
    fig.supylabel(common_ylabel, x=0.02, fontsize=plt.rcParams["axes.labelsize"])
    fig.supxlabel(common_xlabel, y=0.05, fontsize=plt.rcParams["axes.labelsize"])

    # Adjust legend position and layout
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),  # Move legend very close to plots
        ncol=len(labels),
        frameon=True,
        framealpha=1.0,
        edgecolor="0.8",
    )

    # Adjust subplot parameters to minimize whitespace
    fig.subplots_adjust(
        left=0.08,
        bottom=0.15,
        right=0.95,
        top=0.95,
        hspace=0.2,
    )

    comparison_type = "optimal" if compare_to_optimal else "golden"
    golden_suffix = (
        "_with_golden_axes" if compare_to_optimal and show_golden_axes else ""
    )
    plot_path = PLOT_SAVE_DIR / f"target_plot_from_{comparison_type}{golden_suffix}.png"
    plot_path.parent.mkdir(exist_ok=True)
    fig.savefig(plot_path, dpi=400, bbox_inches="tight")
    logger.info(f"Target plot saved as '{plot_path}'")


def enhance_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance the DataFrame with additional columns for analysis.
    """
    # Group by task to find golden values (before training)
    golden_values = (
        df[df["stage"] == "Golden"].drop_duplicates(subset="task").set_index("task")
    )

    # Calculate normalized distances from golden (before training) values
    df["golden_train_distance"] = df.apply(
        lambda row: (
            row["train_enc_len"] - golden_values.loc[row["task"], "train_enc_len"]
        ),
        axis=1,
    )

    df["golden_test_distance"] = df.apply(
        lambda row: (
            row["test_enc_len"] - golden_values.loc[row["task"], "test_enc_len"]
        ),
        axis=1,
    )

    # Add normalized distances (as ratios relative to golden values)
    df["golden_train_distance_norm"] = df.apply(
        lambda row: row["golden_train_distance"]
        / golden_values.loc[row["task"], "train_enc_len"],
        axis=1,
    )

    df["golden_test_distance_norm"] = df.apply(
        lambda row: row["golden_test_distance"]
        / golden_values.loc[row["task"], "test_enc_len"],
        axis=1,
    )

    # Optimal distances
    df["optimal_train_distance"] = df.apply(
        lambda row: (row["train_enc_len"] - row["optimal_train_enc_len"]),
        axis=1,
    )

    df["optimal_test_distance"] = df.apply(
        lambda row: (row["test_enc_len"] - row["optimal_test_enc_len"]),
        axis=1,
    )

    df["optimal_train_distance_norm"] = df.apply(
        lambda row: row["optimal_train_distance"] / row["optimal_train_enc_len"],
        axis=1,
    )

    df["optimal_test_distance_norm"] = df.apply(
        lambda row: row["optimal_test_distance"] / row["optimal_test_enc_len"],
        axis=1,
    )

    return df


def create_latex_table(df: pd.DataFrame, show_full_table: bool = False) -> None:
    """
    Create a LaTeX table similar to the simulation analysis table.
    """
    golden_df = df[df["stage"] == "Golden"].copy()
    df_table = df[df["stage"] == "After Training"].copy()

    if show_full_table:
        latex_string = "\\begin{tabular}{llccccccc}\n"
        latex_string += "\\toprule\n"
        columns = [
            COLUMN_NAME_MAP["task"],
            COLUMN_NAME_MAP["experiment"],
            COLUMN_NAME_MAP["approx_g_enc_len"],
            COLUMN_NAME_MAP["l1_value"],
            COLUMN_NAME_MAP["l2_value"],
            COLUMN_NAME_MAP["train_enc_len"],
            COLUMN_NAME_MAP["test_enc_len"],
            COLUMN_NAME_MAP["optimal_train_distance_norm"],
            COLUMN_NAME_MAP["optimal_test_distance_norm"],
        ]
    else:
        latex_string = "\\begin{tabular}{llccccc}\n"
        latex_string += "\\toprule\n"
        columns = [
            COLUMN_NAME_MAP["task"],
            COLUMN_NAME_MAP["experiment"],
            COLUMN_NAME_MAP["approx_g_enc_len"],
            COLUMN_NAME_MAP["train_enc_len"],
            COLUMN_NAME_MAP["test_enc_len"],
            COLUMN_NAME_MAP["optimal_train_distance_norm"],
            COLUMN_NAME_MAP["optimal_test_distance_norm"],
        ]

    latex_string += " & ".join(columns) + " \\\\\n"
    latex_string += "\\midrule\n"

    tasks = df_table["task"].unique()
    first_task = True

    for task in tasks:
        task_df = df_table[df_table["task"] == task].copy()
        golden_row = golden_df[golden_df["task"] == task].iloc[0]

        if not first_task:
            latex_string += "\\midrule\n"
        first_task = False

        min_train_distance = task_df["optimal_train_distance_norm"].abs().min()
        min_test_distance = task_df["optimal_test_distance_norm"].abs().min()

        # First, add the golden row
        train_str = f"{golden_row['train_enc_len']:.2f}"
        test_str = f"{golden_row['test_enc_len']:.2f}"

        if show_full_table:
            golden_row_str_values = [
                TASK_NAME_MAP.get(task, task),
                "(Golden)",
                f"({golden_row['approx_g_enc_len']})",
                f"({golden_row['l1_value']:.2f})",
                f"({golden_row['l2_value']:.2f})",
                f"({train_str})",
                f"({test_str})",
                f"({golden_row['optimal_train_distance_norm'] * 100:.1f})",
                f"({golden_row['optimal_test_distance_norm'] * 100:.1f})",
            ]
        else:
            golden_row_str_values = [
                TASK_NAME_MAP.get(task, task),
                "(Golden)",
                f"({golden_row['approx_g_enc_len']})",
                f"({train_str})",
                f"({test_str})",
                f"({golden_row['optimal_train_distance_norm'] * 100:.1f})",
                f"({golden_row['optimal_test_distance_norm'] * 100:.1f})",
            ]

        latex_string += " & ".join(golden_row_str_values) + " \\\\\n"

        # Then add all other rows
        for _, row in task_df.iterrows():
            # Format values, applying bolding if close to the minimum
            test_dg_str = f"{row['test_enc_len']:.2f}"
            train_dg_str = f"{row['train_enc_len']:.2f}"

            train_dist_val = row["optimal_train_distance_norm"]
            train_dist_str = f"{train_dist_val * 100:.1f}"
            if np.isclose(abs(train_dist_val), min_train_distance):
                train_dist_str = f"\\textbf{{{train_dist_str}}}"

            test_dist_val = row["optimal_test_distance_norm"]
            test_dist_str = f"{test_dist_val * 100:.1f}"
            if np.isclose(abs(test_dist_val), min_test_distance):
                test_dist_str = f"\\textbf{{{test_dist_str}}}"

            if show_full_table:
                row_str_values = [
                    EXPERIMENT_NAME_MAP.get(row["experiment"], row["experiment"]),
                    f"{row['approx_g_enc_len']}",
                    f"{row['l1_value']:.2f}",
                    f"{row['l2_value']:.2f}",
                    train_dg_str,
                    test_dg_str,
                    train_dist_str,
                    test_dist_str,
                ]
            else:
                row_str_values = [
                    EXPERIMENT_NAME_MAP.get(row["experiment"], row["experiment"]),
                    f"{row['approx_g_enc_len']}",
                    train_dg_str,
                    test_dg_str,
                    train_dist_str,
                    test_dist_str,
                ]

            latex_string += "& " + " & ".join(row_str_values) + " \\\\\n"

    latex_string += "\\bottomrule\n"
    latex_string += "\\end{tabular}"

    print(latex_string)


def main():
    if not EXPERIMENTS_RESULTS_CSV.exists():
        logger.error(f"Results file not found: {EXPERIMENTS_RESULTS_CSV}")
        logger.error("Please run the training script first.")
        return

    df = pd.read_csv(EXPERIMENTS_RESULTS_CSV)
    df = enhance_df(df)

    create_latex_table(df)
    create_latex_table(df, show_full_table=True)
    create_target_plot(df, compare_to_optimal=True, show_golden_axes=True)


if __name__ == "__main__":
    main()
