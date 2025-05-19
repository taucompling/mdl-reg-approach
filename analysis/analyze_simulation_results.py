import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
NETWORKS_PATH = BASE_DIR / "networks"
SIMULATIONS_CSV = BASE_DIR / "new_obj_function_simulations.csv"
OUTPUT_DIR = BASE_DIR / "obj_function_analysis_results"
OUTPUT_RESULT_FILE = OUTPUT_DIR / "new_obj_function_simulation_results.csv"


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


TASK_NAME_MAP = {
    "an_bn": "a$^n$b$^n$",
    "an_bn_cn": "a$^n$b$^n$c$^n$",
    "dyck_1": "Dyck-1",
    "dyck_2": "Dyck-2",
    "arithmetic": "Arithmetic",
    "toy_english": "Toy English",
}

OBJECTIVE_NAME_MAP = {
    "MDL": "MDL (|H|)",
    "|D:G| + L1": "$L_1$",
    "|D:G| + L2": "$L_2$",
    "|D:G| + Limit |G|": "None (Lim. $|H|$)",
    "|D:G|": "None",
    "Golden": "Golden",
}

LEGEND_ORDER = [
    "Golden",
    "MDL",
    "|D:G| + L1",
    "|D:G| + L2",
    "|D:G| + Limit |G|",
    "|D:G|",
]
OBJECTIVE_TO_COLOR = {
    "MDL": (*sns.color_palette("colorblind")[2], 0.8),
    "Golden": (*sns.color_palette("colorblind")[1], 0.8),
    "|D:G| + L1": (*sns.color_palette("colorblind")[0], 0.8),
    "|D:G| + L2": (*sns.color_palette("colorblind")[3], 0.8),
    "|D:G| + Limit |G|": (*sns.color_palette("colorblind")[4], 0.8),
}


def sort_objectives(objectives):
    """Sort objectives based on the predefined LEGEND_ORDER."""
    sort_key_func = (
        lambda obj: LEGEND_ORDER.index(obj)
        if obj in LEGEND_ORDER
        else len(LEGEND_ORDER)
    )
    return sorted(list(objectives), key=sort_key_func)


COLUMN_NAME_MAP = {
    "g": "$|H|$",
    "l1_reg": "$L_1$",
    "l2_reg": "$L_2$",
    "train_d_given_g": "Train $|D:H|$",
    "test_d_given_g": "Test $|D:H|$",
    "test_d_given_g_no_overlap": "Test $|D:H|$ (NoOverlap)",
    "optimal_train_distance_norm": "$\\Delta^{Train}_{Optim}$ (\\%)",
    "optimal_test_distance_norm": "$\\Delta^{Test}_{Optim}$ (\\%)",
    "optimal_test_distance_no_overlap_norm": "$\\Delta^{Test}_{Optim,NoOverlap}$ (\\%)",
}


def create_target_plot(
    df: pd.DataFrame, compare_to_optimal: bool = False, show_golden_axes: bool = False
) -> None:
    set_neurips_style()

    layout = [["an_bn", "dyck_1", "arithmetic"], ["an_bn_cn", "dyck_2", "toy_english"]]
    n_rows, n_cols = len(layout), max(len(row) for row in layout)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.5), squeeze=False
    )

    if compare_to_optimal:
        x_column = "optimal_train_distance_norm"
        y_column = "optimal_test_distance_norm"
        reference_point = "Optimal"
        # Get all unique objectives initially
        temp_objectives = df["objective"].unique()
        if (
            show_golden_axes
        ):  # If also showing golden axes, filter out Golden from points
            objectives_to_plot = [obj for obj in temp_objectives if obj != "Golden"]
        else:  # Otherwise, include Golden as a point if present
            objectives_to_plot = temp_objectives
    else:
        x_column = "golden_train_distance_norm"
        y_column = "golden_test_distance_norm"
        reference_point = "Golden"
        objectives_to_plot = [
            obj for obj in df["objective"].unique() if obj != "Golden"
        ]

    objectives_to_plot = sort_objectives(objectives_to_plot)

    for row_idx, row_tasks in enumerate(layout):
        for col_idx, task in enumerate(row_tasks):
            ax = axes[row_idx, col_idx]
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
            for objective_point in (
                objectives_to_plot
            ):  # objectives_to_plot already accounts for show_golden_axes
                obj_df = task_df[task_df["objective"] == objective_point]
                if (
                    not obj_df.empty
                    and x_column in obj_df.columns
                    and y_column in obj_df.columns
                ):
                    x_val, y_val = (
                        obj_df[x_column].iloc[0] * 100.0,
                        obj_df[y_column].iloc[0] * 100.0,
                    )
                    if not (
                        pd.isna(x_val)
                        or pd.isna(y_val)
                        or not np.isfinite(x_val)
                        or not np.isfinite(y_val)
                    ):
                        points_to_draw.append(
                            {"x": x_val, "y": y_val, "objective": objective_point}
                        )
                        max_abs_val = max(max_abs_val, abs(x_val), abs(y_val))

            golden_x_val, golden_y_val = None, None
            if compare_to_optimal and show_golden_axes:
                golden_df = task_df[task_df["objective"] == "Golden"]
                if (
                    not golden_df.empty
                    and "optimal_train_distance_norm" in golden_df.columns
                    and "optimal_test_distance_norm" in golden_df.columns
                ):
                    _gx = golden_df["optimal_train_distance_norm"].iloc[0] * 100.0
                    _gy = golden_df["optimal_test_distance_norm"].iloc[0] * 100.0
                    if not (
                        pd.isna(_gx)
                        or pd.isna(_gy)
                        or not np.isfinite(_gx)
                        or not np.isfinite(_gy)
                    ):
                        golden_x_val, golden_y_val = _gx, _gy
                        max_abs_val = max(
                            max_abs_val, abs(golden_x_val), abs(golden_y_val)
                        )

            limit = max_abs_val * 1.2 if max_abs_val > 0 else 10
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal", adjustable="box")

            if compare_to_optimal:
                # Arrow parameters to indicate overfitting region (top-left)
                # Start from near origin, point towards top-left
                arrow_origin_x = 0.0
                arrow_origin_y = 0.0

                arrow_length_factor = 0.45  # How far the arrow extends
                arrow_dx = -arrow_length_factor * limit
                arrow_dy = arrow_length_factor * limit

                # Arrow style (less dramatic)
                head_w = 0.035 * limit  # head_width relative to plot limit
                head_l = 0.055 * limit  # head_length relative to plot limit

                ax.arrow(
                    arrow_origin_x,
                    arrow_origin_y,
                    arrow_dx,
                    arrow_dy,
                    head_width=head_w,
                    head_length=head_l,
                    fc="dimgray",  # fill color of head
                    ec="dimgray",  # edge color of head and line
                    lw=1.2,  # line width
                    alpha=0.6,
                    zorder=3,  # Below points and circles
                    length_includes_head=True,  # dx, dy define up to tip of arrow
                )

                # Text "Overfit"
                # Position text near the center of the arrow
                mid_arrow_x = arrow_origin_x + arrow_dx * 0.5
                mid_arrow_y = arrow_origin_y + arrow_dy * 0.5

                # Small offset to position text slightly "above" (top-right) the arrow line
                # This helps avoid the text sitting directly on the arrow if rotation makes it hard to read
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
                    rotation=-45,  # Aligns with an arrow from (close to) origin to top-left
                    zorder=4,  # Above arrow, below points/circles
                )

            circle_radii_proportions = [0.25, 0.5, 0.75, 1.0]

            for i, prop in enumerate(circle_radii_proportions):
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

            for point_data in points_to_draw:
                obj_df = task_df[
                    task_df["objective"] == point_data["objective"]
                ]  # Re-fetch for safety, though data is in point_data
                x, y = point_data["x"], point_data["y"]

                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=10,
                    color=OBJECTIVE_TO_COLOR[point_data["objective"]],
                    markeredgewidth=0.5,
                    markeredgecolor="black",
                    linestyle="",
                    label=point_data["objective"]
                    if row_idx == 0 and col_idx == 0
                    else "",
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
                    color=OBJECTIVE_TO_COLOR["Golden"],
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                    zorder=5,
                )
                ax.axhline(
                    y=golden_y_val,
                    color=OBJECTIVE_TO_COLOR["Golden"],
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                    zorder=5,
                )

            ax.axhline(
                0, color="black", linestyle="-", alpha=0.0, linewidth=1.5, zorder=1
            )
            ax.axvline(
                0, color="black", linestyle="-", alpha=0.0, linewidth=1.5, zorder=1
            )
            ax.grid(False)

    for row_idx in range(n_rows):
        for col_idx in range(len(layout[row_idx]), n_cols):
            axes[row_idx, col_idx].set_visible(False)

    actual_objectives_for_legend_points = sort_objectives(list(objectives_to_plot))

    handles = []
    labels = []
    for obj_name in actual_objectives_for_legend_points:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=OBJECTIVE_TO_COLOR[obj_name],
                markeredgewidth=1.0,
                markeredgecolor="black",
                markersize=8,
                linestyle="",
            )
        )
        labels.append(OBJECTIVE_NAME_MAP.get(obj_name, obj_name))

    if compare_to_optimal and show_golden_axes:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=OBJECTIVE_TO_COLOR["Golden"],
                linestyle="--",
                linewidth=2.5,
            )
        )
        labels.append('Manual "Golden" Network')

    common_xlabel = f"Δ Train |D:H| from {reference_point} (%)"
    common_ylabel = f"Δ Test |D:H| from {reference_point} (%)"
    fig.supylabel(common_ylabel, x=0.01, fontsize=plt.rcParams["axes.labelsize"])
    fig.supxlabel(common_xlabel, y=0.05, fontsize=plt.rcParams["axes.labelsize"])

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(labels),
        frameon=True,
        framealpha=1.0,
        edgecolor="0.8",
    )

    fig.tight_layout()

    bottom_margin = 0.10 if labels else 0.08
    left_margin = 0.06
    right_margin = 0.95
    top_margin = 0.95
    fig.subplots_adjust(
        left=left_margin,
        bottom=bottom_margin,
        right=right_margin,
        top=top_margin,
        hspace=0.2,
    )

    comparison_type = "optimal" if compare_to_optimal else "golden"
    golden_suffix = (
        "_with_golden_axes" if compare_to_optimal and show_golden_axes else ""
    )
    plot_path = OUTPUT_DIR / f"target_plot_from_{comparison_type}{golden_suffix}.png"
    fig.savefig(plot_path, dpi=400)
    logger.info(f"Target plot saved as '{plot_path}'")


def create_latex_table(df: pd.DataFrame, show_full_table: bool = False) -> None:
    golden_df = df[df["objective"] == "Golden"]
    regular_df = df[df["objective"] != "Golden"]

    latex_table = _build_table_header(show_full_table)

    tasks = regular_df["task"].unique()
    for i, task in enumerate(tasks):
        if i > 0:
            latex_table += "\\midrule\n"

        task_df = regular_df[regular_df["task"] == task].copy()
        golden_row = golden_df[golden_df["task"] == task].iloc[0]

        min_values = _get_minimum_values(task_df)
        latex_table += _format_golden_row(task, golden_row, show_full_table)

        for _, row in task_df.iterrows():
            latex_table += _format_regular_row(row, min_values, show_full_table)

    latex_table += "\\bottomrule\n\\end{tabular}"
    print(latex_table)


def _build_table_header(show_full_table: bool = True) -> str:
    if show_full_table:
        columns = [
            "Task",
            "Regularizer",
            COLUMN_NAME_MAP["g"],
            COLUMN_NAME_MAP["l1_reg"],
            COLUMN_NAME_MAP["l2_reg"],
            COLUMN_NAME_MAP["train_d_given_g"],
            COLUMN_NAME_MAP["test_d_given_g"],
            COLUMN_NAME_MAP["optimal_train_distance_norm"],
            COLUMN_NAME_MAP["optimal_test_distance_norm"],
        ]
        header = "\\begin{tabular}{llccccccc}\n"
    else:
        columns = [
            "Task",
            "Regularizer",
            COLUMN_NAME_MAP["g"],
            COLUMN_NAME_MAP["train_d_given_g"],
            COLUMN_NAME_MAP["test_d_given_g"],
            COLUMN_NAME_MAP["optimal_train_distance_norm"],
            COLUMN_NAME_MAP["optimal_test_distance_norm"],
        ]
        header = "\\begin{tabular}{llccccc}\n"

    header += "\\toprule\n"
    header += " & ".join(columns) + " \\\\\n"
    header += "\\midrule\n"
    return header


def _get_minimum_values(task_df: pd.DataFrame) -> dict:
    return {
        "train_distance": task_df["optimal_train_distance_norm"].abs().min(),
        "test_distance": task_df["optimal_test_distance_norm"].abs().min(),
    }


def _format_metric_value(value: float, min_value: float, precision: int = 1) -> str:
    scaled_value = value * 100.0
    formatted = f"{scaled_value:.{precision}f}"
    if np.isclose(abs(value), min_value):
        return f"\\textbf{{{formatted}}}"
    return formatted


def _format_dg_value(value: float, is_infinite: bool) -> str:
    formatted = f"{value:.2f}"
    if is_infinite:
        formatted += "$^*$"
    return formatted


def _format_golden_row(task: str, row: pd.Series, show_full_table: bool = True) -> str:
    train_str = _format_dg_value(row["train_d_given_g"], row["is_inf_train_d_given_g"])
    test_str = _format_dg_value(row["test_d_given_g"], row["is_inf_test_d_given_g"])

    if show_full_table:
        values = [
            TASK_NAME_MAP[task],
            f"({OBJECTIVE_NAME_MAP['Golden']})",
            f"({row['g']})",
            f"({row['l1_reg']:.2f})",
            f"({row['l2_reg']:.2f})",
            f"({train_str})",
            f"({test_str})",
            f"({row['optimal_train_distance_norm'] * 100.0:.1f})",
            f"({row['optimal_test_distance_norm'] * 100.0:.1f})",
        ]
    else:
        values = [
            TASK_NAME_MAP[task],
            f"({OBJECTIVE_NAME_MAP['Golden']})",
            f"({row['g']})",
            f"({train_str})",
            f"({test_str})",
            f"({row['optimal_train_distance_norm'] * 100.0:.1f})",
            f"({row['optimal_test_distance_norm'] * 100.0:.1f})",
        ]

    return " & ".join(values) + " \\\\\n"


def _format_regular_row(
    row: pd.Series, min_values: dict, show_full_table: bool = True
) -> str:
    train_dg_str = _format_dg_value(
        row["train_d_given_g"], row["is_inf_train_d_given_g"]
    )
    test_dg_str = _format_dg_value(row["test_d_given_g"], row["is_inf_test_d_given_g"])

    train_dist_str = _format_metric_value(
        row["optimal_train_distance_norm"], min_values["train_distance"], precision=1
    )

    test_dist_str = _format_metric_value(
        row["optimal_test_distance_norm"], min_values["test_distance"], precision=1
    )

    if show_full_table:
        values = [
            OBJECTIVE_NAME_MAP[row["objective"]],
            f"{row['g']}",
            f"{row['l1_reg']:.2f}",
            f"{row['l2_reg']:.2f}",
            train_dg_str,
            test_dg_str,
            train_dist_str,
            test_dist_str,
        ]
    else:
        values = [
            OBJECTIVE_NAME_MAP[row["objective"]],
            f"{row['g']}",
            train_dg_str,
            test_dg_str,
            train_dist_str,
            test_dist_str,
        ]

    return "& " + " & ".join(values) + " \\\\\n"


def enhance_df(df: pd.DataFrame) -> pd.DataFrame:
    # Distance from optimal values
    df["optimal_train_distance"] = df["train_d_given_g"] - df["optimal_train_d_given_g"]
    df["optimal_test_distance"] = df["test_d_given_g"] - df["optimal_test_d_given_g"]
    df["optimal_test_distance_no_overlap"] = (
        df["test_d_given_g_no_overlap"] - df["optimal_test_d_given_g_no_overlap"]
    )

    df["optimal_train_distance_norm"] = (
        df["optimal_train_distance"] / df["optimal_train_d_given_g"]
    )
    df["optimal_test_distance_norm"] = (
        df["optimal_test_distance"] / df["optimal_test_d_given_g"]
    )
    df["optimal_test_distance_no_overlap_norm"] = (
        df["optimal_test_distance_no_overlap"] / df["optimal_test_d_given_g_no_overlap"]
    )

    # Distance from golden values
    golden_values = (
        df[df["objective"].eq("Golden")]
        .loc[
            :,
            ["task", "train_d_given_g", "test_d_given_g", "test_d_given_g_no_overlap"],
        ]
        .rename(
            columns={
                "train_d_given_g": "golden_train_d_given_g",
                "test_d_given_g": "golden_test_d_given_g",
                "test_d_given_g_no_overlap": "golden_test_d_given_g_no_overlap",
            }
        )
        .set_index("task")
    )

    df["golden_train_d_given_g"] = df["task"].map(
        golden_values["golden_train_d_given_g"]
    )
    df["golden_test_d_given_g"] = df["task"].map(golden_values["golden_test_d_given_g"])
    df["golden_test_d_given_g_no_overlap"] = df["task"].map(
        golden_values["golden_test_d_given_g_no_overlap"]
    )

    df["golden_train_distance"] = df["train_d_given_g"] - df["golden_train_d_given_g"]
    df["golden_test_distance"] = df["test_d_given_g"] - df["golden_test_d_given_g"]
    df["golden_test_distance_no_overlap"] = (
        df["test_d_given_g"] - df["golden_test_d_given_g_no_overlap"]
    )

    df["golden_train_distance_norm"] = (
        df["golden_train_distance"] / df["golden_train_d_given_g"]
    )
    df["golden_test_distance_norm"] = (
        df["golden_test_distance"] / df["golden_test_d_given_g"]
    )
    df["golden_test_distance_no_overlap_norm"] = (
        df["golden_test_distance_no_overlap"] / df["golden_test_d_given_g_no_overlap"]
    )

    return df


def main():
    df = pd.read_csv(OUTPUT_RESULT_FILE)
    df = enhance_df(df)

    create_latex_table(df)
    create_latex_table(df, show_full_table=True)
    create_target_plot(df, compare_to_optimal=True, show_golden_axes=True)


if __name__ == "__main__":
    main()
