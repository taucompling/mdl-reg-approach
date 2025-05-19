import csv
import os
import pathlib

from loguru import logger

_SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")

_CELL_CHARACTER_LIMIT = 50_000

RESULTS_COLS = [
    "Simulation",
    "Configuration",
    "Seed",
    "Params",
    "Status",
    "Generation",
    "Best island",
    "MDL",
    "|G|",
    "|D:G|",
    "Units",
    "Connections",
    "Accuracy",
    "Deterministic accuracy",
    "Dot",
    "Simulation id",
    "Commit",
    "Started",
    "Finished",
    "Last update",
    "Comments",
    "Training set optimal D:G",
    "Training set num. chars",
    "Average train D:G per character",
    "Optimal average train D:G per character",
    "Test set optimal D:G",
    "Test set num. chars",
    "Test set D:G",
    "Average test D:G per character",
    "Optimal average test D:G per character",
    "Test set accuracy",
    "Test set deterministic accuracy",
    "Test set params",
    "Tag",
    "max_correct",
    "num_correct",
]

_RNN_RESULTS_COLS = [
    "Corpus",
    "Params",
    "Corpus name",
    "Corpus params",
    "Net params",
    "Seed",
    "Status",
    "Network Type",
    "Hidden Units",
    "Regularization",
    "Epochs",
    "Train Accuracy",
    "Train deterministic accuracy",
    "Train symbolic accuracy",
    "Train Loss",
    "Train Cross Entropy",
    "Train Optimal CE",
    "Train num. chars",
    "Average train CE per character",
    "Optimal average train CE per character",
    "Test num. chars",
    "Test CE",
    "Test optimal CE",
    "Test Loss",
    "Average test CE per character",
    "Optimal average test CE per character",
    "Test accuracy",
    "Test deterministic accuracy",
    "Test symbolic accuracy",
    "Simulation id",
    "Started",
    "Finished",
    "Test set params",
    "Comments",
    "Tag",
    "max_correct",
    "num_correct",
]

_COLUMN_TITLES = {
    "results": RESULTS_COLS,
}


def _col_num_to_letter(col):
    # 1 => A
    if col > 26:
        return "A" + _col_num_to_letter(col - 26)
    return chr(64 + col)


def get_spreadsheet():
    import gspread

    return gspread.service_account(".spreadsheet_account.json").open_by_key(
        _SPREADSHEET_ID
    )


def _next_available_row(worksheet):
    str_list = list(filter(None, worksheet.col_values(1)))
    return str(len(str_list) + 1)


def write_to_spreadsheet(worksheet_name, data):
    import gspread
    from requests import exceptions

    column_titles = _COLUMN_TITLES[worksheet_name.lower()]
    column_to_idx = {x.lower(): i + 1 for i, x in enumerate(column_titles)}
    try:
        worksheet = get_spreadsheet().worksheet(worksheet_name)
        simulation_id = data["simulation id"]
        try:
            row = worksheet.find(simulation_id).row
            logger.warning(
                f"Simulation row already exists in spreadsheet: {simulation_id}"
            )
        except (AttributeError, gspread.exceptions.CellNotFound):
            row = _next_available_row(worksheet)

        cell_range = worksheet.range(row, 1, row, len(column_titles))
        for key, val in data.items():
            if type(val) == str:
                val = val[:_CELL_CHARACTER_LIMIT]
            try:
                col = column_to_idx[key.lower()]
                cell_range[col - 1].value = val
            except KeyError:
                logger.error(f"Can't find column {key}")
        worksheet.update_cells(cell_range)
    except (gspread.exceptions.APIError, exceptions.RequestException):
        logger.exception("Google API error")


def log_stats_to_csv(simulation_id, stats_dict):
    csv_dict = {}
    for column in RESULTS_COLS:
        val = str(stats_dict.get(column.lower(), ""))
        csv_dict[column.lower()] = val

    csv_path = pathlib.Path(f"logs/{simulation_id}.csv")
    write_header = False
    if not csv_path.exists():
        write_header = True
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="") as f:
        csv_writer = csv.DictWriter(
            f, fieldnames=tuple(csv_dict.keys()), quoting=csv.QUOTE_MINIMAL
        )
        if write_header:
            csv_writer.writeheader()
        csv_writer.writerow(csv_dict)
