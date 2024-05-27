# imports
import sys
import os
import csv
import json

import click
from reinvent.config_parse import read_smiles_csv_file

from linkinvent_sampler import LinkinventSampler


# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# This arguments is reserved for testing or
# running model locally or in notebook.
is_debug = sys.argv[3] == "True" if len(sys.argv) > 3 else False

# Name of the log file.
# Only write if `is_debug` is True.
log_file = output_file + ".json"

batch_size = 1000
num_input_smiles = 0
input_smiles = None


if os.path.exists(input_file):
    input_smiles = read_smiles_csv_file(input_file, columns=0, header=True)
    num_input_smiles = len(input_smiles)

else:
    click.echo(click.style(f"[INPUT_FILE]: {input_file} doesn't exist.", fg="red"))


if not os.path.exists(os.path.dirname(os.path.abspath(output_file))):
    click.echo(
        click.style(
            f"[OUTPUT_DIR]: {os.path.dirname(output_file)} doesn't exist.", fg="red"
        )
    )

linkinvent_sampler = LinkinventSampler(batch_size=batch_size, is_debug=is_debug)

if is_debug:
    click.echo(click.style("Running linkinvent prior", fg="white", bg="green"))

outputs, _, log_linkinvent = linkinvent_sampler.generate(input_smiles=input_smiles)


input_len = len(input_smiles)
output_len = len(outputs)

assert input_len == output_len

HEADER = [f"smi_{x}" for x in range(batch_size)]

with open(output_file, "w", newline="") as fp:
    csv_writer = csv.writer(fp)
    # First Row: Header
    # Second Row: Generated Smiles (Output)
    csv_writer.writerows([HEADER])
    csv_writer.writerows(outputs)


if is_debug:
    log = {
        "start": log_linkinvent["start"],
        "end": log_linkinvent["end"],
        "input_smiles": log_linkinvent["input_smiles"],
        "total": log_linkinvent["total"],
        "expected": batch_size * num_input_smiles,
    }

    with open(os.path.abspath(log_file), "w", newline="\n") as fp:
        json.dump(log, fp)
