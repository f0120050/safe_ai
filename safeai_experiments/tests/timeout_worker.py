"""Synthetic worker for testing batch timeout handling."""

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--mode")
parser.add_argument("--attack")
parser.add_argument("--repetition")
args = parser.parse_args()

print(
    f"Simulated worker started: repetition={args.repetition}",
    flush=True,
)

# Deliberately exceed the batch timeout.
time.sleep(30)

print("Unexpected: simulated worker completed.", flush=True)
