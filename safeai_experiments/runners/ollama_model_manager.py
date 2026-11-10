"""Manage Ollama model residency before SafeAI experiments."""

import json
import re
import subprocess


def loaded_models():
    """Return currently loaded Ollama model information."""
    result = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )

    lines = result.stdout.strip().splitlines()

    if len(lines) <= 1:
        return []

    # Parse model names from the first column.
    return [line.split()[0] for line in lines[1:] if line.strip()]


def stop_other_models(target_model):
    """Unload all models except the requested target."""
    models = loaded_models()
    stopped = []

    for model in models:
        if model == target_model:
            continue

        subprocess.run(
            ["ollama", "stop", model],
            check=True,
            timeout=30,
        )
        stopped.append(model)

    return stopped



def get_model_processor(target_model):
    """Return Ollama's reported processor allocation for a loaded model."""
    result = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )

    lines = result.stdout.splitlines()

    # Header-only output means no Ollama models are loaded.
    if len(lines) <= 1:
        return None

    header = lines[0]
    match = re.search(r"\bPROCESSOR\b", header)

    if match is None:
        raise RuntimeError("PROCESSOR column not found in ollama ps.")

    start = match.start()
    end = header.find("CONTEXT", match.end())

    for line in lines[1:]:
        columns = line.split()

        if columns and columns[0] == target_model:
            if end == -1:
                return line[start:].strip()
            return line[start:end].strip()

    return None


def gpu_ready(target_model, minimum_gpu_percent=80):
    """Evaluate GPU allocation reported by Ollama."""
    processor = get_model_processor(target_model)

    if processor is None:
        return False, "Model is not loaded."

    mixed = re.fullmatch(
        r"(\d+)%/(\d+)% CPU/GPU", processor
    )

    if mixed:
        gpu_percent = int(mixed.group(2))
    elif processor == "100% GPU":
        gpu_percent = 100
    elif processor == "100% CPU":
        gpu_percent = 0
    else:
        return False, f"Unknown processor format: {processor}"

    return (
        gpu_percent >= minimum_gpu_percent,
        f"GPU allocation: {gpu_percent}%",
    )



def prepare_model(target_model, minimum_gpu_percent=80):
    """Load an Ollama model and verify GPU placement."""

    # Free GPU memory occupied by other Ollama models.
    stopped = stop_other_models(target_model)

    # Ollama /api/generate supports preloading without a prompt.
    import requests

    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": target_model,
            "keep_alive": "15m",
        },
        timeout=120,
    )
    response.raise_for_status()

    ready, reason = gpu_ready(
        target_model,
        minimum_gpu_percent=minimum_gpu_percent,
    )

    return {
        "model": target_model,
        "stopped_models": stopped,
        "gpu_ready": ready,
        "reason": reason,
    }


def print_model_status():
    """Display current model residency without modifying anything."""
    result = subprocess.run(
        ["ollama", "ps"],
        text=True,
        check=True,
        timeout=15,
    )
    return result.returncode


if __name__ == "__main__":
    print_model_status()


def stop_all_models():
    """Unload every currently loaded Ollama model."""
    models = loaded_models()
    stopped = []

    for model in models:
        subprocess.run(
            ["ollama", "stop", model],
            check=True,
            timeout=30,
        )
        stopped.append(model)

    return stopped
