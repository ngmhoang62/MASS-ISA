from datetime import datetime


def log_message(config, stage: str, message: str):
    """Print a timestamped message and append it to the training log."""

    config.log_dir.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%H:%M:%S')}] [{stage}] {message}"
    print(line)
    with (config.log_dir / "training.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
