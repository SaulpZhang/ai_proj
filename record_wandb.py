import multiprocessing as mp
import uuid

import wandb


run = None


def get_wandb_run():
    global run
    if run is not None:
        return run

    # Avoid initializing wandb in spawned worker processes.
    if mp.current_process().name != "MainProcess":
        return None

    run_id = f"funsearch_{uuid.uuid4().hex[:10]}"
    run = wandb.init(
        id=run_id,
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="qijizhang5-city-university-of-hong-kong",
        # Set the wandb project where this run will be logged.
        project="funsearch",
        # Track hyperparameters and run metadata.
        name="funsearch_ori",
    )
    return run


def log_metrics(metrics: dict) -> None:
    current_run = get_wandb_run()
    if current_run is not None:
        current_run.log(metrics)
