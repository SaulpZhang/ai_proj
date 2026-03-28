import wandb

run = None

# Start a new wandb run to track this script.


def get_wandb_run():
    global run
    if run is None:
        run = wandb.init(
            id = "funsearch_ori",
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="qijizhang5-city-university-of-hong-kong",
            # Set the wandb project where this run will be logged.
            project="funsearch",
            # Track hyperparameters and run metadata.
            name="funsearch_ori",
        )
    return run