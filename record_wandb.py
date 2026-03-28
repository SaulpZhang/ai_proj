import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="qijizhang5-city-university-of-hong-kong",
    # Set the wandb project where this run will be logged.
    project="funsearch",
    # Track hyperparameters and run metadata.
    name="funsearch_ori",
)

def get_wandb_run():
    return run