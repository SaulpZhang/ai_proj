from funsearch.implementation import config as config_lib
from funsearch.implementation import funsearch
import dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    with open("templete/bin_online.txt", "r") as f:
        specification = f.read()
    
    or3 = dataset.get_dataset_or3()

    logging.info("Starting funsearch...")
    funsearch.main(specification=specification,inputs=[or3], config=config_lib.Config())
    