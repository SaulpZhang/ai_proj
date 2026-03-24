from funsearch.implementation import config as config_lib
from funsearch.implementation import funsearch

if __name__ == "__main__":
    with open("funsearch/implementation/specification_nonsymmetric_admissible_set.txt", "r") as f:
        specification = f.read()
    
    funsearch.main(specification=specification,inputs=[(1,2),(3,4)], config=config_lib.Config())
    