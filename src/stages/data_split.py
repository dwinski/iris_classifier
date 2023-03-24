import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Text


from src.utils.logs import get_logger


def parse_args():
    """ Get command line arguments """
    parser = argparse.ArgumentParser(description='Split featurized data in train and test sets and store as csv files')
    # optional (keyword) argument with '-i' flag to accept pre-existing input file(s)
    parser.add_argument('-i',
                        '--input',
                        help='Path to featurized data used as input',
                        type=Path,
                        default=None,
                        required=True)

    # optional (keyword) argument with '-o' flag to specify path to output file(s) that will be generated
    parser.add_argument('-o',
                        '--output',
                        help='Paths of the csv files that will store the split train.csv and test.csv files ',
                        type=Path,
                        default=None,
                        required=True,
                        nargs=2)
    args = parser.parse_args()
    return args


def main() -> None:
   
    args = parse_args()

    logger = get_logger('DATA_SPLIT', log_level='INFO')

    logger.info('Load features')
    dataset = pd.read_csv(args.input)

    logger.info('Split features into train and test sets')
    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size = 0.2,
        random_state = 42
    )

    logger.info('Save train and test sets')
    #print(args.output)
    train_csv_path = args.output[0]
    test_csv_path = args.output[1]
    train_dataset.to_csv(train_csv_path, index=False)
    test_dataset.to_csv(test_csv_path, index=False)

if __name__ == '__main__':

    main()