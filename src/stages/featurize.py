import argparse
import pandas as pd
from pathlib import Path
from typing import Text


from src.utils.logs import get_logger

def parse_args():
    """ Get command line arguments """
    parser = argparse.ArgumentParser(description='Featurize raw data and stored featurized data')
    # optional (keyword) argument with '-i' flag to accept pre-existing input file(s)
    parser.add_argument('-i',
                        '--input',
                        help='Path to raw data input csv',
                        type=Path,
                        default=None,
                        required=True)

    # optional (keyword) argument with '-o' flag to specify path to output file(s) that will be generated
    parser.add_argument('-o',
                        '--output',
                        help='Path to store the featurized data as csv within project',
                        type=Path,
                        default=None,
                        required=True)
    args = parser.parse_args()
    return args


def main() -> None:

    args = parse_args()

    logger = get_logger('FEATURIZE', log_level='INFO')

    logger.info('Load raw data')
    dataset = pd.read_csv(args.input)

    logger.info('Extract features')
    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length']/ dataset['petal_width']
    featured_dataset = dataset[[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
        'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
        'target'
    ]]

    logger.info('Save features')
    features_path = args.output
    featured_dataset.to_csv(features_path, index=False)

if __name__ == '__main__':

    main()