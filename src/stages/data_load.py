import argparse
import pandas as pd
from pathlib import Path
from typing import Text


from src.utils.logs import get_logger

def parse_args():
    """ Get command line arguments """
    parser = argparse.ArgumentParser(description='Load raw data from external source and store in data/raw folder inside project')

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
                        help='Path to store the raw data csv within project',
                        type=Path,
                        default=None,
                        required=True)
    args = parser.parse_args()
    return args


def main() -> None:

    # parse cli arguments
    args = parse_args()

    logger = get_logger('DATA_LOAD', log_level='INFO')

    logger.info('Get dataset')

    dataset = pd.read_csv(args.input)
    dataset.rename(
        columns=lambda colname: colname.strip(' (cm)').replace(' ', '_'),
        inplace=True
    )

    logger.info('Save raw data')
    dataset.to_csv(args.output, index=False)


if __name__ == '__main__':

    main()

