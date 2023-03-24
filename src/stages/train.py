import argparse
import joblib
import pandas as pd
from pathlib import Path
from typing import Text

from src.utils.train import train
from src.utils.logs import get_logger


def parse_args():
    """ Get command line arguments """
    parser = argparse.ArgumentParser(description='Train classifier model on training data and store model')
    # optional (keyword) argument with '-i' flag to accept pre-existing input file(s)
    parser.add_argument('-i',
                        '--input',
                        help='Path to training data input csv',
                        type=Path,
                        default=None,
                        required=True)

    # optional (keyword) argument with '-o' flag to specify path to output file(s) that will be generated
    parser.add_argument('-o',
                        '--output',
                        help='Path to store the generated trained model as a .joblib file within project',
                        type=Path,
                        default=None,
                        required=True)
    args = parser.parse_args()
    return args


def main() -> None:
    
    args = parse_args()

    logger = get_logger('TRAIN', log_level="INFO")

    # Get the name of an estimator (i.e. model)
    logger.info('Get estimator name')
    estimator_name = 'logreg'
    logger.info(f'Estimator: {estimator_name}')

    # Load training data
    logger.info('Load train dataset')
    train_df = pd.read_csv(args.input)

    """
    train:

        cv: 10
        estimator_name: 'logreg'  # sets which single model is selected
        estimators:
            logreg: # sklearn.linear_model_LogisticRegression
            param_grid: # params of GridSearchCV constructor
                C: [0.001]
                max_iter: [100]
                solver: ['lbfgs']
                multi_class: ['multinomial']
            svm: # sklearn.svm.SVC
            param_grid: 
                C: [0.1, 1.0]
                kernel: ['rbf', 'linear']
                gamma: ['scale']
                degree: [3, 5]
        model_path: models/model.joblib
    """

    # Train the model-- see train\train.py for details
    logger.info('Train model')

    # params for GridSeachCV constructor
    param_grid = {
                'C': [0.001],
                'max_iter': [100],
                'solver': ['lbfgs'],
                'multi_class': ['multinomial']
                }

    model = train(
        df= train_df,
        target_column= 'target',
        estimator_name= estimator_name,
        param_grid= param_grid,
        cv= 10
    )

    logger.info(f'Best score: {model.best_score_}')

    # Save the model
    logger.info('Save model')
    models_path= args.output
    joblib.dump(model, models_path)


if __name__ == '__main__':

    main()