import argparse
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from typing import Text, Dict


from src.utils.visualize import plot_confusion_matrix
from src.utils.logs import get_logger

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--input", help="input file", nargs=2, required=True)
#     parser.add_argument("-o", "--output", help="output file", nargs=2, required=True)
#     return parser.parse_args()

def parse_args():
    """ Get command line arguments """
    parser = argparse.ArgumentParser(description='Evaluate performance of trained model on test set')
    # optional (keyword) argument with '-i' flag to accept pre-existing input file(s)
    parser.add_argument('-i',
                        '--input',
                        help='Path to trained model and csv file with test data, respectively',
                        type=Path,
                        default=None,
                        required=True,
                        nargs=2)

    # optional (keyword) argument with '-o' flag to specify path to output file(s) that will be generated
    parser.add_argument('-o',
                        '--output',
                        help='Paths of the .json file and .png file that will store the model metrics and confusion matrix, respectively',
                        type=Path,
                        default=None,
                        required=True,
                        nargs=2)
    args = parser.parse_args()
    return args


def main() -> None:

    args = parse_args()

    logger = get_logger('EVALUATE', log_level='INFO')

    logger.info('Load model')
    model_path = args.input[0]
    model = joblib.load(model_path)

    logger.info('Load test dataset')
    test_df = pd.read_csv(args.input[1])

    logger.info('Evaluate (build reports)')
    target_column='target'
    y_test = test_df.loc[:, target_column].values
    X_test = test_df.drop(target_column, axis=1).values

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=prediction, normalize=True)
    precision = precision_score(y_true=y_test, y_pred=prediction, average='macro')
    recall = recall_score(y_true=y_test, y_pred=prediction, average='macro')
    f1 = f1_score(y_true=y_test, y_pred=prediction, average='macro')
    cm = confusion_matrix(prediction, y_test)
    report = {
        'target_names': (np.sort(test_df[target_column].unique())).tolist(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm,
        'actual': y_test,
        'predicted': prediction
    }

    logger.info('Save model metrics')
    # save metrics file
    metrics_path = args.output[0]

    json.dump(
        obj={
            'accuracy': report['accuracy'],
            'precision': report['precision'],
            'recall': report['recall'],
            'f1_score': report['f1']
        },
        fp=open(metrics_path, 'w')
    )

    logger.info(f'Model metrics file saved to : {metrics_path}')


    logger.info('Save confusion matrix')

    # save confusion_matrix.png
    plt = plot_confusion_matrix(cm=report['cm'],
                                target_names= report['target_names'],
                                normalize=False
    )

    confusion_matrix_png_path = args.output[1]
    plt.savefig(confusion_matrix_png_path)
    logger.info(f'Confusion matrix saved to: {confusion_matrix_png_path}')


if __name__ == '__main__':

    main()








