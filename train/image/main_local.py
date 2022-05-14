import argparse 
import pandas as pd
import warnings
# Local imports
from training import Training
from utils import get_column_dtype, check_and_create_folder, get_column_datetime, remove_large_categorical_columns, datetime_to_int, int_to_datetime

warnings.filterwarnings("ignore")

# Utils code
def convert_to_list(string):
    str_list = string.split(',')
    str_list = [x.strip() for x in str_list]
    return str_list 


def preprocessing(dataset):
    dataset, datetime_cols = get_column_datetime(dataset)
    dataset = datetime_to_int(dataset, datetime_cols)
    categorical_col, numerical_col, integer_col = get_column_dtype(dataset, threshold=15)
    dataset, removed_categorical_cols = remove_large_categorical_columns(dataset, categorical_col, threshold=50)
    return dataset, categorical_col, integer_col, datetime_cols

def model_training(args):
    """
    Model training
    """
    data_path = args.raw_data_path
    problem_type = convert_to_list(args.problem_type)
    problem_type = {problem_type[0]: problem_type[1]}
    epochs = args.epochs
    save_weights = args.save_weights
    save_frequency = args.save_frequency
    save_path = args.save_path

    dataset = pd.read_csv(data_path)
    dataset, categorical_columns, integer_columns, datetime_columns= preprocessing(dataset)
    dataset.to_csv(data_path, index=False)
    mixed_columns = {}
    log_columns = []

    check_and_create_folder(save_path)

    model_training = Training(
        input_csv_path=data_path,
        categorical_columns=categorical_columns,
        log_columns=log_columns,
        mixed_columns=mixed_columns,
        integer_columns=integer_columns,
        problem_type=problem_type,
        epochs=epochs,
        save_weights=save_weights,
        save_frequency=save_frequency,
        save_path=save_path,
        model_name=None,
    )
    model_training.init_model()
    model_training.train()

    synthetic_data = pd.read_csv(save_path + '/sample_syn_data.csv')
    synthetic_data = int_to_datetime(synthetic_data, datetime_columns)
    synthetic_data.to_csv(save_path + '/sample_syn_data.csv', index=False)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or infer')
    parser.add_argument('--raw_data_path', type=str, help='raw data path')
    parser.add_argument('--problem_type', type=str, help='problem type')
    parser.add_argument('--epochs', type=int, help='epochs')
    parser.add_argument('--save_weights', type=bool, help='save weights')
    parser.add_argument('--save_frequency', type=int, help='save frequency')
    parser.add_argument('--save_path', type=str, help='save path')
    parser.add_argument('--model_name', type=str, help='model name')
    args = parser.parse_args()

    if args.mode == 'train':
        model_training(args)
