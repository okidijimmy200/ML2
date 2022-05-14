import argparse
import pandas as pd
import warnings
# Local imports
from inference import Inference
from utils import get_column_dtype
from utils import get_column_dtype, get_column_datetime, int_to_datetime

warnings.filterwarnings("ignore")




def model_inference(args):
    """
    Model training
    """
    pre_trained_paths = {
        "generator": args.generator_path,
        "discriminator": args.discriminator_path,
        "classifier": args.classifier_path,
    }


    dataset = pd.read_csv(args.raw_data_path)
    categorical_columns, integer_columns, _ = get_column_dtype(dataset)
    print("Categorical columns:", categorical_columns)
    print("Integer columns:", integer_columns)
    model_training = Inference(
        raw_data = args.raw_data_path,
        categorical_columns=categorical_columns,
        integer_columns=integer_columns,
        pre_trained_paths = pre_trained_paths,
        synthetic_data_path = args.synthetic_data_path,
    )
    model_training.init_model()
    model_training.get_synthetic_data(args.num_rows)
    syn_data = pd.read_csv(args.synthetic_data_path)
    syn_data, datetime_columns = get_column_datetime(syn_data)
    syn_data = int_to_datetime(syn_data, datetime_columns)
    syn_data.to_csv(args.synthetic_data_path, index=False)
    print(syn_data.head())
    print(syn_data.info())
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='infer', help='train or infer')
    parser.add_argument('--raw_data_path', type=str, help='path to raw data')
    parser.add_argument('--generator_path', type=str, help='generator path')
    parser.add_argument('--discriminator_path', type=str, default="", help='discriminator path')
    parser.add_argument("--classifier_path", type=str, default="", help="classifier path")
    parser.add_argument('--synthetic_data_path', type=str, help='synthetic data path')
    parser.add_argument('--num_rows', type=int, help='number of rows')
    args = parser.parse_args()

    if args.mode == 'infer':
        model_inference(args)
