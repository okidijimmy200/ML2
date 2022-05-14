import argparse
import re
import pandas as pd
import warnings
# Local imports
from inference import Inference
from utils import get_column_dtype
from utils import get_column_dtype, get_column_datetime, int_to_datetime

# MinIO Imports
from minio import Minio
from minio.error import InvalidResponseError

warnings.filterwarnings("ignore")

#----------Code for MinIO-------------------
def download(client, bucketname:str, filename:str, filepath:str):
    # Get a full object
    try:
        client.fget_object(bucketname, filename, filepath)
    except InvalidResponseError as err:
        print(err)

def get_models_from_minio(args, client):
    model_paths = {"generator": "generator.h5", "discriminator": "discriminator.h5", "classifier": "classifier.h5"}
    download(client, "client-demo", f"/{args.user_id}/{args.train_id}/generator.h5", model_paths["generator"])
    return model_paths

def get_raw_data_from_minio(args, client):
    download(client, "client-demo", str(args.user_id) + "/" + str(args.train_id) + "/" + str(args.gen_id) + "/" + args.raw_data_path, "./raw_data.csv")
    return "./raw_data.csv"

def upload_synthetic_data_to_minio(args, client, synthetic_data_path):
    client.fput_object("client-demo", f"/{args.user_id}/{args.train_id}/{args.gen_id}/synthetic_data.h5", synthetic_data_path)

#-------------------------------------------


#----------Code from main_local.py----------
def model_inference(args):
    """
    Model training
    """
    pre_trained_paths = {
        "generator": args["generator_path"],
        "discriminator": "",
        "classifier": "",
    }

    dataset = pd.read_csv(args["raw_data_path"])
    categorical_columns, integer_columns, _ = get_column_dtype(dataset)
    print("Categorical columns:", categorical_columns)
    print("Integer columns:", integer_columns)
    model_training = Inference(
        raw_data = args["raw_data_path"],
        categorical_columns=categorical_columns,
        integer_columns=integer_columns,
        pre_trained_paths = pre_trained_paths,
        synthetic_data_path = args["synthetic_data_path"],
    )
    model_training.init_model()
    model_training.get_synthetic_data(args["num_rows"])
    syn_data = pd.read_csv(args["synthetic_data_path"])
    syn_data, datetime_columns = get_column_datetime(syn_data)
    syn_data = int_to_datetime(syn_data, datetime_columns)
    syn_data.to_csv(args["synthetic_data_path"], index=False)
#-------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=int, required=True)
    parser.add_argument("--train_id", type=int, required=True)
    parser.add_argument("--gen_id", type=int, required=True)
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument("--num_rows", type=int, required=True)
    parser.add_argument("--minio_creds", type=str, required=True, help="minio credentials")
    args = parser.parse_args()

    creds= args.minio_creds
    creds_list=creds.split(",")
    ip=creds_list[0]
    access_key=creds_list[1]
    secret_key=creds_list[2]

    client = Minio(
    ip,
    access_key= access_key,
    secret_key=secret_key,
    secure=False
    )

    #---------Download the trained models
    dataset_path = get_raw_data_from_minio(args, client)
    model_paths = get_models_from_minio(args, client)
    synthetic_data = "synthetic_data.csv"

    inference_args = {"generator_path": model_paths["generator"], "discriminator_path": model_paths["discriminator"], "classifier_path": model_paths["classifier"], "raw_data_path": dataset_path, "synthetic_data_path": synthetic_data, "num_rows": args.num_rows}
    model_inference(inference_args)

    #---------Upload the synthetic data
    upload_synthetic_data_to_minio(args, client, synthetic_data)

