import argparse 
import pandas as pd
import warnings
import json
# Local imports
from training import Training
from utils import get_column_dtype, check_and_create_folder, get_column_datetime, remove_large_categorical_columns, datetime_to_int, int_to_datetime
import time
# MinIO Imports
from minio import Minio
from minio.error import InvalidResponseError

warnings.filterwarnings("ignore")

#----------Code for MinIO----------

def download(client, bucketname:str, filename:str, filepath:str):
    # Get a full object
    try:
        client.fget_object(bucketname, filename, filepath)
    except InvalidResponseError as err:
        print(err)

def get_json_from_minio(args, client):
    download(client, "client-demo", str(args.user_id) + "/" + str(args.train_id) + "/" + args.json_filename, "./params.json")
    return "./params.json"

def get_raw_data_from_minio(args, client):
    download(client, "client-demo", str(args.user_id) + "/" + str(args.train_id) + "/" + args.raw_data_filename, "./train.csv")
    return "./train.csv"

def parser_json(json_file):
    with open(json_file, 'r') as j:
        params = json.loads(j.read())
    return params


def upload_trained_models_and_data(args, client, generator_path):
    client.fput_object("client-demo", f"/{args.user_id}/{args.train_id}/generator.h5", generator_path)

#----------Code from main_local.py----------
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
    data_path = args["raw_data_path"]
    problem_type = args["problem_type"]
    epochs = args["epochs"] #args["epochs"]
    save_weights = args["save_weights"]
    save_frequency = args["epochs"] #args["epochs"]
    save_path = args["save_path"]
    model_name = "model"

    dataset = pd.read_csv(data_path)
    dataset, categorical_columns, integer_columns, datetime_columns= preprocessing(dataset)
    dataset.to_csv(data_path, index=False)


    mixed_columns = args["mixed_columns"]
    log_columns = args["log_columns"]

    check_and_create_folder(save_path)
    print("epochs:", epochs)
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
        model_name=model_name,
    )
    print("step 1")
    model_training.init_model()
    print("step 2")
    start = time.time()
    model_training.train()
    print("step 3")
    end = time.time()
    print(end - start)



#-------------------------------------------



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=int, required=True)
    parser.add_argument("--train_id", type=int, required=True)
    parser.add_argument("--raw_data_filename", type=str, required=True, help="URL of raw data")
    parser.add_argument("--json_filename", type=str, required=True, help="URL of json")
    parser.add_argument("--minio_creds", type=str, required=True, help="minio credentials")
    args= parser.parse_args()
    #----------Code for MinIO----------
    #client = Minio(
    #"10.1.54.248:9000",
    #access_key="admin",
    #secret_key="password",
    #secure=False
    #)
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

    dataset_path = get_raw_data_from_minio(args, client)
    json_file_path = get_json_from_minio(args, client)
    arguments = parser_json(json_file_path)
    arguments.update({"raw_data_path": dataset_path})
    arguments.update({"save_path": "./model_artificats"})
    
    #----------Code from main_local.py----------
    model_training(arguments)
    #-------------------------------------------
    generator_path =  arguments["save_path"] + f'/generator-epoch-{arguments["epochs"]}.h5'
    #----------Uploading the models and data-----
    upload_trained_models_and_data(args, client, generator_path)
