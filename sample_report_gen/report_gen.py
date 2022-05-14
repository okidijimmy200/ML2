import argparse
import json
from minio import Minio
from minio.error import InvalidResponseError
import requests
import warnings
import json


def download(client, bucketname:str, filename:str, filepath:str):
    try:
        client.fget_object(bucketname, filename, filepath)
    except InvalidResponseError as err:
        print(err)


def get_json(args, client):
    download(client, "client-demo",  str(args.user_id) + "/" + str(args.train_id) + "/" + args.json_filename, "./params.json")
    return "/params.json" 

def parser_json(json_file):
    with open(json_file, 'r') as j:
        params=json.loads(j.read())
        return params

def sample_report_gen(args, client):
    headers = {
            'Content-Type': 'application/json',
        }
    #download json for params
    json_file=get_json(args, client)
    params=parser_json(json_file)

    #call inference api to gen synthetic data.csv
    
    num_rows=params["num_rows"]
    
    data_inf={"user_id": args.user_id, "train_id": args.train_id, "gen_id": args.train_id, "raw_data_path": args.raw_data_file_name, "num_rows":num_rows}
    requests.post('http://localhost:8086/deploy', headers=headers, json=data_inf)

    #call metrics api to gen sample report
    synthetic_data_path=f"/{args.user_id}/{args.train_id}/synthetic_data.h5"
    target_string=params["target_string"]
    #random.choice(choicelist)
    data_rep={"user_id": args.user_id, "gen_id": args.train_id, "real_data_url": args.raw_data_file_name, "fake_data_url":synthetic_data_path, "target_string": target_string}
    requests.post('http://localhost:8087/metrics', headers=headers, json=data_rep)


if __name__ =="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--user_id", type=int, required=True)
    parser.add_argument("--train_id", type=int, required=True)
    parser.add_argument("--raw_data_file_name", type=str, required=True)
    parser.add_argument("--json_filename", type=str, required=True)
    parser.add_argument("--minio_creds", type=str, required=True, help="minio credentials")
    args=parser.parse_args()

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

    sample_report_gen(args, client)

    #docker image name will be : localhost:5000/samplereportgen

