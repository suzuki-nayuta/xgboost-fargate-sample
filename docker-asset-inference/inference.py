import json
import logging
import os
import random

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb


logging.basicConfig(level=logging.INFO)
SEED = 46

SOURCE_BUCKET = os.environ["SOURCE_BUCKET"]
MODEL_KEY = os.environ["MODEL_KEY"]
TARGET_DATA_KEY = os.environ["TARGET_DATA_KEY"]

DESTINATION_BUCKET = os.environ["DESTINATION_BUCKET"]
DESTINATION_OBJECKT_DIR = os.environ["DESTINATION_OBJECKT_DIR"]

s3 = boto3.resource("s3")


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def download_data_from_s3(target_key, local_file_path):
    source_bucket = s3.Bucket(SOURCE_BUCKET)

    try:
        source_bucket.download_file(target_key, local_file_path)
    except Exception as e:
        logging.error("Downloading failed.")
        logging.error(e)


def read_csv():
    return pd.read_csv("./input.csv")


def read_xgb_param():
    with open("./xgb_param.json") as f:
        xgb_param = json.load(f)
    return xgb_param


def save_result(result):
    np.save('./result', result)


def inference(target_data, model):
    fix_seed(SEED)
    target_data = xgb.DMatrix(target_data)
    result = model.predict(target_data)
    return result


def load_model():
    model = xgb.Booster()
    try:
        model.load_model("./model.txt")
    except Exception as e:
        logging.error("Loading failed.")
        logging.error(e)
    return model


def upload_result_to_s3():
    destination_bucket = s3.Bucket(DESTINATION_BUCKET)
    model_key = os.path.join(DESTINATION_OBJECKT_DIR, "result.npy")
    try:
        destination_bucket.upload_file("./result.npy", model_key)
    except Exception as e:
        logging.error("Result uploading failed.")
        logging.error(e)


def main():
    fix_seed(SEED)

    download_data_from_s3(TARGET_DATA_KEY, "./input.csv")
    download_data_from_s3(MODEL_KEY, "./model.txt")

    target_data = read_csv()
    model = load_model()

    result = inference(target_data, model)
    save_result(result)
    upload_result_to_s3()


if __name__ == "__main__":
    main()
