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
TRAIN_DATA_KEY = os.environ["TRAIN_DATA_KEY"]
XGBOOST_PARAM_KEY = os.environ["XGBOOST_PARAM_KEY"]

DESTINATION_BUCKET = os.environ["DESTINATION_BUCKET"]
DESTINATION_OBJECKT_DIR = os.environ["DESTINATION_OBJECKT_DIR"]

s3 = boto3.resource("s3")


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def download_data_from_s3(bucket, target_key, local_file_path):
    source_bucket = s3.Bucket(bucket)

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


def save_model(model):
    try:
        model.save_model("./model.txt")
    except Exception as e:
        logging.error("Saving failed.")
        logging.error(e)
    return model


def train(train_data, xgb_params):
    fix_seed(SEED)
    unused_cols = ["target"]
    y = train_data[unused_cols]
    X = train_data.drop(unused_cols, errors='ignore', axis=1)
    dtrain = xgb.DMatrix(X, label=y,
                         feature_names=X.columns)
    model = xgb.train(xgb_params, dtrain)
    return model


def upload_model_to_s3():
    destination_bucket = s3.Bucket(DESTINATION_BUCKET)
    model_key = os.path.join(DESTINATION_OBJECKT_DIR, "model.txt")
    try:
        destination_bucket.upload_file("./model.txt", model_key)
    except Exception as e:
        logging.error("Model uploading failed.")
        logging.error(e)


def main():
    fix_seed(SEED)

    download_data_from_s3(SOURCE_BUCKET, TRAIN_DATA_KEY, "./input.csv")
    download_data_from_s3(SOURCE_BUCKET, XGBOOST_PARAM_KEY, "./xgb_param.json")

    train_data = read_csv()
    xgb_param = read_xgb_param()

    model = train(train_data, xgb_param)
    save_model(model)
    upload_model_to_s3()


if __name__ == "__main__":
    main()
