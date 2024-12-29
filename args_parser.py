import argparse
import os
from datetime import datetime

def int2bool(v):
    v = int(v)
    if v == 0:
        return False
    else:
        return True

def str2datetime(v):
    if len(v.split("T")) == 2:
        return datetime.strptime(v, "%Y-%m-%dT%H:%M:%S")
    else:
        return datetime.strptime(v, "%Y-%m-%d")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="output/forecast")
    parser.add_argument("--exp_name", type=str, default="exp_default")

    parser.add_argument("--data_path", type=str, default="dataset/btc.csv")
    parser.add_argument("--index_col", type=str, default="timeOpen")
    parser.add_argument("--target_cols", type=str, default=["close"])

    parser.add_argument("--train_start_date", type=str2datetime, default=None)
    parser.add_argument("--train_end_date", type=str2datetime, default=None)
    parser.add_argument("--test_start_date", type=str2datetime, default=None)
    parser.add_argument("--test_end_date", type=str2datetime, default=None)
    
    parser.add_argument("--train_data_length", type=int, default=60)
    parser.add_argument("--test_data_length", type=int, default=30)
    parser.add_argument("--val_data_length", type=int, default=0)
    parser.add_argument("--split_type", type=str, default="datetime")

    parser.add_argument("--prediction_length", type=int, default=30)
    parser.add_argument("--context_length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--num_batches", type=int, default=64)
    parser.add_argument("--num_parallel_samples", type=int, default=1000)

    parser.add_argument("--is_pre_scaling", type=int2bool, default=True)

    parser.add_argument("--is_model_scaling", type=int2bool, default=False)
    parser.add_argument("--add_time_features", type=int2bool, default=False)
    parser.add_argument("--add_extention_features", type=int2bool, default=False)

    args = parser.parse_args()

    if not os.path.exists(f"{args.output_path}/images/{args.exp_name}"):
        os.makedirs(f"{args.output_path}/images/{args.exp_name}")

    if not os.path.exists(f"{args.output_path}/logs"):
        os.makedirs(f"{args.output_path}/logs")

    if args.split_type == "datetime":
        if (args.train_end_date - args.train_start_date).days < args.context_length + args.prediction_length:
            raise ValueError(f"train data must be longer than {args.context_length + args.prediction_length}: {args.train_end_date - args.train_start_date}")

        if (args.test_end_date - args.test_start_date).days < args.prediction_length:
            raise ValueError(f"test data must be longer than {args.prediction_length}: {args.test_end_date - args.test_start_date}")

    return args