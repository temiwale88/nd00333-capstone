# Import libraries
import argparse
import os
from azureml.core import Run

# Get data
run = Run.get_context()
raw_data = run.input_datasets["input_data"]

parser = argparse.ArgumentParser("cleanse")
parser.add_argument("--output_cleanse", type=str, help="cleaned data directory")
parser.add_argument("--useful_columns", type=str, help="useful columns to keep")

args = parser.parse_args()

# print(args.useful_columns)
print("Argument 1(columns to keep): %s" % str(args.useful_columns.strip("[]").split("\;")))
useful_columns = [s.strip().strip("'") for s in args.useful_columns.strip("[]").split("\;")]


# Creating new df with no nulls and only useful columns
new_df = (raw_data.to_pandas_dataframe()
          .dropna(how='all'))[useful_columns]

new_df.reset_index(inplace=True, drop=True)

if not (args.output_cleanse is None):
    os.makedirs(args.output_cleanse, exist_ok=True)
    print("%s created" % args.output_cleanse)
    path = args.output_cleanse + "/processed.parquet"
    write_df = new_df.to_parquet(path)