
# Importing the libraries
import argparse
import os
import pandas as pd
from azureml.core import Run

run = Run.get_context()

cleansed_data = run.input_datasets["cleansed_data"]
cleansed_df = cleansed_data.to_pandas_dataframe()

parser = argparse.ArgumentParser("encoded")
parser.add_argument("--encoded_output", type=str, help="columns geography and gender are encoded")

args = parser.parse_args()

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
cleansed_df["Gender"] = labelencoder.fit_transform(cleansed_df["Gender"])

# Encoding country
columns_to_encode = ["Geography"]

encoded_df = pd.get_dummies(cleansed_df, prefix=columns_to_encode, columns=columns_to_encode, drop_first=True)
encoded_df.reset_index(inplace=True, drop=True)
print(encoded_df.head(10))

if not (args.encoded_output is None):
    os.makedirs(args.encoded_output, exist_ok=True)
    print("%s created" % args.encoded_output)
    path = args.encoded_output + "/processed.parquet"
    write_df = encoded_df.to_parquet(path)