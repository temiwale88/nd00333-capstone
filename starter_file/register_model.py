import argparse
import json
import os
import sklearn
import azureml.core
from azureml.core import Workspace, Experiment, Model
from azureml.core import Run
from azureml.train.hyperdrive import HyperDriveRun
from shutil import copy2

parser = argparse.ArgumentParser()
parser.add_argument('--saved-model', type=str, dest='saved_model', help='path to saved model file')
args = parser.parse_args()

# Saved in blobstorage
model_output_dir = './model/'

# Looks like we're copying model from child run folders (output/model/saved_model.joblib) to blobstorage folder space
# that contains all our experiments folders and files
os.makedirs(model_output_dir, exist_ok=True)
copy2(args.saved_model, model_output_dir) #Model is copied to ./model/save_model.joblib

ws = Run.get_context().experiment.workspace

description = "My Udacity Capstone Model"
model = Model.register(workspace=ws, 
                        description=description,
                        model_framework=Model.Framework.SCIKITLEARN,  # Framework used to create the model.
                        model_framework_version=sklearn.__version__,  # Version of scikit-learn used to create the model 
                        model_name='udacity_capstone', 
                        model_path=model_output_dir,
                        tags={"model_name": "LGBM-capstone-model"}
                    )