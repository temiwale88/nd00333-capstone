# See Azure sample notebooks "deploy-to-cloud/score.py" for more information
# Also see here: https://bit.ly/3sPmIgT, https://bit.ly/2MpmoVC & https://bit.ly/2LRXqOM for debugging codes

import joblib
import numpy as np
import os
import json
import pandas as pd

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"CreditScore": pd.Series([0.0], dtype="float64"), "Gender": pd.Series([0.0], dtype="float64"), "Age": pd.Series([0.0], dtype="float64"), "Tenure": pd.Series([0.0], dtype="float64"), "Balance": pd.Series([0.0], dtype="float64"), "NumOfProducts": pd.Series([0.0], dtype="float64"), "HasCrCard": pd.Series([0.0], dtype="float64"), "IsActiveMember": pd.Series([0.0], dtype="float64"), "EstimatedSalary": pd.Series([0.0], dtype="float64"), "Geography_Germany": pd.Series([0.0], dtype="float64"), "Geography_Spain": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])

# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    try:
        global model
        global model_path

        # The AZUREML_MODEL_DIR environment variable indicates
        # a directory containing the model file you registered.
        model_filename = 'saved_model'
        model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)

        model = joblib.load(model_path)
        # return model
    except Exception as e:
        result = str(e)
        return {"init error": result}

# The run() method is called each time a request is made to the scoring API.
#
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators on your
# run() method parses and validates the incoming payload against
# the example input you provide here. This will also generate a Swagger
# API document for your web service.

# Check your child run saved_model.joblib for expected array input e.g:
# feature_names=CreditScore Gender Age Tenure Balance NumOfProducts HasCrCard IsActiveMember EstimatedSalary Geography_Germany Geography_Spain
# feature_infos=[350:850] [0:1] [18:92] [0:10] [0:250898.09] [1:4] [0:1] [0:1] [11.58:199992.48000000001] [0:1] [0:1]

# TO-DO: Better define the parameters including COLUMN names
# @input_schema('data', NumpyParameterType(np.array([[350, 0, 18, 10, 0.1, 4, 0, 1, 11.58, 0, 1]])))
# @output_schema(NumpyParameterType(np.array([1])))

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))

def run(data):
    try:
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        # ONLY Return error in Dev not production
        return {"run error": result}