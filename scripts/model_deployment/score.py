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

# You can get the below input and output samples from an automl run (inference(folder)/score.py)
# Change data types, e.g. INT, where necessary
input_sample = pd.DataFrame({"CreditScore": pd.Series([0.0], dtype="float64"), "Gender": pd.Series([0.0], dtype="float64"), "Age": pd.Series([0.0], dtype="float64"), "Tenure": pd.Series([0.0], dtype="float64"), "Balance": pd.Series([0.0], dtype="float64"), "NumOfProducts": pd.Series([0.0], dtype="float64"), "HasCrCard": pd.Series([0.0], dtype="float64"), "IsActiveMember": pd.Series([0.0], dtype="float64"), "EstimatedSalary": pd.Series([0.0], dtype="float64"), "Geography_Germany": pd.Series([0.0], dtype="float64"), "Geography_Spain": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])

# The init() method is called once, when the web service starts up.

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