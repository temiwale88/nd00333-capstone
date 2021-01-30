# See Azure sample notebooks "deploy-to-cloud/score.py" for more information
# Also see here: https://bit.ly/2McMH1p, https://bit.ly/3sPmIgT, https://bit.ly/2MpmoVC & https://bit.ly/2LRXqOM for debugging codes

import json
import time
import sys
import os
import pandas as pd
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
from azureml.core.model import Model
import os.path as path


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
        global session

        # a directory containing the model file you registered.
        two_up =  path.abspath(path.join('__file__' ,"../.."))
        model_filename = '/saved_models/saved_model.onnx'
        model_path = path.abspath(path.join(two_up, "../."+model_filename))
        session = onnxruntime.InferenceSession(model_path)

    except Exception as e:
        result = str(e)
        return {"init error": result}
        # return {"init_error": '', "session": session, "model": model, "onnxruntime": onnxruntime}

def preprocess(input_data_json):
    # convert the JSON data into the tensor input
    return np.array(json.loads(input_data_json)['data']).astype('float32')

def postprocess(result):
    return np.array(result).tolist()

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
        start = time.time()   # start timer
        # input_data = preprocess(data)
        input_name = session.get_inputs()[0].name  # get the id of the first input of the model   
        result = session.run([], {input_name: data})
        end = time.time()     # stop timer
        return {"result": postprocess(result), "time": end - start}
    except Exception as e:
        result = str(e)
        return {"run error": result}
        # return {"run_error": '', "input_name": input_name, "result": result}

# def run(data):
#     try:
#         result = model.predict(data)
#         # You can return any data type, as long as it is JSON serializable.
#         return json.dumps({"result": result.tolist()})
#     except Exception as e:
#         error = str(e)
#         # ONLY Return error in Dev not production
#         return json.dumps({"error": error})