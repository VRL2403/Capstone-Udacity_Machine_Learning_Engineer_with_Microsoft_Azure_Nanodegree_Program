# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.externals import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"instant": pd.Series(["1"], dtype="int64"), "date": pd.Series(["2011-01-01T00:00:00.000Z"], dtype="datetime64[ns]"), "season": pd.Series(["1"], dtype="int64"), "yr": pd.Series(["0"], dtype="int64"), "mnth": pd.Series(["1"], dtype="int64"), "weekday": pd.Series(["6"], dtype="int64"), "weathersit": pd.Series(["2"], dtype="int64"), "temp": pd.Series(["0.344167"], dtype="float64"), "atemp": pd.Series(["0.363625"], dtype="float64"), "hum": pd.Series(["0.805833"], dtype="float64"), "windspeed": pd.Series(["0.160446"], dtype="float64"), "casual": pd.Series(["331"], dtype="int64"), "registered": pd.Series(["654"], dtype="int64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    try:
        model = joblib.load(model_path)
    except Exception as e:
        path = os.path.normpath(model_path)
        path_split = path.split(os.sep)
        log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
