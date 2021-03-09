import os
import io
import json
import zipfile
from collections import OrderedDict
import numpy as np

def save_to_zip(save_path, data=None, params=None):
    """
    Save model to a .zip archive.

    Copied and adapated from Stable Baselines.

    :param save_path: (str or file-like) Where to store the model
    :param data: (dict) Class parameters being stored
    :param params: (OrderedDict) Model parameters being stored
    """
    # data/params can be None, so do not
    # try to serialize them blindly
    if data is not None:
        try:
            serialized_data = json.dumps(data, indent=4)
        except TypeError:
            for k,v in data.items:
                try:
                    _ = json.dumps(v)
                except TypeError:
                    raise TypeError("Can't JSON serialize " + str(v))
            raise Exception("Can't serialize dict but can serialize each value?")

    if params is not None:
        serialized_params = serialize_param_dict(params)
        # We also have to store list of the parameters
        # to store the ordering for OrderedDict.
        # We can trust these to be strings as they
        # are taken from the Tensorflow graph.
        serialized_param_list = json.dumps(
            list(params.keys()),
            indent=4
        )

    # Check postfix if save_path is a string
    if isinstance(save_path, str):
        _, ext = os.path.splitext(save_path)
        if ext == "":
            save_path += ".zip"

    # Create a zip-archive and write our objects
    # there. This works when save_path
    # is either str or a file-like
    with zipfile.ZipFile(save_path, "w") as file_:
        # Do not try to save "None" elements
        if data is not None:
            file_.writestr("data", serialized_data)
        if params is not None:
            file_.writestr("parameters", serialized_params)
            file_.writestr("parameter_list", serialized_param_list)

    return save_path

def load_from_zip(load_path):
    """
    Load model from a .zip archive

    Copied and adapted from Stable Baselines.

    :param load_path: (str or file-like) Where to load model from
    :return: (dict, OrderedDict) Class parameters and model parameters
    """
    # Check if file exists if load_path is
    # a string
    if isinstance(load_path, str):
        if not os.path.exists(load_path):
            if os.path.exists(load_path + ".zip"):
                load_path += ".zip"
            else:
                raise ValueError("Error: the file {} could not be found".format(load_path))

    # Open the zip archive and load data.
    with zipfile.ZipFile(load_path, "r") as file_:
        namelist = file_.namelist()
        # If data or parameters is not in the
        # zip archive, assume they were stored
        # as None.
        data = None
        params = None
        if "data" in namelist:
            # Load class parameters and convert to string
            # (Required for json library in Python 3.5)
            json_data = file_.read("data").decode()
            data = json.loads(json_data)

        if "parameters" in namelist:
            # Load parameter list and and parameters
            parameter_list_json = file_.read("parameter_list").decode()
            parameter_list = json.loads(parameter_list_json)
            serialized_params = file_.read("parameters")
            params = deserialize_param_dict(
                serialized_params, parameter_list
            )

    return data, params

# These functions were adapted from Stable Baselines code.
def serialize_param_dict(params):
    byte_file = io.BytesIO()
    np.savez(byte_file, **params)
    serialized_params = byte_file.getvalue()
    return serialized_params

def deserialize_param_dict(serialized_params, param_list):
    byte_file = io.BytesIO(serialized_params)
    params = np.load(byte_file)
    param_dict = OrderedDict()
    for param_name in param_list:
        param_dict[param_name] = params[param_name]
    return param_dict

def serialize_ndarray(array):
    byte_file = io.BytesIO()
    np.save(byte_file, array)
    serialized_array = byte_file.getvalue()
    return serialized_array

def deserialize_ndarray(serialized_array):
    byte_file = io.BytesIO(serialized_array)
    array = np.load(byte_file)
    return array


