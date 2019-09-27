"""
Copyright 2019 Marco Lattuada

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import os

class GpuPredictionError(Exception):
    """ Raised when an error occurs in this module"""

def error(message):
    logging.error(message)
    raise GpuPredictionError(message)

class GpuPrediction:
    """
    Class performing prediction of GPU training execution time

    Attributes
    ----------
    _directory: string
        The path where performance models are stored

    _available_gpus: dict of string-tuple
        The GPU types for which information (i.e., computational power and disk speed is available)

    _gpus_configuration: dict of string-set
        The set of GPUs configuration to be considered (GPU_type-GPU_number)

    Methods
    -------
    generate_predictions
        Generate predictions of execution time on different combinations of GPU-type and GPUs number

    """

    _directory = None

    _available_gpus = None

    _gpus_configurations = None

    def __init__(self, directory, gpus_configurations=None):
        """
        Parameters
        ----------
        directory: string
            The path where models are stored

        gpus_configurations: dict of string-set
            The configurations to be evaluated. Keys of the dict is the gpus to be considred, values are the number of gpus for each gpu type
        """
        if not os.path.exists(directory):
            error(directory + " does not exist")
        else:
            self._directory = directory

            gpus_information_file_name = os.path.join(self._directory, "gpus_information.csv")
            if os.path.exists(gpus_information_file_name):
                self._available_gpus = {}
                for line in open(gpus_information_file_name, "r"):
                    if "GFlops" in line:
                        continue
                    tokens = line.split(",")
                    if len(tokens) != 3:
                        error("Unexpected pattern in line " + line + " of " + gpus_information_file_name)
                    self._available_gpus[tokens[0]] = (int(tokens[1]), int(tokens[2]))
            else:
                error(gpus_information_file_name + " does not exist")
        if gpus_configurations:
            self._gpus_configurations = gpus_configurations
        else:
            self._gpus_configurations = {}
            for gpu_type in self._available_gpus:
                self._gpus_configurations[gpu_type] = {1, 2, 4, 8}

    def generate_predictions(self, application, data_types, samples_number, batch_size, iterations_number, deadline):
        """
        Generate the predictions of the training time of the input applications on all the gpus configurations

        Parameters
        ----------
        application: string
            The type of application to be estimated

        data_types: string
            The type of data used by the application

        samples_number: integer
            The number of samples used in the training

        batch_size: integer
            The size of the minibatches used during the training

        iterations_number: integer
            The number of iterations of the training process

        deadline: integer
            The deadline (in seconds) of the training process
        """
        model_file_name = os.path.join(self._directory, application + ".csv")
        if not os.path.exists(model_file_name):
            error(model_file_name + " does not exist")

        model = {}
        for term in open(model_file_name, "r"):
            tokens = term.replace("\n", "").split(",")
            if len(tokens) != 2:
                error("Unexpected pattern in line " + term + " of " + model_file_name)
            model[tokens[1]] = float(tokens[0])

        predictions = {}

        feature_i = int(iterations_number)
        feature_j = 1 / feature_i
        feature_t = 4
        feature_l = 1 / feature_t
        feature_j = 1 / feature_i
        feature_b = int(batch_size)
        feature_c = 1 / feature_b

        for gpu_type in self._gpus_configurations:
            predictions[gpu_type] = {}
            if not gpu_type in self._available_gpus:
                error("Information about gpu_type is not available")
            gpu_characteristics = self._available_gpus[gpu_type]
            feature_p = gpu_characteristics[0]
            feature_q = 1 / feature_p
            feature_d = gpu_characteristics[1]
            feature_e = 1 / feature_d
            for gpu_number in self._gpus_configurations[gpu_type]:
                feature_g = gpu_number
                feature_m = 1 / feature_g
                prediction = 0
                for term in model:
                    product = 1
                    for character in term:
                        if character == "1":
                            product = 1
                        elif character == "B":
                            product = product * feature_b
                        elif character == "C":
                            product = product * feature_c
                        elif character == "D":
                            product = product * feature_d
                        elif character == "E":
                            product = product * feature_e
                        elif character == "G":
                            product = product * feature_g
                        elif character == "I":
                            product = product * feature_i
                        elif character == "J":
                            product = product * feature_j
                        elif character == "L":
                            product = product * feature_l
                        elif character == "M":
                            product = product * feature_m
                        elif character == "P":
                            product = product * feature_p
                        elif character == "Q":
                            product = product * feature_q
                        elif character == "T":
                            product = product * feature_t
                        else:
                            error("Unexpected character in model: " + character)

                    term_contribution = product * model[term]
                    prediction = prediction + term_contribution
                predictions[gpu_type][gpu_number] = prediction
        return predictions
