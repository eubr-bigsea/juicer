# coding: utf-8
"""
Copyright 2019 Danilo Ardagna
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
from math import ceil
from math import inf
import os
import pickle

import numpy as np

#import matplotlib.pyplot as plt
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'

class OptIcError(Exception):
    """ Raised when an error occurs in this module"""

def error(message):
    logging.error(message)
    raise OptIcError(message)

class PerformanceModel:
    """
    Performance model based on XGBoost based on black box features

    Attributes
    ----------

    _ml_model : XXX
        XGBoost Machine learning model

    _perfHashTable : Dictionary
        Dictionary storing execution time predictions given number of cores

    _directory : string
        Path storing the models to be loaded
    """

    _ml_model = None
    _perfHashTable = {}
    _app_name = None
    _data_size = None
    _directory = None

    def _create_hash_table(self):
        #simulated_data = self.load_data_sim (app_name)
        #self._perfHashTable = simulated_data[app_name][str(data_size)]
        n_cores_max = 44
        n_cores_min = 1
        for n_cores in range(n_cores_min, n_cores_max+1):
            self._perfHashTable[n_cores] = self._predict_perf_ml_model(n_cores)

    def _monotone_decreasing(self):
        """
        Make the performance prediction function monotone non incresiing in the number of cores
        """
        n_cores_max = max(self._perfHashTable.keys())
        n_cores_min = min(self._perfHashTable.keys())

        prev_val = self._perfHashTable[n_cores_min]

        for n_cores in range(n_cores_min+1, n_cores_max+1):
            if self._perfHashTable[n_cores] > prev_val:
                self._perfHashTable[n_cores] = prev_val
            else:
                prev_val = self._perfHashTable[n_cores]

    def _predict_perf_ml_model(self, n_cores):
        #features order
        #"dataSize","nContainers","1OverContainers","DataOverContainers","Log2Containers"
        x_value = [self._data_size, n_cores, 1/ n_cores, self._data_size/n_cores, np.log2(n_cores)]
        return self._ml_model.predict(x_value)[0]

    def __init__(self, app_name, data_size, directory):
        """
        Parameters
        ----------
        app_name : string
            file name storing data

        data_size : integer
            data size of the target configuration

        _directory : string
            Path storing the models to be loaded
        """
        self._app_name = app_name
        self._data_size = data_size
        self._directory = directory

        # load ML model binary file
        #features order
        #"data_size","n_cores","1_over_n_corers",
        #"data_size_over_n_cores","Log2 n_cores"
        ml_file_name = os.path.join(self._directory, app_name + ".pickle")
        if not os.path.exists(ml_file_name):
            error("Model for " + app_name + " not found")
        ml_file = open(ml_file_name, 'rb')
        self._ml_model = pickle.load(ml_file)
        ml_file.close()

        self._create_hash_table()
        # print(self._perfHashTable)

        self._monotone_decreasing()
        # print(self._perfHashTable)

    def predict(self, n_cores):
        """
        Returns performance estimate for nCores configutation
        """
        # if n_cores larger than the one available  return the best performance

        max_cores_in_dict = max(self._perfHashTable.keys())
        min_cores_in_dict = min(self._perfHashTable.keys())

        if n_cores > max_cores_in_dict:
            return self.best_performance()
        elif n_cores < min_cores_in_dict:
            return self._perfHashTable[min_cores_in_dict]
        else:
            return self._perfHashTable[n_cores]

    def best_performance(self):
        """
        Return the best performance achieved with the maximum cores number
        available in the dictionary
        """
        return min(self._perfHashTable.values())

class Optimizer:
    """
    Optimizer computing initial configuration for a Spark application

    Attributes
    ----------

    _performance_model : PerformanceModel
        Performance model of the Spark application
    _app_name : string
        Application name
    _data_size : int
        data_size of the target configuration

    """

    _performance_model = None
    _app_name = None
    _data_size = None

    def __init__(self, app_name, data_size, directory):
        """
        Parameters
        ----------
        app_name : string
            file name storing data

        data_size : integer
            data size of the target configuration

        """
        self._app_name = app_name
        self._data_size = data_size
        self._performance_model = PerformanceModel(app_name, data_size, directory)

    def solve(self, vm_n_cores, deadline):
        """
        Implement dichotomic search

        Parameters
        ----------
        vm_n_cores : int
            number of cores available in the VM of the target deployment

        deadline : float
            deadline for the Spark application execution

        Returns the minimum number of VMs (configured with vm_n_cores)
        required to fullfill the deadline

        """

        n_nodes_max = 200
        n_nodes_min = 1

        # if the evaluated time is less than the deadline with just one virtual machine
        # the result is 1
        tmin = self._performance_model.predict(vm_n_cores)
        #print("Tmin",tmin)
        if tmin < deadline:
            #print("Ncores = ", vm_n_cores, " -> time = ", tmin, " ms")
            return 1

        if self._performance_model.best_performance() > deadline:
            #print(best_performance(perf_data,app,data_size))
            #print("The problem is unfeasible, the deadline is too strict")
            return inf

        while n_nodes_max - n_nodes_min != 1:
            #print("Node min ", n_nodes_min, " Node max ",n_nodes_max)
            n_nodes = ceil((n_nodes_max + n_nodes_min) / 2)
            n_cores = n_nodes * vm_n_cores
            predicted_time = self._performance_model.predict(n_cores)
            #print("Ncores = ", n_cores, " -> time = ", t, " ms")
            if predicted_time > deadline:
                n_nodes_min = n_nodes
            else:
                n_nodes_max = n_nodes
        #print("Node min ", n_nodes_min, " Node max ",n_nodes_max)
        if self._performance_model.predict(n_nodes*vm_n_cores) > deadline:
            return n_nodes + 1
        elif self._performance_model.predict((n_nodes-1) *vm_n_cores) < deadline:
            return n_nodes -1
        else:
            return n_nodes
