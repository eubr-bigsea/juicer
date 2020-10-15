# coding=utf-8
# Author: 'Marissa Saunders' <marissa.saunders@thinkbiganalytics.com>
# License: MIT
# Author: 'Andrey Sapegin' <andrey.sapegin@hpi.de> <andrey@sapegin.org>
# Adapted to Spark Interface by: 'Lucas Miguel Ponce' <lucasmsp@dcc.ufmg.br>

from .base import *
from .util import *

import numpy as np

import time
import random

from pyspark import keyword_only
from pyspark.ml import Estimator, Model
from pyspark.ml.param.shared import HasFeaturesCol, Param, Params, \
    HasPredictionCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import Row
from pyspark.sql import SparkSession


class IncrementalPartitionedKMetaModesParams(HasFeaturesCol,
                                       HasPredictionCol, DefaultParamsReadable,
                                       DefaultParamsWritable):

    n_clusters = Param(Params._dummy(), "n_clusters", "Number of clusters (K)")
    max_dist_iter = Param(Params._dummy(), "max_dist_iter",
                          "Maximum iteration in partitioned K-modes.")
    local_kmodes_iter = Param(Params._dummy(), "local_kmodes_iter",
                              "Maximum iteration of merged K-modes.")
    similarity = Param(Params._dummy(), "similarity",
                       "distance functions for partitioned k-modes")
    metamodessimilarity = Param(Params._dummy(), "metamodessimilarity",
                                "distance functions of merged k-modes.")
    seed = Param(Params._dummy(), "seed", "Seed for select initial clusters.")
    fragmentation = Param(Params._dummy(), "fragmentation",
                          "Reduce fragmentation. If enabled, it will reduce "
                          "the parallelization in favor of the ability to "
                          "handle small databases.")

    @keyword_only
    def __init__(self, n_clusters=3, max_dist_iter=10, local_kmodes_iter=10,
                 similarity="hamming", metamodessimilarity="hamming",
                 fragmentation=False, seed=None):
        super(IncrementalPartitionedKMetaModesParams, self).__init__()
        self._setDefault(n_clusters=3, max_dist_iter=10,
                         local_kmodes_iter=10, similarity="hamming",
                         metamodessimilarity="hamming", fragmentation=False)

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, n_clusters=3, max_dist_iter=10, local_kmodes_iter=10,
                 similarity="hamming", metamodessimilarity="hamming", seed=None,
                 fragmentation=False, featuresCol='features',
                 predictionCol='prediction'):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setFeaturesCol(self, value):
        """
        Sets the value of :py:attr:`featuresCol`.
        """
        return self._set(featuresCol=value)

    def setPredictionCol(self, value):
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        return self._set(predictionCol=value)

    def setK(self, value):
        """
        Sets the value of :py:attr:`k`.
        """
        return self._set(n_clusters=value)

    def getK(self):
        """
        Gets the value of `k` or its default value.
        """
        return self.getOrDefault(self.n_clusters)

    def setMaxDistIter(self, value):
        """
        Sets the value of :py:attr:`max_dist_iter`.
        """
        return self._set(max_dist_iter=value)

    def getMaxDistIter(self):
        """
        Gets the value of `max_dist_iter` or its default value.
        """
        return self.getOrDefault(self.max_dist_iter)

    def setLocalKmodesIter(self, value):
        """
        Sets the value of :py:attr:`local_kmodes_iter`.
        """
        return self._set(local_kmodes_iter=value)

    def getLocalKmodesIter(self):
        """
        Gets the value of `local_kmodes_iter` or its default value.
        """
        return self.getOrDefault(self.local_kmodes_iter)

    def setFragmentation(self, value):
        """
        Sets the value of :py:attr:`fragmentation`.
        """
        return self._set(fragmentation=value)

    def getFragmentation(self):
        """
        Gets the value of `fragmentation` or its default value.
        """
        return self.getOrDefault(self.fragmentation)

    def setSeed(self, value):
        """
        Sets the value of :py:attr:`seed`.
        """
        return self._set(seed=value)

    def getSeed(self):
        """
        Gets the value of `seed` or its default value.
        """
        return self.getOrDefault(self.seed)

    def setSimilarity(self, value):
        """
        Sets the value of :py:attr:`similarity`.
        """
        return self._set(similarity=value)

    def getSimilarity(self):
        """
        Gets the value of `similarity` or its default value.
        """
        return self.getOrDefault(self.similarity)

    def setMetamodesSimilarity(self, value):
        """
        Sets the value of :py:attr:`metamodessimilarity`.
        """
        return self._set(metamodessimilarity=value)

    def getMetamodesSimilarity(self):
        """
        Gets the value of `k` or its default value.
        """
        return self.getOrDefault(self.metamodessimilarity)


class IncrementalPartitionedKMetaModes(Estimator,
                                       IncrementalPartitionedKMetaModesParams):
    """Based on the algorithm proposed by Visalakshi and Arunprabha
    (IJERD, March 2015) to perform K-modes clustering in an ensemble-based way.

        K-modes clustering is performed on each partition of an rdd and the
        resulting clusters are collected to the driver node.
        Local K-modes clustering is then performed on all modes returned
        from all partitions to yield a final set of modes.
    """

    @keyword_only
    def __init__(self, n_clusters=3, max_dist_iter=10, local_kmodes_iter=10,
                 similarity="hamming", metamodessimilarity="hamming", seed=None,
                 fragmentation=False, featuresCol='features',
                 predictionCol='prediction'):
        super(IncrementalPartitionedKMetaModes, self).__init__()
        self._setDefault(n_clusters=3, max_dist_iter=10,
                         local_kmodes_iter=10, similarity="hamming",
                         seed=None, fragmentation=False,
                         metamodessimilarity="hamming")

        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        self.meta_modes = None
        self.modes = None

    @staticmethod
    def get_unique_modes_with_index_hamming(all_modes):
        mode_list = list(enumerate(all_modes))
        uniq_mode_list = []
        indexes = []
        for index, mode in mode_list:
            if len(uniq_mode_list) > 0:
                diss = hamming_dissim(mode.attrs, uniq_mode_list)
                if min(diss) == 0:
                    continue
            uniq_mode_list.append(mode)
            indexes.append(index)
        return list(zip(indexes, uniq_mode_list))

    def _fit(self, kmdata):
        """Compute distributed k-modes clustering."""
        feature_col = self.getFeaturesCol()
        k = self.getK()
        max_local_iter = self.getLocalKmodesIter()
        max_dist_iter = self.getMaxDistIter()
        similarity_func = self.getSimilarity()
        metamode_similarity = self.getMetamodesSimilarity()
        seed = self.getSeed()
        reduce_fragmentation = self.getFragmentation()

        if not reduce_fragmentation and kmdata.rdd.getNumPartitions() == 1:
            spark = SparkSession.builder.getOrCreate()
            available_cores = spark.sparkContext.defaultParallelism
            kmdata = kmdata.repartition(available_cores)

        elif reduce_fragmentation and kmdata.rdd.getNumPartitions() != 1:
            kmdata = kmdata.coalesce(1)

        rdd = kmdata\
            .rdd\
            .map(lambda x: x[feature_col].toArray().tolist())\
            .map(lambda x: k_modes_record(x))

        # Calculate the modes for each partition and return the
        # clusters and an indexed rdd.
        modes = k_modes_partitioned(rdd, k, max_dist_iter,
                                    similarity_func, seed)

        # Calculate the modes for the set of all modes
        # 1) prepare rdd with modes from all partitions
        self.modes = []
        for one_partition_modes in modes:
            for mode in one_partition_modes:
                self.modes.append(mode)

        # 2) run k-modes on single partition
        self.meta_modes = k_metamodes_local(self.modes, k,
                                            max_local_iter,
                                            metamode_similarity,
                                            seed)
        return self._copyValues(
                IncrementalPartitionedKMetaModesModel(models=self.meta_modes))


class IncrementalPartitionedKMetaModesModel(
        IncrementalPartitionedKMetaModesParams, Model):

    def __init__(self, models):
        super(IncrementalPartitionedKMetaModesModel, self).__init__()
        self.models = models

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This creates a deep copy of the embedded
        paramMap,
        and copies the embedded and extra parameters over.
        :param extra: Extra parameters to copy to the new instance
        :return: Copy of this instance
        """
        if extra is None:
            extra = dict()
        newModel = Params.copy(self, extra)
        newModel.models = [model.copy(extra) for model in self.models]
        return newModel

    def _transform(self, data_input):
        feature_col = self.getFeaturesCol()
        similarity_func = self.getSimilarity()
        predict_col = self.getPredictionCol()

        if self.models is None:
            raise Exception("Model must be fitted first.")

        diss_function = get_dissim_function(similarity_func)

        def distance(record, features, centroids, dissim_function, alias):
            drow = record.asDict()
            r = record[features].toArray().tolist()
            diss = list(dissim_function(r, centroids))
            drow[alias] = int(np.argmin(diss))
            return Row(**drow)

        data_with_cluster = data_input\
            .rdd\
            .map(lambda record: distance(record, feature_col,
                                         self.models,
                                         diss_function, predict_col))\
            .toDF()

        return data_with_cluster

    def clusterCenters(self):
        """
        :return: Return all k-metamodes
        """
        centroids = []
        for c in self.models:
            centroids.append(c.attrs)
        return centroids

    def getModes(self):
        """
        returns all modes (not metamodes!) from all partitions
        """
        return self.modes


class k_modes_record:
    """ A single item in the rdd that is used for training the k-modes
    calculation.

        - Initialization:
            - A tuple containing (Index, DataPoint)

        - Structure:
            - the index (.index)
            - the data point (.record)

        - Methods:
            - update_cluster(clusters): determines which cluster centroid is
            closest to the data point and updates the cluster membership lists
            appropriately.  It also updates the frequencies appropriately.
    """

    def __init__(self, record):
        self.record = record
        # index contains the number of the mode, initially record
        # does not belong to any cluster, so it is set to -1
        self.index = -1
        self.mode_id = -1

    def update_cluster(self, clusters, similarity):
        # clusters contains a list of cluster objects.  This function
        # calculates which cluster is closest to the record contained in this
        # object and changes the cluster to contain the index of this mode.
        # It also updates the cluster frequencies.

        if similarity == "hamming":
            diss = hamming_dissim(self.record, clusters)
        else:  # if (similarity == "frequency"):
            diss = frequency_based_dissim(self.record, clusters)

        new_cluster = np.argmin(diss)

        moved = 0

        if self.index == -1:
            # First cycle through
            moved += 1
            self.index = new_cluster
            self.mode_id = clusters[new_cluster].mode_id
            clusters[new_cluster].add_member(self.record)
            clusters[new_cluster].update_mode()
        elif self.index == new_cluster:
            pass
        else:  # self.index != new_cluster:
            if diss[self.index] == 0.0:
                raise Exception(
                        "Warning! Dissimilarity to old mode was 0, but new "
                        "mode with the dissimilarity 0 also found! K-modes "
                        "failed...")
            moved += 1
            clusters[self.index].subtract_member(self.record)
            clusters[self.index].update_mode()
            clusters[new_cluster].add_member(self.record)
            clusters[new_cluster].update_mode()
            self.index = new_cluster
            self.mode_id = clusters[new_cluster].mode_id

        return self, clusters, moved


def iter_k_modes(iterator, similarity, max_iter):
    """
    Function that is used with mapPartitionsWithIndex to perform a single
    iteration of the k-modes algorithm on each partition of data.

        - Inputs

            - *clusters*: is a list of cluster objects for all partitions,
            - *n_clusters*: is the number of clusters to use on each partition

        - Outputs

            - *clusters*: a list of updated clusters,
            - *moved*: the number of data items that changed clusters
    """

    i = 0
    for element in iterator:
        records = element[0]
        partition_clusters = element[1]
        partition_moved = element[2]
        i += 1
    if i != 1:
        raise Exception(
            "More than 1 element in partition! This is not expected!")

    if partition_moved == 0:
        yield records, partition_clusters, partition_moved
    else:

        partition_moved = 0
        partition_records = []
        for _ in range(max_iter):

            # iterator should contain only 1 list of records
            for record in records:
                new_record, partition_clusters, temp_move = record.update_cluster(
                    partition_clusters, similarity)
                partition_records.append(new_record)
                partition_moved += temp_move

        yield partition_records, partition_clusters, partition_moved


def hamming_dissim_records(record, records):
    list_dissim = []
    for record_from_records in records:
        sum_dissim = 0
        for elem1, elem2 in zip(record, record_from_records.record):
            if elem1 != elem2:
                sum_dissim += 1
        list_dissim.append(sum_dissim)
    return list_dissim


def get_unique_records_with_index(partition_records):
    record_list = list(enumerate(partition_records))
    uniq_record_list = []
    indexes = []
    for index, value in record_list:
        if len(uniq_record_list) > 0:
            diss = hamming_dissim_records(value.record, uniq_record_list)
            if min(diss) == 0:
                continue
    uniq_record_list.append(value)
    indexes.append(index)
    return list(zip(indexes, uniq_record_list))


def select_random_modes(pindex, partition_records, n_modes, uniq, seed):
    i = 0
    failed = 0
    partition_clusters = []
    indexes = []
    if uniq:
        record_list = get_unique_records_with_index(partition_records)
    else:
        record_list = list(enumerate(partition_records))

    try:
        random.seed(seed)
        sample = random.sample(record_list, n_modes)
    except ValueError:
        raise Exception(
            "Error in Select Random Modes. Possibly, there are many records "
            "with same values.")

    for index, value in sample:
        # check if there is a mode with same counts already in modes:

        if len(partition_clusters) > 0:
            diss = hamming_dissim(partition_records[index].record,
                                  partition_clusters)
            if min(diss) == 0:
                print("Warning! Two modes with distance between each other "
                      "equals to 0 were randomly selected. KMetaModes can "
                      "fail! Retrying random metamodes selection...")

                failed = 1
                break

        indexes.append(index)
        partition_records[index].mode_id = pindex * n_modes + i
        partition_records[index].index = i
        partition_clusters.append(Mode(partition_records[index].record,
                                       partition_records[index].mode_id))
        i += 1
    return partition_clusters, failed, indexes


def partition_to_list(pindex, iterator, n_modes, seed):
    # records
    partition_records = []
    uniq = False
    for record in iterator:
        partition_records.append(record)

    # modes
    # try to select modes randomly 3 times
    for trial in range(3):
        partition_clusters, failed, indexes = \
            select_random_modes(pindex, partition_records, n_modes, uniq, seed)
        # if modes were sucessfully selected, break the loop
        if failed == 0:
            break
        else:
            if trial == 1:
                uniq = True
            # if it was the last iteration, raise an exception
            if trial == 2:
                raise Exception('KMetaModes failed! Cannot initialise a set '
                                'of unique modes after 3 tries... ', pindex)
            # if selection of modes failed, reset records' indexes in
            # partition records before next iteration
            for i in indexes:
                partition_records[i].mode_id = -1
                partition_records[i].index = -1
    # if exception was not raised:
    partition_moved = 1
    yield partition_records, partition_clusters, partition_moved


def k_modes_partitioned(rdd, n_clusters, max_iter, similarity, seed=None):
    """
    Perform a k-modes calculation on each partition of data.

        - Input:
            - *data_rdd*: in the form (index, record). Make sure that the data
               is partitioned appropriately: i.e. spread across partitions,
               and a relatively large number of data points per partition.
            - *n_clusters*: the number of clusters to use on each partition
            - *max_iter*: the maximum number of iterations
            - *similarity*: the type of the dissimilarity function to use
            - *seed*:  controls the sampling of the initial clusters

        - Output:
            - *clusters*: the final clusters for each partition
            - *rdd*: rdd containing the k_modes_record objects
    """

    rdd = rdd\
        .mapPartitionsWithIndex(
            lambda i, it: partition_to_list(i, it, n_clusters, seed))

    rdd = rdd.mapPartitions(lambda it: iter_k_modes(it, similarity, max_iter))
    new_clusters = rdd.map(lambda x: x[1]).collect()

    return new_clusters


def get_unique_modes_with_index(all_modes):
    mode_list = list(enumerate(all_modes))
    uniq_mode_list = []
    indexes = []
    for index, mode in mode_list:
        if len(uniq_mode_list) > 0:
            diss = all_frequency_based_dissim_for_modes(mode, uniq_mode_list)
            if min(diss) == 0:
                continue
        uniq_mode_list.append(mode)
        indexes.append(index)
    return list(zip(indexes, uniq_mode_list))


def select_random_metamodes(all_modes, n_clusters, uniq, similarity, seed):
    i = 0
    failed = 0
    metamodes = []
    indexes = []
    if uniq:
        modes_list = get_unique_modes_with_index(all_modes)
    else:
        modes_list = list(enumerate(all_modes))

    random.seed(seed)
    for index, value in random.sample(modes_list, n_clusters):
        indexes.append(index)
        if all_modes[index].nmembers == 0:
            print("Warning! Mode without members identified!")
            print("Attributes: ", all_modes[index].attrs)
            print("Attribute frequencies: ", all_modes[index].attr_frequencies)
            print("Counts: ", all_modes[index].count)
            print("Frequencies: ", all_modes[index].freq)
            print()
        if all_modes[index].freq is None:
            all_modes[index].calculate_freq()
        all_modes[index].index = i
        # check if there is a metamode with same counts already in metamodes:
        if len(metamodes) > 0:
            sim_function = get_dissim_function(similarity)
            diss = sim_function(all_modes[index].attrs, metamodes)

            if min(diss) == 0:
                print("Warning! Two metamodes with distance between each "
                      "other equals to 0 were randomly selected. KMetaModes "
                      "can fail! Retrying random metamodes selection...")
                failed = 1
        metamodes.append(Metamode(all_modes[index]))
        i += 1
    return metamodes, failed, indexes


def k_metamodes_local(all_modes, n_clusters, max_iter, similarity, seed=None):
    uniq = False
    metamodes = None
    for trial in range(3):
        metamodes, failed, indexes = \
            select_random_metamodes(all_modes, n_clusters, uniq, similarity,
                                    seed)
        # if metamodes were sucessfully selected, break the loop
        if failed == 0:
            break
        else:
            if trial == 1:
                uniq = True
            # if it was the last iteration, raise an exception
            if trial == 2:
                raise Exception('KMetaModes failed! Cannot initialise a set '
                                'of unique metamodes after 3 tries... ')
            # if selection of metamodes failed, reset modes' indexes in
            # partition records before next iteration
            for i in indexes:
                all_modes[i].index = -1

    # do an iteration of k-modes analysis, passing back the final metamodes.
    # Repeat until no points move
    moved = 1
    iter_count = 0
    while moved != 0:
        moved = 0

        print("Iteration ", iter_count)
        iter_count += 1
        iteration_start = time.time()
        for mode in all_modes:
            metamodes, temp_move = mode.update_metamode(metamodes, similarity)
            moved += temp_move

        print("Iteration ", iter_count - 1, "finished within ",
              time.time() - iteration_start, ", moved = ", moved)

        if iter_count >= max_iter:
            break

    return metamodes
