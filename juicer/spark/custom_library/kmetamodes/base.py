# coding=utf-8
# Author: 'Marissa Saunders' <marissa.saunders@thinkbiganalytics.com>
# License: MIT
# Author: 'Andrey Sapegin' <andrey.sapegin@hpi.de> <andrey@sapegin.org>
# Adapted to Spark Interface by: 'Lucas Miguel Ponce' <lucasmsp@dcc.ufmg.br>

from copy import deepcopy
from collections import defaultdict

from .util import *
import numpy as np


class Metamode:
    def __init__(self, mode):
        # Initialisation of metamode object
        self.attrs = deepcopy(mode.attrs)
        # the metamode is initialised with frequencies,
        # it means that the metamode will have 1 element right after
        # initialisation. So, frequencies are copied from the mode
        self.attr_frequencies = deepcopy(mode.attr_frequencies)
        # The count and freq are different from frequencies of mode attributes.
        # They contain frequencies/counts for all values in the cluster,
        # and not just frequencies of the most frequent attributes
        # (stored in the mode)
        self.count = deepcopy(mode.count)
        # used only to calculate distance to modes
        self.freq = deepcopy(mode.freq)
        # Number of members (modes) of this metamode, initially set to 1
        # (contains mode from which initialisation was done)
        self.nmembers = 1
        # number of all records in all modes of this metamode
        self.nrecords = deepcopy(mode.nmembers)

    def calculate_freq(self):
        # create frequencies from counts by dividing each count on total
        # number of values for corresponding attribute for corresponding
        # cluster of this mode
        self.freq = [defaultdict(float) for _ in range(len(self.attrs))]
        for i in range(len(self.count)):
            self.freq[i] = {k: v / self.nrecords for k, v in
                            self.count[i].items()}

    def add_member(self, mode):
        self.nmembers += 1
        self.nrecords += mode.nmembers
        for i in range(len(self.count)):
            # sum and merge mode count to metamode count
            self.count[i] = {
                k: self.count[i].get(k, 0) + mode.count[i].get(k, 0) for k in
                set(self.count[i]) | set(mode.count[i])}

    def subtract_member(self, mode):
        self.nmembers -= 1
        self.nrecords -= mode.nmembers
        if self.nmembers == 0:
            print(
                    "Last member removed from metamode! "
                    "This situation should never happen in incremental "
                    "k-modes! "
                    "Reason could be non-unique modes/metamodes or same "
                    "distance "
                    "from mode to two or more metamodes.")

        for i in range(len(self.count)):
            # substract and merge mode count from metamode count
            self.count[i] = {
                k: self.count[i].get(k, 0) - mode.count[i].get(k, 0) for k in
                set(self.count[i]) | set(mode.count[i])}

    def update_metamode(self):
        new_mode_attrs = []
        new_mode_attr_freqs = []
        for ind_attr, val_attr in enumerate(self.attrs):
            key, value = get_max_value_key(self.count[ind_attr])
            new_mode_attrs.append(key)
            new_mode_attr_freqs.append(value / self.nrecords)

        self.attrs = new_mode_attrs
        self.attr_frequencies = new_mode_attr_freqs
        self.calculate_freq()


class Mode:
    """
    This is the k-modes mode object

    - Initialization:
            - just the mode attributes will be initialised
    - Structure:

            - the mode object
            -- consists of mode and frequencies of mode attributes
            - the frequency at which each of the values is observed for each
                category in each variable calculated over the cluster members
                (.freq)

    - Methods:

            - add_member(record): add a data point to the cluster
            - subtract_member(record): remove a data point from the cluster
            - update_mode: recalculate the centroid of the cluster based on the
                frequencies.

    """

    def __init__(self, record, mode_id):
        # Initialisation of mode object
        self.attrs = deepcopy(record)
        # the mode is initialised with frequencies, it means that the cluster
        # contains record already. So, frequencies should be set to 1
        self.attr_frequencies = [1] * len(self.attrs)
        # The count and freq are different from frequencies of mode attributes.
        # They contain frequencies/counts for all values in the cluster,
        # and not just frequencies of the most frequent attributes (stored
        # in the mode)
        self.count = [defaultdict(int) for _ in range(len(self.attrs))]
        for ind_attr, val_attr in enumerate(record):
            self.count[ind_attr][val_attr] += 1
        self.freq = None  # used only to calculate distance to metamodes, will
        # be initialised within a distance function
        # Number of members of the cluster with this mode, initially set to 1
        self.nmembers = 1
        # index contains the number of the metamode, initially mode does not
        # belong to any metamode, so it is set to -1
        self.index = -1
        self.mode_id = mode_id

    def calculate_freq(self):
        # create frequencies from counts by dividing each count on total
        # number of values for corresponding attribute for corresponding
        # cluster of this mode
        self.freq = [defaultdict(float) for _ in range(len(self.attrs))]
        for i in range(len(self.count)):
            self.freq[i] = {k: v / self.nmembers for k, v in
                            self.count[i].items()}

    def add_member(self, record):
        self.nmembers += 1
        for ind_attr, val_attr in enumerate(record):
            self.count[ind_attr][val_attr] += 1

    def subtract_member(self, record):
        self.nmembers -= 1
        for ind_attr, val_attr in enumerate(record):
            self.count[ind_attr][val_attr] -= 1

    def update_mode(self):
        new_mode_attrs = []
        new_mode_attr_freqs = []
        for ind_attr, val_attr in enumerate(self.attrs):
            key, value = get_max_value_key(self.count[ind_attr])
            new_mode_attrs.append(key)
            new_mode_attr_freqs.append(value / self.nmembers)

        self.attrs = new_mode_attrs
        self.attr_frequencies = new_mode_attr_freqs

    def update_metamode(self, metamodes, similarity):
        # metamodes contains a list of metamode objects.  This function
        # calculates which metamode is closest to the mode contained in this
        # object and changes the metamode to contain the index of this mode.
        # It also updates the metamode frequencies.

        if similarity == "hamming":
            diss = hamming_dissim(self.attrs, metamodes)
        elif similarity == "frequency":
            diss = frequency_based_dissim(self.attrs, metamodes)
        else:  # if (similarity == "meta"):
            diss = all_frequency_based_dissim_for_modes(self, metamodes)

        new_metamode_index = np.argmin(diss)

        moved = 0

        if self.index == -1:
            # First cycle through
            moved += 1
            self.index = new_metamode_index
            metamodes[self.index].add_member(self)
            metamodes[self.index].update_metamode()
        elif self.index == new_metamode_index:
            pass
        else:
            if diss[self.index] == 0.0:
                print(
                        "Warning! Mode dissimilarity to old metamode was 0, "
                        "but dissimilarity to another metamode is also 0! "
                        "KMetaModes is going to fail...")
                print("New metamode data: ")
                print("Attributes: ", metamodes[new_metamode_index].attrs)
                print("Attribute frequencies: ",
                      metamodes[new_metamode_index].attr_frequencies)
                print("Number of members: ",
                      metamodes[new_metamode_index].nmembers)
                print("Number of records: ",
                      metamodes[new_metamode_index].nrecords)
                print("Counts: ", metamodes[new_metamode_index].count)
                print()
                print("Old metamode data: ")
                print("Attributes: ", metamodes[self.index].attrs)
                print("Attribute frequencies: ",
                      metamodes[self.index].attr_frequencies)
                print("Number of members: ", metamodes[self.index].nmembers)
                print("Number of records: ", metamodes[self.index].nrecords)
                print("Counts: ", metamodes[self.index].count)
                print()
            moved += 1
            metamodes[self.index].subtract_member(self)
            metamodes[self.index].update_metamode()
            metamodes[new_metamode_index].add_member(self)
            metamodes[new_metamode_index].update_metamode()
            self.index = new_metamode_index

        return metamodes, moved
