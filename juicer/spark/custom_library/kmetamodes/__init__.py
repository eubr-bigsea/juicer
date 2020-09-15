# coding=utf-8
# Author: 'Marissa Saunders' <marissa.saunders@thinkbiganalytics.com>
# License: MIT
# Author: 'Andrey Sapegin <andrey.sapegin@hpi.de> <andrey@sapegin.org>
# Adapted to Spark Interface by: 'Lucas Miguel Ponce' <lucasmsp@dcc.ufmg.br>


"""
Ensemble-based incremental distributed K-modes clustering for PySpark, 
similar to the algorithm proposed by Visalakshi and Arunprabha in 
"Ensemble based Distributed K-Modes Clustering" (IJERD, March 2015) to 
perform K-modes clustering in an ensemble-based way.

In short, k-modes will be performed for each partition in order to identify a 
set of *modes* (of clusters) for each partition. Next, k-modes will be 
repeated to identify modes of a set of all modes from all partitions. 
These modes of modes are called *metamodes* here.

This module uses several different distance functions for k-modes:

1) Hamming distance.
2) Frequency-based dissimilarity proposed by He Z., Deng S., Xu X. in Improving
 K-Modes Algorithm Considering Frequencies of Attribute Values in Mode.
3) Andrey Sapegin dissimilarity function, which is used for calculation of 
 metamodes only. This distance function keeps track of and takes into account 
 all frequencies of all unique values of all attributes in the cluster, and 
 NOT only most frequent values that became the attributes of the mode/metamode.
"""

from .kmetamodes import IncrementalPartitionedKMetaModes
