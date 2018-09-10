# -*- coding: utf-8 -*-
from datetime import timedelta
# from geopy.distance import great_circle
import numpy as np


def st_dbscan(df, col_latitude, col_longitude, col_datetime, alias,
              spatial_threshold=500, temporal_threshold=60, min_neighbors=15):
    """
    Python st-dbscan implementation.
    INPUTS:
        df={o1,o2,...,on} Set of objects
        spatial_threshold = Maximum geographical coordinate (spatial) distance
        value
        temporal_threshold = Maximum non-spatial distance value
        min_neighbors = Minimun number of points within Eps1 and Eps2 distance
    OUTPUT:
        C = {c1,c2,...,ck} Set of clusters
    """
    cluster_label = 0
    noise = -1
    unmarked = 777777
    stack = []

    # initialize each point with unmarked
    df[alias] = unmarked

    # for each point in database
    for index, point in df.iterrows():
        if df.loc[index][alias] == unmarked:
            neighborhood = retrieve_neighbors(index, df, col_latitude,
                                              col_longitude, col_datetime,
                                              spatial_threshold,
                                              temporal_threshold)

            if len(neighborhood) < min_neighbors:
                df.at[index, alias] = noise
            else:  # found a core point
                cluster_label += 1
                # assign a label to core point
                df.at[index, alias] = cluster_label

                # assign core's label to its neighborhood
                for neig_index in neighborhood:
                    df.at[neig_index, alias] = cluster_label
                    stack.append(neig_index)  # append neighborhood to stack

                # find new neighbors from core point neighborhood
                while len(stack) > 0:
                    current_point_index = stack.pop()
                    new_neighborhood = retrieve_neighbors(
                        current_point_index, df, col_latitude, col_longitude,
                        col_datetime, spatial_threshold, temporal_threshold)

                    # current_point is a new core
                    if len(new_neighborhood) >= min_neighbors:
                        for neig_index in new_neighborhood:
                            neig_cluster = df.loc[neig_index][alias]
                            if any([neig_cluster == noise,
                                    neig_cluster == unmarked]):
                                df.at[neig_index, alias] = cluster_label
                                stack.append(neig_index)
    return df


def retrieve_neighbors(index_center, df, col_latitude, col_longitude,
                       col_datetime, spatial_threshold, temporal_threshold):
    neigborhood = []

    center_point = df.loc[index_center]

    # filter by time
    min_time = center_point[col_datetime]-timedelta(seconds=temporal_threshold)
    max_time = center_point[col_datetime]+timedelta(seconds=temporal_threshold)
    df = df[(df[col_datetime] >= min_time) & (df[col_datetime] <= max_time)]

    # filter by distance
    for index, point in df.iterrows():
        if index != index_center:
            distance = great_circle(
                (center_point[col_latitude], center_point[col_longitude]),
                (point[col_latitude], point[col_longitude])).meters
            if distance <= spatial_threshold:
                neigborhood.append(index)

    return neigborhood


def great_circle(a, b):
    """Great-circle.

    The great-circle distance or orthodromic distance is the shortest
    distance between two points on the surface of a sphere, measured
    along the surface of the sphere (as opposed to a straight line
    through the sphere's interior).

    :Note: use cython in the future
    :returns: distance in meters.
    """
    import math
    earth_radius = 6371.009
    lat1, lng1 = np.radians(a[0]), np.radians(a[1])
    lat2, lng2 = np.radians(b[0]), np.radians(b[1])

    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)

    d = math.atan2(np.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                           (cos_lat1 * sin_lat2 -
                           sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                   sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

    return (earth_radius * d) * 1000