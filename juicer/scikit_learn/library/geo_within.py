# -*- coding: utf-8 -*-

import pyqtree
import pandas as pd
from matplotlib.path import Path
import numpy as np


def GeoWithinOperation(data_input, shp_object, lat_long,
                       attributes, alias, polygon_col, col_lat, col_long):

    if len(attributes) == 0:
        attributes = shp_object.columns

    if lat_long:
        LAT_pos = 0
        LON_pos = 1
    else:
        LAT_pos = 1
        LON_pos = 0

    xmin = float('+inf')
    ymin = float('+inf')
    xmax = float('-inf')
    ymax = float('-inf')

    for i, sector in shp_object.iterrows():
        for point in sector[polygon_col]:
            xmin = min(xmin, point[LON_pos])
            ymin = min(ymin, point[LAT_pos])
            xmax = max(xmax, point[LON_pos])
            ymax = max(ymax, point[LAT_pos])

    # create the main bound box
    spindex = pyqtree.Index(bbox=[xmin, ymin, xmax, ymax])

    # than, insert all sectors bbox
    for inx, sector in shp_object.iterrows():
        points = []
        xmin = float('+inf')
        ymin = float('+inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for point in sector[polygon_col]:
            points.append((point[LON_pos], point[LAT_pos]))
            xmin = min(xmin, point[LON_pos])
            ymin = min(ymin, point[LAT_pos])
            xmax = max(xmax, point[LON_pos])
            ymax = max(ymax, point[LAT_pos])
        spindex.insert(item=inx, bbox=[xmin, ymin, xmax, ymax])

    sector_position = []
    if len(data_input) > 0:
        for i, point in data_input.iterrows():
            y = float(point[col_lat])
            x = float(point[col_long])

            # (xmin,ymin,xmax,ymax)
            matches = spindex.intersect([x, y, x, y])

            for shp_inx in matches:
                row = shp_object.loc[shp_inx]
                polygon = Path(row[polygon_col])
                if polygon.contains_point([y, x]):
                    content = [i] + row[attributes].tolist()
                    sector_position.append(content)

    if len(sector_position) > 0:
        attributes = ['index_geoWithin'] + list(attributes)
        cols = ["%s%s" % (a, alias) for a in attributes]
        tmp = pd.DataFrame(sector_position, columns=cols)

        key = 'index_geoWithin' + alias
        data_input = pd.merge(data_input, tmp,
                              left_index=True, right_on=key)
        data_input = data_input.drop([key], axis=1)
    else:
        for a in [a + alias for a in attributes]:
            data_input[a] = np.nan

    data_input = data_input.reset_index(drop=True)
    return data_input
