# -*- coding: utf-8 -*-


import shapefile
from io import BytesIO, StringIO
import pandas as pd


def ReadShapefile(polygon, lat_long, header, shp, dbf):

    shp_content = open(shp, "rb")
    dbf_content = open(dbf, "rb")

    # shp_io = BytesIO(shp_path)
    # dbf_io = BytesIO(dbf_path)

    shp_object = shapefile.Reader(shp=shp_content, dbf=dbf_content)
    records = shp_object.records()
    sectors = shp_object.shapeRecords()

    fields = dict()  # column: position
    for i, f in enumerate(shp_object.fields):
        fields[f[0]] = i
    del fields['DeletionFlag']

    if len(header) == 0:
        header = [f for f in fields]

    # position of each selected field
    num_fields = [fields[f] for f in header]

    header.append(polygon)
    data = []
    for i, sector in enumerate(sectors):
        attributes = []
        r = records[i]
        for t in num_fields:
            attributes.append(r[t - 1])

        points = []
        for point in sector.shape.points:
            if lat_long:
                points.append([point[1], point[0]])
            else:
                points.append([point[0], point[1]])
        attributes.append(points)
        data.append(attributes)

    return pd.DataFrame(data, columns=header)