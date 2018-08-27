# -*- coding: utf-8 -*-
import ast
from textwrap import dedent

import pytest
from juicer.sklearn.geo_operation import ReadShapefileOperation, GeoWithinOperation


from tests import compare_ast, format_code_comparison


def debug_ast(code, expected_code):
    print("""
    Code
    {sep}
    {code}
    {sep}
    Expected
    {sep}
    {expected}
    """.format(code=code, sep='-' * 20, expected=expected_code))


#############################################################################
#   ReadShapefile Operation
def test_readshapefile_minimal_params_success():
    params = {
        'polygon': 'points',
        'shapefile': 'shapefile.shp'
    }
    n_out = {'geo data': 'out'}

    instance = ReadShapefileOperation(parameters=params,
                                      named_inputs={},
                                      named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        import shapefile
        from io import BytesIO, StringIO

        polygon = 'points'
        lat_long = True
        header = []

        shp_content = open(shapefile.shp, "rb")
        dbf_content = open(shapefile.dbf, "rb")
    
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
                attributes.append(r[t-1])

            points = []
            for point in sector.shape.points:
                if lat_long:
                    points.append([point[1], point[0]])
                else:
                    points.append([point[0], point[1]])
            attributes.append(points)
            data.append(attributes)

        out = pd.DataFrame(data, columns=header)
    """)
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_readshapefile_params_success():
    params = {
        'polygon': 'points',
        'shapefile': 'shapefile.shp',
        'attributes': ['BAIRR', 'UF']
    }
    n_out = {'geo data': 'out'}

    instance = ReadShapefileOperation(parameters=params,
                                      named_inputs={},
                                      named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        import shapefile
        from io import BytesIO, StringIO

        polygon = 'points'
        lat_long = True
        header = ['BAIRR', 'UF']

        shp_content = open(shapefile.shp, "rb")
        dbf_content = open(shapefile.dbf, "rb")

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
                attributes.append(r[t-1])

            points = []
            for point in sector.shape.points:
                if lat_long:
                    points.append([point[1], point[0]])
                else:
                    points.append([point[0], point[1]])
            attributes.append(points)
            data.append(attributes)

        out = pd.DataFrame(data, columns=header)
    """)
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_readshapefile_missing_columns_failure():
    params = {
    }

    with pytest.raises(ValueError):
        n_out = {'geo data': 'out'}
        ReadShapefileOperation(parameters=params, named_inputs={},
                               named_outputs=n_out)


#############################################################################
#   Geo Within Operation
def test_geowithin_params_success():
    params = {GeoWithinOperation.POLYGON_ATTR_COLUMN_PARAM: ['col1'],
              GeoWithinOperation.POLYGON_POINTS_COLUMN_PARAM: ['points'],
              GeoWithinOperation.TARGET_LAT_COLUMN_PARAM: ['LATITUDE'],
              GeoWithinOperation.TARGET_LON_COLUMN_PARAM: ['LONGITUDE']}

    n_in = {'input data': 'df1', 'geo data': 'shape'}
    n_out = {'output data': 'out'}
    class_name = GeoWithinOperation
    instance = class_name(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
       def GeoWithinOperation(data_input, shp_object):                                         
           import pyqtree

           lat_long = True
           attributes = ['col1']
           if len(attributes) == 0:
               attributes = shp_object.columns
           alias = "_shp"
           polygon_col = "points"
           col_lat = "LATITUDE"
           col_long = "LONGITUDE"

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

           from matplotlib.path import Path

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

               key = 'index_geoWithin'+alias
               data_input = pd.merge(data_input, tmp,
                                     left_index=True, right_on=key)
               data_input = data_input.drop([key], axis=1)
           else:
               for a in [a + alias for a in attributes]:
                   data_input[a] = np.nan

           data_input = data_input.reset_index(drop=True)  
           return data_input

       out = GeoWithinOperation(df1, shape)                                   
       """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_geowithin_missing_columns_failure():
    params = {
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'df1', 'geo data': 'geo'}
        n_out = {'output data': 'out'}
        GeoWithinOperation(params, named_inputs=n_in,
                           named_outputs=n_out)