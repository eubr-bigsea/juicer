# -*- coding: utf-8 -*-
import ast
from textwrap import dedent

import pytest
from juicer.scikit_learn.geo_operation import ReadShapefileOperation, \
    GeoWithinOperation, STDBSCANOperation


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


'''
    ReadShapefile Operation
'''


def test_readshapefile_minimal_params_success():
    params = {
        ReadShapefileOperation.POLYGON_ATTR_PARAM: 'points',
        ReadShapefileOperation.SHAPEFILE_PARAM: 'shapefile.shp',
    }
    n_out = {'geo data': 'out'}

    instance = ReadShapefileOperation(parameters=params,
                                      named_inputs={},
                                      named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        polygon = '{polygon}'
        lat_long = True
        attributes = []
        {out} = ReadShapefile(polygon, lat_long, 
        attributes, 'shapefile.shp', 'shapefile.dbf')
    """.format(polygon=params['polygon'], out=n_out['geo data']))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_readshapefile_params_success():
    params = {
        ReadShapefileOperation.POLYGON_ATTR_PARAM: 'points',
        ReadShapefileOperation.SHAPEFILE_PARAM: 'shapefile.shp',
        ReadShapefileOperation.ATTRIBUTES_PARAM: ['BAIRR', 'UF'],
        ReadShapefileOperation.LAT_LONG_PARAM: False
    }
    n_out = {'geo data': 'out'}

    instance = ReadShapefileOperation(parameters=params,
                                      named_inputs={},
                                      named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        polygon = '{polygon}'
        lat_long = False
        attributes = ['BAIRR', 'UF']
        {out} = ReadShapefile(polygon, lat_long, 
        attributes, 'shapefile.shp', 'shapefile.dbf')
    """.format(polygon=params['polygon'], out=n_out['geo data']))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_readshapefile_missing_columns_failure():
    params = {
    }

    with pytest.raises(ValueError):
        n_out = {'geo data': 'out'}
        ReadShapefileOperation(parameters=params, named_inputs={},
                               named_outputs=n_out)


'''
    Geo Within Operation
'''


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
        lat_long = True
        attributes = ['col1']
        alias = '_shp'
        polygon_col = 'points'
        col_lat = "LATITUDE"
        col_long = "LONGITUDE"     

        {out} = GeoWithinOperation({input}, {shape}, lat_long,
                       attributes, alias, polygon_col, col_lat, col_long)                  
       """.format(out=n_out['output data'], input=n_in['input data'],
                  shape=n_in['geo data']))

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


'''
    STDBSCAN Operation
'''


def test_stdbscan_minimal_params_success():
    params = {
        STDBSCANOperation.DATETIME_PARAM: ['date'],
        STDBSCANOperation.LON_PARAM: ['lon'],
        STDBSCANOperation.LAT_PARAM: ['lat'],
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}

    instance = STDBSCANOperation(parameters=params, named_inputs=n_in,
                                 named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
    out = st_dbscan(df1, 'lat', 'lon', 
            'date', 'cluster', spatial_threshold=500.0, 
             temporal_threshold=60, min_neighbors=15)
    """.format())
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_stdbscan_params_success():
    params = {
        STDBSCANOperation.DATETIME_PARAM: ['date1'],
        STDBSCANOperation.LON_PARAM: ['lon1'],
        STDBSCANOperation.LAT_PARAM: ['lat1'],
        STDBSCANOperation.ALIAS_PARAM: 'result',
        STDBSCANOperation.TMP_THRESHOLD_PARAM: 100,
        STDBSCANOperation.SPA_THRESHOLD_PARAM: 150,
        STDBSCANOperation.MIN_SAMPLE_PARAM: 10
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}

    instance = STDBSCANOperation(parameters=params, named_inputs=n_in,
                                 named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        out = st_dbscan(df1, 'lat1', 'lon1', 
            'date1', 'result', spatial_threshold=150.0, 
             temporal_threshold=100, min_neighbors=10)
    """)
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_stdbscan_missing_columns_failure():
    params = {
    }

    with pytest.raises(ValueError):
        n_in = {'input data': 'df1'}
        n_out = {'output data': 'out'}
        STDBSCANOperation(parameters=params, named_inputs=n_in,
                          named_outputs=n_out)
