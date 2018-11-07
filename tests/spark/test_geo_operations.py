# coding=utf-8
from __future__ import absolute_import

import ast
import json
from textwrap import dedent

from juicer.spark.data_operation import DataReaderOperation
from juicer.spark.geo_operation import ReadShapefile, GeoWithin
from mock import patch
from tests import compare_ast, format_code_comparison


def set_ds_param(self, params):
    self.metadata = {'updated': None, 'url': 'http://limonero/shape1.shp'}


@patch.object(DataReaderOperation, '_set_data_source_parameters',
              side_effect=set_ds_param, autospec=True)
def test_read_shapefile_success(mock_set_ds_param):
    params = {
        ReadShapefile.DATA_SOURCE_ID_PARAM: 2017,
    }
    n_out = {'geodata': 'output_1'}
    instance = ReadShapefile(params, named_inputs={}, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent("""
         import shapefile
         import zipfile
         from io import BytesIO

         shp_file = '{url}'
         shp_io = None
         dbf_io = None
         shx_io = None
         dbf_file = re.sub('.shp$', '.dbf', shp_file)
         shp_content = spark_session.sparkContext.binaryFiles(
             shp_file).collect()
         dbf_content = spark_session.sparkContext.binaryFiles(
             dbf_file).collect()
         shp_io = BytesIO(shp_content[0][1])
         dbf_io = BytesIO(dbf_content[0][1])

         shp_object = shapefile.Reader(shp=shp_io, dbf=dbf_io, shx=shx_io)
         records = shp_object.shapeRecords()
         header = []
         header = types.StructType(
             [types.StructField(h, types.StringType(), False)
                          for h in header])
         header.add(types.StructField('points', types.ArrayType(
             types.ArrayType(types.DoubleType()))))
         data = []
         for shape_record in records:
             data.append(shape_record.record +
                 [shape_record.shape.points])
         {output} = spark_session.createDataFrame(data, header)
            """.format(url=instance.metadata['url'],
                       output=n_out['geodata'], ))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_geo_within_success():
    params = {
        GeoWithin.POLYGON_POINTS_COLUMN_PARAM: ['polygon'],
        GeoWithin.POLYGON_ATTRIBUTES_COLUMN_PARAM: ['attribute'],
        GeoWithin.POLYGON_ALIAS_COLUMN_PARAM: 'alias',
        GeoWithin.TARGET_LAT_COLUMN_PARAM: 'latitude',
        GeoWithin.TARGET_LON_COLUMN_PARAM: 'longitude'
    }
    n_out = {'output data': 'output_1'}
    n_in = {'input data': 'input_1', 'geo data': 'geo_data'}
    instance = GeoWithin(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent("""
        from matplotlib.path import Path
        import pyqtree
        attributes_to_add = {attributes}

        schema = [s.name for s in {geo}.schema]
        shp_object = {geo}.select(attributes_to_add +
                ['{points}']).collect()
        bcast_shapefile = spark_session.sparkContext.broadcast(shp_object)

        f_min = functions.udf(
               lambda v, index: min([item[index] for item in v]),
                   types.DoubleType())
        f_max = functions.udf(
            lambda v, index: max([item[index] for item in v]),
                types.DoubleType())

        boundaries = {geo}.select(
            (f_min('{points}', functions.lit(1))).alias('x_min'),
            (f_min('{points}', functions.lit(0))).alias('y_min'),
            (f_max('{points}', functions.lit(1))).alias('x_max'),
            (f_max('{points}', functions.lit(0))).alias('y_max'),
        ).collect()

        global_min_x = float('+inf')
        global_min_y = float('+inf')
        global_max_x = float('-inf')
        global_max_y = float('-inf')

        to_update = []
        for inx, row in enumerate(boundaries):
            x_min = row['x_min']
            y_min = row['y_min']
            x_max = row['x_max']
            y_max = row['y_max']
            to_update.append({{
                'item': inx,
                'bbox': [x_min, y_min, x_max, y_max]
            }})
            global_min_x = min(global_min_x, x_min)
            global_min_y = min(global_min_y, y_min)
            global_max_x = max(global_max_x, x_max)
            global_max_y = max(global_max_y, y_max)

        sp_index = pyqtree.Index(
                bbox=[global_min_x, global_min_y, global_max_x, global_max_y])

        for item in to_update:
            sp_index.insert(**item)

        broad_casted_sp_index = spark_session.sparkContext.broadcast(
            sp_index)

        def get_first_polygon(lat, lng):
            x = float(lat)
            y = float(lng)
            bcast_index = broad_casted_sp_index.value
            matches = bcast_index.intersect([x, y, x, y])
            for shp_inx in matches:
                row = bcast_shapefile.value[shp_inx]
                p_polygon = Path(row['{points}'])
                # Here it uses longitude, latitude
                if p_polygon.contains_point([y, x]):
                    return [c for c in row]
            return [None] * len(bcast_shapefile.value[0])

        udf_get_first_polygon = functions.udf(
            get_first_polygon, types.ArrayType(types.StringType()))
        within = input_1.withColumn(
            "tmp_polygon_data", udf_get_first_polygon(functions.col('l'),
                                                    functions.col('l')))
        aliases = {aliases}
        {output} = within.select(within.columns +
            [within.tmp_polygon_data[i].alias(aliases.pop())
                for i, col in enumerate(schema)
                if col in attributes_to_add])
        {output} = {output}.drop('tmp_polygon_data')

           """.format(
        aliases=json.dumps(
            params[GeoWithin.POLYGON_ALIAS_COLUMN_PARAM].split(',')),
        output=n_out['output data'], geo=n_in['geo data'],
        points=params[GeoWithin.POLYGON_POINTS_COLUMN_PARAM][0],
        attributes=params[
            GeoWithin.POLYGON_ATTRIBUTES_COLUMN_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)
