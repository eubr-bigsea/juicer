# -*- coding: utf-8 -*-
import ast
import json
import pprint
from textwrap import dedent

from juicer.dist.metadata import MetadataGet
from juicer.service import limonero_service
from juicer.spark.operation import Operation

class ReadShapefile(Operation):
    """
    Reads a shapefile.
    Parameters:
        - File location
        - List of target columns with position and alias
        - Row filter expression
    """
    DATA_SOURCE_ID_PARAM = 'shapefile'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        if self.DATA_SOURCE_ID_PARAM in parameters:
            self.database_id = parameters[self.DATA_SOURCE_ID_PARAM]
            metadata_obj = MetadataGet('123456')
            self.metadata = metadata_obj.get_metadata(self.database_id)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.DATA_SOURCE_ID_PARAM, self.__class__))

    def generate_code(self):
        code = """
            import shapefile

            from matplotlib.path import Path
            from io import BytesIO
            reload(sys)
            sys.setdefaultencoding('utf-8')

            metadata = {}
            shp_file = metadata['url']
            dbf_file = re.sub('.shp$', '.dbf', shp_file)
            shp_content = spark.sparkContext.binaryFiles(shp_file).collect()
            dbf_content = spark.sparkContext.binaryFiles(dbf_file).collect()
            shp_io = BytesIO(shp_content[0][1])
            dbf_io = BytesIO(dbf_content[0][1])
            shp_object = shapefile.Reader(shp=shp_io, dbf=dbf_io)
            records = shp_object.records()
            sectors = shp_object.shapeRecords()
            header = []
            for record in metadata['attributes']:
                header.append(json.dumps(record['name']).strip('"'))
            header.append('points')


            data = []
            for i in range(0,len(sectors)):
                attributes = []
                for r in records[i]:
                    attributes.append(str(r))
                points = []
                for point in sectors[i].shape.points:
                    pair = []
                    pair.append(point[0])
                    pair.append(point[1])
                    points.append(pair)
                attributes.append(points)
                data.append(attributes)

            {} = SparkSession.createDataFrame(spark, data, header)
        """.format(self.metadata, self.outputs[0])

        return dedent(code)
