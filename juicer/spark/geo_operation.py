# -*- coding: utf-8 -*-
from itertools import izip_longest
from textwrap import dedent

from juicer.dist.metadata import MetadataGet
from juicer.operation import Operation


class ReadShapefile(Operation):
    """
    Reads a shapefile.
    Parameters:
        - File location
        - List of target columns with position and alias
        - Row filter expression
    """
    DATA_SOURCE_ID_PARAM = 'shapefile'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs,
                           named_inputs, named_outputs)
        if self.DATA_SOURCE_ID_PARAM in parameters:
            self.database_id = parameters[self.DATA_SOURCE_ID_PARAM]
            metadata_obj = MetadataGet('123456')
            self.metadata = metadata_obj.get_metadata(self.database_id)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.DATA_SOURCE_ID_PARAM, self.__class__))

    def generate_code(self):
        """ We still have to add a parameter to define whether the points are
        expressed as (LAT,LON) or (LON,LAT). This will change the way the points
        are read.

        LON,LAT:
            points.append([point[0], point[1]])
        LAT,LON:
            points.append([point[1], point[0]])
        """

        code = """
            import shapefile
            from io import BytesIO
            reload(sys)
            sys.setdefaultencoding('utf-8')
            metadata = {}
            shp_file = metadata['url']
            dbf_file = re.sub('.shp$', '.dbf', shp_file)
            shp_content = spark_session.sparkContext.binaryFiles(shp_file)\
                .collect()
            dbf_content = spark_session.sparkContext.binaryFiles(dbf_file)\
                .collect()
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
            for i, sector in enumerate(sectors):
                attributes = []
                for r in records[i]:
                    attributes.append(str(r))
                points = []
                for point in sector.shape.points:
                    points.append([point[1], point[0]])
                attributes.append(points)
                data.append(attributes)
            {} = spark_session.createDataFrame(data, header)
        """.format(self.metadata, self.outputs[0])

        return dedent(code)


class GeoWithin(Operation):
    POLYGON_POINTS_COLUMN_PARAM = 'polygon'
    POLYGON_ATTRIBUTES_COLUMN_PARAM = 'polygon_attributes'
    POLYGON_ALIAS_COLUMN_PARAM = 'alias'
    TARGET_LAT_COLUMN_PARAM = 'latitude'
    TARGET_LON_COLUMN_PARAM = 'longitude'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs,
                           named_inputs, named_outputs)
        self.polygon_column = parameters[self.POLYGON_POINTS_COLUMN_PARAM]
        self.attributes = parameters[self.POLYGON_ATTRIBUTES_COLUMN_PARAM]

        self.alias = [
            alias.strip() for alias in
            parameters.get(self.POLYGON_ALIAS_COLUMN_PARAM, '').split(',')]

        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _indexed.
        self.alias = [x[1] or '{}_alias'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.lat_column = parameters[self.TARGET_LAT_COLUMN_PARAM]
        self.lon_column = parameters[self.TARGET_LON_COLUMN_PARAM]

        if len(self.lat_column) == 0 or len(self.lon_column) == 0 or len(
                self.polygon_column) == 0:
            raise ValueError(
                'Values for latitude and longitude columns must be informed')

    def generate_code(self):
        code = """
            from matplotlib.path import Path
            import pyqtree

            shp_object = {0}.collect()
            broad_shapefile_{0} = spark_session.sparkContext.broadcast(
                shp_object)

            xmin = float('+inf')
            ymin = float('+inf')
            xmax = float('-inf')
            ymax = float('-inf')
            for sector in shp_object:
                for point in sector['{1}']:
                    xmin = min(xmin, point[1])
                    ymin = min(ymin, point[0])
                    xmax = max(xmax, point[1])
                    ymax = max(ymax, point[0])
            #
            spindex = pyqtree.Index(bbox=[xmin, ymin, xmax, ymax])
            broad_casted_spindex = spark_session.sparkContext.broadcast(spindex)

            def get_first_sector(lat, lng):
                for row in broad_shapefile_{0}.value:
                    polygon = Path(row['{1}'])
                    if polygon.contains_point([float(lat), float(lng)]):
                        return [col for col in row]
                return [None]*len(broad_shapefile_{0}.value[0])

            shapefile_features_count_{0}= len(broad_shapefile_{0}.value[0])
            udf_get_first_sector = functions.udf(
                get_first_sector, types.ArrayType(types.StringType()))
            within_{0} = {2}.withColumn(
                "sector_position", udf_get_first_sector(functions.col('{3}'),
                                                        functions.col('{4}')))
            {5} = within_{0}.select(within_{0}.columns +
                [within_{0}.sector_position[i]
                    for i in xrange(shapefile_features_count_{0})])
        """.format(self.named_inputs['geo data'], self.polygon_column[0],
                   self.named_inputs['input data'], self.lat_column[0],
                   self.lon_column[0], self.output)
        return dedent(code)
