# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from itertools import izip_longest
from textwrap import dedent

from jinja2 import Environment, BaseLoader
from juicer.operation import Operation
from juicer.spark.data_operation import DataReaderOperation


class ReadShapefile(DataReaderOperation):
    """
    Reads a shapefile.
    Parameters:
        - File location
        - List of target columns with position and alias
        - Row filter expression
    """
    DATA_SOURCE_ID_PARAM = 'shapefile'

    def __init__(self, parameters, named_inputs, named_outputs):
        DataReaderOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        if self.DATA_SOURCE_ID_PARAM in parameters:
            self._set_data_source_parameters(parameters)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}".format(
                    self.DATA_SOURCE_ID_PARAM, self.__class__)))
        self.output = self.named_outputs.get('geodata',
                                             'out_{}'.format(self.order))

    def generate_code(self):
        """ We still have to add a parameter to define whether the points are
        expressed as (LAT,LON) or (LON,LAT). This will change the way the points
        are read.

        LON,LAT:
            points.append([point[0], point[1]])
        LAT,LON:
            points.append([point[1], point[0]])
        """

        code_template = dedent("""
            import shapefile
            import zipfile
            from io import BytesIO
            shp_file = '{{url}}'
            shp_io = None
            dbf_io = None
            shx_io = None
            {%- if zipped %}
            # shp and dbf files must be present in the zip file
            # shx is optional
            memory_data = BytesIO(spark_session.sparkContext.binaryFiles(
                shp_file).collect()[0][1])

            z = zipfile.ZipFile(memory_data)
            for file_name in z.namelist():
                if file_name.endswith('.shp'):
                    shp_io = BytesIO(z.open(file_name).read())
                elif file_name.endswith('.shx'):
                    shx_io = BytesIO(z.open(file_name).read())
                elif file_name.endswith('.dbf'):
                    dbf_io = BytesIO(z.open(file_name).read())
            if not all([shp_io, dbf_io]):
                raise ValueError('{{invalid_shp}}')
            {%- else %}
            dbf_file = re.sub('.shp$', '.dbf', shp_file)
            shp_content = spark_session.sparkContext.binaryFiles(
                shp_file).collect()
            dbf_content = spark_session.sparkContext.binaryFiles(
                dbf_file).collect()
            shp_io = BytesIO(shp_content[0][1])
            dbf_io = BytesIO(dbf_content[0][1])
            {%- endif %}

            shp_object = shapefile.Reader(shp=shp_io, dbf=dbf_io, shx=shx_io)
            records = shp_object.shapeRecords()
            header = [
                {%- for attr in attrs %}
                '{{attr.name}}',
                {%- endfor %}
            ]
            header = types.StructType(
                [types.StructField(h, types.StringType(), False)
                             for h in header])
            header.add(types.StructField('points', types.ArrayType(
                types.ArrayType(types.DoubleType()))))
            data = []
            for shape_record in records:
                data.append(shape_record.record +
                    [shape_record.shape.points])
            {{out}} = spark_session.createDataFrame(data, header)
        """)
        ctx = dict(
            zipped=self.metadata['url'].endswith('.zip'),
            invalid_shp=_('Invalid zipped shapefile. It must contains '
                          'both *.shp and *.dbf files.'),
            url=self.metadata['url'], out=self.output,
            attrs=self.metadata.get('attributes', []))
        template = Environment(loader=BaseLoader).from_string(code_template)
        return template.render(ctx)


class GeoWithin(Operation):
    POLYGON_POINTS_COLUMN_PARAM = 'polygon'
    POLYGON_ATTRIBUTES_COLUMN_PARAM = 'polygon_attributes'
    POLYGON_ALIAS_COLUMN_PARAM = 'alias'
    TARGET_LAT_COLUMN_PARAM = 'latitude'
    TARGET_LON_COLUMN_PARAM = 'longitude'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.points_column = parameters[self.POLYGON_POINTS_COLUMN_PARAM]
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

        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))
        if len(self.lat_column) == 0 or len(self.lon_column) == 0 or len(
                self.points_column) == 0:
            raise ValueError(
                _('Values for latitude and longitude columns must be informed'))

    def generate_code(self):
        code = """
            from matplotlib.path import Path
            import pyqtree

            schema = [s.name for s in {geo}.schema]
            shp_object = {geo}.collect()
            bcast_shapefile = spark_session.sparkContext.broadcast(
                shp_object)

            x_min = float('+inf')
            y_min = float('+inf')
            x_max = float('-inf')
            y_max = float('-inf')
            for i, polygon in enumerate(shp_object):
                for point in polygon['points']:
                    x_min = min(x_min, point[1])
                    y_min = min(y_min, point[0])
                    x_max = max(x_max, point[1])
                    y_max = max(y_max, point[0])
            #
            sp_index = pyqtree.Index(bbox=[x_min, y_min, x_max, y_max])
            for inx, polygon in enumerate(shp_object):
                points = []
                x_min = float('+inf')
                y_min = float('+inf')
                x_max = float('-inf')
                y_max = float('-inf')
                for point in polygon['points']:
                    points.append((point[0], point[1]))
                    x_min = min(x_min, point[0])
                    y_min = min(y_min, point[1])
                    x_max = max(x_max, point[0])
                    y_max = max(y_max, point[1])
                sp_index.insert(item=inx, bbox=[x_min, y_min, x_max, y_max])

            broad_casted_sp_index = spark_session.sparkContext.broadcast(
                sp_index)

            counter = 0
            def get_first_polygon(lat, lng):
                x = float(lat)
                y = float(lng)
                bcast_index = broad_casted_sp_index.value
                # Here it uses longitude, latitude
                matches = bcast_index.intersect([y, x, y, x])

                for shp_inx in matches:
                    row = bcast_shapefile.value[shp_inx]
                    p_polygon = Path(row['{points_column}'])
                    # Here it uses longitude, latitude
                    if p_polygon.contains_point([y, x]):
                        return [c for c in row]
                return [None] * len(bcast_shapefile.value[0])

            udf_get_first_polygon = functions.udf(
                get_first_polygon, types.ArrayType(types.StringType()))
            within = {input}.withColumn(
                'tmp_polygon_data', udf_get_first_polygon(
                    functions.col('{lat}'), functions.col('{lng}')))
            aliases = {aliases}
            attributes_to_add = {attributes}
            {out} = within.select(within.columns +
               [within.tmp_polygon_data[i].alias(aliases.pop())
                for i, col in enumerate(schema)
                if col in attributes_to_add])

            {out} = {out}.drop('tmp_polygon_data')
        """.format(geo=self.named_inputs['geo data'],
                   points_column=self.points_column[0],
                   input=self.named_inputs['input data'],
                   lat=self.lat_column[0],
                   lng=self.lon_column[0], out=self.output,
                   aliases=repr(self.alias),
                   attributes=self.attributes
                   )
        return dedent(code)
