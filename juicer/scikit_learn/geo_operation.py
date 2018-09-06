# -*- coding: utf-8 -*-
from textwrap import dedent
from itertools import izip_longest
import re
from juicer.operation import Operation


class ReadShapefileOperation(Operation):
    """Reads a shapefile.

    Parameters:
        - File location
        - Alias

    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = 'shapefile' in parameters
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('shapefile', self.__class__))

        self.shp_file = parameters['shapefile']
        if '.shp' not in self.shp_file:
            self.shp_file = self.shp_file+'.shp'

        self.dbf_file = re.sub('.shp$', '.dbf', self.shp_file)
        self.alias = parameters.get('polygon', 'points')
        self.lat_long = True
        self.attributes = parameters.get('attributes', []) or []
        self.output = self.named_outputs.get(
                'geo data', 'output_data_{}'.format(self.order))
        self.has_import = \
            "from juicer.scikit_learn.library.read_shapefile " \
            "import ReadShapefile\n"

    def generate_code(self):
        """Generate code."""

        code = """
        polygon = '{polygon}'
        lat_long = {lat_long}
        attributes = {attributes}
        {out} = read_shapefile(polygon, lat_long, attributes, '{shp}', '{dbf}')

        """.format(shp=self.shp_file,
                   dbf=self.dbf_file,
                   lat_long=self.lat_long,
                   polygon=self.alias,
                   attributes=self.attributes,
                   out=self.output)
        return dedent(code)


class GeoWithinOperation(Operation):
    """GeoWithinOperation.
    """

    POLYGON_POINTS_COLUMN_PARAM = 'polygon'
    POLYGON_ATTR_COLUMN_PARAM = 'polygon_attributes'
    TARGET_LAT_COLUMN_PARAM = 'latitude'
    TARGET_LON_COLUMN_PARAM = 'longitude'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if len(named_inputs) == 2:
            self.has_code = True

            attributes = [self.TARGET_LAT_COLUMN_PARAM,
                          self.TARGET_LON_COLUMN_PARAM,
                          self.POLYGON_POINTS_COLUMN_PARAM]

            for att in attributes:
                if att not in parameters:
                    raise ValueError(
                        _("Parameter '{}' must be informed for task {}")
                        .format(att, self.__class__))

            self.lat_column = parameters[self.TARGET_LAT_COLUMN_PARAM]
            self.lon_column = parameters[self.TARGET_LON_COLUMN_PARAM]
            self.polygon_column = parameters.get(
                    self.POLYGON_POINTS_COLUMN_PARAM, 'points')
            self.attributes = parameters.get(self.POLYGON_ATTR_COLUMN_PARAM, [])
            if len(self.attributes) == 0:
                self.attributes = []

            self.alias = parameters.get('alias', '_shp')

            self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))
            self.has_import = \
                "from juicer.scikit_learn.library.geo_within " \
                "import GeoWithinOperation\n"

    def generate_code(self):
        """Generate code."""
        code = """
        lat_long = {lat_long}
        attributes = {att}
        alias = '{alias}'
        polygon_col = '{polygon}'
        col_lat = "{LAT}"
        col_long = "{LON}"     

        {out} = GeoWithinOperation({input}, {shape}, lat_long,
                       attributes, alias, polygon_col, col_lat, col_long)    
                                
        """.format(shape=self.named_inputs['geo data'],
                   polygon=self.polygon_column[0],
                   att=self.attributes,  alias=self.alias,
                   input=self.named_inputs['input data'],
                   LAT=self.lat_column[0], LON=self.lon_column[0],
                   out=self.output, lat_long=True)
        return dedent(code)


class STDBSCANOperation(Operation):
    """STDBSCANOperation.

    Perform a ST-DBSCAN with the geospatial data
    """

    LAT_PARAM = 'LAT'
    LON_PARAM = 'LON'
    DATETIME_PARAM = 'DATETIME'
    MIN_SAMPLE_PARAM = 'min_sample'
    SPA_THRESHOLD_PARAM = 'spatial_threshold'
    TMP_THRESHOLD_PARAM = 'thresold_temporal'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(named_inputs) == 1 and self.contains_results()
        if self.has_code:

            if any([self.LAT_PARAM not in parameters,
                    self.LON_PARAM not in parameters,
                    self.DATETIME_PARAM not in parameters]):
                raise ValueError(
                    _('Parameters {}, {} and {} must be informed for task {}.')
                    .format('Latitude', 'Longitude', 'Datetime',
                            self.__class__))

            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))

            self.lat_col = parameters.get(self.LAT_PARAM)[0]
            self.lon_col = parameters.get(self.LON_PARAM)[0]
            self.datetime_col = parameters.get(self.DATETIME_PARAM)[0]

            self.alias = parameters.get(self.ALIAS_PARAM, 'cluster')
            self.min_pts = parameters.get(self.MIN_SAMPLE_PARAM, 15) or 15
            self.spatial_thr = parameters.get(self.SPA_THRESHOLD_PARAM,
                                              500) or 500
            self.temporal_thr = parameters.get(self.TMP_THRESHOLD_PARAM,
                                               60) or 60

            self.min_pts = abs(int(self.min_pts))
            self.spatial_thr = abs(float(self.spatial_thr))
            self.temporal_thr = abs(int(self.temporal_thr))
            self.has_import = "from juicer.scikit_learn.library.stdbscan " \
                              "import st_dbscan\n"

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = st_dbscan({data_input}, '{col_latitude}', '{col_longitude}', 
            '{col_datetime}', '{alias}', spatial_threshold={spatial}, 
            temporal_threshold={temporal}, min_neighbors={minPts})
            """.format(data_input=self.named_inputs['input data'],
                       output=self.output, minPts=self.min_pts,
                       spatial=self.spatial_thr, temporal=self.temporal_thr,
                       col_latitude=self.lat_col, col_longitude=self.lon_col,
                       col_datetime=self.datetime_col, alias=self.alias)

        return dedent(code)