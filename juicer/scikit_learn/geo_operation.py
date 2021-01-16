# -*- coding: utf-8 -*-
from textwrap import dedent

import re
from juicer.operation import Operation


class ReadShapefileOperation(Operation):
    """Reads a shapefile.

    Parameters:
        - File location
        - Alias

    """
    SHAPEFILE_PARAM = 'shapefile'
    POLYGON_ATTR_PARAM = 'polygon'
    ATTRIBUTES_PARAM = 'attributes'
    LAT_LONG_PARAM = 'lat_lon'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = self.SHAPEFILE_PARAM in parameters
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.SHAPEFILE_PARAM, self.__class__))

        self.shp_file = parameters[self.SHAPEFILE_PARAM]
        if '.shp' not in self.shp_file:
            self.shp_file = self.shp_file+'.shp'

        self.dbf_file = re.sub('.shp$', '.dbf', self.shp_file)
        self.alias = parameters.get(self.POLYGON_ATTR_PARAM, 'points')
        self.lat_long = parameters.get(self.LAT_LONG_PARAM,
                                       True) in (1, '1', True)
        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        self.output = self.named_outputs.get(
                'geo data', 'output_data_{}'.format(self.order))

        from juicer.scikit_learn.library.read_shapefile import ReadShapefile
        self.transpiler_utils.add_custom_function("ReadShapefile",
                                                  ReadShapefile)

    def generate_code(self):
        """Generate code."""

        code = """
        polygon = '{polygon}'
        lat_long = {lat_long}
        attributes = {attributes}
        {out} = ReadShapefile(polygon, lat_long, attributes, '{shp}', '{dbf}')
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
            from juicer.scikit_learn.library.geo_within \
                import GeoWithinOperation
            self.transpiler_utils.add_custom_function("GeoWithinOperation",
                                                      GeoWithinOperation)

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

        self.has_code = len(named_inputs) == 1 and any([self.contains_results(),
                                                        len(named_outputs) > 0])
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
            from juicer.scikit_learn.library.stdbscan import STDBSCAN
            self.transpiler_utils.add_custom_function("STDBSCAN", STDBSCAN)

    def generate_code(self):
        """Generate code."""
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        code = """
        {output} = {data_input}{copy_code}
            
        st_dbscan = STDBSCAN(spatial_threshold={spatial}, 
            temporal_threshold={temporal}, min_neighbors={minPts})
                         
        {output} = st_dbscan.fit_transform({data_input}, 
                col_lat='{col_latitude}', col_lon='{col_longitude}',
                col_time='{col_datetime}', col_cluster='{alias}')

        """.format(copy_code=copy_code,
                   data_input=self.named_inputs['input data'],
                   output=self.output, minPts=self.min_pts,
                   spatial=self.spatial_thr, temporal=self.temporal_thr,
                   col_latitude=self.lat_col, col_longitude=self.lon_col,
                   col_datetime=self.datetime_col, alias=self.alias)

        return dedent(code)


class CartographicProjectionOperation(Operation):
    """ Cartographic Projection Operation

    Converts different cartographic projections to each other.
    """

    SRC_PROJ_PARAM = 'src_projection'
    DST_PROJ_PARAM = 'dst_projection'
    LAT_PARAM = 'col_lat'
    LON_PARAM = 'col_lon'
    LAT_ALIAS_PARAM = 'alias_lat'
    LON_ALIAS_PARAM = 'alias_lon'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(named_inputs) == 1 and any([self.contains_results(),
                                                        len(named_outputs) > 0])
        if self.has_code:

            for att in [self.SRC_PROJ_PARAM, self.DST_PROJ_PARAM,
                        self.LAT_PARAM, self.LON_PARAM]:
                if att not in self.parameters:
                    raise ValueError(
                        _('Parameters {} must be informed for task {}.')
                        .format(att, self.__class__))

            self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

            self.lat_col = parameters.get(self.LAT_PARAM)[0]
            self.lon_col = parameters.get(self.LON_PARAM)[0]

            self.lat_alias = parameters.get(self.LAT_ALIAS_PARAM, self.lat_col)
            self.lon_alias = parameters.get(self.LON_ALIAS_PARAM, self.lon_col)

            self.src_prj = parameters.get(self.SRC_PROJ_PARAM)
            self.dst_prj = parameters.get(self.DST_PROJ_PARAM)

    def generate_code(self):
        """Generate code."""
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        code = """
        import pyproj
        {output} = {data_input}{copy_code}

        old_proj = pyproj.Proj({src_epsg}, preserve_units=True)
        new_proj = pyproj.Proj({dst_epsg}, preserve_units=True)

        lon = {data_input}['{col_lon}'].to_numpy()
        lat = {data_input}['{col_lat}'].to_numpy()
        x1, y1 = old_proj(lon, lat)
        x2, y2 = pyproj.transform(old_proj, new_proj, x1, y1)
        
        {output}['{alias_lon}'] = x2
        {output}['{alias_lat}'] = y2
        """.format(copy_code=copy_code,
                   data_input=self.named_inputs['input data'],
                   output=self.output,
                   col_lat=self.lat_col, col_lon=self.lon_col,
                   alias_lon=self.lon_alias, alias_lat=self.lat_alias,
                   src_epsg=self.src_prj, dst_epsg=self.dst_prj)

        return dedent(code)
