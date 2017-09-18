# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class ReadShapefileOperation(Operation): # ok
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

    def generate_code(self):
        code = """

        """
        return dedent(code)

class GeoWithinOperation(Operation):
    POLYGON_POINTS_COLUMN_PARAM = 'polygon'
    POLYGON_ATTRIBUTES_COLUMN_PARAM = 'polygon_attributes'
    POLYGON_ALIAS_COLUMN_PARAM = 'alias'
    TARGET_LAT_COLUMN_PARAM = 'latitude'
    TARGET_LON_COLUMN_PARAM = 'longitude'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.polygon_column = parameters[self.POLYGON_POINTS_COLUMN_PARAM]
        self.attributes = parameters[self.POLYGON_ATTRIBUTES_COLUMN_PARAM]

        self.alias = [alias.strip() for alias in parameters.get(self.POLYGON_ALIAS_COLUMN_PARAM, '').split(',')]

        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _indexed.
        self.alias = [x[1] or '{}_alias'.format(x[0]) for x in izip_longest(self.attributes, elf.alias[:len(self.attributes)])]

        self.lat_column = parameters[self.TARGET_LAT_COLUMN_PARAM]
        self.lon_column = parameters[self.TARGET_LON_COLUMN_PARAM]

        self.has_import = "from functions.geo.geo_functions import GeoWithinOperation\n"
        if len(self.lat_column) == 0 or len(self.lon_column) == 0 or len(self.polygon_column) == 0:
            raise ValueError('Values for latitude and longitude columns must be informed')

    def generate_code(self):
        code = """
            settings = dict()
            settings['lat_col']  = "{LAT}"
            settings['long_col'] = "{LON}"
            settings['id_col']   = {ids}
            settings['polygon']  = '{polygon}'
            {out} = GeoWithinOperation({input}, {shape}, settings, numFrag)
            """.format(shape   = self.named_inputs['geo data'],
                       polygon = self.polygon_column[0],
                       ids     = self.attributes,
                       input   = self.named_inputs['input data'],
                       LAT     = self.lat_column[0],
                       LON     = self.lon_column[0],
                       out     = self.named_outputs['output data'])
        return dedent(code)



