# coding=utf-8
import decimal
from textwrap import dedent

import datetime

from juicer.operation import Operation
from juicer.service import limonero_service

try:
    from urllib.request import urlopen
    from urllib.parse import urlparse, parse_qs
except ImportError:
    from urlparse import urlparse, parse_qs
    from urllib import urlopen


class DataReaderOperation(Operation):
    HEADER_PARAM = 'header'
    SEPARATOR_PARAM = 'separator'
    QUOTE_PARAM = 'quote'
    INFER_SCHEMA_PARAM = 'infer_schema'
    NULL_VALUES_PARAM = 'None_values'  # Must be compatible with other platforms
    MODE_PARAM = 'mode'
    DATA_SOURCE_ID_PARAM = 'data_source'

    INFER_FROM_LIMONERO = 'FROM_LIMONERO'
    INFER_FROM_DATA = 'FROM_VALUES'
    DO_NOT_INFER = 'NO'

    SEPARATORS = {
        '{tab}': '\\t',
        '{new_line}': '\\n',
    }

    LIMONERO_TO_PANDAS_DATA_TYPES = {
        "CHARACTER": 'object',
        "DATETIME": 'object',
        "DATE": 'datetime.date',
        "DOUBLE": 'np.float64',
        "DECIMAL": 'np.float64',
        "FLOAT": 'np.float64',
        "LONG": 'np.int64',
        "INTEGER": 'np.int32',
        "TEXT": 'object',
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any(
            [len(self.named_outputs) > 0, self.contains_results()])

        self.header = False
        if self.has_code:
            if self.DATA_SOURCE_ID_PARAM in parameters:
                self._set_data_source_parameters(parameters)
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.DATA_SOURCE_ID_PARAM, self.__class__))

            # Test if data source was changed since last execution and
            # invalidate cache if so.
            self._set_data_source_parameters(parameters)
            data_source_updated = self.metadata.get('updated')

            if data_source_updated:
                data_source_updated = datetime.datetime.strptime(
                    data_source_updated[0:19], '%Y-%m-%dT%H:%M:%S')
            self.supports_cache = (
                parameters.get('execution_date') is not None and
                data_source_updated < parameters['execution_date'])

            self.output = named_outputs.get('output data',
                                            'out_task_{}'.format(self.order))

    def _set_data_source_parameters(self, parameters):

        self.data_source_id = int(parameters[self.DATA_SOURCE_ID_PARAM])
        # Retrieve metadata from Limonero.
        limonero_config = self.parameters['configuration']['juicer'][
            'services']['limonero']
        url = limonero_config['url']
        token = str(limonero_config['auth_token'])

        # Is data source information cached?
        self.metadata = self.parameters.get('workflow', {}).get(
            'data_source_cache', {}).get(self.data_source_id)
        if self.metadata is None:
            self.metadata = limonero_service.get_data_source_info(
                url, token, self.data_source_id)
            self.parameters['workflow']['data_source_cache'][
                self.data_source_id] = self.metadata
        if not self.metadata.get('url'):
            raise ValueError(
                _('Incorrect data source configuration (empty url)'))

        self.header = parameters.get(
            self.HEADER_PARAM, False) not in ('0', 0, 'false', False)
        self.null_values = [v.strip() for v in parameters.get(
            self.NULL_VALUES_PARAM, '').split(",") if v.strip()]

        self.sep = parameters.get(
            self.SEPARATOR_PARAM,
            self.metadata.get('attribute_delimiter', ',')) or ','
        self.quote = parameters.get(self.QUOTE_PARAM,
                                    self.metadata.get('text_delimiter'))
        if self.quote == '\'':
            self.quote = '\\\''
        if self.sep in self.SEPARATORS:
            self.sep = self.SEPARATORS[self.sep]
        self.infer_schema = parameters.get(self.INFER_SCHEMA_PARAM,
                                           self.INFER_FROM_LIMONERO)
        self.mode = parameters.get(self.MODE_PARAM, 'FAILFAST')

    def generate_code(self):
        code = []
        infer_from_data = self.infer_schema == self.INFER_FROM_DATA
        infer_from_limonero = self.infer_schema == self.INFER_FROM_LIMONERO
        if self.has_code:
            if infer_from_limonero:
                self.header = self.metadata.get('is_first_line_header')
                if 'attributes' in self.metadata:
                    code.append('columns = {}')
                    parse_dates = []
                    converters = {}
                    attrs = self.metadata.get('attributes', [])
                    if attrs:
                        for attr in attrs:
                            if attr['type'] == 'DATETIME':
                                parse_dates.append(attr['name'])
                            elif attr['type'] == 'DECIMAL':
                                converters[attr['name']] = 'decimal.Decimal'
                            self._add_attribute_to_schema(attr, code)
                    else:
                        code.append("columns['value'] = '11'")
                    code.append('parse_dates = {}'.format(repr(parse_dates)))

                    def custom_repr(elems):
                        result = '{{{d}}}'.format(d=','.join(
                            ['"{}": {}'.format(k, v) for k, v in
                             elems.items()]))
                        return result

                    code.append('converters = {}'.format(custom_repr(
                        converters)))
                    code.append("")
                else:
                    raise ValueError(
                        _("Metadata do not include attributes information"))
            else:
                code.append('columns = None')

            if self.metadata['format'] in ['CSV', 'TEXT']:
                code.append("url = '{url}'".format(url=self.metadata['url']))

                null_values = self.null_values
                if self.metadata.get('treat_as_missing'):
                    null_values.extend([x.strip() for x in self.metadata.get(
                        'treat_as_missing').split(',') if x.strip()])
                null_option = ''.join(
                    [".option('nullValue', '{}')".format(n) for n in
                     set(null_values)]) if null_values else ""

                if self.metadata['format'] == 'CSV':
                    encoding = self.metadata.get('encoding', 'utf-8') or 'utf-8'
                    parsed = urlparse(self.metadata['url'])
                    if parsed.scheme in ('hdfs', 'file'):
                        if parsed.scheme == 'hdfs':
                            open_code = dedent("""
                            fs = pa.hdfs.connect('{hdfs_server}', {hdfs_port})
                            f = fs.open('{path}', 'rb')""".format(
                                path=parsed.path,
                                hdfs_server=parsed.hostname,
                                hdfs_port=parsed.port,
                            ))
                        else:
                            open_code = "f = open('{path}', 'rb')".format(
                                path=parsed.path)
                        code.append(open_code)
                        code_csv = dedent("""
                            header = {header}
                            {output} = pd.read_csv(f, sep='{sep}',
                                                   encoding='{encoding}',
                                                   header=header,
                                                   na_values={na_values},
                                                   dtype=columns,
                                                   parse_dates=parse_dates,
                                                   converters=converters)
                            f.close()
                            if header is None:
                                {output}.columns = ['col_{{col}}'.format(
                                    col=col) for col in {output}.columns]
                        """).format(output=self.output,
                                    input=parsed.path,
                                    sep=self.sep,
                                    encoding=encoding,
                                    header=0 if self.header else 'None',
                                    na_values=self.null_values if len(
                                        self.null_values) else 'None')
                        code.append(code_csv)
                    else:
                        raise ValueError(_('Not supported'))
                else:
                    code_csv = dedent("""
                    columns = {}
                    columns['value'] = object
                    {output} = spark_session.read{null_option}.schema(
                        columns).option(
                        'treatEmptyValuesAsNulls', 'true').text(
                            url)""".format(output=self.output,
                                           null_option=null_option))
                    code.append(code_csv)
            elif self.metadata['format'] == 'PARQUET_FILE':
                raise ValueError(_('Not supported'))
            elif self.metadata['format'] == 'JSON':
                code_json = dedent("""
                    columns = {{'value': object}}
                    {output} = spark_session.read.option(
                        'treatEmptyValuesAsNulls', 'true').json(
                        '{url}')""".format(output=self.output,
                                           url=self.metadata['url']))
                code.append(code_json)
            elif self.metadata['format'] == 'LIB_SVM':
                self._generate_code_for_lib_svm(code, infer_from_data)
            elif self.metadata['format'] == 'JDBC':
                self._generate_code_for_jdbc(code)

        if self.metadata.get('privacy_aware', False):
            raise ValueError(_('Not supported'))

        return '\n'.join(code)

        # url = self.name_file
        # parsed = urlparse(url)
        # if parsed.scheme == 'hdfs':
        #     code = """
        #     import pyarrow as pa
        #
        #     fs = pa.hdfs.connect('{hdfs_server}', {hdfs_port})
        #     header = {header}
        #     with fs.open('{path}', 'rb') as f:
        #         {output} = pd.read_csv(
        #             f, sep='{sep}', encoding='utf-8', header=header,
        #             na_values={na_values})
        #         if header == 0:
        #             {output}.columns = ['col_{{col}}'.format(col=col)
        #                 for col in {output}.columns]
        #     """.format(path=parsed.path,
        #                output=self.output,
        #                hdfs_server=parsed.hostname,
        #                hdfs_port=parsed.port,
        #                input=parsed.path,
        #                sep=self.separator,
        #                header=1 if self.header else 0,
        #                na_values=self.null_values if len(
        #                    self.null_values) else 'None')
        # else:
        #
        #     code = "{output} = pd.read_csv('{input}', sep='{sep}', " \
        #            "encoding='utf-8'".format(output=self.output,
        #                                      input=self.name_file,
        #                                      sep=self.separator)
        #
        #     if not self.header:
        #         code += ", header=None"
        #
        #     if len(self.null_values) > 0:
        #         code += ", na_values= {}".format(self.null_values)
        #     code += ")\n"
        #
        #     if not self.header:
        #         code += "{out}.columns = ['col_'+str(col) " \
        #                 "for col in {out}.columns]".format(out=self.output)
        # return dedent(code)

    def _generate_code_for_jdbc(self, code):

        parsed = urlparse(self.metadata['url'])
        qs_parsed = parse_qs(parsed.query)
        driver = self.SUPPORTED_DRIVERS.get(parsed.scheme)
        if driver is None:
            raise ValueError(
                _('Database {} not supported').format(parsed.scheme))
        if not self.metadata.get('command'):
            raise ValueError(
                _('No command nor table specified for data source.'))
        code_jdbc = dedent("""
            query = '{table}'
            if query.strip()[:6].upper() == 'SELECT':
                # Add parentheses required by Spark
                query = '(' + query + ') AS tb'
            {out} = spark_session.read.format('jdbc').load(
                driver='{driver}',
                url='jdbc:{scheme}://{server}:{port}{db}',
                user='{user}', password='{password}',
                dbtable=query)
                """.format(scheme=parsed.scheme, server=parsed.hostname,
                           db=parsed.path,
                           port=parsed.port,
                           driver=driver, user=qs_parsed.get('user', [''])[0],
                           password=qs_parsed.get('password', [''])[0],
                           table=self.metadata.get('command'),
                           out=self.output))
        code.append(code_jdbc)

    # noinspection PyMethodMayBeStatic
    def _generate_code_for_lib_svm(self, code, infer_from_data):
        raise ValueError(_('Not supported'))

    def _add_attribute_to_schema(self, attr, code):
        data_type = self.LIMONERO_TO_PANDAS_DATA_TYPES[attr['type']]

        # Metadata is not supported
        # metadata = {'sources': [
        #     '{}/{}'.format(self.data_source_id, attr['name'])
        # ]}
        code.append("columns['{name}'] = {dtype}".format(
            name=attr['name'],
            dtype=data_type))


class SaveOperation(Operation):
    """
    Saves the content of the DataFrame at the specified path.
    """
    NAME_PARAM = 'name'
    PATH_PARAM = 'path'
    STORAGE_ID_PARAM = 'storage'
    FORMAT_PARAM = 'format'
    TAGS_PARAM = 'tags'
    OVERWRITE_MODE_PARAM = 'mode'
    HEADER_PARAM = 'header'

    MODE_ERROR = 'error'
    MODE_APPEND = 'append'
    MODE_OVERWRITE = 'overwrite'
    MODE_IGNORE = 'ignore'

    FORMAT_PICKLE = 'PICKLE'
    FORMAT_CSV = 'CSV'
    FORMAT_JSON = 'JSON'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        for att in [self.NAME_PARAM, self.FORMAT_PARAM, self.PATH_PARAM]:
            if att not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        att, self.__class__))

        self.name = parameters.get(self.NAME_PARAM)
        self.format = parameters.get(self.FORMAT_PARAM, self.FORMAT_CSV)
        self.path = parameters.get(self.PATH_PARAM, '.')
        self.mode = parameters.get(self.OVERWRITE_MODE_PARAM, self.MODE_ERROR)

        if self.mode == "overwrite":
            self.mode = "w"
        elif self.mode == "append":
            self.mode = "a"
        else:
            # NEED CHECK
            self.mode = "w"

        self.header = parameters.get(self.HEADER_PARAM, False) in (1, '1', True)

        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(
                                                 self.order))

        self.filename = self.name
        # if len(self.path) > 0:
        #     self.filename = '/'+self.name
        # else:
        #     self.filename = '/'+self.path+'/'+self.name

        self.has_code = len(self.named_inputs) == 1

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):

        code_save = ""
        if self.FORMAT_CSV == self.format:
            code_save = """
                {input}.to_csv('{output}', sep=',', mode='{mode}',
                    header={header}, index=False, encoding='utf-8')
                
                {tmp} = None
                """.format(tmp=self.output,
                           output=self.filename,
                           input=self.named_inputs['input data'],
                           mode=self.mode,
                           header=self.header)
        elif self.FORMAT_JSON == self.format:
            code_save = """
            {input}.to_json('{output}', orient='records', encoding='utf-8')
            {tmp} = None
            """.format(tmp=self.output, output=self.filename,
                       input=self.named_inputs['input data'])

        return dedent(code_save)
