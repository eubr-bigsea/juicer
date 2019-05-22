# coding=utf-8
import json
import uuid
from textwrap import dedent

import datetime

from jinja2 import Environment, BaseLoader
from juicer.operation import Operation
from juicer.service import limonero_service

try:
    from urllib.request import urlopen
    from urllib.parse import urlparse, parse_qs
except ImportError:
    from urllib.parse import urlparse, parse_qs
    from urllib.request import urlopen


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

    SUPPORTED_DRIVERS = {
        'mysql': dedent("""
            import pymysql
            query = '{table}'
            connection = pymysql.connect(
                host='{host}',
                port={port},
                user='{user}',
                password='{password}',
                db='{db}',
                charset='utf8')
            {out} = pd.read_sql(query, con=connection)
        """)
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
                            # elif attr['type'] == 'DECIMAL':
                            #    converters[attr['name']] = 'decimal.Decimal'
                            self._add_attribute_to_schema(attr, code)
                    else:
                        code.append("columns['value'] = '11'")
                    code.append('parse_dates = {}'.format(repr(parse_dates)))

                    def custom_repr(elems):
                        result = '{{{d}}}'.format(d=','.join(
                            ['"{}": {}'.format(k, v) for k, v in
                             list(elems.items())]))
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

    def _generate_code_for_jdbc(self, code):

        parsed = urlparse(self.metadata['url'])
        qs_parsed = parse_qs(parsed.query)
        if parsed.scheme not in self.SUPPORTED_DRIVERS:
            raise ValueError(
                _('Database {} not supported').format(parsed.scheme))
        if not self.metadata.get('command'):
            raise ValueError(
                _('No command nor table specified for data source.'))

        code_jdbc = self.SUPPORTED_DRIVERS[parsed.scheme].format(
            scheme=parsed.scheme, host=parsed.hostname,
            db=parsed.path[1:],
            port=parsed.port or 3306,
            user=qs_parsed.get('user', [''])[0],
            password=qs_parsed.get('password', [''])[0],
            table=self.metadata.get('command'),
            out=self.output)
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
    FORMAT_PARQUET = 'PARQUET'

    USER_PARAM = 'user'
    WORKFLOW_ID_PARAM = 'workflow_id'

    PANDAS_TO_LIMONERO_DATA_TYPES = {
        'object': "CHARACTER",
        'datetime64[ns]': "DATETIME",
        'float64': "DOUBLE",
        'float32': "FLOAT",
        'int64': "LONG",
        'int32': "INTEGER",
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        for att in [self.NAME_PARAM, self.FORMAT_PARAM, self.PATH_PARAM]:
            if att not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        att, self.__class__))

        self.name = parameters.get(self.NAME_PARAM)
        self.tags = parameters.get(self.TAGS_PARAM)
        self.format = parameters.get(self.FORMAT_PARAM, self.FORMAT_CSV)
        self.path = parameters.get(self.PATH_PARAM, '.')
        self.mode = parameters.get(self.OVERWRITE_MODE_PARAM, self.MODE_ERROR)
        self.storage_id = parameters.get(self.STORAGE_ID_PARAM)
        self.user = parameters.get(self.USER_PARAM)
        self.workflow_id = parameters.get(self.WORKFLOW_ID_PARAM)

        self.header = parameters.get(self.HEADER_PARAM, False) in (1, '1', True)

        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(
                                                 self.order))

        self.filename = self.name
        self.has_code = len(self.named_inputs) == 1

    def generate_code(self):

        limonero_config = \
            self.parameters['configuration']['juicer']['services']['limonero']
        url = '{}'.format(limonero_config['url'], self.mode)
        token = str(limonero_config['auth_token'])
        storage = limonero_service.get_storage_info(url, token, self.storage_id)

        if storage['type'] != 'HDFS':
            raise ValueError(_('Storage type not supported: {}').format(
                storage['type']))

        if storage['url'].endswith('/'):
            storage['url'] = storage['url'][:-1]
        if self.path.endswith('/'):
            self.path = self.path[:-1]

        if self.path.startswith('/'):
            self.path = self.path[1:]

        final_url = '{}/limonero/user_data/{}/{}/{}'.format(
            storage['url'],
            self.user['id'],
            self.path,
            self.name.replace(' ', '_'))

        if self.format == self.FORMAT_CSV and not final_url.endswith('.csv'):
            final_url += '.csv'
        elif self.format == self.FORMAT_JSON and not final_url.endswith(
                '.json'):
            final_url += '.json'
        elif self.format == self.FORMAT_PARQUET and not final_url.endswith(
                '.parquet'):
            final_url += '.parquet'

        df_input = self.named_inputs['input data']
        code_template = """
            path = '{{path}}'
            {%- if scheme == 'hdfs' %}
            fs = pa.hdfs.connect('{{hdfs_server}}', {{hdfs_port}})
            exists = fs.exists(path)
            {%- elif scheme == 'file' %}
            exists = os.path.exists(path)
            {%- endif %}

            mode = '{{mode}}'
            if mode not in ('error', 'ignore', 'overwrite'):
                raise ValueError('{{error_invalid_mode}}')
            if exists:
                if mode == 'error':
                    raise ValueError('{{error_file_exists}}')
                elif mode == 'ignore':
                    emit_event(name='update task',
                        message='{{warn_ignored}}',
                        status='COMPLETED',
                        identifier='{{task_id}}')

            if not exists or mode == 'overwrite':
                {%- if scheme == 'hdfs'%}
                fs.delete(path, False)
                f = fs.open(path, 'wb')
                {%- elif scheme == 'file' %}
                if exists:
                    os.remove(path)
                parent_dir = os.path.dirname(path)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)
                f = open(path, 'wb')
                {%- endif %}
                {%- if format == FORMAT_CSV %}
                {{input}}.to_csv(f, sep=str(','), mode='w',
                    header={{header}}, index=False, encoding='utf-8')
                {%- elif format == FORMAT_PARQUET %}
                {{input}}.to_parquet(f, engine='pyarrow')
                {%- elif format == FORMAT_JSON %}
                {{input}}.to_json(f, orient='records')
                {%- endif %}
                f.close()


            # Code to update Limonero metadata information
            from juicer.service.limonero_service import register_datasource
            types_names = {{data_types}}

            attributes = []
            for attr in {{input}}.columns:
                type_name = {{input}}.dtypes[attr]
                precision = None
                scale = None
                attributes.append({
                  'enumeration': 0,
                  'feature': 0,
                  'label': 0,
                  'name': attr,
                  'type': types_names[str(type_name)],
                  'nullable': True,
                  'metadata': None,
                  'precision': precision,
                  'scale': scale
                })
            parameters = {
                'name': "{{name}}",
                'enabled': 1,
                'is_public': 0,
                'format': "{{format}}",
                'storage_id': {{storage}},
                'description': "{{description}}",
                'user_id': "{{user_id}}",
                'user_login': "{{user_login}}",
                'user_name': "{{user_name}}",
                'workflow_id': "{{workflow_id}}",
                'task_id': '{{task_id}}',
                'url': "{{final_url}}",
                'attributes': attributes,
                'tags': ','.join({{tags}})
            }
            register_datasource('{{url}}', parameters, '{{token}}', 'overwrite')
        """
        parsed = urlparse(final_url)
        template = Environment(loader=BaseLoader).from_string(
            code_template)
        path = parsed.path

        ctx = dict(path=path,
                   hdfs_server=parsed.hostname,
                   hdfs_port=parsed.port,
                   scheme=parsed.scheme,
                   name=self.name,
                   mode=self.mode,
                   storage=self.storage_id,
                   description=_('Data source generated by workflow {}').format(
                       self.workflow_id),
                   workflow_id=self.workflow_id,
                   format=self.format,
                   header=self.header,
                   user_name=self.user['name'],
                   user_id=self.user['id'],
                   user_login=self.user['login'],
                   tags=repr(self.tags),
                   FORMAT_CSV=self.FORMAT_CSV,
                   FORMAT_PICKLE=self.FORMAT_PICKLE,
                   FORMAT_JSON=self.FORMAT_JSON,
                   FORMAT_PARQUET=self.FORMAT_PARQUET,
                   data_types=json.dumps(self.PANDAS_TO_LIMONERO_DATA_TYPES),
                   final_url=final_url,
                   input=df_input,
                   token=token,
                   url=url,
                   error_file_exists=_('File already exists'),
                   warn_ignored=_('File not written (already exists)'),
                   error_invalid_mode=_('Invalid mode {}').format(self.mode),
                   uuid=uuid.uuid4().get_hex(),
                   storage_url=storage['url'],
                   task_id=self.parameters['task_id'],
                   )

        return dedent(template.render(ctx))
