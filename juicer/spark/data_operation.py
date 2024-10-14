import datetime
import json
import uuid
from gettext import gettext
from textwrap import dedent, indent
from urllib.parse import parse_qs, urlparse
from typing import List
from juicer import auditing
from juicer.deploy import Deployment, DeploymentFlow, DeploymentTask
from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.util.template_util import strip_accents
from juicer.util.variable import handle_variables

class DataReaderOperation(Operation):
    """
    Reads a database.
    Parameters:
    - Limonero database ID
    """
    DATA_SOURCE_ID_PARAM = 'data_source'
    HEADER_PARAM = 'header'
    SEPARATOR_PARAM = 'separator'
    QUOTE_PARAM = 'quote'
    INFER_SCHEMA_PARAM = 'infer_schema'
    NULL_VALUES_PARAM = 'null_values'
    MODE_PARAM = 'mode'

    INFER_FROM_LIMONERO = 'FROM_LIMONERO'
    INFER_FROM_DATA = 'FROM_VALUES'
    DO_NOT_INFER = 'NO'
    NO_IF_PARQUET = 'NO_IF_PARQUET'

    LIMONERO_TO_SPARK_DATA_TYPES = {
        "BINARY": 'types.BinaryType',
        "CHARACTER": 'types.StringType',
        "DATETIME": 'types.TimestampType',
        "DATE": 'types.DateType',
        "DOUBLE": 'types.DoubleType',
        "DECIMAL": 'types.DecimalType',
        "FLOAT": 'types.FloatType',
        "LONG": 'types.LongType',
        "INTEGER": 'types.IntegerType',
        "TEXT": 'types.StringType',
    }

    SUPPORTED_DRIVERS = {
        'mysql': 'com.mysql.jdbc.Driver'
    }
    DATA_TYPES_WITH_PRECISION = {'DECIMAL'}

    SEPARATORS = {
        '{tab}': '\\t',
        '{new_line \\n}': '\n',
        '{new_line \\r\\n}': '\r\n'
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code: bool = any(
            [len(self.named_outputs) > 0, self.contains_results()],)

        if self.has_code:
            if self.DATA_SOURCE_ID_PARAM in parameters:
                self._set_data_source_parameters(parameters)
            else:
                raise ValueError(
                    gettext("Parameter '{}' must be informed for task {}").format(
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
        self.connection_factory_function_name = parameters.get(
                'connection_factory_function_name')
        self.schema_var: str = 'schema'

    def _set_data_source_parameters(self, parameters):

        self.data_source_id = int(parameters[self.DATA_SOURCE_ID_PARAM])
        # Retrieve metadata from Limonero.
        limonero_config = \
            self.parameters['configuration']['juicer']['services'][
                'limonero']
        url = limonero_config['url']
        token = str(limonero_config['auth_token'])
        # Is data source information cached?
        self.metadata = self.parameters.get('workflow', {}).get(
            'data_source_cache', {}).get(self.data_source_id)
        if self.metadata is None:
            self.metadata = limonero_service.get_data_source_info(
                url, token, self.data_source_id)
            # Expand variables
            (self.metadata['url'], self.metadata['command']) = handle_variables(
                None,
                [self.metadata.get('url'), self.metadata.get('command')],
                parameters['workflow']['expanded_variables'],
                parse_date=False
            )
            self.parameters['workflow']['data_source_cache'][
                self.data_source_id] = self.metadata

        # URL can be empty for HIVE
        # if not self.metadata.get('url'):
        #    raise ValueError(
        #        gettext('Incorrect data source configuration (empty url)'))

        self.header: bool = parameters.get(
            self.HEADER_PARAM, False) not in ('0', 0, 'false', False)
        self.null_values = [v.strip() for v in parameters.get(
            self.NULL_VALUES_PARAM, '').split(",")]

        self.record_delimiter: str = self.metadata.get('record_delimiter')
        self.sep: str = parameters.get(
            self.SEPARATOR_PARAM,
            self.metadata.get('attribute_delimiter', ',')) or ','
        self.quote: str = parameters.get(self.QUOTE_PARAM,
                                    self.metadata.get('text_delimiter'))
        if self.quote == '\'':
            self.quote = '\\\''
        if self.sep in self.SEPARATORS:
            self.sep = self.SEPARATORS[self.sep]
        self.infer_schema = parameters.get(self.INFER_SCHEMA_PARAM,
                                           self.INFER_FROM_LIMONERO)
        self.mode: str = parameters.get(self.MODE_PARAM, 'FAILFAST')
        self.use_hive_warehouse_connector: bool = (
            self.metadata['storage']['type'] == 'HIVE_WAREHOUSE')
        self.use_hdfs: bool = (
            self.metadata['storage']['type'] == 'HDFS')
        self.use_s3: bool = (
            self.metadata['storage']['type'] == 'S3')

        self.extra_options = {}
        if self.use_hive_warehouse_connector:
            self.transpiler_utils.add_import(
                'from pyspark_llap import HiveWarehouseSession')
        elif self.use_hdfs:
            self.transpiler_utils.add_import('import os')
        elif self.use_s3:
            s3_url = self.metadata["storage"]["url"]
            extra_params = json.loads(self.metadata["storage"]["extra_params"])
            self.extra_options["fs.s3a.endpoint"] = (s3_url, True)
            self.extra_options["fs.s3a.access.key"] = (
                extra_params.get("access_key"),
                True,
            )
            self.extra_options["fs.s3a.secret.key"] = (
                extra_params.get("secret_key"),
                True,
            )
            self.extra_options["fs.s3a.path.style.access"] = ("true", False)
            self.extra_options["fs.s3a.impl"] = (
                "org.apache.hadoop.fs.s3a.S3AFileSystem",
                False,
            )

    def generate_code(self):

        code = []
        infer_from_data: bool = self.infer_schema == self.INFER_FROM_DATA
        infer_from_limonero: bool = (
            self.infer_schema == self.INFER_FROM_LIMONERO
        )
        read_options: List[str] = [
            f'"{k}": "{v}", { "#@HIDE_INFO@" if p else ""}'
            for k, (v, p) in self.extra_options.items()
        ]

        if self.has_code:
            is_hive: bool = (
                self.metadata['format'] not in ['HIVE', 'HIVE_WAREHOUSE']
            )
            date_format = "yyyy/MM/dd HH:mm:ss"
            if infer_from_limonero and is_hive:
                if 'attributes' in self.metadata:
                    code.append(f'{self.schema_var} = types.StructType()')
                    attrs = self.metadata.get('attributes', [])
                    if attrs:
                        for attr in attrs:
                            self._add_attribute_to_schema(attr, code)
                            if attr.get('type') == 'DATETIME':
                                date_format = attr.get('format') or \
                                    "yyyy/MM/dd HH:mm:ss"
                    else:
                        code.append(
                            f"{self.schema_var}.add('value', "
                            "types.StringType(), True, None)")

                    code.append("")
                else:
                    raise ValueError(gettext(
                        "Metadata do not include attributes information"))
            else:
                code.append(f'{self.schema_var} = None')

            url: str = self.metadata['url']
            if self.parameters.get('export_notebook', False) and False:
                # Protect URL
                url = 'hdfs://xxxxx:0000/path/name'
                code.append("# URL is protected, please update it")
            if self.metadata['format'] in ['CSV', 'TEXT']:

                if self.metadata['storage']['type'] != 'S3':
                    code.append(
                        f"url = '/'.join({repr(url.split('/'))}) #@HIDE_INFO@")
                else:
                    code.append(f"url = 's3a://{url}' #@HIDE_INFO@")

                code.append("read_options = { # These options are from storage")
                for opt in read_options:
                    code.append(f"{(4*' ')}{opt}")
                code.append("}")
                if self.metadata['storage'].get('extra_params'):
                    extra_params = json.loads(
                            self.metadata['storage']['extra_params'])
                    if 'user' in extra_params:
                        user = extra_params.get('user')
                        if user:
                            code.append('jvm = spark_session._jvm')
                            code.append(
                                'jvm.java.lang.System.setProperty('
                                f'"HADOOP_USER_NAME", "{user}")')
                            code.append(
                                f"os.environ['HADOOP_USER_NAME'] = '{user}'")

                # Multiple values not supported yet! See SPARK-17878, SPARK-31747
                null_values = self.null_values
                if self.metadata.get('treat_as_missing'):
                    null_values.extend([x.strip() for x in self.metadata.get(
                        'treat_as_missing').split(',')])
                null_option = ''.join(
                    [f".option('nullValue', '{n}')" for n in
                     set(null_values)]) if null_values else ""

                if self.metadata['format'] == 'CSV':
                    # Spark does not works with encoding + multiline options
                    # See https://github.com/databricks/spark-csv/issues/448
                    # And there is no way to specify record delimiter!!!!!
                    encoding: str = (
                        self.metadata.get('encoding', 'UTF-8') or 'UTF-8'
                    )
                    header: bool = self.header or self.metadata.get(
                            'is_first_line_header', False)
                    quote: str = (
                        "None" if self.quote is None else f"'{self.quote}'"
                    )
                    multiline:bool = encoding in ('UTF-8', 'UTF8', '')
                    code_csv = dedent(f"""
                        {self.output} = (spark_session.read{null_option}
                            .option('treatEmptyValuesAsNulls', 'true')
                            .option('wholeFile', True)
                            .option('multiLine', {multiline})
                            .option('escape','"')
                            .option('timestampFormat', {repr(date_format)})
                            .options(**read_options)
                            .csv(
                                url,
                                schema={self.schema_var},
                                quote={quote},
                                ignoreTrailingWhiteSpace=True, # Handles \r
                                encoding='{encoding}',
                                header={header},
                                sep='{self.sep}',
                                inferSchema={infer_from_data},
                                mode='{self.mode}')
                        )""")
                    code.append(code_csv)
                else:
                    code_csv = dedent(f"""
                    {self.schema_var} = types.StructType()
                    {self.schema_var}.add('value', types.StringType(), True)
                    {self.output} = (spark_session.read{null_option}
                        .schema({self.schema_var})
                        .option('treatEmptyValuesAsNulls', 'true')
                        .text(url)
                    )""")
                    code.append(code_csv)
            elif self.metadata['format'] == 'PARQUET':
                if self.infer_schema == 'NO_IF_PARQUET':
                    infer_from_data = True
                    infer_from_limonero = False
                self._generate_code_for_parquet(code, infer_from_data,
                                                infer_from_limonero,
                                                read_options)
            elif self.metadata['format'] == 'HIVE':
                # import pdb; pdb.set_trace()
                # parsed = urlparse(self.metadata['url'])
                code_hive = self._get_hwc_code()
                code.append(code_hive)
            elif self.metadata['format'] == 'JSON':
                code_json = dedent(f"""
                    {self.schema_var} = types.StructType()
                    {self.schema_var}.add('value', types.StringType(), True)
                    {self.output} = spark_session.read.option(
                        'treatEmptyValuesAsNulls', 'true').json(
                        '{url}')""")
                code.append(code_json)
                # FIXME: Evaluate if it is good idea to always use cache
                code.append('{}.cache()'.format(self.output))

            elif self.metadata['format'] == 'LIB_SVM':
                self._generate_code_for_lib_svm(code, infer_from_data)
            elif self.metadata['format'] == 'JDBC':
                self._generate_code_for_jdbc(code)

        code.append('{}.cache()'.format(self.output))
        return '\n'.join(code)

    def _get_hwc_code(self):
        """
        Generates code to connect to Hive Warehouse Connector
        """
        if self.use_hive_warehouse_connector:
            sql: str = self.metadata.get('command').replace('\n', ' ')
            # A factory function name is set when using Meta Plataform
            if self.connection_factory_function_name:
                code_hive = dedent(f"""
                    hive = {self.connection_factory_function_name}(spark_session)
                    {self.output} = hive.executeQuery('''{sql}''')
                """)
            else:
                missing_config_msg = gettext(
                            'Cluster is not configured for Hive Warehouse')
                code_hive = dedent(f"""
                    if spark_session.conf.get(
                            'spark.sql.hive.hiveserver2.jdbc.url') is None:
                        raise ValueError('{missing_config_msg}')
                    hive = HiveWarehouseSession.session(spark_session).build()
                    {self.output} = hive.executeQuery('''{sql}''')
                    """)
        else:
            # Notifies the transpiler that Hive is required.
            # In order to support Hive, SparkSession must be
            # correctly configured.
            self.parameters['transpiler'].on('requires-hive', self.metadata)
            code_hive = f"{self.output} = spark_session.sql('''{sql}''')"

        return code_hive

    def _generate_code_for_jdbc(self, code):

        parsed = urlparse(self.metadata['url'])
        # qs_parsed = parse_qs(parsed.query)
        driver = self.SUPPORTED_DRIVERS.get(parsed.scheme)
        if driver is None:
            raise ValueError(
                gettext('Database {} not supported').format(parsed.scheme))
        if not self.metadata.get('command'):
            raise ValueError(
                gettext('No command nor table specified for data source.'))
        final_url: str = (
            f'{parsed.scheme}://{parsed.server}:{parsed.port}{parsed.db}'
        )
        code_jdbc: str = dedent(f"""
            query = '''{self.metadata.get('command').strip()}'''
            if query[:6].upper() == 'SELECT':
                # Add parentheses required by Spark
                query = '(' + query + ') AS tb'
            {self.output} = (spark_session.read
                .format('jdbc')
                .load(driver='{driver}',
                      url='jdbc:{final_url}',
                      user='{parsed.username}',
                      password='{parsed.password}',
                      dbtable=query)
            )""")
        code.append(code_jdbc)

    def _generate_code_for_parquet(self, code, infer_from_data,
                                   infer_from_limonero,
                                   read_options):
        code.append(f"url = 's3a://{self.metadata['url']}' #@HIDE_INFO@")
        code.append("read_options = { # These options are from storage")
        for opt in read_options:
            code.append(f"{(4*' ')}{opt}")
        code.append("}")

        if infer_from_limonero:
            code_parquet = f"""
                {self.output} = (spark_session.read
                    .options(**read_options)
                    .format('parquet')
                    .schema({self.schema_var})
                    .load(url))
                # Drop index columns
                #{self.output} = {self.output}.drop('__index_level_0__')
            """
        else:
            code_parquet = f"""
                {self.output} = (spark_session.read
                    .options(**read_options)
                    .format('parquet').load(url))
                # Drop index columns
                {self.output} = {self.output}.drop('__index_level_0__')
            """
        code.append(dedent(code_parquet))

    def _generate_code_for_lib_svm(self, code, infer_from_data):
        """"""
        # # FIXME: Evaluate if it is good idea to always use cache
        code.append(
            "url_{0} = '{1}'".format(self.output, self.metadata['url']))
        code_csv = """
            {0} = spark_session.read.format('libsvm').load(
                url_{0}, mode='DROPMALFORMED')""".format(
            self.output)
        code.append(dedent(code_csv))

    def _add_attribute_to_schema(self, attr, code):
        data_type = self.LIMONERO_TO_SPARK_DATA_TYPES[attr['type']]
        if attr['type'] in self.DATA_TYPES_WITH_PRECISION:
            data_type = '{}({}, {})'.format(
                data_type,
                (attr.get('precision', 0) or 0) + 3,  # extra precision
                (attr.get('scale', 0) or 0) or 0)
        else:
            data_type = '{}()'.format(data_type)

        # Notice: According to Spark documentation, nullable
        # option of StructField is just a hint and when
        # loading CSV file, it won't work. So, we are adding
        # this information in metadata.
        # Not used.
        # metadata = {k: attr[k] for k in
        #             ['nullable', 'type', 'size', 'precision', 'enumeration',
        #              'missing_representation'] if attr[k]}
        code.append(
            f"{self.schema_var}.add('{attr['name']}', {data_type}, {attr.get('nullable', True)})")

    def get_output_names(self, sep=", "):
        return self.output

    def get_data_out_names(self, sep=','):
        return self.output

    def to_deploy_format(self, id_mapping):
        params = self.parameters['task']['forms']
        result = Deployment()

        forms = [(k, v['category'], v['value']) for k, v in list(params.items()) if v]

        task = self.parameters['task']
        task_id = task['id']

        deploy = DeploymentTask(task_id) \
            .set_operation(slug="external-input") \
            .set_properties(forms) \
            .set_pos(task['top'], task['left'], task['z_index'])

        result.add_task(deploy)
        id_mapping[task_id] = deploy.id
        for flow in self.parameters['workflow']['flows']:
            if flow['source_id'] == task_id:
                flow['source_port'] = 129  # FIXME how to avoid hard coding ?
                flow['source_id'] = deploy.id
                result.add_flow(DeploymentFlow(**flow))
        return result

    @property
    def is_data_source(self):
        return True

class DataSourceOperation(DataReaderOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        DataReaderOperation.__init__(self, parameters, named_inputs, named_outputs)

class SaveOperation(Operation):
    """
    Saves the content of the DataFrame at the specified path
    and generate the code to call the Limonero API.
    Parameters:
        - Database name
        - Path for storage
        - Storage ID
        - Database tags
        - Workflow that generated the database
    """
    SPARK_TO_LIMONERO_DATA_TYPES = {
        'types.StringType': "CHARACTER",
        'types.TimestampType': "DATETIME",
        'types.DateType': "DATETIME",
        'types.DoubleType': "DOUBLE",
        'types.DecimalType': "DECIMAL",
        'types.FloatType': "FLOAT",
        'types.LongType': "LONG",
        'types.IntegerType': "INTEGER",
        'types.BinaryType': "BINARY",

        'StringType': "CHARACTER",
        'TimestampType': "DATETIME",
        'DateType': "DATETIME",
        'DoubleType': "DOUBLE",
        'DecimalType': "DECIMAL",
        'FloatType': "FLOAT",
        'LongType': "LONG",
        'IntegerType': "INTEGER",
        'BinaryType': "BINARY",
    }

    NAME_PARAM = 'name'
    PATH_PARAM = 'path'
    STORAGE_ID_PARAM = 'storage'
    FORMAT_PARAM = 'format'
    TAGS_PARAM = 'tags'
    MODE_PARAM = 'mode'
    HEADER_PARAM = 'header'

    MODE_ERROR = 'error'
    MODE_APPEND = 'append'
    MODE_OVERWRITE = 'overwrite'
    MODE_IGNORE = 'ignore'

    FORMAT_PARQUET = 'PARQUET'
    FORMAT_CSV = 'CSV'
    FORMAT_JSON = 'JSON'
    WORKFLOW_JSON_PARAM = 'workflow_json'
    USER_PARAM = 'user'
    WORKFLOW_ID_PARAM = 'workflow_id'
    WORKFLOW_VERSION_PARAM = 'workflow_version'
    SAVE_TO_LIMONERO = 'save_to_limonero'
    USE_STORAGE_PATH = 'use_storage_path' # Do not add limonero extra path

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = parameters.get(self.NAME_PARAM)
        if self.name is None or not self.name.strip():
            raise ValueError(gettext(
                'You must specify a name for new data source.'))

        self.format = parameters.get(self.FORMAT_PARAM, '') or ''
        valid_formats = (self.FORMAT_PARQUET, self.FORMAT_CSV, self.FORMAT_JSON)
        if not self.format.strip() or self.format not in valid_formats:
            raise ValueError(gettext('You must specify a valid format.'))

        self.url = parameters.get(self.PATH_PARAM)
        self.storage_id = parameters.get(self.STORAGE_ID_PARAM)
        if not self.storage_id:
            raise ValueError(gettext('You must specify a storage for saving data.'))

        self.tags = parameters.get(self.TAGS_PARAM, [])
        self.path = parameters.get(self.PATH_PARAM)
        #if self.path is None or not self.path.strip():
        #    raise ValueError(gettext('You must specify a path for saving data.'))

        self.workflow_json = parameters.get(self.WORKFLOW_JSON_PARAM, '')

        self.mode = parameters.get(self.MODE_PARAM, self.MODE_ERROR)
        self.header = parameters.get(self.HEADER_PARAM, True) in (1, '1', True)

        self.user = parameters.get(self.USER_PARAM)
        self.workflow_id = parameters.get(self.WORKFLOW_ID_PARAM)
        self.workflow_version = parameters.get(self.WORKFLOW_VERSION_PARAM)
        self.has_code = len(self.named_inputs) == 1
        self.supports_cache = False
        self.save_to_limonero = parameters.get(self.SAVE_TO_LIMONERO, True) in (
            1, '1', True)
        self.use_storage_path = parameters.get(self.USE_STORAGE_PATH, False) in (
            1, '1', True)

        self.transpiler_utils.add_import(
            'from juicer.service.limonero_service import register_datasource')

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return self.output

    def get_audit_events(self):
        return [auditing.SAVE_DATA]

    def generate_code(self):
        # Retrieve Storage URL

        limonero_config = \
            self.parameters['configuration']['juicer']['services']['limonero']
        url =limonero_config['url']
        token = str(limonero_config['auth_token'])
        storage = limonero_service.get_storage_info(url, token, self.storage_id)

        if self.use_storage_path:
            storage_url = storage['url'].strip('/')
            path = self.path.strip('/')
            name = strip_accents(self.name.replace(' ', '_')).strip('/')
            final_url = f'{storage_url}/{path}/{name}'
        else:
            final_url = '{}/limonero/user_data/{}/{}/{}'.format(
                storage['url'].strip('/'), self.user['id'],
                self.path.strip('/'),
                strip_accents(self.name.replace(' ', '_')).strip('/'))

        hdfs_user = 'hadoop'
        if storage.get('extra_params'):
            extra_params = json.loads(storage['extra_params'])
            if 'user' in extra_params:
                hdfs_user = extra_params.get('user')

        register_in_limonero = True
        code_save = ''
        if storage['type'] == 'HIVE_WAREHOUSE':
            parts = self.name.split('.')
            if len(parts) > 1:
                database_name, table_name = parts[0:2]
            else:
                database_name = '' # Uses current database
                table_name = parts[0]
            self.format = 'HIVE'

            template_str = dedent("""
                # Write results
                from pyspark_llap import HiveWarehouseSession
                if spark_session.conf.get(
                    'spark.sql.hive.hiveserver2.jdbc.url') is None:
                    raise ValueError('{{ missing_config }}')
                hive = HiveWarehouseSession.session(spark_session).build();
                {%- if database_name %}
                hive.setDatabase('{{database_name}}');
                {%- endif %}
                {{ input_data }}.write.mode('{{ mode }}').format(
                    HiveWarehouseSession.HIVE_WAREHOUSE_CONNECTOR).option(
                        'table', '{{table_name}}').save();
                """)

            code_save = self.render_str_template(template_str, dict(
                database_name=database_name,
                table_name=table_name,
                input_data=self.named_inputs['input data'],
                mode=self.mode,
                missing_config=gettext(
                    'Cluster is not configured for Hive Warehouse')
            ))
            register_in_limonero = False

        elif storage['type'] == 'KAFKA':
            topic = self.path
            self.format = 'KAFKA'
            parsed = urlparse(storage['url'])

            code_save = dedent("""
                 {input}.select(functions.to_json(functions.struct("*")).alias(
                     "value")).selectExpr(
                     "CAST(value AS STRING)").write.format("kafka").option(
                     'kafka.bootstrap.servers', '{kafka_server}').option(
                     'topic', '{topic}').save()
            """).format(topic=topic,
                        kafka_server=f'{parsed.hostname}:{parsed.port}',
                        input=self.named_inputs['input data'])
            # A data source will not be registered in Limonero
            register_in_limonero = False

        elif self.format == self.FORMAT_CSV:
            code_save = dedent("""
            cols = []
            for attr in {input}.schema:
                if attr.dataType.typeName() in ['array']:
                    cols.append(functions.concat_ws(
                        ', ', {input}[attr.name]).alias(attr.name))
                else:
                    cols.append({input}[attr.name])

            {input} = {input}.select(*cols)
            mode = '{mode}'

            # Write in a temporary directory
            # Header configuration will be handled by LemonadeFileUtil class
            {input}.write.csv('{url}{uuid}',
                         header=False, mode=mode)
            # Merge files using Hadoop HDFS API
            conf = spark_session._jsc.hadoopConfiguration()
            jvm = spark_session._jvm
            jvm.java.lang.System.setProperty("HADOOP_USER_NAME", "{hdfs_user}")

            fs = jvm.org.apache.hadoop.fs.FileSystem.get(
                jvm.java.net.URI('{storage_url}'), conf)

            path = jvm.org.apache.hadoop.fs.Path('{url}')
            tmp_path = jvm.org.apache.hadoop.fs.Path(
                '{url}{uuid}')
            write_header = {header}
            # org.apache.hadoop.fs.FileUtil do not handle files with header
            header = None
            if write_header:
                header = ','.join([attr.name for attr in {input}.schema])

            fs_util = jvm.br.ufmg.dcc.lemonade.ext.io.LemonadeFileUtil
            if fs.exists(path):
                if mode == 'error':
                    raise ValueError('{error_file_exists}')
                elif mode == 'ignore':
                    emit_event(name='update task',
                        message='{warn_ignored}',
                        status='COMPLETED',
                        identifier='{task_id}')
                elif mode == 'overwrite':
                    fs.delete(path, False)
                    fs_util.copyMergeWithHeader(fs, tmp_path, fs, path, True,
                        conf, header)
                elif mode != 'append':
                    raise ValueError('{error_invalid_mode}')
            else:
                fs_util.copyMergeWithHeader(fs, tmp_path, fs, path, True, conf,
                    header)
            """.format(
                input=self.named_inputs['input data'],
                url=final_url, header=self.header, mode=self.mode,
                uuid=uuid.uuid4().hex,
                hdfs_user=hdfs_user,
                storage_url=storage['url'],
                task_id=self.parameters['task_id'],
                error_file_exists=gettext('File already exists'),
                warn_ignored=gettext('File not written (already exists)'),
                error_invalid_mode=gettext('Invalid mode {}').format(self.mode)
            ))
            # Need to generate an output, even though it is not used.
        elif self.format == self.FORMAT_PARQUET:
            code_save = dedent("""
            {}.write.parquet('{}', mode='{}')""".format(
                self.named_inputs['input data'],
                final_url, self.mode))
            # Need to generate an output, even though it is not used.
            code_save += '\n{0}_tmp = {0}'.format(
                self.named_inputs['input data'])
            code_save += '\nwrite_header = False'
        elif self.format == self.FORMAT_JSON:
            code_save = dedent("""
            {}.write.json('{}', mode='{}')""".format(
                self.named_inputs['input data'],
                final_url, self.mode))

        code = dedent(code_save)
        if register_in_limonero and self.save_to_limonero:
            code_api = dedent("""
            # Code to update Limonero metadata information
            types_names = {data_types}

            # nullable information is also stored in metadata
            # because Spark ignores this information when loading CSV files
            attributes = []

            for att in {input}.schema:
                type_name = str(att.dataType)
                if type_name.endswith('()'):
                    type_name = type_name[:-2]
                precision = None
                scale = None
                found = isinstance(att.dataType, types.DecimalType)
                if found:
                    type_name = 'DecimalType'
                    precision = att.dataType.precision
                    scale = att.dataType.scale
                attributes.append({{
                  'enumeration': 0,
                  'feature': 0,
                  'label': 0,
                  'name': att.name,
                  'type': types_names[str(type_name)],
                  'nullable': att.nullable,
                  'metadata': att.metadata,
                  'precision': precision,
                  'scale': scale
                }})
            parameters = {{
                'attribute_delimiter': ',',
                'is_first_line_header': write_header,
                'name': "{name}",
                'enabled': 1,
                'is_public': 0,
                'format': "{format}",
                'storage_id': {storage},
                'description': "{description}",
                'user_id': "{user_id}",
                'user_login': "{user_login}",
                'user_name': "{user_name}",
                'workflow_id': "{workflow_id}",
                'workflow_verson': "{workflow_version}",
                'tags': '{tags}',
                'url': "{final_url}",
                'attributes': attributes
            }}
            register_datasource('{url}', parameters, '{token}', 'overwrite')
            """).format(
                input=self.named_inputs['input data'],
                name=self.name,
                format=self.format,
                storage=self.storage_id,
                description=gettext('Data source generated by workflow {}').format(
                    self.workflow_id),
                user_name=self.user['name'],
                user_id=self.user['id'],
                user_login=self.user['login'],
                workflow_id=self.workflow_id,
                workflow_version=self.workflow_version,
                final_url=final_url,
                token=token,
                url=url,
                tags=', '.join(self.tags or []),
                mode=self.mode,
                data_types=json.dumps(self.SPARK_TO_LIMONERO_DATA_TYPES, indent=2))

            code += dedent(code_api)
        elif self.save_to_limonero:
            task_id=self.parameters['task_id']
            warn_ignored=gettext('Data was written, but was '
                'not registered as a data source in Lemonade (unsupported).')
            code += dedent(f"""
                emit_event(name='update task',
                        message='{warn_ignored}',
                        status='COMPLETED',
                        identifier='{task_id}')\n""")



        # No return
        if self.save_to_limonero:
            code += '{} = None'.format(self.output)

        # return dedent(self.render_template(ctx))
        return code


class ReadCSVOperation(Operation):
    """
    Reads a CSV file without HDFS.
    The purpose of this operation is to read files in
    HDFS without using the Limonero API.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.url = parameters['url']
        try:
            self.header = parameters['header']
        except KeyError:
            self.header = "True"
        try:
            self.separator = parameters['separator']
        except KeyError:
            self.separator = ";"

    def generate_code(self):
        code = """{} = spark_session.read.csv('{}',
            header={}, sep='{}',inferSchema=True)""".format(
            self.named_outputs['output data'], self.url, self.header,
            self.separator)
        return dedent(code)


class ChangeAttributeOperation(Operation):
    ATTRIBUTES_PARAM = 'attributes'
    IS_FEATURE_PARAM = 'is_feature'
    IS_LABEL_PARAM = 'is_label'
    NULLABLE_PARAM = 'nullable'
    NEW_NAME_PARAM = 'new_name'
    NEW_DATA_TYPE_PARAM = 'new_data_type'
    KEEP_VALUE = 'keep'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(gettext(
                "Parameter '{}' must be informed for task {}").format(
                self.ATTRIBUTES_PARAM, self.__class__))
        self.has_code = len(self.named_inputs) == 1

    def generate_code(self):
        del self.parameters['workflow_json']
        code = []
        if self.parameters.get(self.NEW_DATA_TYPE_PARAM,
                               'keep') == self.KEEP_VALUE:
            # Do not require processing data frame, change only meta data
            code.append('{0} = {1}'.format(self.output,
                                           self.named_inputs['input data']))

            for attr in self.attributes:
                code.append(
                    "\ninx_{0} = [i for i, _ in enumerate({0}.schema) "
                    "if _.name.lower() == '{1}']".format(self.output,
                                                         attr.lower()))

                nullable = self.parameters.get(self.NULLABLE_PARAM,
                                               self.KEEP_VALUE)
                if nullable != self.KEEP_VALUE:
                    code.append(
                        ChangeAttributeOperation.change_meta(
                            self.output, attr, 'nullable', nullable == 'true'))

                feature = self.parameters.get(self.IS_FEATURE_PARAM,
                                              self.KEEP_VALUE)
                if feature != self.KEEP_VALUE:
                    code.append(
                        ChangeAttributeOperation.change_meta(
                            self.output, attr, 'feature', feature == 'true'))

                label = self.parameters.get(self.IS_LABEL_PARAM,
                                            self.KEEP_VALUE)
                if label != self.KEEP_VALUE:
                    code.append(
                        ChangeAttributeOperation.change_meta(
                            self.output, attr, 'label', label == 'true'))

            format_name = self.parameters[self.NEW_NAME_PARAM]
            if format_name:
                rename = [
                    "withColumnRenamed('{}', '{}')".format(
                        attr,
                        ChangeAttributeOperation.new_name(format_name, attr))
                    for
                    attr in self.attributes]

                code.append('{0} = {0}.{1}'.format(
                    self.output, '\\\n              .'.join(rename)))

        else:
            # Changing data type requires to rebuild data frame
            pass
        return '\n'.join(code)  # json.dumps(self.parameters)

    @staticmethod
    def new_name(format_new_name, name):
        return format_new_name.format(name)

    @staticmethod
    def change_meta(output, attr_name, meta_name, value):
        return dedent("if inx_{output}:\n"
                      "    {output}.schema.fields[inx_{ouput}[0]]"
                      ".metadata['{meta_name}'] = {value}")


class ExternalInputOperation(Operation):
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.output) > 0

    def generate_code(self):
        code = """{out} = None""".format(out=self.output)
        return code


class StreamConsumerOperation(DataReaderOperation):
    SOURCE_TYPE_PARAM = 'source_type'
    WINDOW_TYPE_PARAM = 'window_type'
    WINDOW_SIZE_PARAM = 'window_size'
    BROKER_URL_PARAM = 'broker_url'
    TOPIC_PARAM = 'topic'
    GROUP_PARAM = 'group'

    SOURCE_TYPES = ['kafka', 'hdfs']
    WINDOW_TYPES = ['seconds', 'size']

    def __init__(self, parameters, named_inputs, named_outputs):
        DataReaderOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.source_type = parameters.get(self.SOURCE_TYPE_PARAM, 'hdfs')
        if self.source_type not in self.SOURCE_TYPES:
            raise ValueError(gettext("Invalid value '{}' for parameter '{}'".format(
                self.source_type, self.SOURCE_TYPE_PARAM)))

        self.window_type = parameters.get(self.WINDOW_TYPE_PARAM, 'seconds')
        if self.window_type not in self.WINDOW_TYPES:
            raise ValueError(gettext("Invalid value '{}' for parameter '{}'".format(
                self.window_type, self.WINDOW_TYPE_PARAM)))

        self.broker_url = parameters.get(self.BROKER_URL_PARAM)
        if not self.broker_url:
            raise ValueError(
                gettext("Parameter '{}' must be informed for task {}").format(
                    self.BROKER_URL_PARAM, self.__class__))

        self.window_size = parameters.get(self.WINDOW_SIZE_PARAM, 5)
        self.topic = parameters.get(self.TOPIC_PARAM)
        self.group = parameters.get(self.GROUP_PARAM)

        self.stream_context_ref = parameters.get('stream_context')
        self.has_code = len(self.output) > 0

    def generate_code(self):

        code = []
        if 'attributes' in self.metadata:
            code.append(f'{self.schema_var} = types.StructType()')
            attrs = self.metadata.get('attributes')
            for attr in attrs:
                self._add_attribute_to_schema(attr, code)
            else:
                code.append(
                    f"{self.schema_var}.add('value', types.StringType(), True, None)")
            code.append("")
        else:
            raise ValueError(
                gettext("Metadata do not include attributes information"))

        code.append(dedent("""
            from pyspark.streaming.kafka import KafkaUtils

            source_type = '{source_type}'
            window_type = '{window_type}'
            scc = {stream_ctx}

            if source_type == 'kafka':
                kvs = KafkaUtils.createStream(
                    ssc, broker, '{group}', {{'{topic}': 1}})
            lines = kvs.map(lambda x: x[1])

            counts = lines.flatMap(
                lambda line: [w.lower() for w in line.split() if w.strip()]).map(
                lambda word: (word, 1)).reduceByKey(
                lambda a, b: a + b).transform(
                lambda rdd: rdd.sortBy(lambda x: x[1], False))
        """.format(source_type=self.source_type, window_type=self.window_type,
                   stream_ctx=self.stream_context_ref, topic=self.topic,
                   group=self.group)))
        return code

    @property
    def is_stream_consumer(self):
        return True
