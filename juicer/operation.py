# -*- coding: utf-8 -*-


import logging
from collections import namedtuple
from gettext import gettext

from juicer.transpiler import TranspilerUtils

try:
    from itertools import zip_longest as zip_longest
except ImportError:
    from itertools import zip_longest as zip_longest

from juicer.deploy import Deployment, DeploymentFlow
from juicer.deploy import DeploymentTask
from juicer.runner import configuration

log = logging.getLogger()
log.setLevel(logging.DEBUG)

TraceabilityData = namedtuple(
    'TraceabilityData',
    'input, attribute, derived_from, was_value')


# noinspection PyClassHasNoInit
class ResultType:
    VISUALIZATION = 'VISUALIZATION'
    MODEL = 'MODEL'

class SampleConfiguration(object):
    """ Allow to set configuration options for operation sampling. """
    __slots__ = ('size', 'infer', 'use_types', 'describe', 'page')
    def __init__(self, size=50, infer=False, describe=False, use_types=None, page=1):
        if use_types is None:
            self.use_types = []
        self.size = size
        self.infer = infer
        self.describe = describe
        self.page = page

    def get_config(self):
        return repr([self.size, self.infer, self.describe, self.use_types])

class Operation(object):
    """ Defines an operation in Lemonade """
    __slots__ = ('parameters', 'named_inputs', 'output',
                 'named_outputs', 'multiple_inputs', 'has_code',
                 'expected_output_ports', 'out_degree', 'order',
                 'supports_cache', 'config', 'deployable', 'plain',
                 'transpiler_utils', 'sample_configuration', 'template')

    def __init__(self, parameters, named_inputs, named_outputs):
        self.parameters = parameters
        self.named_inputs = named_inputs
        self.named_outputs = named_outputs
        self.multiple_inputs = False
        self.out_degree = 0
        self.plain = False
        self.transpiler_utils: TranspilerUtils = (
            parameters.get('transpiler_utils', TranspilerUtils()))

        self.config = configuration.get_config()
        # Assume default as 1, useful for testing.
        self.order = parameters.get('order', 1)

        # Should data be cached between job executions?
        # Exception to this rule includes visualization operations.
        self.supports_cache = True

        # Indicate if operation generates code or not. Some operations, e.g.
        self.has_code = len(self.named_inputs) > 0 or len(
            self.named_outputs) > 0

        self.deployable = False
        # How many output ports the operation has
        self.expected_output_ports = 1

        # self.output = 'out_task_{order}'.format(order=parameters['order'])
        # # @!CHECK-ME inspect this part of code.
        # if len(self.named_inputs) > 0:
        #     outputs = self.named_outputs.keys()
        #     self.output = outputs[0] if len(
        #         # self.outputs) > 0 else '{}_tmp_{}'.format(
        #         # self.inputs[0], parameters['task']['order'])
        #         # Used for tests, not correct.
        #         self.outputs) > 0 else '{}_tmp_{}_{}'.format(
        #         self.inputs[0], self.inputs[0],
        #         self.parameters.get('task', {}).get('order', ''))
        #     # Some cases this string to _tmp_ doesn't work in the spark code generation
        #     #  parameters['task']['order']
        # elif len(self.outputs) > 0:
        #     self.output = self.outputs[0]
        # else:
        #     self.output = "NO_OUTPUT_WITHOUT_CONNECTIONS"

        # Subclasses should override this
        self.output = self.named_outputs.get(
            'output data', 'out_task_{}'.format(self.order))

        self.sample_configuration = SampleConfiguration(
            infer=parameters.get('infer_sample') in [1, '1', 'true', True],
            size=int(parameters.get('sample_size', 50)),
            page=int(parameters.get('sample_page', 1)),
            describe=parameters.get('describe_sample') in [1, '1', 'true', True],
            use_types=parameters.get('use_types_in_sample'))

    def generate_code(self):
        raise NotImplementedError(
            _("Method generate_code should be implemented "
              "in {} subclass").format(self.__class__))

    def render_template(self, context: dict):
        return self.transpiler_utils.render_template(self.template, context)

    # noinspection PyMethodMayBeStatic
    def get_auxiliary_code(self):
        return []

    def get_generated_results(self):
        """
         Returns results generated by a task executing an operation.
         Results can be models and visualizations (for while).
        """
        return []

    def get_port_multiplicity(self, port):
        return self.parameters.get('multiplicity',{}).get('input data', 1)

    @property
    def enabled(self):
        return self.parameters.get('task', {}).get('enabled', True)

    @property
    def is_data_source(self):
        """ Operation is a data source and must be audited? """
        return False

    @property
    def is_stream_consumer(self):
        return False

    @property
    def supports_pipeline(self):
        return False

    @property
    def get_inputs_names(self):
        return ', '.join(list(self.named_inputs.values()))

    def get_audit_events(self):
        return []

    def set_plain(self, value):
        self.plain = value

    def get_output_names(self, sep=", "):
        if self.output:
            return self.output
        else:
            return sep.join(list(self.named_outputs.values()))

    def get_data_out_names(self, sep=','):
        return self.get_output_names(sep)

    @property
    def contains_sample(self):
        forms = self.parameters.get('task', {}).get('forms', {})
        return forms.get('display_sample', {}).get('value') in (1, '1')

    @property
    def contains_schema(self):
        forms = self.parameters.get('task', {}).get('forms', {})
        return forms.get('display_schema', {}).get('value') in (1, '1')

    def contains_results(self):
        return self.contains_sample or self.contains_schema

    def must_be_executed(self, is_satisfied, ignore_out_degree=False,
                         ignore_has_code=False):
        consider_degree = self.out_degree == 0 or ignore_out_degree
        info_or_data = self.contains_results()
        return (((self.has_code or ignore_has_code) and is_satisfied and
                 consider_degree) or info_or_data)

    # noinspection PyMethodMayBeStatic
    def attribute_traceability(self):
        """
        Handle attribute traceability. This is the default implementation.
        """
        return []

    def render_template(self, context: dict):
        return self.transpiler_utils.render_template(self.template, context)

    def to_deploy_format(self, id_mapping):
        params = self.parameters['task']['forms']
        result = Deployment()

        forms = [(k, v['category'], v['value']) for k, v in list(params.items())
                 if v]
        task = self.parameters['task']
        task_id = task['id']

        deploy = DeploymentTask(task_id) \
            .set_operation(slug=task['operation']['slug']) \
            .set_properties(forms) \
            .set_pos(task['top'], task['left'], task['z_index'])
        result.add_task(deploy)

        id_mapping[task_id] = deploy.id
        for flow in self.parameters['workflow']['flows']:
            if flow['source_id'] == task_id:
                flow['source_id'] = deploy.id
                result.add_flow(DeploymentFlow(**flow))
            elif flow['target_id'] == task_id:
                flow['target_id'] = deploy.id
                result.add_flow(DeploymentFlow(**flow))

        # All leaf output port with interface Data defined is considered
        # an output
        candidates = [p for p in
                      list(self.parameters['task']['operation'][
                               'ports'].values())
                      if 'Data' in p['interfaces'] and p[
                          'slug'] not in self.named_outputs and p[
                          'type'] == 'OUTPUT']
        # for p in self.parameters['task']['operation']['ports'].values():
        #     import sys
        #     print >> sys.stderr, self.parameters['task']['operation']['slug'],
        #     print >> sys.stderr, 'Data' in p['interfaces'],
        #     print >> sys.stderr, p['slug'] not in self.named_outputs,
        #     print >> sys.stderr, p['type']
        for i, candidate in enumerate(candidates):
            # FIXME Evaluate form
            service_out = DeploymentTask(task_id) \
                .set_operation(slug="service-output") \
                .set_properties(forms) \
                .set_pos(task['top'] + 140 * i + 140, task['left'],
                         task['z_index'])
            result.add_task(service_out)
            result.add_flow(DeploymentFlow(
                deploy.id, candidate['id'], candidate['slug'],
                service_out.id, 40,
                'input data'))

        return result

    def get_required_parameter(self, parameters, name):
        if name not in parameters:
            raise ValueError(
                gettext('Missing required parameter: {}').format(
                    name))
        else:
            return parameters.get(name)

# noinspection PyAbstractClass
class ReportOperation(Operation):
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)


# noinspection PyAbstractClass
class NoOp(Operation):
    """ Null operation """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = False


# noinspection PyAbstractClass
class TransformModelOperation(Operation):
    """
    Base class for operations that produce a transform model.
    """

    @staticmethod
    def _get_aliases(attributes, aliases, suffix):
        aliases = [alias.strip() for alias in aliases]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name suffixed by _indexed.
        return [x[1] or '{}_{}'.format(x[0], suffix) for x in
                zip_longest(attributes, aliases[:len(attributes)])]
