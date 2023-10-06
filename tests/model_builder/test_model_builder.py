from io import StringIO
import sys
from juicer.meta.meta_minion import MetaMinion
from juicer.workflow.workflow import Workflow
from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SortOperation
import pytest
from juicer.transpiler import GenerateCodeParams
from mock import patch, MagicMock
from .fixtures import * # Must be * in order to import fixtures
import hashlib

config = {
        'juicer': {
            'auditing': False,
            'services':{
                'limonero': {
                    'url': 'http://localhost',
                    'auth_token': 111
                }
            }

        }
    }
def test_sample(builder_params: any):
    for k in ['type', 'fraction', 'seed', 'value']:
        print(k, builder_params.sample.parameters.get(k))

    print(builder_params.sample.enabled)
    builder_params.sample.parameters['seed'] = 7777
    builder_params.sample.parameters['value'] = 30000
    builder_params.sample.parameters['fraction'] = .5
    builder_params.sample.parameters['type'] = 'percent'
    # Test type parameter:
    # - None: no code is generated (no sampling)
    # - percent: 
    # - value: 
    # - head: 

    print(builder_params.sample.model_builder_code())

def test_split(builder_params: any):
    for k in ['strategy', 'seed', 'ratio']:
        print(k, builder_params.split.parameters.get(k))

    builder_params.split.has_code = True
    builder_params.split.strategy = 'cross_validation'
    builder_params.split.seed = 302324
    builder_params.split.ratio = .7
    # Test method parameter:
    # - split: 
    # - cross_validation: Not implemented :(

    print(builder_params.split.model_builder_code())

def test_evaluator(builder_params: any):
    for k in ['multi_metric', 'bin_metric', 'reg_metric', 'clust_metric']:
        print(k, builder_params.evaluator.parameters.get(k))
    print(builder_params.evaluator.parameters.keys())

    # Type of task is stored in the workflow. May change in the future.

    builder_params.evaluator.has_code = True
    print('Task type: ', builder_params.evaluator.parameters
        .get('workflow')
        .get('forms').get('$meta').get('value')
        .get('taskType'))
    builder_params.evaluator.method = 'pca'
    builder_params.evaluator.k = 30
    # Test taskType and metric* parameters:

    print(builder_params.evaluator.model_builder_code())

def test_features(builder_params: any):
    # Many tests to be implemented! Maybe convert the FeaturesOperation
    # to use a Jinja template

    # for k in ['method', 'number', 'k']:
    #     print(k, builder_params.features.parameters.get(k))
    # print(builder_params.features.parameters.keys())

    # builder_params.features.method = 'pca'
    # builder_params.features.has_code = True
    # builder_params.features.k = 30
    # Test method parameter:
    # - disable: no features
    # - pca: Uses pca

    print(builder_params.features.model_builder_code())

def test_grid(builder_params: any):
    for k in ['strategy', 'random_grid', 'seed', 'max_iterations',
            'max_search_time', 'parallelism']:
        print(k, builder_params.grid.parameters.get(k))

    builder_params.grid.has_code = True
    builder_params.grid.strategy = 'random'
    builder_params.grid.max_iterations = 30
    # Test strategy parameter:
    # - grid: ok
    # - random: ok

    print(builder_params.grid.model_builder_code())


def test_reduction(builder_params: any):
    for k in ['method', 'number', 'k']:
        print(k, builder_params.reduction.parameters.get(k))
    print(builder_params.reduction.parameters.keys())

    builder_params.reduction.method = 'pca'
    builder_params.reduction.has_code = True
    builder_params.reduction.k = 30
    # Test method parameter:
    # - disable: no reduction
    # - pca: Uses pca

    print(builder_params.reduction.model_builder_code())

def test_estimators(builder_params: any):

    print('Estimators',  builder_params.estimators)
    for estimator in builder_params.estimators:
        for k in ['method', 'number', 'k']:
            print(k, estimator.parameters.get(k))
            # print(estimator.parameters.keys())

        estimator.has_code = True
        # Test method parameter:
        # - disable: no estimator
        # - pca: Uses pca
        print('-'*20)
        estimator.grid_info.get('value')['strategy'] = 'random'
        estimator.grid_info.get('value')['max_iterations'] = 32 
        print(estimator.model_builder_code())
        print(estimator.generate_hyperparameters_code())
        # print(estimator.generate_random_hyperparameters_code())


def test_builder_params(sample_workflow: dict, builder_params: any):
    
    loader = Workflow(sample_workflow, config, lang='en')
    instances = loader.workflow['tasks']

    minion = MetaMinion(None, config=config, workflow_id=sample_workflow['id'],
                        app_id=sample_workflow['id'])

    job_id = 1
    opt = GenerateCodeParams(loader.graph, job_id, None, {}, 
        ports={}, state={}, task_hash=hashlib.sha1(), 
        workflow=loader.workflow,
        tasks_ids=list(loader.graph.nodes.keys()))
    instances, _ = minion.transpiler.get_instances(opt)

    builder_params = minion.transpiler.prepare_model_builder_parameters(ops=instances.values())

    print(dir(builder_params))
    print(builder_params.read_data.model_builder_code())
    print(builder_params.sample.model_builder_code())
    print(builder_params.split.model_builder_code())
    print(builder_params.evaluator.model_builder_code())
    print(builder_params.features.model_builder_code())
    print(builder_params.reduction.model_builder_code())
    print(builder_params.grid.model_builder_code())
    
def test_generate_run_code_success(sample_workflow: dict, builder_params: any):

    job_id = 1
    estimator = builder_params.estimators[0]
    estimator.has_code = True
    estimator.grid_info.get('value')['strategy'] = 'random'
    estimator.grid_info.get('value')['strategy'] = 'grid'
    estimator.grid_info.get('value')['max_iterations'] = 32 

    # import pdb; pdb.set_trace()
    loader = Workflow(sample_workflow, config, lang='en')

    loader.handle_variables({'job_id': job_id})
    out = StringIO()

    minion = MetaMinion(None, config=config, workflow_id=sample_workflow['id'],
                        app_id=sample_workflow['id'])

    minion.transpiler.transpile(loader.workflow, loader.graph,
                                config, out, job_id,
                                persist=False)
    out.seek(0)
    code = out.read()
    with open('/tmp/juicer_app_1_1_1.py', 'w') as f:
        f.write(code)

    print(code, file=sys.stderr)

    result = util.execute(code, {'df': 'df'})
    print('Executed')
    # assert not result['out'].equals(test_df)
    # assert """out = df.sort_values(by=['sepalwidth'], ascending=[False])""" == \
    #        instance.generate_code()
