# !/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
from gettext import gettext, translation
import logging.config
import os
import json
import sys

import yaml
from io import StringIO
from juicer.runner import configuration
from juicer.service.tahiti_service import query_tahiti
from juicer.workflow.workflow import Workflow
import juicer.plugin.util as plugin_util

logging.config.fileConfig('logging_config.ini')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def generate(workflow_id, template_name, lang='en'):
    juicer_config = {}
    result = {}
    # import pdb; pdb.set_trace()

    locales_path = os.path.join(os.path.dirname(__file__), 'i18n', 'locales')
    t = translation('messages', locales_path, [lang],
                    fallback=True)
    t.install()

    if 'JUICER_CONF' in os.environ:
        with open(os.environ['JUICER_CONF']) as config_file:
            juicer_config = yaml.load(config_file.read(),
                                      Loader=yaml.FullLoader)
        out = StringIO()
        try:
            _generate(workflow_id,
                      False,
                      {},
                      juicer_config,
                      out=out,
                      export_notebook=template_name == 'notebook',
                      plain=template_name == 'python', lang=lang)
            result['code'] = str(out.getvalue())
            result['status'] = 'OK'
        except Exception as e:
            result['status'] = 'ERROR'
            result['message'] = str(e)

    else:
        result['status'] = 'ERROR'
        result['message'] = gettext('Server is not correctly configured.')

    return result


def _get_lookups(tahiti_conf, workflow_id, resp, lang):
    ops = query_tahiti(
        base_url=tahiti_conf['url'], item_path='/operations',
        token=str(tahiti_conf['auth_token']), item_id='',
        qs='lang={}&fields=id,slug,ports.id,ports.slug,ports.interfaces&'
        'platform={}&workflow={}&code_gen=1'.format(
            lang, resp['platform']['id'], workflow_id)).get('data')
    slug_to_op_id = dict([(op['slug'], op['id']) for op in ops])
    port_id_to_port = dict([(p['id'], p) for op in ops for p in op['ports']])

    return ops, slug_to_op_id, port_id_to_port


def _generate(workflow_id, job_id, execute_main, params, config, out=sys.stdout,
              deploy=False, export_notebook=False, plain=False,
              custom_vars=None, lang='en', from_meta=False, variant=None,
            json_file=None):
    log.debug(gettext(
        'Generating code for workflow %s, notebook=%s, plain=%s'),
        workflow_id,
        export_notebook,
        plain
    )
    tahiti_conf = config['juicer']['services']['tahiti']
    if json_file is None:
        resp = query_tahiti(base_url=tahiti_conf['url'],
                        item_path='/workflows',
                        token=str(tahiti_conf['auth_token']),
                        item_id=workflow_id, qs=f'lang={lang}')
    else:
        with open(json_file) as f:
            resp = json.loads(f.read().strip())
    loader = Workflow(resp, config, lang=lang)
    loader.handle_variables(custom_vars)
    if variant is not None:
        config['variant'] = variant

    configuration.set_config(config)

    ops, slug_to_op_id, port_id_to_port = _get_lookups(
        tahiti_conf, workflow_id, resp, lang)

    try:
        if loader.platform['slug'] == "spark":
            from juicer.spark.transpiler import SparkTranspiler

            transpiler = SparkTranspiler(configuration.get_config(),
                                         slug_to_op_id, port_id_to_port)
        elif loader.platform['slug'] == "compss":
            from juicer.compss.transpiler import COMPSsTranspiler
            transpiler = COMPSsTranspiler(configuration.get_config())
        elif loader.platform['slug'] == "scikit-learn":
            from juicer.scikit_learn.transpiler import ScikitLearnTranspiler
            transpiler = ScikitLearnTranspiler(configuration.get_config())
        elif loader.platform['slug'] == 'keras':
            from juicer.keras.transpiler import KerasTranspiler
            transpiler = KerasTranspiler(configuration.get_config())
        elif loader.platform.get('plugin'):
            plugin_factories = plugin_util.prepare_and_get_plugin_factory(
                configuration.get_config(), loader.platform.get('id'))
            factory = plugin_factories.get(loader.platform['id'])
            transpiler = factory.get_transpiler(configuration.get_config())
        elif loader.platform['slug'] == 'meta':
            from juicer.meta.transpiler import MetaTranspiler
            transpiler = MetaTranspiler(configuration.get_config())
        else:
            raise ValueError(
                gettext('Invalid platform value: {}').format(loader.platform))

        params['execute_main'] = execute_main
        transpiler.execute_main = execute_main
        if loader.platform['slug'] == 'meta' and from_meta:
            out1 = StringIO()

            loader = Workflow(resp, config, lang=lang)
            targets = {'spark': {'id': 1, 'slug': 'spark'},
                       'scikit-learn': {'id': 4, 'slug': 'scikit-learn'}}
            loader.workflow['target_meta_platform'] = targets[from_meta]
            transpiler.transpile(
                loader.workflow, loader.graph, params=params, deploy=deploy,
                export_notebook=export_notebook, plain=plain, job_id=job_id,
                out=out1)
            out1.seek(0)

            if loader.workflow.get('type') == 'MODEL_BUILDER':
                out.write(out1.read())
            else:
                resp = json.loads(out1.read())
                target_loader = Workflow(resp, config, lang=lang)
                target_loader.handle_variables(custom_vars)

                ops, slug_to_op_id, port_id_to_port = _get_lookups(
                    tahiti_conf, 0, resp, lang)

                if transpiler.target_meta.get('slug') == 'spark':
                    from juicer.spark.transpiler import SparkTranspiler
                    final_transpiler = SparkTranspiler(
                        configuration.get_config())
                elif transpiler.target_meta.get('slug') == 'scikit-learn':
                    from juicer.scikit_learn.transpiler import (
                        ScikitLearnTranspiler)
                    final_transpiler = ScikitLearnTranspiler(
                        configuration.get_config())
                else:
                    raise ValueError('Invalid target platform')
                final_transpiler.sample_style = 'DATA_EXPLORER'
                final_transpiler.transpile(
                    target_loader.workflow, target_loader.graph,
                    params=params, deploy=deploy,
                    export_notebook=export_notebook,
                    plain=plain, job_id=job_id, out=out)

        else:
            transpiler.transpile(
                loader.workflow, loader.graph, params=params, deploy=deploy,
                export_notebook=export_notebook, plain=plain, job_id=job_id,
                out=out)

    except ValueError as ve:
        log.exception(
            gettext("At least one parameter is missing"), exc_info=ve)
    except:
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False,
                        help="Configuration file")

    parser.add_argument("-w", "--workflow", type=int, required=True,
                        help="Workflow identification number")

    parser.add_argument("-j", "--job_id", type=int, required=False,
                        help="Job identification number", default=0)

    parser.add_argument("-e", "--execute-main", action="store_true",
                        help="Write code to run the program (it calls main()")
    parser.add_argument("-d", "--deploy", action="store_true", default=False,
                        help="Generate deployment workflow")

    parser.add_argument("-n", "--notebook", action="store_true",
                        help="Generate Jupyter Notebook")

    parser.add_argument("-m", "--meta",
                        help="Convert from Meta Plataform")

    parser.add_argument("--lang", help="Minion messages language (i18n)",
                        required=False, default="en")
    parser.add_argument("--vars",
                        help="Path to a YAML file with the extra variables",
                        required=False)
    parser.add_argument("-v", "--variant", type=str, required=False,
                        help="Variant used to code generation in platform")
    parser.add_argument("--json", type=str, required=False,
                        help="Path to JSON file containing workflow code (do not read from Tahiti API)")
    parser.add_argument(
        "-p", "--plain", required=False, action="store_true",
        help="Indicates if workflow should be plain Python, "
             "without Lemonade extra code")
    args = parser.parse_args()

    locales_path = os.path.join(os.path.dirname(__file__), 'i18n', 'locales')
    t = translation('messages', locales_path, [args.lang],
                    fallback=True)
    t.install()

    juicer_config = {}
    if args.config:
        with open(args.config) as config_file:
            juicer_config = yaml.load(config_file.read(),
                                      Loader=yaml.FullLoader)
    custom_vars = {'job_id': 9999}
    if args.vars:
        with open(args.vars) as vars_file:
            custom_vars.extend(yaml.load(vars_file.read(),
                                    Loader=yaml.FullLoader))

    _generate(args.workflow, args.job_id, args.execute_main,
              {"plain": args.plain},
              config=juicer_config, deploy=args.deploy,
              export_notebook=args.notebook, plain=args.plain,
              custom_vars=custom_vars, lang=args.lang, from_meta=args.meta,
              variant=args.variant, json_file=args.json)
