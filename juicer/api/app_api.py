#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from flask_cors import CORS, cross_origin
from flask import Flask, session
from flask_restful import Api

from models import db
from execution_api import ExecutionListApi, ExecutionDetailApi

import json

app = Flask(__name__)
app.secret_key = 'l3m0n4d1-juicer'

app.config['SQLALCHEMY_POOL_SIZE'] = 10
app.config['SQLALCHEMY_POOL_RECYCLE'] = 300

# CORS
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

mappings = {
    '/executions': ExecutionListApi,
    '/executions/<int:execution_id>': ExecutionDetailApi,
}
for path, view in mappings.iteritems():
    api.add_resource(view, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file")

    args = parser.parse_args()
    config_file = '/scratch/walter/juicer/juicer/api/config.json' if args.config is None else args.config
    if config_file:
        with open(config_file) as f:
            config = json.load(f)

        app.config["RESTFUL_JSON"] = {"cls": app.json_encoder}

        server_config = config.get('servers', {})
        app.config['SQLALCHEMY_DATABASE_URI'] = server_config.get(
            'database_url')
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

        db.init_app(app)
        with app.app_context():
            db.create_all()

        if server_config.get('environment', 'dev') == 'dev':

            app.run(debug=True)
    else:
        parser.print_usage()
main()
