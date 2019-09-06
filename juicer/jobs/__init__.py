# coding=utf-8
import os
import tempfile
from urllib.parse import urlparse

import pyarrow as pa
from juicer.atmosphere import opt_ic


def estimate_time_with_performance_model(payload):
    # retrieves the model
    model = payload['model']
    storage = model['storage']
    result = float('inf')
    if storage['type'] == 'HDFS':
        parsed = urlparse(storage['url'])
        if parsed.scheme == 'file':
            path = os.path.abspath(os.path.join(parsed.path, model['path']))
            optimizer = opt_ic.Optimizer(
                os.path.splitext(os.path.basename(path))[0],
                20, os.path.dirname(path))
            # Deadline is in seconds, model expects milliseconds
            result = optimizer.solve(2, payload['deadline'] * 1000)
        else:
            tmp_path = os.path.join(tempfile.gettempdir(),
                                    os.path.basename(model['path']))
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                fs = pa.hdfs.connect(parsed.hostname, parsed.port)
                with open(tmp_path, 'wb') as temp:
                    with fs.open(model['path'], 'rb') as f:
                        temp.write(f.read())
                fs.close()

            optimizer = opt_ic.Optimizer(
                os.path.splitext(os.path.basename(tmp_path))[0],
                20, os.path.dirname(tmp_path))
            # Deadline is in seconds, model expects milliseconds
            result = optimizer.solve(2, payload['deadline'] * 1000)
    else:
        pass
    print(result)
    return result
