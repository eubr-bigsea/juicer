# coding=utf-8

from textwrap import dedent


def copy_from_input(operation):
    code = []
    if len(operation.named_inputs) == 1:
        in_df = operation.get_inputs_names()
        for out_df in operation.named_outputs:
            code.append(dedent("""
                juicer.traceability.copy({in_df}, {out_df})
            """.format(out_df=out_df, in_df=in_df)))
    elif len(operation.named_inputs) > 1:
        raise ValueError(_(
            'This implementation of traceability do not support multiple inputs'
        ))
    return '\n'.join(code)


def copy(source, target):
    for attr in source.schema:
        for k, v in list(attr.metadata.items()):
            target.schema[attr.name].metadata[k] = v
