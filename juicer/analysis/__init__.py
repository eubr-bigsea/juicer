# coding=utf-8



def flatten(l):
    return [item for sublist in l for item in sublist]


def only_field(task, field, many):
    result = []
    if not many:
        result.extend(
            [f.strip() for f in task['forms'][field]['value'].split(', ')])
    elif task['forms'][field]:
        result.extend(task['forms'][field]['value'])

    return result


def copy_input_add_field(task, field, many, default_value):
    if not many:
        result = flatten(task['uiPorts']['inputs'])
    else:
        result = []

    if task['forms'][field] and task['forms'][field]['value']:
        result.append(task['forms'][field]['value'])
    elif default_value:
        result.append(default_value)
    else:
        result.extend(flatten(task['uiPorts']['inputs']))
    if task['forms'][field] and task['forms'][field]['value'].length:
        result.extend(task['forms'][field]['value'])
    elif default_value:
        result.append(default_value)
    return sorted(result)


def copy_all_inputs_remove_duplicated(task):
    attrs = set(flatten(task['uiPorts']['inputs']))
    return list(attrs)


def copy_input(task):
    return sorted(flatten(task['uiPorts']['inputs']))


def join_suffix_duplicated_attributes(task):
    """ Group attributes by name
    :param task:
    :return:
    """
    return []


def copy_input_add_attributes_split_alias(task, attributes, alias,
                                          suffix):
    result = flatten(task['uiPorts']['inputs'])
    aliases = []
    if task['forms'][alias] and task['forms'][alias]['value']:
        aliases = [a.strip() for a in task['forms'][alias]['value'].split(',')]
    else:
        aliases = []
    if task['forms'][attributes] and task['forms'][attributes]['value']:
        while task['forms'][attributes]['value'].length > len(aliases):
            attr = task['forms'][attributes]['value'][len(aliases)]
            aliases.append(attr + suffix)
    else:
        print('Alias ' + alias + ' does not exist for task ' +
              task.id + '(' + task.set_operation.slug + ')')

    result.extend(aliases)


def copy_from_only_one_input(task, _id):
    inputs = [inp for inp in task['uiPorts']['inputs'] if
              inp['targetPortId'] == _id]
    return flatten(inputs).sort()
