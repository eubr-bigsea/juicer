# coding=utf-8
import collections

from juicer import privaaas
from juicer.service import limonero_service


class PrivacyWorkflow(object):
    __slots__ = ('config', '_query_data_sources', 'workflow')

    def __init__(self, workflow_data, config,
                 query_data_sources=None):
        self._query_data_sources = query_data_sources
        self.config = config
        self.workflow = workflow_data

    def _build_privacy_restrictions(self):
        if 'juicer' not in self.config or \
                        'services' not in self.config['juicer']:
            return
        limonero_config = self.config['juicer']['services']['limonero']
        data_sources = []
        if self.workflow['platform']['slug'] != 'spark':
            return
        for t in self.workflow['tasks']:
            if t['operation'].get('slug') == 'data-reader':
                if self._query_data_sources:
                    ds = next(self._query_data_sources())
                else:
                    ds = limonero_service.get_data_source_info(
                        limonero_config['url'],
                        str(limonero_config['auth_token']),
                        t['forms']['data_source']['value'])
                data_sources.append(ds)

        privacy_info = {}
        attribute_group_set = collections.defaultdict(list)
        data_source_cache = {}
        for ds in data_sources:
            data_source_cache[ds['id']] = ds
            attrs = []
            privacy_info[ds['id']] = {'attributes': attrs}
            for attr in ds['attributes']:
                privacy = attr.get('attribute_privacy', {}) or {}
                attribute_privacy_group_id = privacy.get(
                    'attribute_privacy_group_id')
                privacy_config = {
                    'id': attr['id'],
                    'name': attr['name'],
                    'type': attr['type'],
                    'details': privacy.get('hierarchy'),
                    'privacy_type': privacy.get('privacy_type'),
                    'anonymization_technique': privacy.get(
                        'anonymization_technique'),
                    'attribute_privacy_group_id': attribute_privacy_group_id
                }
                attrs.append(privacy_config)
                if attribute_privacy_group_id:
                    attribute_group_set[attribute_privacy_group_id].append(
                        privacy_config)
                    # print('#' * 40)
                    # print(attr.get('name'), attr.get('type'))
                    # print(privacy.get('privacy_type'),
                    #       privacy.get('anonymization_technique'),
                    #       privacy.get('attribute_privacy_group_id'))

        def sort_attr_privacy(a):
            return privaaas.ANONYMIZATION_TECHNIQUES[a.get(
                'anonymization_technique', 'NO_TECHNIQUE')]

        for attributes in list(attribute_group_set.values()):
            more_restrictive = sorted(
                attributes, key=sort_attr_privacy, reverse=True)[0]
            # print(json.dumps(more_restrictive[0], indent=4))
            # Copy all privacy config from more restrictive one
            for attribute in attributes:
                attribute.update(more_restrictive)

        self.workflow['data_source_cache'] = data_source_cache
        self.workflow['privacy_restrictions'] = privacy_info
