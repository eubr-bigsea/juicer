import sys
import yaml
import os
from juicer.service import tahiti_service 

def prepare_and_get_plugin_factory(config, platform_id):
    # retrieve complementary data for platform
    base_url = config['juicer']['services']['tahiti'].get('url')
    token = config['juicer']['services']['tahiti'].get('auth_token')

    if not all([base_url, token]):
        raise ValueError(_('Juicer is not correctly configured '
            '(missing Tahiti service paramter(s).'))

    platform = tahiti_service.get_platform(base_url, str(token), platform_id)
    plugin_base_dir = os.path.join(os.environ.get('JUICER_HOME', '.'), 'plugins')

    if platform.get('plugins'):
        plugin_factories = {}
        for plugin in platform.get('plugins', []):
            plugin_dir = os.path.join(plugin_base_dir, str(plugin['id']))
            if not os.path.exists(plugin_dir):
                from git import Repo
                repo = Repo.clone_from(
                    plugin['url'], plugin_dir, branch='master')
            else:
                # TODO: Verify version/commit
                pass
            # Adds new source code tree to the path
            sys.path.append(os.path.join(plugin_dir, 'src'))
            from importlib import import_module
            manifest = yaml.load(plugin.get('manifest'), 
                    Loader=yaml.FullLoader)['plugin']

            # import pdb; pdb.set_trace()
            module_path, class_name = manifest.get('provides').get(
                    'factory').get('name').rsplit('.', 1)
            module = import_module(module_path)
            factory = getattr(module, class_name)
            plugin_factories[platform.get('id')] = factory
    else:
        raise ValueError(_('Platform does not have any plugin.'))
    return plugin_factories

