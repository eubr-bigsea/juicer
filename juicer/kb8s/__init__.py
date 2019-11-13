# coding=utf-8
import json

from kubernetes import client
from kubernetes.client.rest import ApiException


def create_k8s_job(workflow_id, minion_cmd, cluster):
    configuration = client.Configuration()
    configuration.host = cluster['address']
    configuration.verify_ssl = False
    configuration.debug = False
    if 'general_parameters' not in cluster:
        raise ValueError('Incorrect cluster config.')

    cluster_params = {}
    for parameter in cluster['general_parameters'].split(','):
        key, value = parameter.split('=')
        if key.startswith('kubernetes'):
            cluster_params[key] = value
    env_vars = {
        'HADOOP_CONF_DIR': '/usr/local/juicer/conf',
    }

    token = cluster['auth_token']
    configuration.api_key = {"authorization": "Bearer " + token}
    # noinspection PyUnresolvedReferences
    client.Configuration.set_default(configuration)

    job = client.V1Job(api_version="batch/v1", kind="Job")
    name = 'job-{}'.format(workflow_id)
    container_name = 'juicer-job'
    container_image = cluster_params['kubernetes.container']
    namespace = cluster_params['kubernetes.namespace']

    job.metadata = client.V1ObjectMeta(namespace=namespace, name=name)
    job.status = client.V1JobStatus()

    # Now we start with the Template...
    template = client.V1PodTemplate()
    template.template = client.V1PodTemplateSpec()

    # Passing Arguments in Env:
    env_list = []
    for env_name, env_value in env_vars.items():
        env_list.append(client.V1EnvVar(name=env_name, value=env_value))

    # Subpath implies that the file is stored as a config map in kb8s
    volume_mounts = [
        client.V1VolumeMount(
            name='juicer-config', sub_path='juicer-config.yaml',
            mount_path='/usr/local/juicer/conf/juicer-config.yaml'),
        client.V1VolumeMount(
            name='hdfs-site', sub_path='hdfs-site.xml',
            mount_path='/usr/local/juicer/conf/hdfs-site.xml'),
        client.V1VolumeMount(
            name='hdfs-pvc',
            mount_path='/srv/storage/'),
    ]
    pvc_claim = client.V1PersistentVolumeClaimVolumeSource(
        claim_name='hdfs-pvc')

    # resources = {'limits': {'nvidia.com/gpu': 1}}
    resources = {}

    container = client.V1Container(name=container_name,
                                   image=container_image,
                                   env=env_list, command=minion_cmd,
                                   image_pull_policy='Always',
                                   volume_mounts=volume_mounts,
                                   resources=resources)

    volumes = [
        client.V1Volume(name='juicer-config',
                        config_map=client.V1ConfigMapVolumeSource(
                            name='juicer-config')),
        client.V1Volume(name='hdfs-site',
                        config_map=client.V1ConfigMapVolumeSource(
                            name='hdfs-site')),
        client.V1Volume(name='hdfs-pvc',
                        persistent_volume_claim=pvc_claim),
    ]
    template.template.spec = client.V1PodSpec(
        containers=[container], restart_policy='Never', volumes=volumes)

    # And finally we can create our V1JobSpec!
    job.spec = client.V1JobSpec(ttl_seconds_after_finished=10,
                                template=template.template)
    api = client.ApiClient(configuration)
    batch_api = client.BatchV1Api(api)

    try:
        batch_api.create_namespaced_job(namespace, job, pretty=True)
    except ApiException as e:
        print("Exception when calling BatchV1Api->: {}\n".format(e))
