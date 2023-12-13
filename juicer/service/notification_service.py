import socketio

def notify(config: dict, room: str, msg_name: str, namespace: str, 
        message: dict, **kwargs):
    mgr = socketio.RedisManager(
            config['juicer']['servers']['redis_url'],
            'job_output', write_only=True)
    data = {}
    data.update(message)
    data.update(kwargs)
    # sio = socketio.Server(client_manager=mgr)
    mgr.emit(msg_name, data=data, room=str(room), namespace=namespace,
        broadcast=True)
    return mgr

