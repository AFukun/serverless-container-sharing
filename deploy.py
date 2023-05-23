import docker

HOST_URL = "ssh://luosf@blockchain2"
# HOST_URL = "ssh://luosf@t630-1080"
REFRESH_VOLUME = False


def _print_logs(logs):
    for chunk in logs:
        if "stream" in chunk:
            for line in chunk["stream"].splitlines():
                print(line)


def deploy_pre_experiments():
    client = docker.DockerClient(base_url=HOST_URL)
    _, logs = client.images.build(rm=True, path="pre_experiment", tag="pre-experiment")
    _print_logs(logs)


def deploy_functions():
    client = docker.DockerClient(base_url=HOST_URL)
    _, logs = client.images.build(rm=True, path="server", tag="server")
    _print_logs(logs)


def deploy_models():
    client = docker.DockerClient(base_url=HOST_URL)
    _, logs = client.images.build(rm=True, path="setup", tag="setup")
    _print_logs(logs)

    if REFRESH_VOLUME:
        client.volumes.get("data").remove()
        client.volumes.create("data")

    container = client.containers.create(
        "setup",
        command="python setup.py -D /data/",
        volumes=["data:/data"],
    )
    try:
        container.start()
        container.wait()
    finally:
        print(container.logs().decode())
        container.remove()


if __name__ == "__main__":
    deploy_pre_experiments()
    deploy_functions()
    deploy_models()
