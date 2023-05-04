import docker

HOST_URL = "ssh://luosf@blockchain2"


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
    print(
        client.containers.run(
            "setup",
            command="python setup.py -D /data/",
            volumes=["data:/data"],
            auto_remove=True,
        ).decode("utf-8")
    )


if __name__ == "__main__":
    deploy_pre_experiments()
    deploy_functions()
    deploy_models()
