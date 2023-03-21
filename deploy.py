import docker

HOST_URL = "ssh://luosf@blockchain2"


def _print_logs(logs):
    for chunk in logs:
        if "stream" in chunk:
            for line in chunk["stream"].splitlines():
                print(line)


def deploy_functions():
    client = docker.DockerClient(base_url=HOST_URL)
    _, logs = client.images.build(
        rm=True, path="server", tag="tensorflow-with-functions"
    )
    _print_logs(logs)


def deploy_models():
    client = docker.DockerClient(base_url=HOST_URL)
    _, logs = client.images.build(
        rm=True, path="setup", tag="tensorflow-generate-model-data"
    )
    _print_logs(logs)
    client.volumes.create(name="data")
    print(
        client.containers.run(
            "tensorflow-generate-model-data",
            command="python setup.py",
            volumes=["data:/data"],
            auto_remove=True,
        ).decode("utf-8")
    )


if __name__ == "__main__":
    deploy_functions()
    deploy_models()
