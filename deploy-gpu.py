import docker

HOST_URL = "ssh://luosf@t630-1080"


def _print_logs(logs):
    for chunk in logs:
        if "stream" in chunk:
            for line in chunk["stream"].splitlines():
                print(line)


def deploy_functions():
    client = docker.DockerClient(base_url=HOST_URL)
    _, logs = client.images.build(rm=True, path="server-gpu", tag="server")
    _print_logs(logs)


if __name__ == "__main__":
    deploy_functions()
