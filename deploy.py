import docker

client = docker.DockerClient(base_url="tcp://172.18.166.229:2375")
client.images.build(path="container", tag="tensorflow-with-models")
