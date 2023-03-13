import docker

client = docker.DockerClient(base_url="tcp://blockchain2:2375")
client.images.build(path="containers/compute", tag="tensorflow-with-functions")
