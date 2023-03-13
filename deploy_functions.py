import docker

client = docker.from_env()
client.images.build(rm=True, path="containers/compute", tag="tensorflow-with-functions")
