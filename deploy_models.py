import docker

client = docker.DockerClient(base_url="tcp://blockchain2:2375")
client.images.build(path="containers/data", tag="tensorflow-generate-model-data")
client.volumes.create(name="data")
container = client.containers.run(
    "tensorflow-generate-model-data",
    command="python gen_model_data.py",
    volumes=["data:/data"],
)
container.remove()
