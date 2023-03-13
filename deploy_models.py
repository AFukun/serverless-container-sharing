import docker

client = docker.from_env()
client.images.build(
    rm=True, path="containers/data", tag="tensorflow-generate-model-data"
)
client.volumes.create(name="data")
container = client.containers.run(
    "tensorflow-generate-model-data",
    command="python gen_model_data.py",
    volumes=["data:/data"],
    auto_remove=True,
)
