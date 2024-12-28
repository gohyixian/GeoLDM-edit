# Building the Image

1. Build Docker image
    ```bash
    cd to this directory
    docker build --no-cache -t control-geoldm .
    ```

2. Create a new container based on built image and run. If you wish to directly start the inferencing server, run:
    ```bash
    docker run --gpus all -it --rm -p 7860:7860 control-geoldm /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate geoldm && python -m deployment.main -p -s 7860"
    ```
    However, if you wish to only run the container for inspection, run:
    ```bash
    docker run --gpus all -it --rm control-geoldm
    ```

</br>

# Pushing the built image to DockerHub
```bash
docker login
docker tag control-geoldm yixian02/control-geoldm:latest
docker push yixian02/control-geoldm:latest
```
Verify your push <a href='https://hub.docker.com/u/yixian02'>here</a>.

</br>

# Pulling the built image from DockerHub and run
```bash
docker pull yixian02/control-geoldm:latest

docker run --gpus all -it --rm -p 7860:7860 control-geoldm /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate geoldm && python -m deployment.main -s 7860"
```