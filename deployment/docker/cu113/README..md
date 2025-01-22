# Building and Running the Image

1. Build Docker image
    ```bash
    cd to this directory

    docker build --no-cache -t control-geoldm .
    ```

2. Create a new container based on built image and run. If you wish to directly start the inferencing server, run:
    ```bash
    docker run --gpus all -it --rm -p 7860:7860 control-geoldm /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate geoldm && python -m deployment.main -u -a 0.0.0.0 -p 7860"
    ```
    However, if you wish to only run the container for inspection, run:
    ```bash
    docker run --gpus all -it --rm control-geoldm
    ```

</br>

# Pushing the built image to DockerHub
```bash
docker login

docker tag control-geoldm yixian02/control-geoldm:cu113

docker push yixian02/control-geoldm:cu113
```
Verify your push <a href='https://hub.docker.com/u/yixian02'>here</a>.

</br>

# Pulling and Running the built image from DockerHub
```bash
docker pull yixian02/control-geoldm:cu113

docker run --gpus all -it --rm -p 7860:7860 yixian02/control-geoldm:cu113 /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate geoldm && python -m deployment.main -u -a 0.0.0.0 -p 7860"
```

</br>

# Accessing the Inferencing UI
There are 2 ways to access the UI:
- Locally via <a href='http://127.0.0.1:7860'>http://localhost:7860</a>.
- Public URL hosted by gradio for 72 hours. The URL will be in the form of <code>some-unique-id.gradio.live</code> and will be printed on the terminal when the docker image is run.

</br>

# Hardware Specifications
The hardware running the docker image must be running the <code>x86_64</code> architecure. <code>ARM</code> is not suppported. This is due to the <a href='../../analysis/qvina/qvina2.1'>QuickVina2.1 binaries</a> only supporting the <code>x86_64</code> architecture. The device must also have a <code>NVIDIA</code> GPU with <code>CUDA 11.3</code> installed.