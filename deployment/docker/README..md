# Building the Image

1. Build Docker Image

cd 
docker build --no-cache -t geoldm-image .

2. Create new Container based on built Image

docker run --gpus all -it --rm geoldm-image

docker run --gpus all -it --rm -p 7860:7860 geoldm-image /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate geoldm && python -m deployment.main -p -s 7860"




Push to DockerHub
docker login
docker tag geoldm-image yixian02/geoldm-image:latest
docker push yixian02/geoldm-image:latest

Verify Push:
https://hub.docker.com/u/yixian02


Pull from DockerHub
docker pull johndoe/geoldm-image