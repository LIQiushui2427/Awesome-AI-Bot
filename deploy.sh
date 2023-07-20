# bash or linux shell script
docker build -f Dockerfile.dev -t testdeploy:dev .
# uncomment the following line to run the container with fixed volume
docker run --gpus all  -v datavol:/app/data -v outputsByAIvol:/app/outputsByAI -v outputsByBtvol:/app/outputsByBt -ti testdeploy:dev
# uncomment the following line if use anonymous volume in Dockerfile.dev
# docker run --gpus all -ti testdeploy:dev
