#!/bin/sh
# update yum
yum update -y
# install docker
yum install -y docker
# pull docker image
docker pull liqiushui2427/cotenv:latest
docker run -d -p 80:80 liqiushui2427/cotenv:latest
```