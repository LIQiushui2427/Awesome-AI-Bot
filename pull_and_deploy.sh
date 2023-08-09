docker login -u liqiushui2427 -p Qq6059160

docker pull liqiushui2427/awesome_ai_trader:dev

docker run --gpus all  -v datavol:/app/data -v outputsByAIvol:/app/outputsByAI -v outputsByBtvol:/app/outputsByBt -ti liqiushui2427/awesome_ai_trader:dev
