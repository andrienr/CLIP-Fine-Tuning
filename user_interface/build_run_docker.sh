#!/bin/sh

docker build -t marble_catalogue . && docker run -p 7860:7860 marble_catalogue