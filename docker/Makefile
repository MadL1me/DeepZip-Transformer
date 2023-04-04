help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
PYTHON_VERSION?=3.6
CUDA_VERSION?=11.2
CUDNN_VERSION?=8
TEST=tests/
SRC?=$(shell dirname `pwd`)

build:
	sudo docker build -t keras --build-arg python_version=$(PYTHON_VERSION) --build-arg cuda_version=$(CUDA_VERSION) --build-arg cudnn_version=$(CUDNN_VERSION) -f $(DOCKER_FILE) .

bash: build
	sudo $(DOCKER) run -u root -it -v $(SRC):/src/workspace -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras bash

ipython: build
	sudo $(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras ipython

notebook: build
	sudo $(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --net=host --env KERAS_BACKEND=$(BACKEND) keras

test: build
	sudo $(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras py.test $(TEST)

