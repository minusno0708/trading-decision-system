IMAGE_NAME=cuda117

.PHONY: build run stop

build:
	docker build -t $(IMAGE_NAME) . 

run:
	docker run -it --rm --gpus all -v .:/workspace $(IMAGE_NAME) $(filter-out $@,$(MAKECMDGOALS))

%:
	@: