.PHONY: start

RUN = poetry run
MODE = dev

start:
ifeq ($(MODE), dev)
	$(RUN) flask run --debug
else
	$(RUN) flask run
endif
	