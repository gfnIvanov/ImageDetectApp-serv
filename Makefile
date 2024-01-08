include .env.public

.PHONY: start

RUN = poetry run

start:
	@echo Server start on ${HOST}:${PORT}
	$(RUN) start
