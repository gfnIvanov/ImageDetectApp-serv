FROM python:3.10.13-slim
WORKDIR /flaskapp
COPY pyproject.toml pyproject.toml
COPY .flaskenv .flaskenv
ADD app /flaskapp/app
ADD logs /flaskapp/logs
ADD models /flaskapp/models
ADD params /flaskapp/params
COPY poetry.lock poetry.lock
RUN python -m pip install --upgrade pip
RUN pip install poetry
RUN poetry install
RUN mkdir data && mkdir data/for_check
EXPOSE 5101
CMD poetry run flask run