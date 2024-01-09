FROM python:3.10.13-slim
WORKDIR /flaskapp
COPY pyproject.toml pyproject.toml
COPY .env.public .env.public
COPY .env.secret .env.secret
ADD app /flaskapp/app
ADD logs /flaskapp/logs
ADD params /flaskapp/params
COPY poetry.lock poetry.lock
RUN python -m pip install --upgrade pip
RUN pip install poetry
RUN poetry install
RUN mkdir models \
    && mkdir -p data/for_check data/test_data data/train_data 
EXPOSE 5101
CMD poetry run start