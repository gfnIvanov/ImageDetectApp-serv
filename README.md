# Серверная часть приложения для распознавания изображений

[Фронтенд](https://github.com/gfnIvanov/ImageDetectApp-web/tree/master)

[Сервис обучения](https://github.com/gfnIvanov/ImageDetectApp-train)

## Обзор

Серверная часть обеспечивает функциональность подготовки модели для распознавания изображений животных, а также доступ к данной модели со стороны фронтенда.

Сервер реализует websocket-соединение для отправки клиенту информации о процессе обучения в реальном времени.

Для тренировки модели и подготовки данных использованы два фреймворка машинного обучения [Tensorflow](https://www.tensorflow.org/) и [Pytorch](https://pytorch.org/). Tensorflow не удалось запустить локально на MacBook m1, обучение выполнялось в [Yandex DataSphere](https://cloud.yandex.ru/services/datasphere), результаты можно увидеть в соответствующем [ноутбуке](./notebooks/with_tensorflow.ipynb). Для дальнейшей работы в рамках данного проекта выбран Pytorch.

Параметры обучения модели настраиваются [здесь](./params/process_model.yml).

Результаты обучения модели логируются [здесь](./models/model_log.json).

В качестве фреймворка для сервера используется [Flask](https://flask.palletsprojects.com/en/3.0.x/).

Развертывание приложения (фронт, бэк и сервис обучения) производится при помощи docker-compose.

## Технологии

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)