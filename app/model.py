import os
import pickle
import json
import yaml
import time
import boto3
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import traceback
from flask_socketio import emit
from websockets.sync.client import connect
from app import app, data as data_module
from datetime import datetime
from dotenv import dotenv_values


envs = {
    **dotenv_values(".env.public"),
    **dotenv_values(".env.secret"),
}

app_log = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(params):
    try:
        start_time = time.time()
        data_handler = data_module.Data(params)
        dataloaders = data_handler.process()
        criterion = nn.CrossEntropyLoss()

        with connect(envs["TRAIN_SERV"]) as websocket:
            for epoch in range(int(params["epochs"])):
                running_loss = 0.0
                for i, data in enumerate(dataloaders["train_data"], 0):
                    inputs, labels = data
        
                    websocket.send(pickle.dumps(inputs))
                    res = websocket.recv()
                    if res == "500":
                        raise Exception("Ошибка при обучении модели")
                    loss = criterion(torch.as_tensor(pickle.loads(res), dtype=torch.float32), labels)
                    loss.backward()
                    running_loss += loss.item()

                    if i % 1000 == 999:
                        loss_res = {
                            "epoch": epoch + 1,
                            "iter": i + 1,
                            "loss": f"{running_loss / 100:.3f}"
                        }
                        running_loss = 0.0
                        yield loss_res

            websocket.send("done")
            res = websocket.recv()

            if res == "200":
                emit("check-model")
                
                get_from_s3()

                end_time = round((time.time() - start_time) / 60, 2)

                log_data = check_model(dataloaders, params, end_time)

                emit("train-done", {"status": "200", "result": log_data})
            else:
                raise Exception("Ошибка при сохранении модели")
    except Exception as err:
        app_log.error(err)
        if envs["MODE"] == "dev":
            traceback.print_tb(err.__traceback__)

def use(for_check=False):
    try:
        with open(app.config["TRAIN_PARAM"]) as f:
            params = yaml.safe_load(f)

        net = Net()
        data_handler = data_module.Data(params)
        translate = data_handler.translate
        classes = list(translate.keys())

        model_file = app.config["MODEL_FILE"] if for_check else app.config["MODEL_FILE_EXM"]

        net.load_state_dict(torch.load(model_file))
        net.eval()

        if for_check:
            return net, classes, translate
        
        dataloaders = data_handler.transform_data(app.config["DATA_DIR"], ["for_check"])
        image, _ = next(iter(dataloaders["for_check"]))
        output = net(image)
        _, predicted = torch.max(output, 1)

        return translate[classes[predicted[0]]]
    except Exception as err:
        app_log.error(err)
        if envs["MODE"] == "dev":
            traceback.print_tb(err.__traceback__)

def get_from_s3():
    try:
        session = boto3.session.Session()

        s3 = session.client(
            service_name="s3",
            aws_access_key_id=envs["aws_access_key_id"],
            aws_secret_access_key=envs["aws_secret_access_key"],
            endpoint_url="https://storage.yandexcloud.net"
        )

        resp_object = s3.get_object(Bucket=envs["BUCKET"], Key="model.pth")
        with open(app.config["MODEL_FILE"], "wb") as f:
            f.write(resp_object["Body"].read())
    except Exception as err:
        app_log.error(err)
        if envs["MODE"] == "dev":
            traceback.print_tb(err.__traceback__)

def check_model(dataloaders, params, duration):
    current_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    log_data = {
        "date": current_datetime,
        "params": params,
        "accuracy": [],
        "duration": str(duration) + " min"
    }

    net, classes, translate = use(for_check=True)

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for i, data in enumerate(dataloaders["test_data"], 0):
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        res_str = f"Accuracy for class: {translate[classname]:5s} is {accuracy:.1f} %"
        log_data["accuracy"].append(res_str)

    if not os.path.exists(app.config["MODEL_LOG"]):
        with open(app.config["MODEL_LOG"], "w") as f:
            json.dump({"model_log": [log_data]}, f, ensure_ascii=False, indent=4)
    else:
        with open(app.config["MODEL_LOG"], "r") as f:
            current_log_data = json.load(f)

        current_log_data["model_log"].append(log_data)

        with open(app.config["MODEL_LOG"], "w") as f:
            json.dump(current_log_data, f, ensure_ascii=False, indent=4)

    with open(app.config["MODEL_PARAM"], "w") as f:
        f.write(current_datetime + "\n")
        f.write(json.dumps(params))

    return log_data
