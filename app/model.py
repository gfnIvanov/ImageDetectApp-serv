import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import traceback
from flask_socketio import emit
from dotenv import load_dotenv
from websockets.sync.client import connect
from app import app, data as data_module


load_dotenv()

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
        data_handler = data_module.Data(params)
        dataloaders = data_handler.process()
        criterion = nn.CrossEntropyLoss()

        with connect(os.getenv("TRAIN_SERV")) as websocket:
            for epoch in range(int(params["epochs"])):
                running_loss = 0.0
                for i, data in enumerate(dataloaders["train_data"], 0):
                    inputs, labels = data
        
                    websocket.send(pickle.dumps(inputs))
                    res = websocket.recv()
                    loss = criterion(pickle.loads(res), labels)
                    loss.backward()
                    running_loss += loss.item()

                    if i % 1000 == 999:
                        loss_res = {
                            "epoch": epoch + 1,
                            "iter": i + 1,
                            "loss": round(running_loss / 100, 4)
                        }
                        running_loss = 0.0
                        yield loss_res

            websocket.send("done")
            res = websocket.recv()

            if res == "200":
                emit("train-done")
            else:
                raise Exception("Ошибка при сохранении модели")
    except Exception as err:
        app_log.error(err)
        if os.getenv("MODE") == "dev":
            traceback.print_tb(err.__traceback__)

def use():
    try:
        net = Net()
        data_handler = data_module.Data()
        classes = list(data_handler.translate.keys())
        net.load_state_dict(torch.load(app.config["MODEL_FILE"]))
        dataloaders = data_handler.transform_data(app.config["DATA_DIR"], ["for_check"])
        image, _ = next(iter(dataloaders["for_check"]))
        output = net(image)
        _, predicted = torch.max(output, 1)
        return data_handler.translate[classes[predicted[0]]]
    except Exception as err:
        app_log.error(err)
        if os.getenv("MODE") == "dev":
            traceback.print_tb(err.__traceback__)
