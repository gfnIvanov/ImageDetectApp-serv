import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from app import app, data


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
    data_handler = data.Data()
    dataloaders = data_handler.process()
    for epoch in range(params["epochs"]):
        running_loss = 0.0
        for i, data in enumerate(dataloaders["train_data"], 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

def use():
    try:
        net = Net()
        data_handler = data.Data()
        classes = list(data_handler.translate.keys())
        net.load_state_dict(torch.load(app.config["MODEL_FILE"]))
        dataloaders = data_handler.transform_data(app.config["DATA_DIR"], ["for_check"])
        image, _ = next(iter(dataloaders["for_check"]))
        output = net(image)
        _, predicted = torch.max(output, 1)
        return data_handler.translate[classes[predicted[0]]]
    except Exception as err:
        app_log.error(err)
