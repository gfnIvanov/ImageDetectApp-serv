import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import logging
import torchvision.transforms as transforms
from torchvision import datasets
from app import BASE_DIR, MODEL_FILE, CHECK_IMG_DIR
from pathlib import Path


app_log = logging.getLogger(__name__)

with open(Path.joinpath(BASE_DIR, 'params', 'process_model.yml')) as f:
    params = yaml.safe_load(f)

translate = {
    "cane": "собака", 
    "cavallo": "лошадь", 
    "elefante": "слон", 
    "farfalla": "бабочка", 
    "gallina": "курица", 
    "gatto": "кошка", 
    "mucca": "корова", 
    "pecora": "овца", 
    "scoiattolo": "белка",
    "ragno": "паук"
}

classes = list(translate.keys())


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(18496, 120)
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
    
net = Net()


def use():
    try:
        net.load_state_dict(torch.load(MODEL_FILE))
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(params['img-shape']),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        image_dataset = datasets.ImageFolder(CHECK_IMG_DIR, data_transforms)
        dataloader = torch.utils.data.DataLoader(image_dataset, 
                                                 batch_size=params['batch-size'],
                                                 shuffle=True, 
                                                 num_workers=4)
        image, _ = next(iter(dataloader))
        output = net(image)
        _, predicted = torch.max(output, 1)
        print(predicted)
        print(translate[classes[predicted[0]]])
    except Exception as err:
        app_log.error(err)
