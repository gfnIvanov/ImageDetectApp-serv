import os
import torch
import shutil
import logging
import traceback
from torchvision import datasets
import torchvision.transforms as transforms
from app import app
from dotenv import load_dotenv


load_dotenv()

app_log = logging.getLogger(__name__)

class Data:
    log_data = {
        "date": "",
        "params": "",
        "accuracy": []
    }

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

    train_data = {}

    test_data = {}

    train_data_dir= os.path.join(app.config["DATA_DIR"], "train_data")

    test_data_dir = os.path.join(app.config["DATA_DIR"], "test_data")

    def __init__(self, params):
        self.params = params

    def prepare_files(self):
        try:
            if os.path.exists(self.test_data_dir):
                shutil.rmtree(self.test_data_dir)
                os.mkdir(self.test_data_dir)

            for folder in os.listdir(self.train_data_dir):
                i = 0
                currentFolder = os.path.join(self.train_data_dir, folder)
                if os.path.isdir(currentFolder):
                    for _ in os.listdir(currentFolder):
                        i += 1
                    self.train_data[folder] = i
                    os.mkdir(os.path.join(self.test_data_dir, folder))

            for folder in os.listdir(self.train_data_dir):
                currentFolder = os.path.join(self.train_data_dir, folder)
                equalTestFolder = os.path.join(self.test_data_dir, folder)
                if os.path.isdir(currentFolder):
                    filesList = os.listdir(currentFolder)
                    part = round(self.train_data[folder] * float(self.params["test-size"]))
                    self.test_data[self.translate[folder]] = part
                    for i in range(1, part):
                        shutil.copy(os.path.join(currentFolder, filesList[i]), os.path.join(equalTestFolder, filesList[i]))
        except Exception as err:
            app_log.error(err)
            if os.getenv("MODE") == "dev":
                traceback.print_tb(err.__traceback__)

    def transform_data(self, data_dir, source):
        try:
            data_transforms = transforms.Compose([
                transforms.Resize((int(self.params["img-shape"]), int(self.params["img-shape"]))),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            image_datasets = {
                x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms)
                for x in source
            }

            dataloaders = {
                x: torch.utils.data.DataLoader(image_datasets[x], 
                                               batch_size=int(self.params["batch-size"]),
                                               shuffle=True, 
                                               pin_memory=False,
                                               num_workers=4)
                for x in source
            }

            return dataloaders
        except Exception as err:
            app_log.error(err)
            if os.getenv("MODE") == "dev":
                traceback.print_tb(err.__traceback__)
    
    def process(self):
        self.prepare_files()
        return self.transform_data(app.config["DATA_DIR"], ["train_data", "test_data"])
