import os
import logging
from flask import Flask
from flask_cors import CORS
from pathlib import Path


app = Flask(__name__)

CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent
CHECK_IMG_DIR = os.path.join(BASE_DIR, "data/for_check")
MODEL_FILE = os.path.join(BASE_DIR, "models/model.pth")
MODEL_PARAMS = os.path.join(BASE_DIR, "models/params.txt")

logging.basicConfig(level = logging.DEBUG,
                    filename = os.path.join(BASE_DIR, 'logs/app_log.log'),
                    filemode = "w",
                    format = "%(asctime)s - %(name)s[%(funcName)s(%(lineno)d)] - %(levelname)s - %(message)s")


from app import routes