import os
import logging
from flask import Flask
from pathlib import Path
from flask_socketio import SocketIO
from dotenv import dotenv_values


envs = dotenv_values(".env.public")

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")

app.config["BASE_DIR"] = Path(__file__).resolve().parent.parent
app.config["DATA_DIR"] = os.path.join(app.config["BASE_DIR"], "data")
app.config["UPLOAD_DIR"] = os.path.join(app.config["DATA_DIR"], "for_check")
app.config["MODEL_FILE"] = os.path.join(app.config["BASE_DIR"], "models/model.pth")
app.config["MODEL_FILE_EXM"] = os.path.join(app.config["BASE_DIR"], "models/example/model.pth")
app.config["MODEL_PARAM"] = os.path.join(app.config["BASE_DIR"], "models/params.txt")
app.config["MODEL_LOG"] = os.path.join(app.config["BASE_DIR"], "models/model_log.json")
app.config["TRAIN_PARAM"] = os.path.join(app.config["BASE_DIR"], "params/process_model.yml")

logging.basicConfig(level = logging.DEBUG if envs["MODE"] == "dev" else logging.WARN,
                    filename = os.path.join(app.config["BASE_DIR"], "logs/app_log.log"),
                    filemode = "w",
                    format = "%(asctime)s - %(name)s[%(funcName)s(%(lineno)d)] - %(levelname)s - %(message)s")

from app import routes

def run():
    socketio.run(app, host=envs["HOST"], port=envs["PORT"], allow_unsafe_werkzeug=True)
