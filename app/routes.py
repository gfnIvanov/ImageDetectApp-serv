import os
import shutil
import json
import logging
import traceback
from websockets.sync.client import connect
from app import app, model, socketio
from flask_socketio import emit
from werkzeug.utils import secure_filename
from dotenv import dotenv_values


envs = dotenv_values(".env.public")

app_log = logging.getLogger(__name__)

@socketio.on("get-model-info")
def index(_):
    try:
        model_date = ""
        train_params = ""
        status = 404

        if os.path.exists(app.config["MODEL_FILE_EXM"]):
            status = 200
            model_name = os.path.basename(app.config["MODEL_FILE_EXM"])
            with open(app.config["MODEL_PARAM"], "r") as file:
                param_lines = file.readlines()
                model_date = param_lines[0]
                train_params = param_lines[1]

        response = {
            "status": status,
            "name": model_name, 
            "date": model_date,
            "params": train_params
        }

        return response
    except Exception as err:
        app_log.error(err)
        if envs["MODE"] == "dev":
            traceback.print_tb(err.__traceback__)
        return {"status": 500}
    
@socketio.on("train-model")
def train_model(params):
    try:
        websocket = connect(envs["TRAIN_SERV"])
        websocket.send(json.dumps(params))
        res = websocket.recv()
        websocket.close()
        if res == "200":
            for result in model.train(params):
                emit("train-res", {"status": 200, "result": result})
        else:
            emit("train-res", {"status": 500})
            raise Exception("Ошибка при передаче параметров")
    except Exception as err:
        app_log.error(err)
        return {"status": 500}
    
@socketio.on("use-model")
def use_model(data):
    try:
        file = data["file"]
        filename = secure_filename(data["filename"]) 
        classname = data["class"]
        
        if os.path.exists(app.config["UPLOAD_DIR"]):
            shutil.rmtree(app.config["UPLOAD_DIR"])
            os.mkdir(app.config["UPLOAD_DIR"])

        class_dir = os.path.join(app.config["UPLOAD_DIR"], classname)
        os.mkdir(class_dir)

        with open(os.path.join(class_dir, filename), 'wb') as f:
            f.write(file)

        result = model.use()

        response = {
            "status": 200,
            "result": result
        }

        return response
    except Exception as err:
        app_log.error(err)
        if envs["MODE"] == "dev":
            traceback.print_tb(err.__traceback__)
        return {"status": 500}
