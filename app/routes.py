import os
import shutil
from app import app, model, socketio
import logging
from flask import request, make_response
from flask_socketio import send
from dotenv import load_dotenv
from werkzeug.utils import secure_filename


load_dotenv()

app_log = logging.getLogger(__name__)

@socketio.on("get-model-info")
def index(_):
    try:
        model_date = ""
        train_params = ""
        status = 404

        if os.path.exists(app.config["MODEL_FILE"]):
            status = 200
            model_name = os.path.basename(app.config["MODEL_FILE"])
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
        return { "status": 500 }
    
@app.route("/train-model", methods=["POST"])
def train_model():
    try:
        params = request.data["params"]
        return make_response("OK", 200)
    except Exception as err:
        app_log.error(err)
        return make_response(err, 500)
    
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
        return { "status": 500 }
