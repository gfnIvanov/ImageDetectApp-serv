import os
from app import (
    app, 
    BASE_DIR, 
    MODEL_FILE, 
    MODEL_PARAMS, 
    model
)
import logging
from flask import request, make_response
from dotenv import load_dotenv


load_dotenv()

app_log = logging.getLogger(__name__)

@app.route("/", methods=["GET"])
def index():
    try:
        model_date = ""
        train_params = ""
        status = 404

        if os.path.exists(MODEL_FILE):
            status = 200
            model_name = os.path.basename(MODEL_FILE)
            with open(MODEL_PARAMS, "r") as file:
                param_lines = file.readlines()
                model_date = param_lines[0]
                train_params = param_lines[1]

        response = {
            "name": model_name, 
            "date": model_date,
            "params": train_params
        }

        return make_response(response, status)
    except Exception as err:
        app_log.error(err)
        return make_response(err, 500)
    

@app.route("/use-model", methods=["PUT"])
def use_model():
    try:
       model.use()
       return make_response("OK", 200)
    except Exception as err:
        app_log.error(err)
        return make_response(err, 500)