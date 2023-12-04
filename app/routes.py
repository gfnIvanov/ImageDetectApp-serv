import os
import shutil
from app import app, model
import logging
from flask import request, make_response
from dotenv import load_dotenv
from werkzeug.utils import secure_filename


load_dotenv()

app_log = logging.getLogger(__name__)

@app.route("/", methods=["GET"])
def index():
    try:
        model_date = ""
        train_params = ""
        status = 404

        if os.path.exists(app.config['MODEL_FILE']):
            status = 200
            model_name = os.path.basename(app.config['MODEL_FILE'])
            with open(app.config['MODEL_PARAM'], "r") as file:
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
        file = request.files['file']
        filename = secure_filename(file.filename)
        classname = request.form['class']
        if os.path.exists(app.config['UPLOAD_DIR']):
            shutil.rmtree(app.config['UPLOAD_DIR'])
            os.mkdir(app.config['UPLOAD_DIR'])
        class_dir = os.path.join(app.config['UPLOAD_DIR'], classname)
        os.mkdir(class_dir)
        file.save(os.path.join(class_dir, filename))
        result = model.use()
        return make_response(result, 200)
    except Exception as err:
        app_log.error(err)
        return make_response(err, 500)