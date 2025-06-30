import json
from flask import Flask, render_template,request,jsonify,send_from_directory
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from train_own_dataset import main
import glob
import shutil

app =Flask(__name__)

# supersecretkey2025 SHA-256 UTF 8
app.config['SECRET_KEY'] = '32b0d906e31813a7693f748e8426727c2395ca77484d283f41b1d4d0eb2a8614'
#file uploads configuration
app.config['UPLOAD_FOLDER'] = r'D:\PythonProject YOLO\static\uploads'
app.config['RESULT_FOLDER'] = r'D:\PythonProject YOLO\static\results'

class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/',methods=['GET','POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        print(f"Secure file name: {filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("FINAL path:", filepath)
        file.save(filepath)
        print("File has been uploaded!")

        nutritionVal = main(filepath)

        #find the 'runs/detect/predict*'
        predict_folders = glob.glob('runs/detect/predict*')
        latest_folder = max(predict_folders, key=os.path.getctime)

        #grabs the first image from latest predit directory
        result_images = glob.glob(os.path.join(latest_folder, '*.jpg'))
        result_image_path = result_images[0] if result_images else None

        #copy labeled images to static/results
        if result_image_path:
            result_image_name = os.path.basename(result_image_path)
            final_result_path = os.path.join('static/results', result_image_name)
            shutil.copy(result_image_path, final_result_path)
        with open('nutritionVal.json', 'w', encoding='utf-8') as f:
            json.dump(list(nutritionVal), f, ensure_ascii=False, indent=4)
            return render_template('results.html', fruits=nutritionVal, result_image=result_image_name)
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
