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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Upload File")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data

        if not allowed_file(file.filename):
            flash("Unsupported file type")
            return redirect(request.url)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
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

@app.errorhandler(413)
def too_large(e):  # noqa: D401
    flash("File is too large. Maximum size is 10MB.", "danger")
    return render_template("index.html", form=UploadFileForm()), 413


@app.errorhandler(404)
def not_found(e):  # noqa: D401
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(e):  # noqa: D401
    # Generic 500 handler; detailed errors are already logged.
    return render_template("500.html"), 500

if __name__ == '__main__':
    app.run(debug=True)
