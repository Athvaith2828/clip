from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from interface.pred import main
from config import config as cg

app = Flask(__name__,template_folder='templates')
app.config['UPLOAD_FOLDER'] = cg.upload_path

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Create 'uploads' folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    uploaded_files = request.files.getlist('file')
    filenames = []

    for file in uploaded_files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_path = f'{cg.upload_path}\{filename}'
            filenames.append(file_path)
    caps, results = main(filenames)
    return render_template('results.html', filenames=filenames)


if __name__ == '__main__':
    app.run(debug=True)
