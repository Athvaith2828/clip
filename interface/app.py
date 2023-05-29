from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from interface.pred import main
from config import config as cg

app = Flask(__name__,template_folder='templates',static_folder='templates')
app.config['UPLOAD_FOLDER'] = cg.upload_path

def generate_html(file_paths, captions, similar_images, output_file=cg.result):
    # Open the output HTML file in write mode
    with open(output_file, 'w') as f:
        # Write the HTML header
        f.write('<html>\n<head>\n</head>\n<body>\n')

        # Iterate over each predicting image, caption, and similar images
        for file_path, caption, sim_images in zip(file_paths, captions, similar_images):
            # Write the predicting image and its caption
            f.write(f'<h2>Predicting Image:</h2>\n')
            f.write(f'<img src="{file_path}" width="400" />\n')
            f.write(f'<p>{caption}</p>\n')

            # Write the similar images
            f.write('<h2>Similar Images:</h2>\n')
            f.write('<div style="display: flex; flex-wrap: wrap;">\n')
            for sim_image in sim_images:
                f.write(f'<img src="{sim_image}" width="200" />\n')
            f.write('</div>\n')

        # Write the HTML footer
        f.write('</body>\n</html>')

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
    generate_html(filenames, caps, results)
    return render_template('results.html')


if __name__ == '__main__':
    app.run(debug=True)
