import os
import myGraph
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_PATH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['csv', 'xes'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No model selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename_secure = secure_filename(file.filename)
        filename = "simple_process_model." + filename_secure.split('.', 1)[1]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Model successfully uploaded and displayed below')
        
        G = myGraph.MyGraph()
        model_filename = os.path.join('static', filename)
        image_filename = os.path.join('static', "simple_process_model.png")
        G.create_and_display_graph(image_filename, filename=model_filename)
        
        return render_template('upload.html', filename=image_filename)
    else:
        flash('Allowed image types are -> csv, xes')
        return redirect(request.url)

@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == "__main__":
    app.run(debug=False)