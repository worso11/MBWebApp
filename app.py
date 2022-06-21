import os
import myGraph
import numpy as np
import pandas as pd
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/'
FILE_TYPE = "csv"

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
    global FILE_TYPE
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No model selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename_secure = secure_filename(file.filename)
        FILE_TYPE = filename_secure.split('.', 1)[1]
        filename = "simple_process_model." + FILE_TYPE
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Model successfully uploaded and displayed below')
        
        if filename.endswith('.csv'):
            if detect_csv_cols():
                return render_template('csv_config.html')
        
        G = myGraph.MyGraph()
        model_filename = os.path.join('static', filename)
        image_filename = os.path.join('static', "simple_process_model.png")
        maxSlider, maxSlider2 = G.create_and_display_graph(image_filename, filename=model_filename)
        
        return render_template('upload.html', filename=image_filename, sliderValue=0, sliderValue2=0, sliderValue3=0, sliderValue4=0, maxSlider=maxSlider, maxSlider2=maxSlider2)
    else:
        flash('Allowed image types are -> csv, xes')
        return redirect(request.url)
    
def detect_csv_cols():
    filename = os.path.join('static', 'simple_process_model.csv')  
    df = pd.read_csv(filename)
    
    # if columns we need are already there, we just skip the config and use them
    if set(['Case ID', 'Activity', 'Start Timestamp']).issubset(set(df.columns.values)):
        df = pd.read_csv(filename, sep=',', usecols=['Case ID','Activity','Start Timestamp'])
        df.to_csv(filename, index=False)
        return False
    # if not, we move to specify which columns contain caseID, activity and timestamp
    else:
        return True


@app.route('/config', methods=['POST'])
def config_csv():
    model_filename = os.path.join('static', 'simple_process_model.csv')
    image_filename = os.path.join('static', "simple_process_model.png")
    
    caseId = request.form.get('caseId')
    activity = request.form.get('activity')
    timestamp = request.form.get('timestamp')
    headers = {
        'Case ID' : caseId,
        'Activity' : activity,
        'Start Timestamp' : timestamp
        }
    
    headers = {k: v for k, v in sorted(headers.items(), key=lambda item: item[1])}
    
    try:
        df = pd.read_csv(model_filename, sep=',', 
                         usecols = np.fromiter(headers.values(), dtype=int), 
                         names = np.array(list(headers.keys())), 
                         header=None, skiprows=1)
    except(ValueError):
        return render_template('csv_config.html')
    
    df.to_csv(model_filename)
    
    G = myGraph.MyGraph()
    maxSlider, maxSlider2 = G.create_and_display_graph(image_filename, filename=model_filename)
    
    return render_template('upload.html', filename=image_filename, sliderValue=0, sliderValue2=0, sliderValue3=0, sliderValue4=0, maxSlider=maxSlider, maxSlider2=maxSlider2)
        
@app.route('/filtered', methods=['POST'])
def filter_model():
    lower_filter = int(request.form["lowerFilter"])
    upper_filter = int(request.form["upperFilter"])
    length_min_filter = int(request.form["lengthMinFilter"])
    length_max_filter = int(request.form["lengthMaxFilter"])
    
    if (upper_filter == 0):
        upper_filter = float('inf')
        
    if (length_max_filter == 0):
        length_max_filter = float('inf')
    
    G = myGraph.MyGraph()
    model_filename = os.path.join('static', "simple_process_model." + FILE_TYPE)
    image_filename = os.path.join('static', "simple_process_model.png")
    maxSlider, maxSlider2 = G.create_and_display_graph(image_filename, filename=model_filename, lowerbound=lower_filter, upperbound=upper_filter, min_len=length_min_filter, max_len=length_max_filter)
    
    if (upper_filter == float('inf')):
        upper_filter = 0
        
    if (length_max_filter == float('inf')):
        length_max_filter = 0
    
    return render_template('upload.html', filename=image_filename, sliderValue=lower_filter, sliderValue2=upper_filter, sliderValue3=length_min_filter, sliderValue4=length_max_filter, maxSlider=maxSlider, maxSlider2=maxSlider2)
    

@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == "__main__":
    app.run(debug=False)