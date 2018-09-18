import os
import uuid

from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, url_for)

#from werkzeug.utils import secure_filename
from model.style_transfer import main  # ignore error from IDE


UPLOAD_FOLDER = './static/images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
SECRET_KEY = 'YOUR SECRET KEY FOR FLASK HERE'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

# check if file extension is right
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# force browser to hold no cache. Otherwise old result might return.
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# main flow of programme
@app.route('/neural-style/', methods=['GET', 'POST'])
def upload_file():
    try:
        # remove older files
        os.system("find static/images/ -maxdepth 1 -mmin +5 -type f -delete")
    except OSError:
        pass
    if request.method == 'POST':
        # check if the post request has the file part
        if 'content-file' and 'style-file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        content_file = request.files['content-file']
        style_file = request.files['style-file']
        files = [content_file, style_file]
        content_name = str(uuid.uuid4()) + ".png"
        style_name = str(uuid.uuid4()) + ".png"
        file_names = [content_name, style_name]
        for i, file in enumerate(files):
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_names[i]))
                # return redirect(url_for('uploaded_file',
                #                        filename=filename))
        result_filename = main(file_names[0], file_names[1])
        params={
            'content': "/neural_style/static/images/" + file_names[0],
            'style': "/neural_style/static/images/" + file_names[1],
            'result': "/neural_style/static/images/" + result_filename
        }
        return render_template('success.html', **params)
    return render_template('upload.html')


@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0')
