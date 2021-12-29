from flask import Flask, render_template, request, url_for
import os

from werkzeug.utils import redirect, secure_filename
import tamura
STATIC_FOLDER = './static/uploaded_images'

app = Flask(__name__)

app.config['UPLOADED'] = './static/uploaded_images'


@app.route('/', methods=['GET'])
def index():

    return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'img' in request.files:
        # Lecture des paramètres
        excel = ("feature.xls")
        if request.form['size']:
            size= int(request.form['size'])
        else:
            size=12
        img = request.files["img"]
        imgName = 'a.jpg'
        img.save(os.path.join(app.config['UPLOADED'], imgName))
        # imgPath = '/home/ibrahim/SIM/M2/Traitement_Image_AitKbir/CBIR/static/uploaded_images'
        fullPath = os.path.join(app.config['UPLOADED'], imgName)
        # print(full_filename)
        list_images = tamura.get_similarity(fullPath, excel, size)
        return render_template('index.html', images=list_images, path='./static/coil-100/')
    else:
        return "else"


if __name__ == "__main__":
    app.run(debug=True)
