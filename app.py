import os
import numpy as np
import io
from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename



from PIL import Image, ImageFile
import matplotlib.pyplot as plt

from io import BytesIO
import base64

import detect




app= Flask(__name__)
net = detect.load_model(model_name="u2netp")


@app.route('/')
def home():
	return render_template('home.html')


@app.route('/Prediction', methods = ['GET', 'POST'])
def pred():
	if request.method=='POST':
		 data = request.files['file'].read()
		 img = Image.open(io.BytesIO(data))
		 output = detect.predict(net, np.array(img))
		 output = output.resize((img.size), resample=Image.BILINEAR) # remove resample

		 empty_img = Image.new("RGBA", (img.size), 0)
		 new_img = Image.composite(img, empty_img, output.convert("L"))
		 

		 #uploaded image
		 img_x0=BytesIO()
		 plt.imshow(new_img)
		 plt.savefig(img_x0,format='png')
		 plt.close()
		 img_x0.seek(0)
		 plot_url0=base64.b64encode(img_x0.getvalue()).decode('utf8')

		

		 


	return render_template('pred.html', plot_url0=plot_url0)


if __name__=='__main__':
	app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
