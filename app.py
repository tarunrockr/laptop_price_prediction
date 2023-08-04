from   flask import Flask, request, render_template
import numpy as np 
import pandas as pd 
import pickle

# Importing the pickle file 
pipe = pickle.load(open('pipe.pkl', 'rb'))
df   = pickle.load(open('df.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predictions', methods=['POST', 'GET'])
def input_predictions():
	if request.method == 'GET':
		return render_template('predict_page.html')

	elif request.method == "POST":
		
		company           = request.form.get('company')
		typename          = request.form.get('typename')
		ram               = int(request.form.get('ram'))
		weight            = float(request.form.get('weight'))
		touch_screen      = int(request.form.get('touch_screen'))
		ips               = int(request.form.get('ips'))
		screen_size       = float(request.form.get('screen_size'))
		screen_resolution = request.form.get('screen_resolution')
		cpu_brand         = request.form.get('cpu_brand')
		hdd               = int(request.form.get('hdd'))
		ssd               = int(request.form.get('ssd'))
		gpu_brand         = request.form.get('gpu_brand')
		os                = request.form.get('os')

		# Calculating ppi
		X_res = int(screen_resolution.split('x')[0])
		Y_res = int(screen_resolution.split('x')[1])

		ppi = (X_res**2 * Y_res**2)**0.5/screen_size

		# creating a numpy array from the user input 
		input_arr = np.array([company, typename, ram, weight, touch_screen, ips, ppi, cpu_brand, hdd, ssd, gpu_brand, os])
		print("Input Array: ",input_arr)
		print("Array Shape: ",input_arr.shape)
		input_arr = input_arr.reshape(1,12)
		print("Array Shape After: ",input_arr.shape)
		print("Input Array: ",input_arr)

		predict_output = np.exp(pipe.predict(input_arr))
		predict_output = int(predict_output[0])

		return render_template('predict_page.html', results = predict_output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)


