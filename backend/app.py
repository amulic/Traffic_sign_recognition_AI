from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import keras
import os

# init flask app
app = Flask(__name__)
CORS(app)

# loading the model
model = load_model('C:/Users/mulik/Desktop/GTSRBsigns/training/TSR.keras')

# defining the traffic sign classes 
classes = classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

# route to handle image upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if(file.filename == ''):
        return jsonify({'error': 'No file provided'}), 400

    try: 
        # process the image
        image = Image.open(file)
        image = image.resize((30,30))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)

        # predict the traffic sign
        # prediction = np.argmax(model.predict(image), axis=1)[0]
        # predicted_class = classes[prediction]
        predictions = model.predict(image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = classes[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        return jsonify({'prediction': predicted_class, 'confidence': confidence}),200
    
    except Exception as e:
        return jsonify({'error': 'Unrecognisable image'}), 500

if __name__ == '__main__': 
    app.run(debug=True)
