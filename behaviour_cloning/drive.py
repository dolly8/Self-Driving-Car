import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

 
sio = socketio.Server()
 
app = Flask(__name__) #'__main__'

def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# Simulator will send updates from audio frames
@sio.on('telemetry')
def telemetry(sid, data): # Recieving an update from simulator
	image = Image.open(BytesIO(base64.b64decode(data['image']))) # It will decode the images # ByteIO to mimic our data like a normal file
	image = np.asarray(image) # It is feeded to neural nets
	image = img_preprocess(image)
	image = np.array([image])
	steering_angle = float(model.predict(image))
	send_control(steering_angle, 1.0)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0) # Steering and throttle angles will be determined based on the model predictions
 
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)
 
 
if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
