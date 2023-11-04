# Flask modules
from flask               import Flask, render_template, request, url_for, redirect, send_from_directory, session, Response
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import numpy as np
import cv2
import cvlib as cv
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from tensorflow.keras.utils import img_to_array

global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

video = cv2.VideoCapture(0)

app = Flask(__name__, template_folder='./templates',static_folder='./static')




# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'your secret key'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'facial_recognition_ads_bd'

# Intialize MySQL
mysql = MySQL(app)


@app.route('/', endpoint='HomePage')
def HomePage():
     return render_template("OfficialSite/index.html")

@app.route('/Login', endpoint='Login')
def LoginPage():
     return render_template("Authentification/sign-in.html")

@app.route('/SignUP', endpoint='SignUP')
def SignUpPage():
     return render_template("Authentification/sign-up.html")

@app.route('/Profil', endpoint='Profile')
def ProfilPage():
     if 'loggedin' in session:
        return render_template("Dashboard/pages/profile.html")
     # User is not loggedin redirect to login page
     return redirect(url_for('Login'))

@app.route('/Dashboard', endpoint='Dashboard')
def DashboardPage():
     if 'loggedin' in session:
        return render_template("Dashboard/pages/dashboard.html")
     # User is not loggedin redirect to login page
     return redirect(url_for('Login'))

@app.route('/Dashboard/Facial Recognition Ads', endpoint='FRA')
def DashboardPage():
     if 'loggedin' in session:
        return render_template("Dashboard/pages/FADS.html")
     # User is not loggedin redirect to login page
     return redirect(url_for('Login'))

# Authenticate user
@app.route('/loginSubmit',endpoint='loginSubmit', methods=['GET', 'POST'])
def login():
    # Flask message injected into the page, in case of any errors
    msg = None

    # check if both http method is POST and form is valid on submit
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password'] 

        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()

        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('Dashboard'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'

    return render_template( 'Authentification/sign-in.html', msg=msg )

# Authenticate user
@app.route('/signUpSubmit',endpoint='SignUpSubmit', methods=['GET', 'POST'])
def SignUp():
        # Output message if something goes wrong...
    msg = ''
    errorType = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
            errorType = 'erreur'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
            errorType = 'erreur'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
            errorType = 'erreur'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
            errorType = 'erreur'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO users VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
            errorType = 'success'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
        errorType = 'erreur'
    # Show registration form with message (if any)
    return render_template('Authentification/sign-up.html', msg=msg, errorType=errorType)

@app.route('/Logout', endpoint='Logout')
def Logout():
        # Remove session data, this will log the user out
        session.pop('loggedin', None)
        session.pop('id', None)
        session.pop('username', None)
        # Redirect to login page
        return redirect(url_for('Login'))

def gen(video):
    while True:
        # read frame from webcam
        status, frame = video.read()
        
        # apply face detection
        face, confidence = cv.detect_face(frame)
        
        # loop through detected faces
        for idx, f in enumerate(face):

            # get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            # draw rectangle over face
            cv2.rectangle(frame, (startX.astype(int),startY.astype(int)), (endX.astype(int),endY.astype(int)), (19, 3, 252), 2)

            # crop the detected face region
            face_crop = np.copy(frame[startY:endY,startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # preprocessing for gender detection model
            # convert the image to grayscale
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_crop = cv2.resize(face_crop, (128, 128))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=-1)
            face_crop = np.expand_dims(face_crop, axis=0)
            

            # apply gender detection on face
            #pred = model.predict(face_crop) # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

            #pred_gender = gender_dict[round(pred[0][0][0])]
            #pred_age = round(pred[1][0][0])

            #label = "Gender:"+ pred_gender+ "Age:"+ str(pred_age)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write label and confidence above face rectangle
            cv2.putText(frame, "person", (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (19, 3, 252), 2)
            
            ret, jpeg = cv2.imencode('.jpg', frame)

            frame = jpeg.tobytes()
            
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            
@app.route('/video_feed', endpoint="video_feed")
def video_feed():
		# Set to global because we refer the video variable on global scope, 
		# Or in other words outside the function
    global video

		# Return the result on the web
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)