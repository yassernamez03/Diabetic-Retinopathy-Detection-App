from flask import Flask, render_template, session, url_for, redirect, request, jsonify, make_response
from pymongo import MongoClient
import os
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
from bson.objectid import ObjectId
import random
import cv2

from form import *
from functions import *

import json
import pickle
from ChatBot import Chat
from Bot_functions import *
from keras.models import load_model
#if the model exists we do not train else we do
if not(os.path.exists("./BotData/test_model.h5")):
    with open('./BotData/training.py', 'r') as f:
        code = compile(f.read(), './BotData/training.py', 'exec')
        exec(code)
        
#importing the bot data    
intents = json.loads(open('./BotData/Data.json',encoding='utf-8').read())

words = pickle.load(open('./BotData/words_test.pkl','rb'))
classes = pickle.load(open('./BotData/classes_test.pkl','rb'))

model = load_model('./BotData/test_model.h5')

chat = Chat(intents,words,classes,model)


app = Flask(__name__)
secret_key = os.urandom(12).hex()
app.config['SECRET_KEY'] = secret_key
UPLOAD_FOLDER = 'static/uploads/'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Mongo
client = MongoClient("mongodb://localhost:27017/")
db = client.get_database("db_bot")


user_name = None

pred = False
predMsg = ""

emailSent = False
code = ""
resetPass = False
emailPointed = ""
recFinished = False


@app.route("/")
def home():
    if "id" in session:
        return redirect(url_for('dashboard'))
        
    return render_template("index.html")

@app.route("/login", methods=("POST", "GET"))
def login():
    if "id" in session:
        return redirect(url_for('dashboard'))

    form = LoginForm()
    error = ""
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        # Checking if user with such email exists
        records = db.users
        user = records.find_one({"email": email})
        if user:
            # User Exists -> Checking passwords
            if check_password_hash(user["password"], password):
                # Correct Password
                session["id"] = str(user["_id"])
                chat.get_response("message")
                return redirect(url_for('dashboard'))
            else:
                # Wrong Password
                error = "Wrong Password."
        else:
            error = "User does not exist."

    return render_template("login.html", form=form, error=error)

@app.route("/signup", methods=("POST", "GET"))
def signup():
    if "id" in session:
        return redirect(url_for('dashboard'))
    
    form = SignupForm()
    error = ""
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        confirm = form.conpassword.data

        if password!=confirm:
            error="Passwords do not match."

        # Verifying if data already exist
        records = db.users
        if list(records.find({'email': email})):
            # User Already Exists
            error="User Already Exists."
        else:
            # User Doesn't Exist
            new_user = {
                "username": username,
                "email": email,
                "password": generate_password_hash(password) # Hashing Password
            }
            records.insert_one(new_user) # Inserting User Records to DB
            user = records.find_one({"email": email})
            session["id"] = str(user["_id"])

            # Creating a new chat for the new user
            records = db.chats
            records.insert_one({"user":session["id"]})
            chat.get_response("message")
            return redirect(url_for('homepage'))
            
    return render_template("signup.html", form=form, error=error)


@app.route("/dashboard")
def dashboard():
    if "id" not in session:
        return redirect(url_for('login'))
    return render_template("dashboard.html")

    
@app.route('/binaryclassifier', methods=['GET', 'POST'])
def binaryclassifier():
    if "id" not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the file to the upload folder
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Redirect to the results page with the file path
            return redirect(url_for('binary_results', filename=filename))
    clear_uploads()
    return render_template('binaryclassifier.html')

@app.route('/binaryresults')
def binary_results():
    if "id" not in session:
        return redirect(url_for('login'))
    
    filename = request.args.get('filename')
    if not filename:
        return "Filename is missing", 400

    # Define paths for the original and processed images
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    preprocessed_filename = 'preprocessed_' + filename
    preprocessed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], preprocessed_filename)
    ms_segmented_filename = 'ms_segmented_' + filename
    ms_segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], ms_segmented_filename)
    hae_segmented_filename = 'hae_segmented_' + filename
    hae_segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], hae_segmented_filename)
    hes_segmented_filename = 'hes_segmented_' + filename
    hes_segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], hes_segmented_filename)

    # Check if segmented files already exist
    if not (os.path.exists(preprocessed_image_path) and
            os.path.exists(ms_segmented_image_path) and
            os.path.exists(hae_segmented_image_path) and
            os.path.exists(hes_segmented_image_path)):
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "Image not found", 404

        # Predict image probability_1
        preprocessed_image = preprocess_image(image)
        proba_p = Detect(preprocessed_image)
        probability_p = {'no': 100-int(proba_p * 100), 'yes': int(proba_p * 100)}
        cv2.imwrite(preprocessed_image_path, preprocessed_image * 255)

        # Predict image probability_ms
        ms_segmented, proba_ms = Detect_Segment_Microaneurysms(preprocessed_image)
        probability_ms = {'no': 100-int(proba_ms * 100), 'yes': int(proba_ms * 100)}
        cv2.imwrite(ms_segmented_image_path, ms_segmented * 255)

        # Predict image probability_3
        hae_segmented, proba_hae = Detect_Segment_Haemorrhages(preprocessed_image)
        probability_hae = {'no': 100-int(proba_hae * 100), 'yes': int(proba_hae * 100)}
        cv2.imwrite(hae_segmented_image_path, hae_segmented * 255)

        # Predict image probability_4
        hes_segmented, proba_hes = Detect_Segment_HardExudates(preprocessed_image)
        probability_hes = {'no': 100-int(proba_hes * 100), 'yes': int(proba_hes * 100)}
        cv2.imwrite(hes_segmented_image_path, hes_segmented * 255)
    
    else:
        # If files exist, assume they were already processed and probabilities calculated
        proba_p = Detect(cv2.imread(preprocessed_image_path, cv2.IMREAD_GRAYSCALE))
        probability_p = {'no': 100-int(proba_p * 100), 'yes': int(proba_p * 100)}

        proba_ms = Detect_Segment_Microaneurysms(cv2.imread(ms_segmented_image_path, cv2.IMREAD_GRAYSCALE))[1]
        probability_ms = {'no': 100-int(proba_ms * 100), 'yes': int(proba_ms * 100)}

        proba_hae = Detect_Segment_Haemorrhages(cv2.imread(hae_segmented_image_path, cv2.IMREAD_GRAYSCALE))[1]
        probability_hae = {'no': 100-int(proba_hae * 100), 'yes': int(proba_hae * 100)}

        proba_hes = Detect_Segment_HardExudates(cv2.imread(hes_segmented_image_path, cv2.IMREAD_GRAYSCALE))[1]
        probability_hes = {'no': 100-int(proba_hes * 100), 'yes': int(proba_hes * 100)}

    results = {
        'Preprocessed': {'filename': preprocessed_filename, 'proba': probability_p},
        'Microaneurysms': {'filename': ms_segmented_filename, 'proba': probability_ms},
        'Haemorrhages': {'filename': hae_segmented_filename, 'proba': probability_hae},
        'HardExudates': {'filename': hes_segmented_filename, 'proba': probability_hes}
    }
    print(results)
    return render_template('binaryresults.html', results=results)


@app.route('/segmentationclassifier', methods=['GET', 'POST'])
def segmentationclassifier():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the file to the upload folder
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Redirect to the results page with the file path
            return redirect(url_for('segmentation_results', filename=filename))
    return render_template('segmentationclassifier.html')

@app.route('/segmentationresults')
def segmentation_results():
    if "id" not in session:
        return redirect(url_for('login'))
    
    filename = request.args.get('filename')
    if not filename:
        return "Filename is missing", 400

    # Define paths for the original and segmented images
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    ms_segmented_filename = 'ms_segmented_' + filename
    ms_segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], ms_segmented_filename)
    hae_segmented_filename = 'hae_segmented_' + filename
    hae_segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], hae_segmented_filename)
    hes_segmented_filename = 'hes_segmented_' + filename
    hes_segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], hes_segmented_filename)

    # Check if segmented files already exist
    if not os.path.exists(ms_segmented_image_path) or not os.path.exists(hae_segmented_image_path) or not os.path.exists(hes_segmented_image_path):
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "Image not found", 404

        # Predict image probability_1
        preprocessed_image = preprocess_image(image)

        # Predict image probability_ms
        ms_segmented = Segment_Microaneurysms(preprocessed_image)
        cv2.imwrite(ms_segmented_image_path, ms_segmented * 255)

        # Predict image probability_3
        hae_segmented = Segment_Haemorrhages(preprocessed_image)
        cv2.imwrite(hae_segmented_image_path, hae_segmented * 255)

        # Predict image probability_4
        hes_segmented = Segment_HardExudates(preprocessed_image)
        cv2.imwrite(hes_segmented_image_path, hes_segmented * 255)
    
    results = {
        'Microaneurysms': {'filename': ms_segmented_filename},
        'Haemorrhages': {'filename': hae_segmented_filename},
        'HardExudates': {'filename': hes_segmented_filename}
    }
    
    return render_template('segmentationresults.html', results=results)

@app.route('/multiclassifier', methods=['GET', 'POST'])
def multiclassifier():
    if "id" not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the file to the upload folder
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Redirect to the results page with the file path
            return redirect(url_for('multi_results', filename=filename))
    clear_uploads()
    return render_template('multiclassifier.html')

@app.route('/multiresults')
def multi_results():
    if "id" not in session:
        return redirect(url_for('login'))
    
    filename = request.args.get('filename')
    if not filename:
        return "Filename is missing", 400

    # Define paths for the original and processed images
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    preprocessed_filename = 'preprocessed_' + filename
    preprocessed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], preprocessed_filename)
    ms_segmented_filename = 'ms_segmented_' + filename
    ms_segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], ms_segmented_filename)
    hae_segmented_filename = 'hae_segmented_' + filename
    hae_segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], hae_segmented_filename)
    hes_segmented_filename = 'hes_segmented_' + filename
    hes_segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], hes_segmented_filename)

    # Check if segmented files already exist
    if not (os.path.exists(preprocessed_image_path) and
            os.path.exists(ms_segmented_image_path) and
            os.path.exists(hae_segmented_image_path) and
            os.path.exists(hes_segmented_image_path)):
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "Image not found", 404

        # Predict image probability_1
        preprocessed_image = preprocess_image(image)
        proba_p = np.round(Classifie(preprocessed_image)*100)
        probability_p = {'0': proba_p[0].astype(float), '1': proba_p[1].astype(float), '2': proba_p[2].astype(float), '3': proba_p[3].astype(float), '4': proba_p[4].astype(float)}
        cv2.imwrite(preprocessed_image_path, preprocessed_image * 255)

        # Predict image probability_ms
        ms_segmented, proba_ms = Classifie_Segment_Microaneurysms(preprocessed_image)
        proba_ms = np.round(proba_ms*100)
        probability_ms = {'0': proba_ms[0].astype(float), '1': proba_ms[1].astype(float), '2': proba_ms[2].astype(float), '3': proba_ms[3].astype(float), '4': proba_ms[4].astype(float)}
        cv2.imwrite(ms_segmented_image_path, ms_segmented * 255)

        # Predict image probability_3
        hae_segmented, proba_hae = Classifie_Segment_Haemorrhages(preprocessed_image)
        proba_hae = np.round(proba_hae*100)
        probability_hae = {'0': proba_hae[0].astype(float), '1': proba_hae[1].astype(float), '2': proba_hae[2].astype(float), '3': proba_hae[3].astype(float), '4': proba_hae[4].astype(float)}
        cv2.imwrite(hae_segmented_image_path, hae_segmented * 255)

        # Predict image probability_4
        hes_segmented, proba_hes = Classifie_Segment_HardExudates(preprocessed_image)
        proba_hes = np.round(proba_hes*100)
        probability_hes = {'0': proba_hes[0].astype(float), '1': proba_hes[1].astype(float), '2': proba_hes[2].astype(float), '3': proba_hes[3].astype(float), '4': proba_hes[4].astype(float)}
        cv2.imwrite(hes_segmented_image_path, hes_segmented * 255)
    
    else:
        # If files exist, assume they were already processed and probabilities calculated
        proba_p = Classifie(cv2.imread(preprocessed_image_path, cv2.IMREAD_GRAYSCALE))
        proba_p = np.round(proba_p*100)
        probability_p = {'0': proba_p[0].astype(float), '1': proba_p[1].astype(float), '2': proba_p[2].astype(float), '3': proba_p[3].astype(float), '4': proba_p[4].astype(float)}

        proba_ms = Classifie_Segment_Microaneurysms(cv2.imread(ms_segmented_image_path, cv2.IMREAD_GRAYSCALE))[1]
        proba_ms = np.round(proba_ms*100)
        probability_ms = {'0': proba_ms[0].astype(float), '1': proba_ms[1].astype(float), '2': proba_ms[2].astype(float), '3': proba_ms[3].astype(float), '4': proba_ms[4].astype(float)}

        proba_hae = Classifie_Segment_Haemorrhages(cv2.imread(hae_segmented_image_path, cv2.IMREAD_GRAYSCALE))[1]
        proba_hae = np.round(proba_hae*100)
        probability_hae = {'0': proba_hae[0].astype(float), '1': proba_hae[1].astype(float), '2': proba_hae[2].astype(float), '3': proba_hae[3].astype(float), '4': proba_hae[4].astype(float)}

        proba_hes = Classifie_Segment_HardExudates(cv2.imread(hes_segmented_image_path, cv2.IMREAD_GRAYSCALE))[1]
        proba_hes = np.round(proba_hes*100)
        probability_hes = {'0': proba_hes[0].astype(float), '1': proba_hes[1].astype(float), '2': proba_hes[2].astype(float), '3': proba_hes[3].astype(float), '4': proba_hes[4].astype(float)}

    results = {
        'Preprocessed': {'filename': preprocessed_filename, 'proba': probability_p},
        'Microaneurysms': {'filename': ms_segmented_filename, 'proba': probability_ms},
        'Haemorrhages': {'filename': hae_segmented_filename, 'proba': probability_hae},
        'HardExudates': {'filename': hes_segmented_filename, 'proba': probability_hes}
    }
    print(results) 
    return render_template('multiresults.html',results=results)
    
@app.route("/bot", methods=("POST", "GET"))
def bot():
    # Redirecting Users to Login Page if they aren't logged
    if "id" not in session:
        return redirect(url_for('login'))

    # Storing Message In DB when sent
    if request.method=="POST":
        data = request.get_data().decode("utf-8").split(" ")
        # AJAX SENDS DATA AS FOLLOWING [ACTION] [MESSAGE] [TARGET]
        # WE TRY TO CLASSIFY THE ACTION TYPE BY CHECKING THE VALUE OF [ACTION]
        action = data[0]
        print(data)

        if action=="AddChat":
            #ADD CHAT
            records = db.chats
            records.insert_one({"user":session['id']})
        elif action=="DeleteChat":
            #DELETE CHAT
            target = data[-1]
            records = db.chats
            if(len(list(records.find({"user":session['id']})))>1):
                records.delete_one({"_id":ObjectId(target)})
                # Deleting Messages
                records = db.messages
                records.delete_many({"chat":target})
        else:
            #INSERT MESSAGE
            global pred
            global predMsg
            global user_name
            global item_pred
            message = " ".join(data[1:-1])
            target = data[-1]
            print(message, target)
            records = db.messages
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            records.insert_one({"user": session["id"], "chat": target, "message": message, "type": "user", "date":dt_string})
            
            # AI Thingie Here
            res,tag = chat.get_response(message.lower())

            if tag == 'TimeQuery':
                response = TimeQuery()
                records.insert_one({"bot": session["id"], "chat": target, "message": ['NORMAL REPLY',response], "type": "bot", "date":dt_string})
            elif tag == 'DateQuery': 
                response = DateQuery()
                records.insert_one({"bot": session["id"], "chat": target, "message": ['NORMAL REPLY',response], "type": "bot", "date":dt_string})
            else:       
                records.insert_one({"bot": session["id"], "chat": target, "message": ['NORMAL REPLY',res], "type": "bot", "date":dt_string})

    return render_template("bot.html")

@app.route('/api/<cat>/<val>/<chat>', methods=("POST", "GET"))
@app.route("/api/<cat>/<val>", methods=("POST", "GET"), defaults={'chat': None})
def api(cat, val, chat):
    data = []
    if "id" in session:
        if cat=="msgs":
            records = db.messages
            for record in list(records.find({"chat": chat})):
                data.append((record["message"], record["type"]))
        if cat=="curr" and val=="user":
            data = session['id']
        if cat=="chats":
            records = db.chats
            for record in list(records.find({"user":val})):
                data.append((str(record["_id"])))

    return make_response(jsonify(data))

@app.route('/recovery', methods=("POST", "GET"))
def recovery():
    global code
    global emailSent
    global resetPass
    global emailPointed
    global recFinished

    if 'id' in session:
        return render_template(url_for('bot'))

    form = RecoveryForm()

    if not(recFinished) and form.validate_on_submit():
        emailSent = False
        code = ""
        resetPass = False
        emailPointed = ""

    if not(emailSent) and form.validate_on_submit():
        records = db.users
        email = form.email.data
        emailPointed = email
        if not(len(list(records.find({"email":email})))):
            return render_template('reset.html', form=form, error="User with such records not found.")
        #Generating random 6-digit-code
        code = random.randint(100000, 999999)
        htmlcode = """<link rel="preconnect" href="https://fonts.googleapis.com">
                    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@100;400&display=swap" rel="stylesheet"> 
                    <body>
                    <style>
                    * {
                            font-family: 'Roboto';
                        }
                    </style>
                    <p>Hi, this email contains a recovery code that you can use to change the password of your account. Here's the <strong>Recovery Code</strong>: </p>
                    <center>
                    <p style="font-size:34px;background: white; width: fit-content; padding: .75em 1em; border-radius: 6px; color: #ff123d; font-weight: bold; box-shadow: rgb(0, 0, 0, .25) 0 2px 5px;">"""+str(code)+"""</p>
                    </center>
                    </body>"""
        sendMail(email, 'Account Recovery', htmlcode)
        emailSent = True
        return render_template('verify.html' , form=VerifyForm(), code=code)
    elif emailSent and VerifyForm().validate_on_submit():
        userCode = VerifyForm().code.data
        if(int(userCode)==code):
            resetPass = True
            return render_template("resetpassword.html", form=ResetPasswordForm())
    elif resetPass and ResetPasswordForm().validate_on_submit():
        password = ResetPasswordForm().newpassword.data
        records = db.users
        myquery = { "email": emailPointed }
        newvalues = { "$set": { "password":generate_password_hash(password)  } }
        records.update_one(myquery, newvalues)

        recFinished = True

        return redirect(url_for('login'))

    return render_template('reset.html', form=form)

@app.route("/destroy", methods=("POST", "GET"))
def destroy():
    if 'id' in session:
        session.pop('id')

    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)