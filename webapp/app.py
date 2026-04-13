import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

import torch
import cv2
import torchvision.transforms as T

from google.cloud import firestore

from src.utils.face_detect import crop_faces_from_frame
from src.model import CNNFeatureExtractor, CNN_LSTM_Attention


# ---------------- FILE VALIDATION ---------------- #

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- APP SETUP ---------------- #

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret123'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

db = firestore.Client()


# ---------------- USER ---------------- #

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password


@login_manager.user_loader
def load_user(user_id):
    doc = db.collection("users").document(user_id).get()
    if doc.exists:
        data = doc.to_dict()
        return User(user_id, data["username"], data["password"])
    return None


# ---------------- MODEL ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pth")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = 'cpu'

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

feat_extractor = CNNFeatureExtractor().to(device)
model = CNN_LSTM_Attention(feat_dim=feat_extractor.out_dim).to(device)

checkpoint = torch.load(MODEL_PATH, map_location='cpu')
feat_extractor.load_state_dict(checkpoint['feat_state'])
model.load_state_dict(checkpoint['model_state'])

feat_extractor.eval()
model.eval()

print("✅ Model loaded correctly")


# ---------------- PREDICTION (OPTIMIZED) ---------------- #

def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []

    ok, frame = cap.read()
    idx = 0
    frame_skip = 3   # 🔥 optimization

    while ok and idx < 24:
        if idx % frame_skip == 0:
            faces = crop_faces_from_frame(frame)

            if len(faces) > 0:
                frames.append(faces[0])

        ok, frame = cap.read()
        idx += 1

    cap.release()

    if len(frames) < 2:
        return "NO FACE DETECTED", 0

    # 🔥 limit frames
    frames = frames[:6]

    xs = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(device)

    with torch.no_grad():
        B, S, C, H, W = xs.shape
        seqs = xs.view(B * S, C, H, W)

        feats = feat_extractor(seqs)
        feats = feats.view(B, S, -1)

        logits, _ = model(feats)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # 🔥 free memory
    del xs, feats, logits

    label = "FAKE" if probs[1] > probs[0] else "REAL"
    confidence = max(probs) * 100

    return label, round(confidence, 2)


# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        db.collection("users").document(username).set({
            "username": username,
            "password": hashed_password
        })

        flash("Registered Successfully")
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        doc = db.collection("users").document(username).get()

        if doc.exists:
            data = doc.to_dict()

            if check_password_hash(data["password"], password):
                user = User(username, username, password)
                login_user(user)
                return redirect(url_for('dashboard'))

        flash("Invalid credentials")

    return render_template('login.html')


@app.route('/dashboard', methods=['GET','POST'])
@login_required
def dashboard():
    if request.method == 'POST':

        # 🔥 size check
        if request.content_length > 50 * 1024 * 1024:
            flash("File too large!")
            return redirect(url_for('dashboard'))

        file = request.files['video']

        if file.filename == '':
            flash("No file selected")
            return redirect(url_for('dashboard'))

        if not allowed_file(file.filename):
            flash("Invalid file type. Only MP4, AVI, MOV allowed.")
            return redirect(url_for('dashboard'))

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 🔥 timeout handling
        start_time = time.time()

        label, confidence = predict_video(filepath)

        if time.time() - start_time > 20:
            flash("Processing too slow. Try shorter video.")
            return redirect(url_for('dashboard'))

        return render_template('result.html',
                               label=label,
                               confidence=confidence)

    return render_template('dashboard.html', username=current_user.username)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/health')
def health():
    return "OK", 200


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
