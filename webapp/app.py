import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

import torch
import cv2
import tempfile
import torchvision.transforms as T

from src.utils.face_detect import crop_faces_from_frame
from src.model import CNNFeatureExtractor, CNN_LSTM_Attention


# ---------------- PATH SETUP ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pth")  # use best.pth

# ---------------- APP SETUP ---------------- #

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/users.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ---------------- DATABASE ---------------- #

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------- MODEL SETUP ---------------- #

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

feat_extractor = CNNFeatureExtractor().to(device)
model = CNN_LSTM_Attention(feat_dim=feat_extractor.out_dim).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
feat_extractor.load_state_dict(checkpoint['feat_state'])
model.load_state_dict(checkpoint['model_state'])

feat_extractor.eval()
model.eval()

print("✅ Model loaded correctly")


# ---------------- PREDICTION FUNCTION ---------------- #

def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []

    ok, frame = cap.read()
    idx = 0

    while ok and idx < 16:
        faces = crop_faces_from_frame(frame)

        if len(faces) > 0:
            pil_img = faces[0]  # take first detected face
            frames.append(pil_img)

        ok, frame = cap.read()
        idx += 1

    cap.release()

    if len(frames) < 2:
        return "NO FACE DETECTED", 0

    xs = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(device)

    with torch.no_grad():
        B,S,C,H,W = xs.shape
        seqs = xs.view(B*S, C, H, W)

        feats = feat_extractor(seqs)
        feats = feats.view(B, S, -1)

        logits, _ = model(feats)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # SAME LOGIC AS STREAMLIT
    label = "FAKE" if probs[1] > probs[0] else "REAL"
    confidence = max(probs) * 100

    return label, round(confidence, 2)


# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        user = User(
            username=request.form['username'],
            password=request.form['password']
        )
        db.session.add(user)
        db.session.commit()
        flash("Registered Successfully")
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(
            username=request.form['username'],
            password=request.form['password']
        ).first()

        if user:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials")

    return render_template('login.html')


@app.route('/dashboard', methods=['GET','POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        file = request.files['video']

        if file.filename == '':
            flash("No file selected")
            return redirect(url_for('dashboard'))

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        label, confidence = predict_video(filepath)

        return render_template('result.html',
                               label=label,
                               confidence=confidence)

    return render_template('dashboard.html', username=current_user.username)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ---------------- RUN ---------------- #

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    with app.app_context():
        print("Creating DB...")
        db.create_all()
        print("DB created")

    app.run(debug=True)
