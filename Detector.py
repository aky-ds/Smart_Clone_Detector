import os
import uuid
import numpy as np
import cv2
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import joblib
from utils import extract_features

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
video_model = load_model('models/m2.h5')
voice_model = joblib.load("models/lr.pkl")

IMG_SIZE = (128, 128)

# ----- VIDEO CLONING -----
def preprocess_frame(frame):
    frame = cv2.resize(frame, IMG_SIZE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype('float32') / 255.0
    return frame

def process_video(filepath):
    cap = cv2.VideoCapture(filepath)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
    cap.release()

    if not frames:
        return "No frames found", 0.0

    input_data = np.array(frames)
    preds = video_model.predict(input_data)
    avg_pred = np.mean(preds)
    label = "No Cloning Detected" if avg_pred >= 0.5 else "Cloning Detected"
    confidence = avg_pred if avg_pred >= 0.5 else 1 - avg_pred
    return label, float(confidence)

# ----- VOICE CLONING -----
def process_audio(filepath):
    features = extract_features(filepath).reshape(1, -1)
    prediction = voice_model.predict(features)[0]
    confidence = voice_model.predict_proba(features)[0][prediction]
    label = "Real Voice" if prediction == 0 else "Cloned Voice"
    return label, float(confidence)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    video_label, video_conf = None, None
    voice_label, voice_conf = None, None
    video_file = None
    audio_file = None

    if 'video' in request.files and request.files['video'].filename:
        file = request.files['video']
        video_filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        file.save(video_path)
        video_file = f"uploads/{video_filename}"
        video_label, video_conf = process_video(video_path)

    if 'audio' in request.files and request.files['audio'].filename:
        file = request.files['audio']
        audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        file.save(audio_path)
        audio_file = f"uploads/{audio_filename}"
        voice_label, voice_conf = process_audio(audio_path)

    return render_template(
        'result.html',
        video_label=video_label,
        video_conf=video_conf,
        video_file=video_file,
        voice_label=voice_label,
        voice_conf=voice_conf,
        audio_file=audio_file
    )

@app.route('/record', methods=['POST'])
def record_audio():
    duration = 5
    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    filename = f"{uuid.uuid4()}.wav"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    write(audio_path, fs, recording)
    voice_label, voice_conf = process_audio(audio_path)
    return render_template(
        'result.html',
        voice_label=voice_label,
        voice_conf=voice_conf,
        audio_file=f"uploads/{filename}"
    )

if __name__ == '__main__':
    app.run(debug=True)
