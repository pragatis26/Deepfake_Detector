import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from models.detector import DeepFakeDetector
from utils.video_utils import extract_frames
from utils.gradcam import generate_gradcam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max
app.config['SECRET_KEY'] = 'deepfake-detector-fyp-2024'

ALLOWED_IMAGE_EXT = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_VIDEO_EXT = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model once at startup
detector = DeepFakeDetector()

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXT):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG.'}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = detector.predict_image(filepath)
        heatmap_filename = None

        if result['face_detected']:
            heatmap_filename = f"heatmap_{filename}"
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
            generate_gradcam(detector.model, filepath, heatmap_path)

        return jsonify({
            'success': True,
            'label': result['label'],
            'confidence': result['confidence'],
            'face_detected': result['face_detected'],
            'image_url': url_for('static', filename=f'uploads/{filename}'),
            'heatmap_url': url_for('static', filename=f'uploads/{heatmap_filename}') if heatmap_filename else None,
            'details': result.get('details', {})
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not allowed_file(file.filename, ALLOWED_VIDEO_EXT):
        return jsonify({'error': 'Invalid file type. Use MP4, AVI, or MOV.'}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        frames = extract_frames(filepath, max_frames=20)
        if not frames:
            return jsonify({'error': 'Could not extract frames from video.'}), 400

        predictions = []
        for frame_path in frames:
            r = detector.predict_image(frame_path)
            if r['face_detected']:
                predictions.append(r['confidence_raw'])

        if not predictions:
            return jsonify({
                'success': True,
                'label': 'UNKNOWN',
                'confidence': 0,
                'face_detected': False,
                'frames_analyzed': len(frames),
                'frames_with_faces': 0,
                'message': 'No faces found in any frame.'
            })

        avg_conf = float(np.mean(predictions))
        label = 'FAKE' if avg_conf > 0.5 else 'REAL'
        confidence = avg_conf if label == 'FAKE' else (1 - avg_conf)

        # Clean up frame files
        for fp in frames:
            if os.path.exists(fp):
                os.remove(fp)

        return jsonify({
            'success': True,
            'label': label,
            'confidence': round(confidence * 100, 2),
            'face_detected': True,
            'frames_analyzed': len(frames),
            'frames_with_faces': len(predictions),
            'per_frame_scores': [round(p * 100, 2) for p in predictions]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
