import os
import json
import time as tm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from flask import Flask, request, render_template, redirect, url_for, session, send_file
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Enforce TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define custom metrics
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + tf.keras.backend.epsilon()) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + tf.keras.backend.epsilon())

# Register the custom metric
get_custom_objects().update({"dice_coefficient": dice_coefficient})

# Paths to models
classification_model_path = 'breast_ultrasound_classification_model.h5'
segmentation_model_path = 'unet_final_enhanced_lr.h5'  # Ensure this file exists and is correct

# Load the models
classification_model = load_model(classification_model_path)
segmentation_model = load_model(segmentation_model_path, compile=False)

# Recompile the segmentation model with custom metrics
segmentation_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient])

# Assuming the classification model outputs three probabilities for 'benign', 'malignant', 'normal'
class_names = ['benign', 'malignant', 'normal']

# Path to the JSON file for storing patient history
history_file_path = 'static/uploads/history.json'

def load_history():
    if os.path.exists(history_file_path):
        try:
            with open(history_file_path, 'r') as file:
                return json.load(file)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading history: {e}")
            return []
    return []

def save_history(history):
    # Convert float32 to float for JSON serialization
    for record in history:
        if isinstance(record['accuracy'], np.float32):
            record['accuracy'] = float(record['accuracy'])
        if isinstance(record['time'], np.float32):
            record['time'] = float(record['time'])
    with open(history_file_path, 'w') as file:
        json.dump(history, file, indent=4)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    username = request.form.get('username')
    password = request.form.get('password')
    # Simple authentication
    if username == 'admin' and password == 'password':
        session['user'] = username
        return redirect(url_for('main_menu'))
    else:
        return render_template('login.html', error='Invalid Credentials')

@app.route('/main_menu')
def main_menu():
    if 'user' in session:
        return render_template('main_menu.html')
    return redirect(url_for('login'))

@app.route('/new_patient', methods=['GET', 'POST'])
def new_patient():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        session['patient_name'] = request.form.get('patient_name')
        session['patient_id'] = request.form.get('patient_id')
        session['dob'] = request.form.get('dob')
        session['age'] = request.form.get('age')
        session['gender'] = request.form.get('gender')
        return redirect(url_for('upload'))

    return render_template('new_patient.html')

@app.route('/history')
def history():
    if 'user' in session:
        history = load_history()
        return render_template('history.html', history=history, enumerate=enumerate)
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)  # Redirect to the same page if no file is uploaded

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)  # Redirect on empty filename

        if file:
            try:
                start_time = tm.time()
                
                img = Image.open(file).convert('RGB').resize((128, 128))  # Ensure 3 channels (RGB)
                img = np.array(img) / 255.0  # Normalize the image
                img = np.expand_dims(img, axis=0)  # Add batch dimension

                # Classification
                classification_result = classification_model.predict(img)
                print(f"Classification probabilities: {classification_result}")

                # Get the class with the highest probability
                classification_index = np.argmax(classification_result)
                classification = class_names[classification_index]
                accuracy = float(classification_result[0][classification_index])
                end_time = tm.time()
                elapsed_time = end_time - start_time
                
                print(f"Classification: {classification}")
                print(f"Accuracy: {accuracy * 100:.2f}%")
                print(f"Time taken: {elapsed_time:.2f}s")

                # Segmentation
                segmentation_result = segmentation_model.predict(img)
                print(f"Segmentation result shape: {segmentation_result.shape}")

                # Ensure the directory exists
                os.makedirs('static/uploads', exist_ok=True)

                # Check segmentation result shape (assuming single-channel output)
                if len(segmentation_result.shape) == 4 and segmentation_result.shape[1:] == (128, 128, 1):
                    segmentation_image = (segmentation_result[0, :, :, 0] * 255).astype(np.uint8)

                    # Plot input image and predicted mask side by side
                    fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Reduced image size
                    axes[0].imshow(img[0], cmap='gray')
                    axes[0].set_title('Input Image')
                    axes[0].axis('off')

                    axes[1].imshow(segmentation_image, cmap='gray')
                    axes[1].set_title('Predicted Mask')
                    axes[1].axis('off')

                    side_by_side_path = 'static/uploads/side_by_side.png'
                    fig.savefig(side_by_side_path)
                    plt.close(fig)

                    # Save report in session
                    session['report'] = {
                        'patient_name': session.get('patient_name'),
                        'patient_id': session.get('patient_id'),
                        'dob': session.get('dob'),
                        'age': session.get('age'),
                        'gender': session.get('gender'),
                        'classification': classification,
                        'accuracy': round(accuracy * 100, 2),
                        'time': round(elapsed_time, 2),
                        'image_file': 'uploads/side_by_side.png'
                    }

                    # Save to history
                    history = load_history()
                    history.append(session['report'])
                    save_history(history)

                    return render_template(
                        'result.html',
                        classification=classification,
                        accuracy=round(accuracy * 100, 2),
                        time=round(elapsed_time, 2),
                        image_file='uploads/side_by_side.png',
                        patient_name=session.get('patient_name'),
                        patient_id=session.get('patient_id'),
                        dob=session.get('dob'),
                        age=session.get('age'),
                        gender=session.get('gender')
                    )
                else:
                    return render_template('result.html', classification=classification, image_file=None)

            except (IOError, OSError) as e:  # Handle potential image opening errors
                print(f"Error processing image: {e}")
                return redirect(request.url)  # Redirect on error

    return render_template('upload.html')

@app.route('/export', methods=['GET', 'POST'])
def export():
    if 'user' not in session or 'report' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        export_format = request.form.get('format')
        report = session['report']
        
        if export_format == 'pdf':
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt="Patient Report", ln=True, align='C')
            pdf.cell(200, 10, txt="", ln=True, align='C')
            
            pdf.cell(200, 10, txt=f"Patient Name: {report['patient_name']}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Patient ID: {report['patient_id']}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Date of Birth: {report['dob']}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Age: {report['age']}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Gender: {report['gender']}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Classification: {report['classification']}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Accuracy: {report['accuracy']:.2f}%", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Time taken: {report['time']:.2f}s", ln=True, align='L')

            pdf.image(f"static/{report['image_file']}", x=10, y=None, w=100)

            pdf_output = f"static/uploads/report_{report['patient_id']}.pdf"
            pdf.output(pdf_output)

            return send_file(pdf_output, as_attachment=True)

        elif export_format == 'txt':
            txt_output = f"static/uploads/report_{report['patient_id']}.txt"
            with open(txt_output, 'w') as f:
                f.write("Patient Report\n\n")
                f.write(f"Patient Name: {report['patient_name']}\n")
                f.write(f"Patient ID: {report['patient_id']}\n")
                f.write(f"Date of Birth: {report['dob']}\n")
                f.write(f"Age: {report['age']}\n")
                f.write(f"Gender: {report['gender']}\n")
                f.write(f"Classification: {report['classification']}\n")
                f.write(f"Accuracy: {report['accuracy']:.2f}%\n")
                f.write(f"Time taken: {report['time']:.2f}s\n")
                f.write(f"Image File: {report['image_file']}\n")

            return send_file(txt_output, as_attachment=True)

    return render_template('export.html')

@app.route('/export_history', methods=['POST'])
def export_history():
    if 'user' not in session:
        return redirect(url_for('login'))

    patient_id = request.form.get('patient_id')
    export_format = request.form.get('format')
    history = load_history()
    report = next((item for item in history if item['patient_id'] == patient_id), None)

    if report:
        if export_format == 'pdf':
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt="Patient Report", ln=True, align='C')
            pdf.cell(200, 10, txt="", ln=True, align='C')
            
            pdf.cell(200, 10, txt=f"Patient Name: {report['patient_name']}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Patient ID: {report['patient_id']}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Date of Birth: {report['dob']}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Age: {report['age']}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Gender: {report['gender']}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Classification: {report['classification']}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Accuracy: {report['accuracy']:.2f}%", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Time taken: {report['time']:.2f}s", ln=True, align='L')

            pdf.image(f"static/{report['image_file']}", x=10, y=None, w=100)

            pdf_output = f"static/uploads/report_{report['patient_id']}.pdf"
            pdf.output(pdf_output)

            return send_file(pdf_output, as_attachment=True)

    return redirect(url_for('history'))

@app.route('/delete_history', methods=['POST'])
def delete_history():
    if 'user' not in session:
        return redirect(url_for('login'))

    report_id = int(request.form.get('report_id'))
    history = load_history()
    if 0 <= report_id < len(history):
        del history[report_id]
        save_history(history)
    return redirect(url_for('history'))

if __name__ == "__main__":
    app.run(debug=True)
