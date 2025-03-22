import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Configure Upload Folder & Allowed Extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Model Definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load Trained Model
model = SimpleCNN()
model_path = 'C:\\Users\\LENOVO\\Desktop\\project\\save_model\\trained_model.pth'

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Classification Categories
CATEGORIES = ["Fetal Abdomen", "Fetal Brain", "Fetal Thorax", "Fetal Femur", "Maternal Cervix", "Others"]

# Image Validation
def is_fetal_ultrasound(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return False
        
        height, width = image.shape
        if height < 100 or width < 100:
            return False
        
        avg_intensity = np.mean(image)
        if avg_intensity < 30 or avg_intensity > 200:
            return False

        return True
    except Exception as e:
        print(f"Error in image validation: {e}")
        return False

# Flask Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predictions')
def predictions():
    return render_template('predictions.html')

@app.route('/evaluationmetrics')
def evaluationmetrics():
    return render_template('evaluationmetrics.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

@app.route('/prediction_result', methods=['POST'])
def prediction_result():
    try:
        if 'file' not in request.files:
            return render_template('error.html', message='No file uploaded.')
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return render_template('error.html', message='Invalid file format.')

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Validate if the uploaded image is a fetal ultrasound
        if not is_fetal_ultrasound(filepath):
            os.remove(filepath)
            return render_template('error.html', message='Uploaded image is not a fetal ultrasound.')

        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        # Perform Prediction
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        result = CATEGORIES[predicted_class]
        return render_template('prediction_result.html', result=result, image_file=filename)
    except Exception as e:
        return render_template('error.html', message=str(e))

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
