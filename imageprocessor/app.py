from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import os
import torch
from torchvision import models, transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

def advanced_remove_background(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    mask = output.argmax(0).byte().cpu().numpy()
    mask_image = Image.fromarray(mask * 255)
    
    mask_image = mask_image.convert("L")
    processed = Image.composite(image, Image.new("RGB", image.size), mask_image)
    return processed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    original_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(original_path)

    processed_image = advanced_remove_background(original_path)
    
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + file.filename)
    processed_image.save(processed_path)

    return render_template('result.html', original_image=file.filename, processed_image='processed_' + file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
