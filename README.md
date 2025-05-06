🧬 Leukemia Classification System using Deep Learning
A deep learning-based web application that classifies blood smear images as Leukemia or Healthy using a fine-tuned ResNet18 CNN model. The system is built with Flask and provides an easy-to-use UI for real-time prediction and visualization.

📌 Highlights
✅ 99.65% Accuracy in classifying leukemia vs. healthy cells

🧠 Powered by ResNet18 pretrained CNN

🌐 Simple Flask Web Interface for uploading and predicting images

🔍 Includes Grad-CAM visualization for model interpretability

⚙️ Built using PyTorch, Flask, HTML/CSS/JS

📁 Project Structure


app.py               # Flask web app backend
model.pth            # Trained ResNet18 model
static/              # Static files (CSS, JS, images)
templates/           # HTML templates
utils.py             # Image preprocessing and Grad-CAM logic
requirements.txt     # Python dependencies


🧠 Technologies Used
Component                   Technology
Framework	                    Flask
Deep Learning	                PyTorch
Model	                        ResNet18
Visualization               	Grad-CAM
Frontend	                  HTML, CSS, JS


📊 Dataset Info
Total Images: 13,000

-->Training:

7,000 images of ALL (Leukemia)

3,000 images of HEM (Healthy)

-->Testing:

2,000 images of ALL

1,000 images of HEM

🔧 Preprocessing:
Resize to 224x224

Normalize using ImageNet mean & std

Augmentations: rotation, flip, contrast adjustment

🏗️ Model Details
🔹 Architecture: ResNet18 (fine-tuned)

🔹 Loss Function: CrossEntropyLoss

🔹 Optimizer: Adam

🔹 Learning Rate: 0.0001

🔹 Batch Size: 32

🔹 Epochs: 50

✅ Performance:
Metric	             Value
Accuracy       -   	99.65%
Precision	     -     99.65%
Recall	       -     99.65%
F1-Score       -   	99.65%

-->Application Workflow
Upload a blood smear image from your device.

The model processes the image and predicts:
🔴 Leukemia or 🟢 Healthy

View Grad-CAM heatmap to understand which regions influenced the prediction.

Developed with ❤️ using Deep Learning & Flask
For academic and research purposes.
