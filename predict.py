import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

class LeukemiaClassifier:
    def __init__(self, model_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        """Load the trained ResNet model"""
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 4),
            nn.Linear(4, len(self.class_names))
        )
        
        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path):
        """Make prediction on a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                _, pred_idx = torch.max(probabilities, 0)
            
            # Get probabilities for both classes
            all_prob = float(probabilities[0])
            hem_prob = float(probabilities[1])
            confidence = max(all_prob, hem_prob)
            
            # Determine if prediction is confident
            is_confident = confidence > 0.7
            
            return {
                'class': self.class_names[pred_idx.item()],
                'probability': confidence,
                'is_confident': is_confident,
                'all_probabilities': [all_prob, hem_prob]
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'is_valid': False
            }