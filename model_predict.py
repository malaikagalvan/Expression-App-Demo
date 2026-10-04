import joblib
import numpy as np
from PIL import Image
from pathlib import Path

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class EmotionClassifier:
    def __init__(self, models_dir=None):
        base_dir = Path(__file__).resolve().parent
        
        if models_dir is None:
            models_dir = base_dir / "saved_models"
            if not models_dir.exists():
                models_dir = base_dir.parent / "saved_models"

        rf_path = models_dir / "random_forest_baseline.joblib"
        nn_path = models_dir / "nn_baseline.joblib"

        # 1. Load Model 1 (Random Forest)
        loaded_rf = joblib.load(rf_path)
        self.model_1 = loaded_rf['model'] if isinstance(loaded_rf, dict) and 'model' in loaded_rf else loaded_rf

        # 2. Load Model 2 (Custom NN Baseline)
        loaded_nn = joblib.load(nn_path)
        self.model_2 = loaded_nn['model'] if isinstance(loaded_nn, dict) and 'model' in loaded_nn else loaded_nn

        # Extract class names
        if isinstance(loaded_nn, dict) and 'class_names' in loaded_nn:
            self.class_names = loaded_nn['class_names']
        elif isinstance(loaded_rf, dict) and 'class_names' in loaded_rf:
            self.class_names = loaded_rf['class_names']
        else:
            self.class_names = ['angry', 'happy', 'neutral', 'sad']

    def predict_model_2(self, img_flat):
        """Handler for Custom Sequential NN and standard models."""
        # 1. Check for custom Sequential / forward pass
        if hasattr(self.model_2, "forward"):
            logits = self.model_2.forward(img_flat)
            probs = softmax(logits)[0]
            return probs

        # 2. Check for callable custom model
        if callable(self.model_2):
            try:
                logits = self.model_2(img_flat)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = softmax(np.array(logits))[0]
                return probs
            except Exception:
                pass

        # 3. Keras / TensorFlow Sequential
        if hasattr(self.model_2, "predict"):
            probs = self.model_2.predict(img_flat, verbose=0)[0]
            return probs

        # 4. Scikit-Learn MLPClassifier
        if hasattr(self.model_2, "predict_proba"):
            probs = self.model_2.predict_proba(img_flat)[0]
            return probs

        raise TypeError(f"Unknown prediction interface for model type: {type(self.model_2)}")

    def prediction_percentage(self, image_path):
        # 1. Load and resize image (48x48 Grayscale)
        img = Image.open(image_path).convert('L')
        img_resize = img.resize((48, 48))

        # 2. Input for Model 1 (Random Forest): Raw pixel values [0, 255]
        img_flat_rf = np.array(img_resize, dtype=np.float32).flatten().reshape(1, -1)

        # 3. Input for Model 2 (Neural Network): Scaled pixel values [0.0, 1.0]
        img_flat_nn = (np.array(img_resize, dtype=np.float32) / 255.0).flatten().reshape(1, -1)

        # Model 1 Predictions
        probs_1 = self.model_1.predict_proba(img_flat_rf)[0]
        top_idx_1 = int(np.argmax(probs_1))
        scores_1 = {name: f"{prob * 100:.1f}%" for name, prob in zip(self.class_names, probs_1)}

        # Model 2 Predictions (using img_flat_nn)
        probs_2 = self.predict_model_2(img_flat_nn)
        top_idx_2 = int(np.argmax(probs_2))
        scores_2 = {name: f"{prob * 100:.1f}%" for name, prob in zip(self.class_names, probs_2)}

        return (
            self.class_names[top_idx_1], 
            self.class_names[top_idx_2], 
            {"random_forest": scores_1, "neural_network": scores_2}
        )

    def get_result(self, image_path):
        return self.prediction_percentage(image_path)




# import joblib # to get the model
# import numpy as np
# from PIL import Image 
# from pathlib import Path


# class EmotionClassifier:
#     def __init__(self, model_path_1=None, model_path_2=None):
#         base_dir = Path(__file__).resolve().parent
        
#         if model_path_1 is None:
#             model_path_1 = base_dir / "models" / "random_forest_baseline.joblib"
#         if model_path_2 is None:
#             model_path_2 = base_dir / "models" / "nn_baseline.joblib"
        
#         # load model and classes
#         self.model_1=joblib.load(model_path_1)
#         self.model_2=joblib.load(model_path_2)
#         self.class_names=['angry', 'happy', 'neutral', 'sad']
        
#     def predict_expression(self, image_path):
        
#         img=Image.open(image_path).convert('L')
#         img_resize=img.resize((48,48))
#         img_flat=np.array(img_resize).flatten().reshape(1,-1)
        
#         prediction_index_1=self.model_1.predict(img_flat)[0]
#         prediction_label_1=self.class_names[prediction_index_1]
        
#         prediction_index_2=self.model_2.predict(img_flat)[0]
#         prediction_label_2=self.class_names[prediction_index_2]
        
#         return prediction_label_1, prediction_label_2
    
#     def prediction_percentage(self, image_path):
        
#         img=Image.open(image_path).convert('L')
#         img_resize=img.resize((48,48))
#         img_flat=np.array(img_resize).flatten().reshape(1,-1)
        
#         # Probabilities for Model 1
#         probs_1 = self.model_1.predict_proba(img_flat)[0]
#         top_idx_1 = int(np.argmax(probs_1))
#         scores_1 = {name: f"{prob * 100:.1f}%" for name, prob in zip(self.class_names, probs_1)}

#         # Probabilities for Model 2
#         probs_2 = self.model_2.predict_proba(img_flat)[0]
#         top_idx_2 = int(np.argmax(probs_2))
#         scores_2 = {name: f"{prob * 100:.1f}%" for name, prob in zip(self.class_names, probs_2)}

#         return (
#             self.class_names[top_idx_1], 
#             self.class_names[top_idx_2], 
#             {"random_forest": scores_1, "neural_network": scores_2}
#         )

#     def get_result(self, image_path):
#         # Everything can be done in a single image pass
#         return self.prediction_percentage(image_path)

#----------#
    
# import joblib
# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from pathlib import Path

# class EmotionClassifier:
#     def __init__(self, model_path_1=None, model_path_2=None):
#         base_dir = Path(__file__).resolve().parent
        
#         if model_path_1 is None:
#             model_path_1 = base_dir / "models" / "random_forest_baseline.joblib"
#         if model_path_2 is None:
#             model_path_2 = base_dir / "models" / "nn_baseline.joblib"

#         # Load Model 1 (Scikit-Learn Random Forest)
#         loaded_1 = joblib.load(model_path_1)
#         self.model_1 = loaded_1['model'] if isinstance(loaded_1, dict) and 'model' in loaded_1 else loaded_1

#         # Load Model 2 (PyTorch Sequential)
#         loaded_2 = joblib.load(model_path_2)
#         self.model_2 = loaded_2['model'] if isinstance(loaded_2, dict) and 'model' in loaded_2 else loaded_2
        
#         # Set PyTorch model to evaluation mode
#         if hasattr(self.model_2, "eval"):
#             self.model_2.eval()

#         # Extract class names
#         if isinstance(loaded_2, dict) and 'class_names' in loaded_2:
#             self.class_names = loaded_2['class_names']
#         else:
#             self.class_names = ['angry', 'happy', 'neutral', 'sad']

#     def prediction_percentage(self, image_path):
#         # 1. Preprocess Image
#         img = Image.open(image_path).convert('L')
#         img_resize = img.resize((48, 48))
        
#         # NumPy array for Scikit-Learn (shape: [1, 2304])
#         img_flat = np.array(img_resize, dtype=np.float32).flatten().reshape(1, -1)

#         # --- Model 1: Random Forest (Scikit-Learn) ---
#         probs_1 = self.model_1.predict_proba(img_flat)[0]
#         top_idx_1 = int(np.argmax(probs_1))
#         scores_1 = {name: f"{prob * 100:.1f}%" for name, prob in zip(self.class_names, probs_1)}

#         # --- Model 2: Neural Network (PyTorch Sequential) ---
#         # Convert NumPy array to PyTorch Tensor
#         img_tensor = torch.tensor(img_flat, dtype=torch.float32)

#         with torch.no_grad():
#             # Pass tensor directly through PyTorch sequential model
#             outputs = self.model_2(img_tensor)
            
#             # Apply Softmax to convert raw logits to probabilities
#             probs_2_tensor = F.softmax(outputs, dim=1)
#             probs_2 = probs_2_tensor.numpy()[0]

#         top_idx_2 = int(np.argmax(probs_2))
#         scores_2 = {name: f"{prob * 100:.1f}%" for name, prob in zip(self.class_names, probs_2)}

#         return (
#             self.class_names[top_idx_1], 
#             self.class_names[top_idx_2], 
#             {"random_forest": scores_1, "neural_network": scores_2}
#         )

#     def get_result(self, image_path):
#         return self.prediction_percentage(image_path)
   

# import sys
# import joblib
# import numpy as np
# from PIL import Image
# from pathlib import Path

# # Add the parent directory (Capstone I) to sys.path so Python can find `saved_models`
# base_dir = Path(__file__).resolve().parent
# parent_dir = base_dir.parent
# if str(parent_dir) not in sys.path:
#     sys.path.append(str(parent_dir))

# # Import team's custom loader and predict functions
# from models.load_nn_baseline import load_nn_baseline, predict

# class EmotionClassifier:
#     def __init__(self, model_path_1=None):
#         if model_path_1 is None:
#             model_path_1 = base_dir / "models" / "random_forest_baseline.joblib"

#         # Load Model 1 (Random Forest)
#         loaded_1 = joblib.load(model_path_1)
#         self.model_1 = loaded_1['model'] if isinstance(loaded_1, dict) and 'model' in loaded_1 else loaded_1

#         # Load Model 2 (Custom NN via Team's Loader)
#         saved_nn = load_nn_baseline()
#         self.model_2 = saved_nn["model"]
#         self.class_names = saved_nn["class_names"]

#     def prediction_percentage(self, image_path):
#         # Preprocess image
#         img = Image.open(image_path).convert('L')
#         img_resize = img.resize((48, 48))
#         img_flat = np.array(img_resize, dtype=np.float32).flatten().reshape(1, -1)

#         # Model 1 (Random Forest)
#         probs_1 = self.model_1.predict_proba(img_flat)[0]
#         top_idx_1 = int(np.argmax(probs_1))
#         scores_1 = {
#             name: f"{prob * 100:.1f}%" 
#             for name, prob in zip(self.class_names.values(), probs_1)
#         }

#         # Model 2 (Custom Neural Network)
#         predictions_2, probs_2_matrix = predict(self.model_2, img_flat)
#         probs_2 = probs_2_matrix[0]
#         top_idx_2 = int(predictions_2[0])

#         scores_2 = {
#             name: f"{prob * 100:.1f}%" 
#             for name, prob in zip(self.class_names.values(), probs_2)
#         }

#         top_label_1 = self.class_names[top_idx_1]
#         top_label_2 = self.class_names[top_idx_2]

#         return (
#             top_label_1, 
#             top_label_2, 
#             {"random_forest": scores_1, "neural_network": scores_2}
#         )

#     def get_result(self, image_path):
#         return self.prediction_percentage(image_path)
 
# Test on an image file

# classifier=EmotionClassifier()
# class_result=classifier.get_result('images/neutral_img_2.png')
# print(class_result)
# prob_result=classifier.prediction_percentage('images/neutral_img_2.png')
# print(prob_result)