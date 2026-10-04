import joblib # to get the model
import numpy as np
from PIL import Image # manipulate images


class EmotionClassifier:
    def __init__(self, model_path='rf_pipeline_2.joblib'):
        # load model and classes
        self.model=joblib.load(model_path)
        self.class_names=['angry', 'happy', 'neutral', 'sad']
        
    def predict_expression(self, image_path):
        
        img=Image.open(image_path).convert('L') # convert to RGB

        img_resize=img.resize((48,48))

        #flatten the image and convert to numpy array
        img_flat=np.array(img_resize).flatten().reshape(1,-1)
        
        #Run prediction, 0 to get the very first prediction the model gives
        prediction_index=self.model.predict(img_flat)[0]
        
        #Use number model gave to get the corresponding class name
        prediction_label=self.class_names[prediction_index]
        
        return prediction_label
    
    def get_result(self, image_path):
        return self.predict_expression(image_path)
    
    def prediction_percentage(self, image_path):
        img=Image.open(image_path).convert('L') # convert to RGB

        img_resize=img.resize((48,48))

        #flatten the image and convert to numpy array
        img_flat=np.array(img_resize).flatten().reshape(1,-1)
        
        #Run prediction, 0 to get the very first prediction the model gives
        prediction_probabilities=self.model.predict_proba(img_flat)[0]
        
        
        prediction_index = np.argmax(prediction_probabilities)
        scores = {
            name: f"{prob * 100:.1f}%"
            for name, prob in zip(self.class_names, prediction_probabilities)
        }

        return self.class_names[prediction_index], scores
    
# Test on an image file

# classifier=EmotionClassifier()
# class_result=classifier.get_result('images/neutral_img_2.png')
# print(class_result)
# prob_result=classifier.prediction_percentage('images/neutral_img_2.png')
# print(prob_result)