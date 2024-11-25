import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.metadata["vocab"])[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    model = ImageToWordModel(model_path="Output_200 epochs/Models/IAM_Training/202410270542/model.onnx")

    accum_cer = []
    
    image_path = "C:/Users/Shyam/Downloads/arn.jpg"
    
    image = cv2.imread(image_path.replace("\\", "/"))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert image to greyscale
    image = np.stack((image,) * 3, axis=-1) # Expand dims for user input images
    print(image.shape)
    

    prediction_text = model.predict(image)
    print(f"Prediction: {prediction_text}")

    cv2.imshow('Image Window', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()