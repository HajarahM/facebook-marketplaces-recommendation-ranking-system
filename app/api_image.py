import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
import torch
import torch.nn as nn
from pydantic import BaseModel
from image_processor import ProcessImage

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

class ImageClassifier(nn.Module):
    def __init__(self, 
                 num_classes,
                 decoder: dict = None):
        super(ImageClassifier, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # num_classes = 13
        # define layers        
        output_features = self.resnet50.fc.out_features
        self.linear = torch.nn.Linear(output_features, num_classes).to(device)
        self.main = torch.nn.Sequential(self.resnet50, self.linear).to(device)
        self.decoder = decoder

    def forward(self, image):
        x = self.main(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

    def predict_proba(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return torch.softmax(x, dim=1)

    def predict_classes(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return self.decoder[int(torch.argmax(x, dim=1))]

try:
    image_decoder = pickle.load(open('image_decoder.pkl', 'rb'))
    n_classes = len(image_decoder)
    image_model = torch.load('image_model.pt')
    image_classifier = ImageClassifier(num_classes=n_classes, decoder=image_decoder)
    image_classifier.load_state_dict(image_model)
except OSError: 
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
    image_processor = ProcessImage()
except:
    raise OSError("No Image processor found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"  
  return {"message": msg}
    
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    processed_image = image_processor(pil_image)
    prediction = image_classifier.predict(processed_image)
    probs = image_classifier.predict_proba(processed_image)
    classes = image_classifier.predict_classes(processed_image)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(content={
        'Category': classes, 
        'Probabilities': probs.tolist()})
    
if __name__ == '__main__':
  uvicorn.run("api_image:app", host="127.0.0.1", port=8080)