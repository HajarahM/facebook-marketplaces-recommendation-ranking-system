import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
from image_processor import ProcessImage
from text_processor import TextProcessor

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

class TextClassifier(nn.Module):
    def __init__(self,
                ngpu,
                input_size: int = 768,
                decoder: dict = None):
        super(TextClassifier, self).__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(torch.nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.MaxPool1d(kernel_size=2, stride=2),
                                  torch.nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.MaxPool1d(kernel_size=2, stride=2),
                                  torch.nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.MaxPool1d(kernel_size=2, stride=2),
                                  torch.nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(192 , 32))        
        self.decoder = decoder

    def forward(self, text):
        x = self.main(text)
        return x

    def predict(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return x
    
    def predict_proba(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return torch.softmax(x, dim=1)


    def predict_classes(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return self.decoder[int(torch.argmax(x, dim=1))]

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


class CombinedModel(nn.Module):
    def __init__(self, ngpu, input_size = 768, num_classes: int=2, decoder: list = None):
        super(CombinedModel, self).__init__()
        self.ngpu = ngpu
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # define layers        
        output_features = self.resnet50.fc.out_features
        self.image_classifier = torch.nn.Sequential(self.resnet50, torch.nn.Linear(output_features, num_classes)).to(device)
        self.text_classifier = TextClassifier(ngpu=ngpu, input_size=input_size)
        self.main = torch.nn.Sequential(torch.nn.Linear(160, 32))             
        self.decoder = decoder

    def forward(self, image_features, text_features):
        image_features = self.image_classifier(image_features)
        text_features = self.text_classifier(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        x = self.main(combined_features)
        return x

    def predict(self, image_features, text_features):
        with torch.no_grad():
            x = self.forward(image_features, text_features)
            return x
    
    def predict_proba(self, image_features, text_features):
        with torch.no_grad():
            x = self.forward(image_features, text_features)
            return torch.softmax(x, dim=1)

    def predict_classes(self, image_features, text_features):
        with torch.no_grad():
            x = self.forward(image_features, text_features)
            return self.decoder[int(torch.argmax(x, dim=1))]

# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str

try:
    text_decoder = pickle.load(open('text_decoder.pkl', 'rb'))
    n_classes = len(text_decoder)
    text_model = torch.load('./final_models/text_model.pt')
    text_classifier = TextClassifier(decoder=text_decoder)
    text_classifier.load_state_dict(text_model)
except:
    raise OSError("No Text model found. Check that you have the decoder and the model in the correct location")

try:
    image_decoder = pickle.load(open('image_decoder.pkl', 'rb'))
    n_classes = len(image_decoder)
    image_model = torch.load('final_models/image_model.pt', 'rb')
    image_classifier = ImageClassifier(num_classes=n_classes, decoder=image_decoder)
    image_classifier.load_state_dict(image_model)
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
    combined_decoder = pickle.load(open('combined_decoder.pkl', 'rb'))
    n_classes = len(combined_decoder)
    combined_model = torch.load('final_models/combined_model.pt', 'rb')
    combined_classifier = CombinedModel(ngpu=1, input_size=768, num_classes=n_classes, decoder=combined_decoder)
    combined_classifier.load_state_dict(combined_model)
except:
    raise OSError("No Combined model found. Check that you have the encoder and the model in the correct location")

try:
    text_processor = TextProcessor(50)
except:
    raise OSError("No Text processor found. Check that you have the encoder and the model in the correct location")

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

@app.post('/predict/text')
def predict_text(text: TextItem):
    processed_text = text_processor(text.text)
    prediction = text_classifier.predict(processed_text)
    probs = text_classifier.predict_proba(processed_text)
    classes = text_classifier.predict_classes(processed_text)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(content={
        'Category': prediction, 
        'Probabilities': probs.tolist(), 
        'classes': classes})
    
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
        'Category': prediction, 
        'Probabilities': probs.tolist(), 
        'classes': classes})
  
@app.post('/predict/combined')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    processed_image = image_processor(pil_image)
    prediction = combined_classifier.predict(processed_image, text)
    probs = combined_classifier.predict_proba(processed_image, text)
    classes = combined_classifier.predict_classes(processed_image, text)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(content={
        'Category': prediction, 
        'Probabilities': probs.tolist()})
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)