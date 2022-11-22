import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
from pydantic import BaseModel
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
                                  torch.nn.Linear(192 , 32),
                                  nn.ReLU(),
                                  nn.Linear(32, 13))        
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

# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str

try:
    text_decoder = pickle.load(open('text_decoder.pkl', 'rb'))
    n_classes = len(text_decoder)
    text_model = torch.load('text_model.pt')
    text_classifier = TextClassifier(ngpu=2, decoder=text_decoder)
    text_classifier.load_state_dict(text_model)
except:
    raise OSError("No Text model found. Check that you have the decoder and the model in the correct location")

try:
    text_processor = TextProcessor(50)
except:
    raise OSError("No Text processor found. Check that you have the encoder and the model in the correct location")

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
        'Category': classes, 
        'Probabilities': probs.tolist()})
    
if __name__ == '__main__':
  uvicorn.run("api_text:app", host="127.0.0.1", port=8080)