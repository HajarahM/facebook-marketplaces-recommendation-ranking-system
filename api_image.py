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
import image_processor
import text_processor

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
            pass


    def predict_classes(self, text):
        with torch.no_grad():
            pass

class ImageClassifier(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(ImageClassifier, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        num_classes = 13
        # define layers        
        output_features = self.resnet50.fc.out_features
        self.linear = torch.nn.Linear(output_features, num_classes).to(device)
        self.main = torch.nn.Sequential(self.resnet50, self.linear)
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
            pass

    def predict_classes(self, image):
        with torch.no_grad():
            pass


class CombinedModel(nn.Module):
    def __init__(self, ngpu, input_size = 768, num_classes: int=2, decoder: list = None):
        super(CombinedModel, self).__init__()
        self.ngpu = ngpu
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # define layers        
        output_features = self.resnet50.fc.out_features
        self.image_classifier = torch.nn.Sequential(self.resnet50, torch.nn.Linear(output_features, 128)).to(device)
        self.text_classifier = TextClassifier(ngpu=ngpu, input_size=input_size)
        self.main = torch.nn.Sequential(torch.nn.Linear(160, 32))             
        self.decoder = decoder

    def forward(self, image_features, text_features):
        pass

    def predict(self, image_features, text_features):
        with torch.no_grad():
            combined_features = self.forward(image_features, text_features)
            return combined_features
    
    def predict_proba(self, image_features, text_features):
        with torch.no_grad():
            pass

    def predict_classes(self, image_features, text_features):
        with torch.no_grad():
            pass

# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str


try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the text model    #
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# text_decoder.pkl                                           #
##############################################################
    pass
except:
    raise OSError("No Text model found. Check that you have the decoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the image model   #
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# image_decoder.pkl                                          #
##############################################################
    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the combined model#
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# combined_decoder.pkl                                       #
##############################################################
    pass
except:
    raise OSError("No Combined model found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Initialize the text processor that you will use to process #
# the text that you users will send to your API.             #
# Make sure that the max_length you use is the same you used #
# when you trained the model. If you used two different      #
# lengths for the Text and the Combined model, initialize two#
# text processors, one for each model                        #
##############################################################
    pass
except:
    raise OSError("No Text processor found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Initialize the image processor that you will use to process#
# the text that you users will send to your API              #
##############################################################
    pass
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
  
    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the text model   #
    # text.text is the text that the user sent to your API       #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
        "Category": "", # Return the category here
        "Probabilities": "" # Return a list or dict of probabilities here
            })
  
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)

    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
    "Category": "", # Return the category here
    "Probabilities": "" # Return a list or dict of probabilities here
        })
  
@app.post('/predict/combined')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    
    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # In this case, text is the text that the user sent to your  #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
    "Category": "", # Return the category here
    "Probabilities": "" # Return a list or dict of probabilities here
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)