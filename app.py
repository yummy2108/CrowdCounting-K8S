import io 
import flask 
import os
import numpy as np
import torch
import argparse

from model import SASNet
import warnings
import random
import cv2
from torchvision import datasets, transforms
from PIL import Image
warnings.filterwarnings('ignore')
tensor_transform = transforms.ToTensor()
img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# define the GPU id to be used
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    # define the argparse for the script
    parser = argparse.ArgumentParser('Inference setting', add_help=False)
    parser.add_argument('--model_path', type=str, help='path of pre-trained model')
    parser.add_argument('--data_path', type=str, help='root path of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--log_para', type=int, default=1000, help='magnify the target density map')
    parser.add_argument('--block_size', type=int, default=32, help='patch size for feature level selection')

    return parser.parse_args()
app = flask.Flask(__name__) 
@app.route("/")
def hello_world():
    return "Hello, World!"
@app.route("/predict",methods=["POST"]) 
def predict(): 
    if flask.request.method == "POST": 
        if flask.request.files.get("img"): 
            image = Image.open(io.BytesIO(flask.request.files["img"].read())).convert('RGB')
            image = tensor_transform(image)
            image = img_transform(image).unsqueeze(0) 
            print(image.shape)
            model = SASNet(args=get_args())
            # load the trained model
            model.load_state_dict(torch.load("./models/SHHB.pth", map_location='cpu'))
            print("load model success")
            model.eval()
            print("model eval success")
            pred_map = model(image)
            print("model(image) success")
            pred_map = pred_map.data.cpu().numpy()
            print("model eval completed")
            # evaluation over the batch
            for i_img in range(pred_map.shape[0]):
                pred_cnt = np.sum(pred_map[i_img]) / 1000
                print(pred_cnt)
            return str(pred_cnt)
        else: 
            return "Image Upload Error"
if __name__ == '__main__': 
    app.run(host='0.0.0.0') 