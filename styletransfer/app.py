#!/usr/bin/env python

import os
import sys

from flask import Flask, request, jsonify, send_file, render_template
from io import BytesIO
from PIL import Image, ImageOps
import base64
import urllib

import numpy as np
import scipy.misc

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transform import TransformerNet
from vgg import Vgg16

app = Flask(__name__)

models = {
    'candy' : './model/candy.pth',
    'mosaic' : './model/mosaic.pth',
    'rain_princess' : './model/rain_princess.pth',
    'udnie' : './model/udnie.pth',
}

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/upload", methods=['POST'])
def upload_file():

    try:
        img = Image.open(BytesIO(request.files['imagefile'].read())).convert('RGB')
        img = ImageOps.fit(img, (400, 400), Image.ANTIALIAS)
    except:
        error_msg = "Please choose an image file!"
        return render_template('index.html', **locals())

    try:
        style = request.form['options']
    except:
        error_msg = "Please select a style to apply!"
        return render_template('index.html', **locals())

    model = models[style]
    args = {'content_image' : img, 'model' : model}
    out_img, content_img = stylize(args)

    img_io = BytesIO()
    out_img.save(img_io, 'PNG')

    c_img_io = BytesIO()
    content_img.save(c_img_io, 'PNG')

    png_output = base64.b64encode(img_io.getvalue())
    processed_file = urllib.parse.quote(png_output)

    c_png_output = base64.b64encode(c_img_io.getvalue())
    c_processed_file = urllib.parse.quote(c_png_output)

    return render_template('result.html', **locals())

@app.route("/api/styles", methods=['GET'])
def return_styles():
    return jsonify(models)

@app.route("/api/upload/<style>", methods=['POST'])
def api_upload_file(style):
    img = Image.open(BytesIO(request.files['imagefile'].read())).convert('RGB')
    if min(img.size) > 400:
        img = ImageOps.fit(img, (400, 400), Image.ANTIALIAS)

    model = models[style]

    args = {'content_image' : img, 'model' : model}
    out_img = stylize(args)

    return send_pil(out_img)

def send_pil(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=90)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

def stylize(args):
    content_image = utils.load_image(args['content_image'])
    content_img = content_image.copy()

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)

    if torch.cuda.is_available():
        content_image = content_image.cuda()

    content_image = Variable(content_image, volatile=True)

    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(args['model']))

    if torch.cuda.is_available():
        style_model.cuda()

    output = style_model(content_image)

    if torch.cuda.is_available():
        output = output.cpu()

    output_data = output.data[0]
    return utils.save_image(output_data), content_img

if __name__ == '__main__':

    # HOST = os.environ.get('SERVER_HOST', 'localhost')
    # try:
    #     PORT = int(os.environ.get('SERVER_PORT', '5555'))
    # except ValueError:
    #     PORT = 5555

    app.run(host='0.0.0.0', port=5000)
