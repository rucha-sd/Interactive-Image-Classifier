from flask import Flask
from keras.models import load_model
from flask import Flask, request,render_template, jsonify
from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import numpy as np
import base64
# import keras
import pandas as pd
import io
from PIL import Image

# from flask_paginate import Pagination, get_page_parameter
img_data = None
dataset_path = './CUB_200_2011'
attributes_path = dataset_path + '/attributes/attributes.txt'

api = Flask(__name__, template_folder='./templates')

attributes = pd.read_csv(dataset_path + '/attributes/attributes.txt', sep=" ", header=None)
attributes.columns = ['attribute_no', 'attribute_name']
bird_classes = pd.read_csv(dataset_path + "/classes.txt", sep=" ", header=None)
bird_classes.columns = ['class_no','class_names']


Model1 = load_model('./final/model.h5')
Model2 = load_model('./Model2latest.h5')
path = './image.jpg'
img_data = None

@api.route("/", methods=("POST", "GET"))
def index():
    img_data= None
    concepts ={}
    class_probs ={}
    data = {'concepts':concepts, 'class_probabilities':class_probs, 'img_data':img_data, 'table_display':"none"}
    # return render_template('index.html',concepts=concepts, class_probabilities=class_probs, img_data=img_data, table_display="none")
    return render_template('index.html', data=data)


@api.route('/getconcepts', methods=["POST"])
def getconcepts():
    global img_data
    print("In /api")
    print(request.files)
    image = request.files["img"]
    im = Image.open(image)
    img = im.convert('RGB')
    img = img.resize((224,224), Image.NEAREST)
    data = io.BytesIO()
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    img_data= encoded_img_data.decode('utf-8')

    # loaded_image = load_img(image, target_size=(224,224))
    # img_array = np.expand_dims(loaded_image, axis=0)

    predictions = Model1.predict(img_array)
    probabilities = Model2.predict(predictions)

    results = {}

    for i in range(len(attributes)):
        results[attributes.iloc[i]['attribute_no']] = { "attribute" : attributes.iloc[i]['attribute_name'] , "value":predictions[0][i]}

    class_probs = {}
    for i in range(len(probabilities[0])):
        class_probs[bird_classes.iloc[i]['class_names']] = probabilities[0][i]
    

    print("Results", results)

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]['value'], reverse=True)}
    print("Sorted results:", sorted_results)

    sorted_probs = {k: v for k, v in sorted(class_probs.items(), key=lambda item: item[1], reverse=True)}
    # print("Probabilities",class_probs)

    # print("Predictions",predictions)
    data = {'concepts' : sorted_results, 'img_data':img_data, 'class_probabilities':sorted_probs, 'table_display':"block"}
    return render_template('index.html', data=data)
    # return render_template('index.html' , concepts = sorted_results, img_data=img_data, class_probabilities=sorted_probs, table_display="block")

@api.route('/rerun', methods=['POST'])
def rerun():
    global img_data
    class_probs = {}
    '''
    For rendering results on HTML GUI
    '''

    new_concepts = request.form.to_dict()
    # print("Retrieved concepts:")

    new_concepts = {int(k):float(v.strip()) for k,v in new_concepts.items()}

    new_concepts_array = sorted(new_concepts.items())
    print("New concept array after sorting:",new_concepts_array)
    concept_array = [v for k, v in new_concepts_array]
    new_concepts_attributes = {k:{"attribute" : attributes.iloc[k-1]['attribute_name'] , "value":v} for k,v in new_concepts_array}
    print(new_concepts_attributes)
    
    probabilities = Model2.predict(np.array(concept_array).reshape(1,312))
    class_probs = {}

    for i in range(len(probabilities[0])):
        class_probs[bird_classes.iloc[i]['class_names']] = probabilities[0][i]

    sorted_probs = {k: v for k, v in sorted(class_probs.items(), key=lambda item: item[1], reverse=True)}
    sorted_concept_attributes = {k: v for k, v in sorted(new_concepts_attributes.items(), key=lambda item: item[1]['value'], reverse=True)}
    print("Sorted results:", sorted_concept_attributes)

    sorted_probs = {k: v for k, v in sorted(class_probs.items(), key=lambda item: item[1], reverse=True)}
    
    data = {'concepts':sorted_concept_attributes, 'img_data':img_data, 'class_probabilities':sorted_probs, 'table_display':"block"}
    return render_template('index.html', data=data)

    # return render_template('index.html', concepts=sorted_concept_attributes, img_data=img_data, class_probabilities=sorted_probs, table_display="block")


if __name__ == '__main__':
    api.run(debug=True)