import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import glob
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image
import argparse
import matplotlib.pyplot as plt


image_size = 224
class_names = {}


def process_image(image): 
   '''
   Normalize image 
   '''
   image = tf.cast(image, tf.float32)
   image = tf.image.resize(image, (image_size, image_size))
   image = image/255
    
   return image.numpy()


def predict(image_path, model, top_k):
    ''' Process and predict the top K flower classes along with associated probabilities
    '''
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image_exp = np.expand_dims(image,  axis=0)
    model_prob_list = model.predict(image_exp)
        
    prob_list, classes_list = tf.nn.top_k(model_prob_list, k=top_k)

    prob_list = list(prob_list.numpy()[0])
    classes_list = list(classes_list.numpy()[0])
    
    classes = []
    probs = prob_list
    
    for lable in classes_list:
        index = lable + 1
        classes.append(class_names[str(index)].title())
        # probs.append(prob_list[index])
      
    # fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
    # ax1.imshow(image)
    # ax1.axis('off')
    # ax1.set_title('Test Image: {}'.format(img))
    # ax2.barh(np.arange(5), prob_list)
    # ax2.set_aspect(0.1)
    # ax2.set_yticks(np.arange(5))
    # ax2.set_yticklabels(classes_list, size='small')
    # ax2.set_title('Class Probability:')
    # ax2.set_xlim(0, 1.1)
    # plt.tight_layout()
    # plt.show()
  
    return probs, classes


if __name__ == '__main__':
    print('predict.py, running')
    
    ## Arguments
    parser = argparse.ArgumentParser(description="Application that predicts the type of a given flower")
    parser.add_argument('image_path', help="path to the image", type=str)
    parser.add_argument('load_model',help="path to the model", type=str)
    parser.add_argument('--top_k', default=3, help ="top k class probabilities", type=int)
    parser.add_argument('--category_names',default="./label_map.json", help="path to the actual flower names", type=str) 
    
    args = parser.parse_args()
    
   
    print(args)
    
    print('Image path:', args.image_path)
    print('model :', args.load_model)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    
    image_path = args.image_path
    top_k = args.top_k
    category_names = args.category_names

    model = tf.keras.models.load_model(args.load_model ,custom_objects={'KerasLayer':hub.KerasLayer})
    
    top_k = int(top_k)
    
    if top_k is None: 
        top_k = 5

    with open(category_names, 'r') as f:
        class_names = json.load(f)
   
    probs, classes = predict(image_path, model, top_k)
    
    print(probs)
    print(classes)
