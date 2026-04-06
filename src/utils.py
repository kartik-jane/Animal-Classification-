import numpy as np
from tensorflow.keras.preprocessing import image

IMG_SIZE = (224, 224)

def load_and_preprocess(img_path):
    """
    Load image and preprocess for model prediction
    """
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0   # normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array