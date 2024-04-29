from transformers import (
    AutoImageProcessor,
    TFVisionTextDualEncoderModel,
    BertTokenizer,
)
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from django.conf import settings
import os

image_processor = AutoImageProcessor.from_pretrained(
    'google/vit-base-patch16-224')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFVisionTextDualEncoderModel.from_vision_text_pretrained(
    text_model_name_or_path='bert-base-uncased', vision_model_name_or_path='google/vit-base-patch16-224')
model.load_weights(os.path.join(settings.STATIC_ROOT, 'model'))
image_features = np.load(os.path.join(settings.STATIC_ROOT, 'features.npy'))


def get_images(query, num_items):
    result = model.text_model(tokenizer(query, return_tensors='tf'))
    pooled_output = model.text_projection(result['pooler_output'])
    logits = np.dot(image_features, pooled_output.numpy().T)
    probs = tf.nn.softmax(logits, axis=0)
    item_list = [i for i in np.argsort(probs, axis=0).flatten()[
        ::-1][:num_items]]
    return item_list


def get_base64(img_path):
    img = Image.open(img_path)
    img = img.resize((200, 200))
    buff = BytesIO()
    img.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode()
    return img_str
