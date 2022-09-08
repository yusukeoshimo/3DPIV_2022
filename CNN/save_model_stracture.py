from tensorflow.keras.models import load_model
import os
from tensorflow import keras
model_path = input('input model path > ')
save_dir = input('input save dir > ')
model = load_model(model_path)
keras.utils.plot_model(model, os.path.join(save_dir, 'model_stracture.png'), show_shapes=True, show_layer_names=False)