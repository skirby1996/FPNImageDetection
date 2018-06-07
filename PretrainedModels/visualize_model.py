import os
import sys
from keras.utils import plot_model

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
from utils.model.model_init import initialize_model

model_name = "ResNet50"
file_path = os.path.join("PretrainedModels", model_name, (model_name + ".png"))
model = initialize_model(model_name)
plot_model(model, to_file=file_path)