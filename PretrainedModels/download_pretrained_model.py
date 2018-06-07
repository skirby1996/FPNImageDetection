import os
import os.path

### For list of keras supported models see
# https://keras.io/applications/#documentation-for-individual-models
from keras.applications.resnet50 import ResNet50

model_name = "ResNet50"
weights_opt = "imagenet" ## "none" for random init

dir_path = os.path.join("PretrainedModels", model_name)
model_path = os.path.join(dir_path, "model.json")
weights_name = model_name + "_weights.h5"
weights_path = os.path.join(dir_path, "weights.h5")
if os.path.exists(dir_path):
    print("%s model already exists..." % (model_name))
else:
    os.makedirs(dir_path)
    print("Downloading %s with weights=%s" % (model_name, weights_opt))
    model = ResNet50(weights=weights_opt, include_top=True)
    model_json = model.to_json()
    out_file = open(model_path, 'w+')
    out_file.write(model_json)
    out_file.close()
    model.save_weights(weights_path)