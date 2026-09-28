# Configuration of the environment
from os import environ
from tensorflow.keras import mixed_precision
environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_cpu_global_jit'
mixed_precision.set_global_policy("mixed_float16")


"""
INPUTS: 4 variables must be defined:
    - weights_paths: path of the model weights to use
    - mri_paths: nifti files, supposed to be preprocessed
    - id_dicts: dict of list identifying the subjects, will be used for the output csv (e.g. {'sub_id':[2,9,1], 'session':['ses1','ses1','ses2']})
    - csv_dest
"""
weights_path = None # TO DEFINE
mri_paths = None # TO DEFINE
id_dicts = None # TO DEFINE
csv_dest = None # TO DEFINE


## Model initialisation
from model import Model
model = Model()
model.load_weights(weights_path)


## Predictions
import numpy as np
from nibabel import load as load_mri
from tensorflow import convert_to_tensor
import pandas as pd
id_dicts['pred_age'] = np.empty(len(mri_paths))
for i,path in enumerate(mri_paths):
    print(f"Processing image {i+1}/{len(mri_paths)}", end='\r')
    mri = load_mri(path).get_fdata()/500-1
    mri = np.expand_dims(mri, axis=(0,4))
    mri = convert_to_tensor(mri, dtype='float32')
    id_dict['pred_age'][i] = model(mri, training=False).numpy().item()

pd.DataFrame(id_dict).to_csv(csv_dest, index=False)
