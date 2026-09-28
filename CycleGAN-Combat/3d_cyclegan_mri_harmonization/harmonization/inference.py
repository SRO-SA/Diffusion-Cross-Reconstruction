# Configuration of the environment
from os import environ
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")


# INPUTS
mri_paths = # TO DEFINE
dest_paths = # TO DEFINE
weights_path = # TO DEFINE


# instanciation générateur
from model_architectures import Generator
generator = Generator()
generator.load_weights(weights_path)

# harmo et sauvegarde des images
import nibabel as nib
from tensorflow import convert_to_tensor
import numpy as np
for i,(mri_path,dest_path) in enumerate(zip(mri_paths,dest_paths), start=1):
    print(f"Processing image {i}/{len(mri_paths)}", end='\r')
    mri = nib.load(mri_path)
    image = mri.get_fdata()/500 - 1
    mask = image > -1
    #image[~mask] = 0
    image = np.expand_dims(image, axis=(0,4))
    image = convert_to_tensor(image, dtype='float32')
    image = generator(image, training=False).numpy().squeeze()
    image[~mask] = -1
    image = (image+1)*500
    new_mri = nib.Nifti1Image(image, affine=mri.affine, header=mri.header)
    nib.save(new_mri, dest_path)
    