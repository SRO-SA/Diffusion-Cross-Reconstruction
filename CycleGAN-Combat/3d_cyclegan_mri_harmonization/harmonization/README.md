# Training

The script *training.py* enables to train an harmonization model from 2 MRI sets. The two TensorFlow datasets have to be defined at the beginning of the file. For that, the functions defined in *input_pipeline.py* can be used. A path towards initial weights for the generators can also be defined (e.g. initialization to identity).


# Inference

The script *inference.py* enables to harmonize a set of MRIs. The list of MRIs to harmonize (median standardized at 500), the destination paths and the path of the generator weights to use must defined at the beginning of the file.