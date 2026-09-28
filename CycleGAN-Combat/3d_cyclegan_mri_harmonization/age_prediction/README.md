# Training

The script *training.py* enables to train an age prediction model from a MRI set. The TensorFlow dataset must be defined at the beginning of the file. For that, the functions defined in *input_pipeline.py* can be used.


# Inference

The script *inference.py* enables to perform age prediction. The path of the model weights, the paths of the MRIs on which the prediction is performed, the list of the corresponding identifiers and the path of the output CSV must defined at the beginning of the file.