from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay

# Configuration of the environment
from os import environ
from tensorflow.keras import mixed_precision
environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_cpu_global_jit'
mixed_precision.set_global_policy("mixed_float16")

## INPUT PIPELINE
from input_pipeline import dataset_from_pathList, dataset_from_tfrecords
dataset = None # use one of the functions above

## SAVE DIR for model weights and training loss
DEST_DIR_PATH = None # TO DEFINE

from time import time
t_start = time()

## Model instancitation
from model import Model
model = Model()


## Training configuration
N_EPOCHS = 400
INIT_LR = 0.001
END_LR = 0.0001
STEPS_PER_EPOCH = None # TO DEFINE, must correspond to the number of MRIs entirely divided by the batch size (i.e. 16)
optimizer = Adam(learning_rate=PolynomialDecay(INIT_LR, N_EPOCHS*STEPS_PER_EPOCH, END_LR))
loss = 'mae'

model.compile(optimizer=optimizer, loss=loss)

## Training execution
history = model.fit(x=dataset, epochs=N_EPOCHS)


## Saving model weights and training loss
model.save_weights(DEST_DIR_PATH+"/model.h5")
with open(DEST_DIR_PATH+"/training_loss.txt", 'w') as f:
    for loss in history.history['loss']:
        f.write(f"{loss},")
print("End of training")
print(f"Model and stats saved in {DEST_DIR_PATH}")
print(f"Execution time : {time()-t_start:.0f} seconds")