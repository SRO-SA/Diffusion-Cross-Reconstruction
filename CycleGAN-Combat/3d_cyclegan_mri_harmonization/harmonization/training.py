# Configuration of the environment
from tensorflow.config import list_physical_devices, set_logical_device_configuration, LogicalDeviceConfiguration
from tensorflow.keras import mixed_precision
set_logical_device_configuration(list_physical_devices(device_type='GPU')[0], [LogicalDeviceConfiguration(memory_limit=21*1024)])
mixed_precision.set_global_policy("mixed_float16")

from tensorflow import (where as tf_where, concat as tf_concat, Variable as tf_Variable, GradientTape, function as tf_function, TensorSpec,
                        tensor_scatter_nd_update, expand_dims as tf_expand_dims, range as tf_range)
from tensorflow.random import shuffle as tf_shuffle
from tensorflow.math import reduce_mean, abs as tf_abs, square
from tensorflow.keras.optimizers import Adam


# Input pipeline
dsA = # TO DEFINE, functions in input_pipeline.py can be used
dsB = # TO DEFINE, functions in input_pipeline.py can be used
weights_path = None # TO DEFINE IF YOU WANT THE GENERATOR TO BE INITIALIZED WITH SPECIFIC WEIGHTS (e.g. identiy)

iterA = dsA.__iter__()
iterB = dsB.__iter__()

# Save dir for model weights and training stats
DEST_DIR_PATH = # TO DEFINE


from time import time
t_start = time()


# Instanciation of models
from model_architectures import Generator, Discriminator
generator_AtoB = Generator()
generator_BtoA = Generator()
discriminator_A = Discriminator()
discriminator_B = Discriminator()


# Initialisation of generators to identity
if weights_path:
    generator_AtoB.load_weights(weights_path)
    generator_BtoA.load_weights(weights_path)


# Discriminator training
DISC_N_BATCHS = 2
DISC_BUF_SIZE = 50

IMAGE_SHAPE = [192,192,192,1]

class Discriminator_trainer:
    
    def __init__(self, discriminator, generator, optimizer):
        self.discriminator = discriminator
        self.generator = generator
        self.optimizer = optimizer
        
    def translate_images(self, images):
        """
        images -> correspond à un batch
        """
        transformed = self.generator(images, training=False)
        return tf_where(images>-1,transformed,-1)
        
    def init_buffer(self, images_list):
        """
        images_list : liste de DISC_BUF_SIZE batchs d'image de l'autre domaine qui seront translatées et utilisées pour intialiser
        l'historique des images générées.
        """
        images_list = [self.translate_images(images) for images in images_list]
        self.buffer = tf_Variable(tf_concat(images_list, 0), dtype='float32', trainable=False)    
        
    @tf_function(input_signature=(TensorSpec(shape=[DISC_N_BATCHS*2]+IMAGE_SHAPE, dtype="float32"),
                                  TensorSpec(shape=[DISC_N_BATCHS]+IMAGE_SHAPE, dtype="float32")),
                 jit_compile=True)
    def train(self, images1, images2):
        """
        images1 : batch d'images du domaine de la critique
        images2 : batch d'images de l'autre domaine, celui qui sera transformé vers le domaine de la critique par le générateur
        """
        indices = tf_shuffle((tf_range(DISC_BUF_SIZE)))[:DISC_N_BATCHS]
        fakes1 = self.buffer.sparse_read(indices)
        fakes2 = self.translate_images(images2)
        fake_images = tf_concat([fakes1,fakes2], axis=0)
        
        replace_indices = tf_shuffle((tf_range(DISC_BUF_SIZE)))[:DISC_N_BATCHS]
        self.buffer.assign(tensor_scatter_nd_update(self.buffer, indices=tf_expand_dims(replace_indices, axis=1), updates=fakes2))
        
        with GradientTape() as tape:
            disc_real = self.discriminator(images1, training=True)
            disc_fakes = self.discriminator(fake_images, training=True)
            disc_loss = reduce_mean(square(disc_real-1)) + reduce_mean(square(disc_fakes))
            disc_loss_scaled = self.optimizer.get_scaled_loss(disc_loss)
        grads = tape.gradient(disc_loss_scaled, self.discriminator.trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(grads)
        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return disc_loss
    

# Generator training
@tf_function(input_signature=(TensorSpec(shape=[1]+IMAGE_SHAPE, dtype="float32"),
                              TensorSpec(shape=[1]+IMAGE_SHAPE, dtype="float32"),
                              TensorSpec(shape=(), dtype='float32')),
             jit_compile=True)
def train_generators(imagesA, imagesB, lambda_cyc):
    maskA = imagesA>-1
    maskB = imagesB>-1
    with GradientTape(persistent=True) as tape:
        fakesA = tf_where(maskB, generator_BtoA(imagesB, training=True), -1)
        fakesB = tf_where(maskA, generator_AtoB(imagesA, training=True), -1)
        
        # losses adversarielles
        disc_fakesA = discriminator_A(fakesA, training=False)
        disc_fakesB = discriminator_B(fakesB, training=False)
        gen_AtoB_adv_loss = reduce_mean(square(disc_fakesB-1))
        gen_BtoA_adv_loss = reduce_mean(square(disc_fakesA-1))
                
        # loss de reconstruction
        cycledA = tf_where(maskA, generator_BtoA(fakesB, training=True), -1)
        cycledB = tf_where(maskB, generator_AtoB(fakesA, training=True), -1)
        cycle_loss_aba = reduce_mean(tf_abs(imagesA-cycledA))
        cycle_loss_bab = reduce_mean(tf_abs(imagesB-cycledB))
        sum_cycle_loss = cycle_loss_aba + cycle_loss_bab
        
        # losses finales
        gen_AtoB_loss = gen_AtoB_adv_loss + lambda_cyc*sum_cycle_loss
        gen_AtoB_loss = generator_AtoB_optimizer.get_scaled_loss(gen_AtoB_loss)
        gen_BtoA_loss = gen_BtoA_adv_loss + lambda_cyc*sum_cycle_loss
        gen_BtoA_loss = generator_BtoA_optimizer.get_scaled_loss(gen_BtoA_loss)
        
    gradients_gen_AtoB = tape.gradient(gen_AtoB_loss, generator_AtoB.trainable_variables)
    gradients_gen_AtoB = generator_AtoB_optimizer.get_unscaled_gradients(gradients_gen_AtoB)
    generator_AtoB_optimizer.apply_gradients(zip(gradients_gen_AtoB, generator_AtoB.trainable_variables))
    gradients_gen_BtoA = tape.gradient(gen_BtoA_loss, generator_BtoA.trainable_variables)
    gradients_gen_BtoA = generator_BtoA_optimizer.get_unscaled_gradients(gradients_gen_BtoA)
    generator_BtoA_optimizer.apply_gradients(zip(gradients_gen_BtoA, generator_BtoA.trainable_variables))
    
    return gen_AtoB_adv_loss, gen_BtoA_adv_loss, cycle_loss_aba, cycle_loss_bab


# Training configuration
INIT_LR = 0.0002
N_EPOCHS_LR_BASE = 150
N_EPOCHS_LINEAR_DECAY = 150
STEPS_PER_EPOCH = 200
N_STEPS_DISC = 1

DECAY_STEP = 0 if N_EPOCHS_LINEAR_DECAY==0 else INIT_LR / N_EPOCHS_LINEAR_DECAY

epoch = 0 # init to avoid bug
def get_lr():
    s = max(epoch - N_EPOCHS_LR_BASE, 0) # epoch gloabl variable
    return INIT_LR - s*DECAY_STEP

def optimizer():
    opt = Adam(learning_rate=get_lr, beta_1=0.5, beta_2=0.999)
    return mixed_precision.LossScaleOptimizer(opt, dynamic=True)

generator_AtoB_optimizer = optimizer()
generator_BtoA_optimizer = optimizer()
discriminator_A_optimizer = optimizer()
discriminator_B_optimizer = optimizer()

discriminatorA_trainer = Discriminator_trainer(discriminator_A, generator_BtoA, discriminator_A_optimizer)
discriminatorB_trainer = Discriminator_trainer(discriminator_B, generator_AtoB, discriminator_B_optimizer)

# lambda cycle loss
LAMBDA_INIT = 200
LAMBDA_END = 100

N_EPOCHS_TOTAL = N_EPOCHS_LR_BASE + N_EPOCHS_LINEAR_DECAY

def get_lambda(epoch):
    step = (LAMBDA_INIT-LAMBDA_END)/(N_EPOCHS_TOTAL-1)
    return LAMBDA_INIT - epoch*step


# Initialisation of discriminator buffers
discriminatorA_trainer.init_buffer([iterB.get_next() for _ in range(DISC_BUF_SIZE)])
discriminatorB_trainer.init_buffer([iterA.get_next() for _ in range(DISC_BUF_SIZE)])


# Complete training function
@tf_function(input_signature=(TensorSpec(shape=(), dtype='float32'),), jit_compile=False)
def train_step(lambda_cyc):
    for _ in range(N_STEPS_DISC):
        imagesA = tf_concat([iterA.get_next() for _ in range(DISC_N_BATCHS*2)], axis=0)
        imagesB = tf_concat([iterB.get_next() for _ in range(DISC_N_BATCHS)], axis=0)
        discA_loss = discriminatorA_trainer.train(imagesA, imagesB)
        
        imagesB = tf_concat([iterB.get_next() for _ in range(DISC_N_BATCHS*2)], axis=0)
        imagesA = tf_concat([iterA.get_next() for _ in range(DISC_N_BATCHS)], axis=0)
        discB_loss = discriminatorB_trainer.train(imagesB, imagesA)
        
    imagesA = iterA.get_next()
    imagesB = iterB.get_next()
    gen_AtoB_adv_loss, gen_BtoA_adv_loss, cycle_loss_aba, cycle_loss_bab = train_generators(imagesA, imagesB, lambda_cyc)
    
    return {"discA_loss":discA_loss, "discB_loss":discB_loss,
            "gen_AtoB_adv_loss": gen_AtoB_adv_loss, "gen_BtoA_adv_loss": gen_BtoA_adv_loss,
            "cycle_loss_aba":cycle_loss_aba, "cycle_loss_bab":cycle_loss_bab}


# Execution
lambda_cyc = tf_Variable(0, dtype='float32')
record_dict = {}
for epoch in range(N_EPOCHS_TOTAL):
    tmp_record = {}
    for step in range(STEPS_PER_EPOCH):
        lambda_cyc.assign(get_lambda(epoch))
        res = train_step(lambda_cyc)
        
        # save of training results
        if not tmp_record:
            for key in res.keys():
                tmp_record[key] = res[key].numpy()
        else:
            for key in res.keys():
                tmp_record[key] += res[key].numpy()
        
        log = f"Fin step {step+1}/{STEPS_PER_EPOCH} époque {epoch+1}/{N_EPOCHS_TOTAL} | "
        for k,v in res.items():
            log += f"{k} = {v:.4f}, "
        print(log, end=f"{' '*20}\r")
        
    # add stats
    if not record_dict:
        for key in tmp_record:
            record_dict[key] = [tmp_record[key]/STEPS_PER_EPOCH]
    else:
        for key in tmp_record:
            record_dict[key].append(tmp_record[key]/STEPS_PER_EPOCH)
        
    print(f"Fin époque {epoch+1}", end=" -> ")
    for key in record_dict:
        print(f"{key} : {record_dict[key][-1]:.4f}", end=", ")
    print(' '*20)
    

# Save of model and training stats
from json import dump as json_dump
generator_AtoB.save_weights(DEST_DIR_PATH+"/generator_AtoB.h5")
generator_BtoA.save_weights(DEST_DIR_PATH+"/generator_BtoA.h5")
discriminator_A.save_weights(DEST_DIR_PATH+"/discriminator_A.h5")
discriminator_B.save_weights(DEST_DIR_PATH+"/discriminator_B.h5")
json_dump(record_dict, open(DEST_DIR_PATH+"/stats.json", 'w'))
print("End of training")
print(f"Model and stats saved in {DEST_DIR_PATH}")
print(f"Execution time : {time()-t_start:.0f} seconds")