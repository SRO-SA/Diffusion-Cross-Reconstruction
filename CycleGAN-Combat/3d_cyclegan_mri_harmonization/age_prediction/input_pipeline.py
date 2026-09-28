from tensorflow.random import uniform
from tensorflow import py_function, pad as tf_pad, roll as tf_roll, expand_dims as tf_expand_dims, TensorSpec
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.io import parse_single_example, FixedLenFeature
from scipy.ndimage import rotate
from nibabel import load as load_mri

def intensity_norm(mri):
    """
    Sets the min of the image to -1 and its median intensity inside the brain to 0, admitting that the image is preprocessed and its min is currently at 0 and its median intensity at 500.
    """
    return mri/500 - 1


### DATA AUGMENTATION
@tf_function
def random_shift_vec(mri, shift_max, cval):
    """
    mri : 4D (width, height, depth, n_channels) tensor
    Randomly shiftes mri in all dimension (width, height, depth).
    For each dimension, shiftes between 0 and shift_max_voxels
    """
    shifts = uniform((3,), minval=-shift_max, maxval=shift_max+1, dtype="int32")
    mri = tf_roll(mri, shift=shifts, axis=(0,1,2))
    
    # cropping
    cropping = ((shifts[0],0) if shifts[0]>=0 else (0,-shifts[0]),
               (shifts[1],0) if shifts[1]>=0 else (0,-shifts[1]),
               (shifts[2],0) if shifts[2]>=0 else (0,-shifts[2]))
    mri = mri[cropping[0][0]:mri.shape[0]-cropping[0][1],
              cropping[1][0]:mri.shape[1]-cropping[1][1],
              cropping[2][0]:mri.shape[2]-cropping[2][1]:,]
    
    # apply padding
    return tf_pad(mri, paddings=(cropping[0],cropping[1],cropping[2],(0,0)), mode="CONSTANT", constant_values=cval)

def rotation(mri, angle, draw, cval):
    if draw==0: axes = (0,1)
    elif draw==1: axes = (0,2)
    else: axes = (1,2)
    return rotate(mri, angle=angle, axes=axes, reshape=False, mode="constant", cval=cval)

@tf_function
def random_rotation_vec(mri, angle_max, cval):
    """
    mri : 4D (width, height, depth, n_channels)
    Randomly rotates MRIs in a randomly drawn plan (among 3 plans)
    """
    angle = uniform((), minval=-angle_max, maxval=angle_max, dtype="float32")
    draw = uniform((), minval=0, maxval=3, dtype="int32")
        
    mri_shape = mri.shape
    [mri,] = py_function(rotation, inp=[mri, angle, draw, cval], Tout=["float32"])
    mri.set_shape(mri_shape)
    return mri

@tf_function
def augmentation(mri):
    if uniform(shape=(1,), minval=0, maxval=1, dtype="float32") < 0.5:
        mri = random_rotation_vec(mri, angle_max=10, cval=-1)
    return random_shift_vec(mri, shift_max=5, cval=-1)

def dataset_from_pathList(mri_paths, ages, cache, buf_size):
    """
    Parameters:
        mri_paths: list of mri paths
        ages: list of corresponding ages
        cache: boolean indicating whether parsed images are cached
        buf_size: size of buffer for shuffling
    Returns:
        TensorFlow dataset
    """
    if cache:
        def gen_train():
            for path,age in zip(mri_paths,ages):
                mri = intensity_norm(load_mri(path).get_fdata())
                yield tf_expand_dims(mri, axis=3),age
        ds_train = Dataset.from_generator(gen_train, output_signature=(TensorSpec(shape=(192,192,192,1), dtype='float32'), TensorSpec(shape=(), dtype='float32')))
        ds_train = ds_train.cache().shuffle(buf_size)
        ds_train = ds_train.map(lambda mri,age: (augmentation(mri),age), num_parallel_calls=AUTOTUNE, deterministic=False)
    else:
        def gen_train():
            for path,age in zip(mri_paths,ages):
                mri = intensity_norm(load_mri(path).get_fdata())
                yield augmentation(tf_expand_dims(mri, axis=3)), age
        ds_train = Dataset.from_generator(gen_train, output_signature=(TensorSpec(shape=(192,192,192,1), dtype='float32'), TensorSpec(shape=(), dtype='float32')))
        ds_train = ds_train.shuffle(buf_size)
    return ds_train.batch(16, drop_remainder=True).prefetch(AUTOTUNE)

def dataset_from_tfrecords(tfrecords_path, cache, buf_size):
    """
    Function to parse tfrecords files. Admits intensities are already normalized.
    
    Parameters:
        tfrecords_path: path to a tfrecords file or list of such paths.
        cache: boolean indicating whether parsed images are cached
        buf_size: size of buffer for shuffling
    """
    keys_to_feature_train = {"mri": FixedLenFeature(shape=(192,192,192,1), dtype="float32"), 'age':FixedLenFeature(shape=(), dtype='float32')}
    ds_train = TFRecordDataset(tfrecords_path, num_parallel_reads=AUTOTUNE, compression_type="GZIP")
    if cache:
        def parse_record(tfrecord):
            d = parse_single_example(tfrecord, keys_to_feature_train)
            return d['mri'], d['age']
        ds_train = ds_train.map(parse_record, num_parallel_calls=AUTOTUNE, deterministic=False)
        ds_train = ds_train.cache().shuffle(buf_size)
        ds_train = ds_train.map(lambda mri,age: (augmentation(mri),age), num_parallel_calls=AUTOTUNE, deterministic=False)
    else:
        def parse_record(tfrecord):
            d = parse_single_example(tfrecord, keys_to_feature_train)
            return augmentation(d['mri']), age
        ds_train = ds_train.shuffle(buf_size)
        ds_train = ds_train.map(parse_record, num_parallel_calls=AUTOTUNE, deterministic=False)
    return ds_train.batch(16, drop_remainder=True).prefetch(AUTOTUNE)