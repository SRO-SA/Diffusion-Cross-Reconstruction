from nibabel import load as load_mri
from scipy.ndimage import rotate
from tensorflow.data import AUTOTUNE, Dataset, TFRecordDataset
from tensorflow.io import parse_single_example, FixedLenFeature
from tensorflow import (TensorSpec, pad as tf_pad, roll as tf_roll, function as tf_function, expand_dims as tf_expand_dims, numpy_function,
                        constant as tf_constant)
from tensorflow.random import uniform

IMAGE_SHAPE = [192,192,192,1]

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

axes_list = tf_constant([[0,1],[0,2],[1,2]], dtype='int32')

def rotation(mri, angle, axes, cval):
    return rotate(mri, angle=angle, axes=axes, reshape=False, mode="constant", cval=cval)

@tf_function # jit compile incompatible avec numpy_function
def random_rotation_vec(mri, angle_max, cval):
    """
    mri : 4D (width, height, depth, n_channels)
    Randomly rotates MRIs in a randomly drawn plan (among 3 plans)
    """
    angle = uniform((), minval=-angle_max, maxval=angle_max, dtype="float32")
    axes = axes_list[uniform((), minval=0, maxval=3, dtype="int32")]
    mri = numpy_function(rotation, inp=[mri, angle, axes, cval], Tout='float32')
    mri.set_shape(IMAGE_SHAPE)
    return mri

@tf_function
def augmentation(mri):
    # if uniform(shape=(1,), minval=0, maxval=1, dtype="float32") < 0.5:
    #     mri = random_rotation_vec(mri, angle_max=15, cval=-1)
    return random_shift_vec(mri, shift_max=5, cval=-1)    

def dataset_from_pathList(mri_paths, cache, buf_size):
    """
    Parameters:
        mri_paths: list of mri paths
        cache: boolean indicating whether parsed images are cached
        buf_size: size of buffer for shuffling
    Returns:
        TensorFlow dataset
    """
    if cache:
        def gen_train():
            for p in mri_paths:
                mri = load_mri(p).get_fdata()
                yield tf_expand_dims(intensity_norm(mri), axis=3)
        ds_train = Dataset.from_generator(gen_train, output_signature=TensorSpec(shape=(192,192,192,1), dtype='float32'))
        ds_train = ds_train.cache().shuffle(buf_size).repeat()
        ds_train = ds_train.map(augmentation, num_parallel_calls=AUTOTUNE, deterministic=False)
    else:
        def gen_train():
            for p in mri_paths:
                mri = load_mri(p).get_fdata()
                mri = tf_expand_dims(intensity_norm(mri), axis=3) 
                yield augmentation(mri)
        ds_train = Dataset.from_generator(gen_train, output_signature=TensorSpec(shape=(192,192,192,1), dtype='float32'))
        ds_train = ds_train.shuffle(buf_size).repeat()
    return ds_train.batch(1).prefetch(AUTOTUNE)
    
def dataset_biasSample_from_pathLists(mriPath_lists,probas,cache,buf_sizes):
    """
    Parameters:
        mriPath_lists: list of list of mri paths
        probas: list of sampling probabilities associated with each list of mris
        cache: boolean indicating whether parsed images are cached
        buf_sizes: size of buffer for shuffling each mri group
    Returns:
        TensorFlow dataset
    """
    def prepare_single_dataset(mri_paths, buf_size):
        if cache:
            def gen_train():
                for p in mri_paths:
                    mri = load_mri(p).get_fdata()
                    yield tf_expand_dims(intensity_norm(mri), axis=3)
            dataset = Dataset.from_generator(gen_train, output_signature=TensorSpec(shape=(192,192,192,1), dtype='float32'))
            return dataset.cache().shuffle(buf_size).repeat()
        else:
            def gen_train():
                for p in mri_paths:
                    mri = load_mri(p).get_fdata()
                    mri = tf_expand_dims(intensity_norm(mri), axis=3)
                    yield augmentation(mri)
            dataset = Dataset.from_generator(gen_train, output_signature=TensorSpec(shape=(192,192,192,1), dtype='float32'))
            return dataset.shuffle(buf_size).repeat()
    
    datasets = [prepare_single_dataset(*args) for args in zip(mriPath_lists,buf_sizes)]
    ds_train = Dataset.sample_from_datasets(datasets, weights=probas)
    if cache: ds_train = ds_train.map(augmentation, num_parallel_calls=AUTOTUNE, deterministic=False)
    return ds_train.batch(1).prefetch(AUTOTUNE)

def dataset_from_tfrecords(tfrecords_paths, cache, buf_size):
    """
    Function to parse tfrecords files. Admits intensities are already normalized.
    
    Parameters:
        tfrecords_path: path to a tfrecords file or list of such paths.
        cache: boolean indicating whether parsed images are cached
        buf_size: size of buffer for shuffling
    """
    keys_to_feature_train = {"mri": FixedLenFeature(shape=(192,192,192,1), dtype="float32")}
    ds_train = TFRecordDataset(tfrecords_path, num_parallel_reads=AUTOTUNE, compression_type="GZIP")
    if cache:
        ds_train = ds_train.map(lambda t: parse_single_example(t, keys_to_feature_train)['mri'], num_parallel_calls=AUTOTUNE, deterministic=False)
        ds_train = ds_train.cache().shuffle(buf_size).repeat()
        ds_train = ds_train.map(lambda mri: augmentation(mri), num_parallel_calls=AUTOTUNE, deterministic=False)
    else:
        ds_train = ds_train.map(lambda t: augmentation(parse_single_example(t, keys_to_feature_train)['mri']), num_parallel_calls=AUTOTUNE, deterministic=False)
        ds_train = ds_train.shuffle(buf_size).repeat()
    return ds_train.batch(1).prefetch(AUTOTUNE)

def dataset_biasSample_from_tfrecords(tfrecords_paths, probas, cache, buf_sizes):
    """
    Function to parse tfrecords files. Admits intensities are already normalized.
    
    Parameters:
        tfrecords_path: list of paths to a tfrecordsfile or list of list of paths of such paths
        probas: list of sampling probabilities associated with each list of mris
        cache: boolean indicating whether parsed images are cached
        buf_size: size of buffer for shuffling each mri group
    """
    keys_to_feature_train = {"mri": FixedLenFeature(shape=(192,192,192,1), dtype="float32")}
    def prepare_single_dataset(tfrecords_paths, buf_size):
        dataset = TFRecordDataset(tfrecords_paths, num_parallel_reads=AUTOTUNE, compression_type="GZIP")
        if cache:
            dataset = dataset.map(lambda t: parse_single_example(t, keys_to_feature_train)['mri'], num_parallel_calls=AUTOTUNE, deterministic=False)
            return dataset.cache().shuffle(buf_size).repeat()
        else:
            dataset = dataset.map(lambda t: augmentation(parse_single_example(t, keys_to_feature_train)['mri']), num_parallel_calls=AUTOTUNE, deterministic=False)
            return dataset.shuffle(buf_size).repeat()
    datasets = [prepare_single_dataset(*args) for args in zip(tfrecords_paths,buf_sizes)]
    ds_train = Dataset.sample_from_datasets(datasets, weights=probas)
    if cache: ds_train = ds_train.map(augmentation, num_parallel_calls=AUTOTUNE, deterministic=False)
    return ds_train.batch(1).prefetch(AUTOTUNE)