import tensorflow as tf
import numpy as np
import skimage.io as io
import os
from PIL import Image

# Code Goal: How to convert your dataset to .tfrecords file and later on use it as a part of a computational graph.

"""
Intro : In this post we will cover how to convert a dataset into .tfrecord file. Binary files are sometimes easier to 
use, because you don’t have to specify different directories for images and groundtruth annotations. While storing your 
data in binary file, you have your data in one block of memory, compared to storing each image and annotation 
separately. Openning a file is a considerably time-consuming operation especially if you use hdd and not ssd, because 
it involves moving the disk reader head and that takes quite some time. Overall, by using binary files you make it 
easier to distribute and make the data better aligned for efficient reading.


"""

# Part 1: Getting raw data bytes in numpy:

"""
Here we demonstrate how you can get raw data bytes of an image (any ndarray) and how to restore the image back. We point
 out that during this operation the information about the dimensions of the image is lost and we have to use it to 
 recover the original image. This is one of the reasons why we will have to store the raw image representation along 
 with the dimensions of the original image.

In the following examples, we convert the image into the raw representation, restore it and make sure that the original 
image and the restored one are the same.
"""


cat_img = io.imread(os.path.join('tensorflow_notes', 'data', 'imgs', 'cat.jpg'))
io.imshow(cat_img)
print("Shape: ", cat_img.shape)
# Let's convert the picture into string representation
# using the ndarray.tostring() function
cat_string = cat_img.tostring()

# Now let's convert the string back to the image
# Important: the dtype should be specified
# otherwise the reconstruction will be errorness
# Reconstruction is 1d, so we need sizes of image
# to fully reconstruct it.
reconstructed_cat_1d = np.fromstring(cat_string, dtype=np.uint8)

# Here we reshape the 1d representation
# This is the why we need to store the sizes of image
# along with its serialized representation.
reconstructed_cat_img = reconstructed_cat_1d.reshape(cat_img.shape)

# Let's check if we got everything right and compare
# reconstructed array to the original one.
print(np.allclose(cat_img, reconstructed_cat_img))

# Part 2 : Creating a .tfrecord file and reading it without defining a graph:

"""
Here we show how to write a small dataset (three images/annotations from PASCAL VOC) to .tfrrecord file and read it 
without defining a computational graph.

We also make sure that images that we read back from .tfrecord file are equal to the original images. 
Pay attention that we also write the sizes of the images along with the image in the raw format. We showed an example 
on why we need to also store the size in the previous section.
"""

# Get some image/annotation pairs for example:
filename_pairs = []
# % matplotlib inline

# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want this type of behaviour
# consider using skimage.io.imread()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


tfrecords_filename = 'pascal_voc_segmentation.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Let's collect the real images to later on compare to the reconstructed ones:
original_images = []

for img_path, annotation_path in filename_pairs:
    img = np.array(Image.open(img_path))
    annotation = np.array(Image.open(annotation_path))

    # The reason to store image sizes was demonstrated
    # in the previous example -- we have to know sizes
    # of images to later read raw serialized string,
    # convert to 1d array and convert to respective
    # shape that image used to have.
    height = img.shape[0]
    width = img.shape[1]

    # Put in the original images into array
    # Just for future check for correctness
    original_images.append((img, annotation))

    img_raw = img.tostring()
    annotation_raw = annotation.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'mask_raw': _bytes_feature(annotation_raw)}))

    writer.write(example.SerializeToString())

writer.close()

# Reconstruction:

reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    height = int(example.features.feature['height']
                 .int64_list
                 .value[0])

    width = int(example.features.feature['width']
                .int64_list
                .value[0])

    img_string = (example.features.feature['image_raw']
                  .bytes_list
                  .value[0])

    annotation_string = (example.features.feature['mask_raw']
                         .bytes_list
                         .value[0])

    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height, width, -1))

    annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)

    # Annotations don't have depth (3rd dimension)
    reconstructed_annotation = annotation_1d.reshape((height, width))

    reconstructed_images.append((reconstructed_img, reconstructed_annotation))

# Let's check if the reconstructed images match
# the original images

for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
    img_pair_to_compare, annotation_pair_to_compare = zip(original_pair,
                                                          reconstructed_pair)
    print(np.allclose(*img_pair_to_compare))
    print(np.allclose(*annotation_pair_to_compare))