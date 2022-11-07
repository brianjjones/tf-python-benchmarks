import os
import time
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import sys

ml = (sys.argv)[1]

file1 = os.path.abspath("images/banana.jpg")
img1 = tf.keras.utils.load_img(file1, target_size=[224, 224])

file2 = os.path.abspath("images/heron.jpg")
img2 = tf.keras.utils.load_img(file2, target_size=[224, 224])

file3 = os.path.abspath("images/pizza.jpg")
img3 = tf.keras.utils.load_img(file3, target_size=[224, 224])

file4 = os.path.abspath("images/stop.jpg")
img4 = tf.keras.utils.load_img(file4, target_size=[224, 224])

file5 = os.path.abspath("images/train.jpg")
img5 = tf.keras.utils.load_img(file5, target_size=[224, 224])

if ml == "MobileNetV3Small":
    print("Using MobileNetV3Small")
    pretrained_model = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3), weights="imagenet")
    pl = tf.keras.applications.mobilenet_v3.preprocess_input
elif ml == "MobileNetV3Large":
    print("Using MobileNet")
    pretrained_model = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), weights="imagenet")
    pl = tf.keras.applications.mobilenet_v3.preprocess_input
elif ml == "MobileNet":
    print("Using MobileNet")
    pretrained_model = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), weights="imagenet")
    pl = tf.keras.applications.mobilenet.preprocess_input
elif ml == "ResNet50":
    print("Using ResNet50")
    pretrained_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), weights="imagenet")
    pl = tf.keras.applications.resnet50.preprocess_input
elif ml == "InceptionResNetV2":
    print("Using InceptionResNetV2")
    pretrained_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), weights="imagenet")
    pl = tf.keras.applications.inception_resnet_v2.preprocess_input
else:
    print("Unknown model. Options are MobileNetV3Small, MobileNetV3Large, MobileNet, ResNet50, or InceptionResNetV2")
    exit

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

imagenet_labels = np.array(open(labels_path).read().splitlines())

for num in range(1, 5):
    for img in [img1, img2, img3, img4, img5]:
        plt.imshow(img)
        plt.axis('off')
        x = tf.keras.utils.img_to_array(img)
        x = pl(
            x[tf.newaxis,...])

        start_time = time.time_ns()
        result_before_save = pretrained_model(x)
        end_time = time.time_ns()
        total_time = (end_time - start_time) // 1_000_000
        decoded = imagenet_labels[np.argsort(result_before_save)[0,::-1][:5]+1]

        print("Result before saving:\n", decoded)
        print("It took : ", total_time)
        print(start_time)
        print(end_time)

# Saving the model and running appears to make it perform much better.
mobilenet_save_path = os.path.join("./", "savedmodel")
tf.saved_model.save(pretrained_model, mobilenet_save_path)

loaded = tf.saved_model.load(mobilenet_save_path)
print(list(loaded.signatures.keys()))  # ["serving_default"]

infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)
print(pretrained_model.output_names)

for num in range(1, 5):
    for img in [img1, img2, img3, img4, img5]:
        plt.imshow(img)
        plt.axis('off')
        x = tf.keras.utils.img_to_array(img)
        x = pl(
            x[tf.newaxis,...])

        start_time = time.time_ns()
        labeling = infer(tf.constant(x))[pretrained_model.output_names[0]]
        end_time = time.time_ns()
        total_time = (end_time - start_time) // 1_000_000
        print("It took : ", total_time)
        decoded = imagenet_labels[np.argsort(labeling)[0,::-1][:5]+1]

        print("Result after saving and loading:\n", decoded)
