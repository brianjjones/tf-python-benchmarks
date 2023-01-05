import os
import time
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import subprocess

if len(sys.argv) < 4:
    print("Missing arguments. Please enter the model, number of runs, and batch size. ex: python pyexample.py ResNet50 5 15", len(sys.argv))
    sys.exit()

ml = (sys.argv)[1]

if ml == "MobileNetV3Small":
    print("Using MobileNetV3Small")
    pretrained_model = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3), weights="imagenet")
    pl = tf.keras.applications.mobilenet_v3.preprocess_input
elif ml == "MobileNetV3Large":
    print("Using MobileNet")
    pretrained_model = tf.keras.applications.MobileNetV3Large(input_shape=(224, 224, 3), weights="imagenet")
    pl = tf.keras.applications.mobilenet_v3.preprocess_input
elif ml == "MobileNetV2":
    print("Using MobileNetV2")
    pretrained_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3), weights="imagenet")
    pl = tf.keras.applications.mobilenet_v2.preprocess_input
elif ml == "MobileNet":
    print("Using MobileNet")
    pretrained_model = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), weights="imagenet")
    pl = tf.keras.applications.mobilenet.preprocess_input
elif ml == "ResNet50":
    print("Using ResNet50")
    pretrained_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), weights="imagenet")
    pl = tf.keras.applications.resnet50.preprocess_input
elif ml == "ResNet152V2":
    print("Using ResNet152V2")
    pretrained_model = tf.keras.applications.resnet_v2.ResNet152V2(input_shape=(224, 224, 3), weights="imagenet")
    pl = tf.keras.applications.resnet_v2.preprocess_input
elif ml == "InceptionResNetV2":
    print("Using InceptionResNetV2")
    pretrained_model = tf.keras.applications.InceptionResNetV2(input_shape=(299, 299, 3), weights="imagenet")
    pl = tf.keras.applications.inception_resnet_v2.preprocess_input
else:
    print("Unknown model. Options are MobileNetV3Small, MobileNetV3Large, MobileNet, ResNet50, ResNet152V2, or InceptionResNetV2")
    sys.exit()

if ml != "InceptionResNetV2":
    fsize = 224
else:
    fsize = 299

images = []
file1 = os.path.abspath("images/banana.jpg")
images.append(tf.keras.utils.load_img(file1, target_size=[fsize, fsize]))

file2 = os.path.abspath("images/heron.jpg")
images.append(tf.keras.utils.load_img(file2, target_size=[fsize, fsize]))

file3 = os.path.abspath("images/pizza.jpg")
images.append(tf.keras.utils.load_img(file3, target_size=[fsize, fsize]))

file4 = os.path.abspath("images/stop.jpg")
images.append(tf.keras.utils.load_img(file4, target_size=[fsize, fsize]))

file5 = os.path.abspath("images/train.jpg")
images.append(tf.keras.utils.load_img(file5, target_size=[fsize, fsize]))

def run_bench():
    curr_img = 0
    x1 = tf.keras.utils.img_to_array(images[curr_img])
    img_arr = [x1]

    for j in range(1, batch_sz):
        if curr_img < 4:
            curr_img += 1
        else:
            curr_img = 0
        x2 = tf.keras.utils.img_to_array(images[curr_img])
        img_arr.append(x2)

    input_arr = np.array(img_arr)

    x = pl(
        input_arr[tf.newaxis,...])

    start_time = time.time_ns()
    labeling = infer(tf.constant(x, shape=[batch_sz,fsize,fsize,3]))[pretrained_model.output_names[0]]
    end_time = time.time_ns()
    total_time = (end_time - start_time) // 1_000_000
    per_img = total_time / batch_sz

    print("\n****** RESULTS FOR RUN %s ******" % (num))
    print("In total it took : ", total_time)
    print("For a batch size of %s, each image took roughly : %s" % (batch_sz, per_img))
    print("Results after saving and loading:\n")
    for i in range(0, batch_sz):
        decoded = imagenet_labels[np.argsort(labeling)[i,::-1][:3]+1]
        print("%s - %s" % (i, decoded))

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

imagenet_labels = np.array(open(labels_path).read().splitlines())

# Saving the model, loading, and running appears to make it perform much better.
mobilenet_save_path = os.path.join("./", "savedmodel")
subprocess.call("rm -rf savedmodel", shell=True)
subprocess.call("rm -rf savedmodel_optimized", shell=True)
subprocess.call("mkdir savedmodel", shell=True)
tf.saved_model.save(pretrained_model, mobilenet_save_path)

# Call the optimize script and save the new model
subprocess.call("python3 freeze_optimize_v2.py --input_saved_model_dir=savedmodel --output_saved_model_dir=savedmodel_optimized", shell=True)
batch_sz = int((sys.argv)[3])

print("\n")
print("*****************************************")
print("--== Running with unoptimized model ==--")
print("*****************************************")
loaded = tf.saved_model.load(mobilenet_save_path)
print(list(loaded.signatures.keys()))  # ["serving_default"]

infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)
print(pretrained_model.output_names)

for num in range(0, int((sys.argv)[2])):
    run_bench()

print("\n")
print("*********************************************")
print("--== Running with optimized frozen model ==--")
print("**********************************************")
loaded = tf.saved_model.load("savedmodel_optimized")
print(list(loaded.signatures.keys()))  # ["serving_default"]

infer = loaded.signatures["serving_default"]
for num in range(0, int((sys.argv)[2])):
    run_bench()
