'''A phase process'''
from keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
from collections import Counter
series = ['A', 'N', 'V']
serie = 'A'
height = 512
print_txt = True

training_path = r'H:\Wei\liaoxiao\Process\training'
test_path = r'H:\Wei\liaoxiao\Process\test'
data_path = r'H:\Wei\liaoxiao\Process\Processed_Data_png'
# img_info_path = r'H:\Wei\liaoxiao\Process\Beijing_image_info' + serie + '2.csv'
img_info_path = r'H:\Wei\liaoxiao\Process\full_' + serie + '.csv'
# img_info_path = r'H:\Wei\liaoxiao\Process\sameSlice_info' + serie + '.csv'
# img_info_path = r'H:\Wei\liaoxiao\Process\full_sameSlice_info' + serie + '.csv'
# img_info_path = r'H:\Wei\liaoxiao\Process\ProcessToRoi_' + serie + '.csv'
# img_info_path = r'H:\Wei\liaoxiao\Process\Normal_Data2_' + serie + '.csv'

def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)

    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    N = K.sum(1 - y_true)
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    P = K.sum(y_true)
    TP = K.sum(y_pred * y_true)
    return TP/P

'''visually see the errors'''
def visualError(validation_generator, model):

    # Get the filenames from the generator
    fnames = validation_generator.filenames

    # Get the ground truth from generator
    ground_truth = validation_generator.classes

    # Get the label to class mapping from the generator
    label2index = validation_generator.class_indices

    # Getting the mapping from class index to class label
    idx2label = dict((v, k) for k, v in label2index.items())

    # Get the predictions from the model using the generator
    predictions = model.predict_generator(validation_generator,
                                          steps=validation_generator.samples / validation_generator.batch_size,
                                          verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    errors = np.where(predicted_classes != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors), validation_generator.samples))

    # Show the errors
    for i in range(len(errors)):
        pred_class = np.argmax(predictions[errors[i]])
        pred_label = idx2label[pred_class]

        title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[errors[i]].split('/')[0],
            pred_label,
            predictions[errors[i]][pred_class])
        # original = load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
        # plt.figure(figsize=[7, 7])
        # plt.axis('off')
        # plt.title(title)
        # plt.imshow(original)
        # plt.show()

def ori_getdata():
    trainSlices = 822
    testSlices = 405
    train = np.zeros((trainSlices, height, height, 3), dtype=np.uint8)
    test = np.zeros((testSlices, height, height, 3), dtype=np.uint8)
    ori_train_slices, ori_test_slices, ori_trainlabel, ori_testlabel, trainlabel, testlabel = [], [], [], [], [], []

    num = 0
    i = 0
    for label in ['1', '0']:
        patient_folder = os.path.join(training_path, label)
        allpatients = os.listdir(patient_folder)
        for patient in allpatients:
            # print(patient)
            oridata_path = os.path.join(patient_folder, patient, serie, 'ori')
            oriFiles = os.listdir(oridata_path)
            for file in oriFiles:
                image = cv2.imread(os.path.join(oridata_path, file))
                train[i] = image
                i += 1
            num = len(oriFiles) + num
            ori_train_slices.append(len(oriFiles))
        ori_trainlabel.extend([int(label)] * len(allpatients))
        trainlabel.extend([int(label)] * num)
        num = 0
        # print('label: %d, number:%d' % (int(label), num))
    # print(num)

    num = 0
    i = 0
    for label in ['1', '0']:
        patient_folder = os.path.join(test_path, label)
        allpatients = os.listdir(patient_folder)
        for patient in allpatients:
            # print(patient)
            oridata_path = os.path.join(patient_folder, patient, serie, 'ori')
            oriFiles = os.listdir(oridata_path)
            for file in oriFiles:
                image = cv2.imread(os.path.join(oridata_path, file))
                test[i] = image
                i += 1
            num = len(oriFiles) + num
            ori_test_slices.append(len(oriFiles))
        ori_testlabel.extend([int(label)] * len(allpatients))
        testlabel.extend([int(label)] * num)
        num = 0
    print('oritrainlabel: %s, oritestlabel: %s' % (Counter(ori_trainlabel), Counter(ori_testlabel)))
    print('trainlabel: %s, testlabel: %s' % (Counter(trainlabel), Counter(testlabel)))