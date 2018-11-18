import os
import SimpleITK as sitk
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
from matplotlib.pylab import plt
# import utils
from keras import backend as K
from keras.applications import inception_v3, xception, resnet50, vgg16, vgg19, densenet, mobilenet
from keras.applications import InceptionV3, Xception, ResNet50, VGG16, VGG19, DenseNet121, MobileNet
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.noise import GaussianDropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from RocAucMetricCallback import RocAucMetricCallback
from DL_augment_training import plt_aucScore

def plt_score(history):
    plt.figure()
    plt.plot()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc.png')
    # loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''A phase process'''
series = ['A', 'N', 'V']
serie = 'A'
height = 128
trainSlices = 822
testSlices = 405
train = np.zeros((trainSlices, height, height, 3), dtype=np.uint8)
test = np.zeros((testSlices, height, height, 3), dtype=np.uint8)
trainlabel = []
testlabel = []

def slice_divide(train, test, trainlabel, testlabel):
    alldata = np.concatenate([train, test], axis=0)
    label = trainlabel + testlabel
    train, test, trainlabel, testlabel = train_test_split(alldata, label, shuffle=True, test_size=1 / 4, random_state=0, stratify=label)
    # print('trainlabel: %s, testlabel: %s' % (Counter(trainlabel), Counter(testlabel)))
    print(Counter(trainlabel)[1] / Counter(trainlabel)[0], Counter(testlabel)[1] / Counter(testlabel)[0])
    return train, test, trainlabel, testlabel

training_path = r'H:\Wei\liaoxiao\Process\training'
test_path = r'H:\Wei\liaoxiao\Process\test'
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
    testlabel.extend([int(label)] * num)
    num = 0

# train, test, trainlabel, testlabel = slice_divide(train, test, trainlabel, testlabel)
print('trainlabel: %s, testlabel: %s' % (Counter(trainlabel), Counter(testlabel)))
X_train, X_val, y_train, y_val = train_test_split(train, trainlabel, shuffle=True, test_size=0.2, random_state=42, stratify=trainlabel)
print('trainlabel: %s, validatelabel: %s' % (Counter(y_train), Counter(y_val)))

# assert 0

def fine_tune(MODEL, preprocess, height, freeze_till, lr, batch, nb_epoch, weights=None):
    x = Input(shape=(height, height, 3))
    x = Lambda(preprocess)(x)

    base_model = MODEL(include_top=False, input_tensor=x, weights='imagenet', pooling='avg')
    for layer in base_model.layers:
        layer.trainable = True
    for layer in base_model.layers[:freeze_till]:
        layer.trainable = False

    y = Dropout(0.2)(base_model.output)
    y = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)

    model = Model(inputs=base_model.input, outputs=y, name='Transfer_Learning')
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # print('Trainable: %d, Non-Trainable: %d' % get_params_count(model))
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    print('Trainable: %d, Non-Trainable: %d' % (trainable_count, non_trainable_count))
    model.summary()
    if weights is not None:
        model.load_weights(weights)

    # Prepare Callbacks for Model Checkpoint, Early Stopping and Tensorboard.
    log_name = '/Differentiation-EP{epoch:02d}-LOSS{val_loss:.4f}.h5'
    log_dir = datetime.now().strftime('transfer_model_%Y%m%d_%H%M')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    rc = RocAucMetricCallback(predict_batch_size=32, include_on_batch=True)
    es = EarlyStopping(monitor='val_loss', patience=20, verbose=0)
    mc = ModelCheckpoint(log_dir + log_name, monitor='val_loss', save_best_only=True, verbose=0)
    tb = TensorBoard(log_dir=log_dir)

    history = model.fit(x=X_train, y=y_train, batch_size=batch, epochs=nb_epoch, verbose=2,
                        validation_data=(X_val, y_val), callbacks=[es, mc, tb])
    with open('log_txt', 'w') as f:
        f.write(str(history.history))
    # plt_score(history)
    plt_aucScore(history)
    trainscore = model.evaluate(x=train, y=trainlabel)
    testscore = model.evaluate(x=test, y=testlabel)
    # train_pre = model.predict(x=train)
    # test_pre = model.predict(x=test)
    # pd.DataFrame(test_pre).to_excel(output_path)
    print(trainscore, testscore)

def fine_dense(MODEL, preprocess, height, freeze_till, lr, batch, nb_epoch, weights=None):
    x = Input(shape=(height, height, 3))
    x = Lambda(preprocess)(x)

    base_model = MODEL(include_top=False, input_tensor=x, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = True
    for layer in base_model.layers[:freeze_till]:
        layer.trainable = False

    y = GaussianDropout(0.4)(base_model.output)
    y = GlobalAveragePooling2D()(y)
    y = Dense(128, activation='selu', kernel_initializer='he_normal')(y)
    y = GaussianDropout(0.4)(y)
    y = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)

    model = Model(inputs=base_model.input, outputs=y, name='Transfer_Learning')
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # print('Trainable: %d, Non-Trainable: %d' % get_params_count(model))

    # Prepare Callbacks for Model Checkpoint, Early Stopping and Tensorboard.
    log_name = '/Differentiation-EP{epoch:02d}-LOSS{val_loss:.4f}.h5'
    log_dir = datetime.now().strftime('transfer_model_%Y%m%d_%H%M')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    es = EarlyStopping(monitor='val_loss', patience=20)
    mc = ModelCheckpoint(log_dir + log_name, monitor='val_loss', save_best_only=True)
    tb = TensorBoard(log_dir=log_dir)

    history = model.fit(x=X_train, y=y_train, batch_size=batch, epochs=nb_epoch, verbose=2,
                        validation_data=(X_val, y_val), callbacks=[es, mc, tb])
    with open('log_txt', 'w') as f:
        f.write(str(history.history))
    plt_score(history)
    trainscore = model.evaluate(x=train, y=trainlabel)
    testscore = model.evaluate(x=test, y=testlabel)
    # train_pre = model.predict(x=train)
    # test_pre = model.predict(x=test)
    # pd.DataFrame(test_pre).to_excel(output_path)
    print(trainscore, testscore)

# fine_tune(Xception, xception.preprocess_input, height, freeze_till=116, lr=0.045, batch=16, nb_epoch=50)
# fine_tune(InceptionV3, inception_v3.preprocess_input, height, freeze_till=280, lr=0.001, batch=32, nb_epoch=50)
# fine_tune(ResNet50, preprocess=resnet50.preprocess_input, height=height, freeze_till=178, lr=0.045, batch=32, nb_epoch=50)
fine_tune(VGG16, preprocess=vgg16.preprocess_input, height=height, freeze_till=22, lr=0.001, batch=32, nb_epoch=100)
# fine_tune(DenseNet121, preprocess=densenet.preprocess_input, height=height, freeze_till=400, lr=0.045, batch=32, nb_epoch=100)
# fine_tune(MobileNet, preprocess=mobilenet.preprocess_input, height=height, freeze_till=90, lr=0.001, batch=32, nb_epoch=100)

# log = './transfer_model_20181031_1516/Differentiation-EP27-LOSS0.5148.h5'
# fine_tune(VGG16, vgg16.preprocess_input, height, freeze_till=16, lr=0.001, batch=32, nb_epoch=50, weights=log)
# fine_tune(VGG19, vgg19.preprocess_input, height, freeze_till=20, lr=0.001, batch=32, nb_epoch=50, weights=log)

# log = r'G:\PycharmProjects\remoteProjects\Diff\code\transfer_model_20181031_1509\Differentiation-EP07-LOSS0.6279.h5'
# fine_tune(DenseNet121, preprocess=densenet.preprocess_input, height=height, freeze_till=400, lr=0.001, batch=32, nb_epoch=50, weights=log)

# fine_dense(VGG19, vgg19.preprocess_input, height, freeze_till=25, lr=0.0001, batch=32, nb_epoch=100, weights=None)
# log = './transfer_model_20181102_0936/REICST-EP28-LOSS0.6223.h5'
# fine_dense(VGG19, vgg19.preprocess_input, height, freeze_till=20, lr=0.001, batch=32, nb_epoch=50, weights=log)
