import sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import inception_v3, xception, resnet50, vgg16, vgg19, densenet, mobilenet
from keras.applications import InceptionV3, Xception, ResNet50, VGG16, VGG19, DenseNet121, MobileNet
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.noise import GaussianDropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam
# from RocAucMetricCallback import RocAucMetricCallback
from roc_auc_callback import roc_auc_callback
from PIL import Image
from PIL import ImageEnhance
from pylab import array
from config import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def read_data(data, train, trainlabel):
    i = 0
    for patient, rang, label_two in tqdm(zip(data.name, data.range, data.label)):
        # print(patient)
        patient_folder = os.path.join(data_path, patient, serie, 'ori')
        oriFiles = os.listdir(patient_folder)
        for file in oriFiles:
            img = cv2.imread(os.path.join(patient_folder, file))
            # img = cv2.resize(img, (height, height))
            train[i] = img
            i += 1
        if rang == len(oriFiles):
            trainlabel.extend([label_two] * rang)
        else:
            raise ValueError('Error: original files number error!')
    return train, np.array(trainlabel)

def read_allData(data, train, trainlabel):
    """scale using all the patients"""
    i = 0
    train = train.astype(np.float32)
    for patient, rang, label_two in tqdm(zip(data.name, data.range, data.label)):
        # print(patient)
        patient_folder = os.path.join(data_path, patient, serie, 'ori', 'pat.npy')
        X_data = np.load(open(patient_folder, 'rb'))
        slice_num = X_data.shape[0]
        train[i: i+slice_num] = X_data
        i += slice_num
        if rang == slice_num:
            trainlabel.extend([label_two] * rang)
        else:
            raise ValueError('Error: original files number error!')
    return train, np.array(trainlabel)

def read_metainfo(img_info_path):
    img_info = pd.read_csv(img_info_path)
    # img_info['time'] = pd.to_datetime(img_info['time'], infer_datetime_format=True)
    # img_info.sort_values(by='time', axis=0, ascending=True, inplace=True)
    # training_index = int((2 / 3) * (img_info.shape[0]))
    # xtrain = img_info.iloc[:training_index, ]
    # xtest = img_info.iloc[training_index:, ]
    xtrain, xtest = train_test_split(img_info, shuffle=True, test_size=1/4, random_state=42, stratify=img_info.label)
    train_info, val_info = train_test_split(xtrain, shuffle=True, test_size=1/4, random_state=42, stratify=xtrain.label)
    return xtrain, xtest, train_info, val_info

def get_data(img_info_path):
    xtrain, xtest, _, _ = read_metainfo(img_info_path)
    ori_trainlabel = xtrain.label
    ori_testlabel = xtest.label
    print('oritrainlabel: %s, oritestlabel: %s' % (Counter(ori_trainlabel), Counter(ori_testlabel)))
    ori_train_slices = xtrain.range
    ori_test_slices = xtest.range
    trainSlices = np.sum(ori_train_slices)
    testSlices = np.sum(ori_test_slices)

    train = np.zeros((trainSlices, height, height, 3), dtype=np.uint8)
    test = np.zeros((testSlices, height, height, 3), dtype=np.uint8)
    trainlabel = []
    testlabel = []
    train, trainlabel = read_data(data=xtrain, train=train, trainlabel=trainlabel)
    test, testlabel = read_data(data=xtest, train=test, trainlabel=testlabel)
    return train, test, trainlabel, testlabel, ori_train_slices, ori_test_slices, ori_trainlabel, ori_testlabel

def get_val_data(img_info_path):
    _, _, xtrain, xval = read_metainfo(img_info_path)
    ori_trainlabel = xtrain.label
    ori_vallabel = xval.label
    print('oritrainlabel: %s, orivallabel: %s' % (Counter(ori_trainlabel), Counter(ori_vallabel)))
    ori_train_slices = xtrain.range
    ori_val_slices = xval.range
    trainSlices = np.sum(ori_train_slices)
    valSlices = np.sum(ori_val_slices)

    train = np.zeros((trainSlices, height, height, 3), dtype=np.uint8)
    val = np.zeros((valSlices, height, height, 3), dtype=np.uint8)
    trainlabel = []
    vallabel = []
    train, trainlabel = read_data(data=xtrain, train=train, trainlabel=trainlabel)
    val, vallabel = read_data(data=xval, train=val, trainlabel=vallabel)
    return train, val, trainlabel, vallabel, ori_train_slices, ori_val_slices, ori_trainlabel, ori_vallabel


def auctoAcc(trainlabel, train_predpro, testlabel, test_predpro, prnconfu=None):
    '''测试集阈值'''
    trainauc = roc_auc_score(trainlabel, train_predpro)
    testauc = roc_auc_score(testlabel, test_predpro)
    trainpre = np.zeros(len(trainlabel))
    testpre = np.zeros(len(testlabel))
    fpr, tpr, thresholds = roc_curve(trainlabel, train_predpro, pos_label=1, drop_intermediate=True)
    clf_thres = thresholds[np.argmax(1-fpr+tpr)]
    # clf_thres = np.array(0.5, dtype=np.float32)
    trainpre[train_predpro >= clf_thres] = 1
    # fpr, tpr, thresholds = roc_curve(testlabel, test_predpro, pos_label=1, drop_intermediate=True)
    # clf_thres = thresholds[np.argmax(1-fpr+tpr)]
    testpre[test_predpro >= clf_thres] = 1
    trainacc = accuracy_score(trainlabel, trainpre)
    testacc = accuracy_score(testlabel, testpre)
    '''get samples truely classified and wrongly classified'''
    train_index = np.logical_not(np.array(trainlabel) ^ trainpre.astype('uint8')).astype('uint8')
    test_index = np.logical_not(np.array(testlabel) ^ testpre.astype('uint8')).astype('uint8')
    train_trueIndex = np.where(train_index == 1)[0]
    test_trueIndex = np.where(test_index == 1)[0]
    tn, fp, fn, tp = confusion_matrix(trainlabel, trainpre).ravel()
    train_sen = tp/(tp + fn)
    train_spe = tn/(tn + fp)
    tn, fp, fn, tp = confusion_matrix(testlabel, testpre).ravel()
    test_sen = tp/(tp + fn)
    test_spe = tn/(tn + fp)
    if prnconfu == True:
        print("training confusion_matrix:\n %s " % confusion_matrix(trainlabel, trainpre))
        print("test confusion_matrix:\n %s " % confusion_matrix(testlabel, testpre))
    return trainauc, trainacc, train_sen, train_spe, testauc, testacc, test_sen, test_spe, trainpre, testpre, train_trueIndex, test_trueIndex

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

def plt_aucScore(history):
    plt.figure()
    plt.plot()
    plt.plot(history.history['roc_auc'])
    plt.plot(history.history['roc_auc_val'])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('auc.png')
    # loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')

def true_performance(train_slices, test_slices, train_label, test_label, train_pre, test_pre):
    lwoindex = 0
    ori_train_pre, ori_test_pre = [], []
    for k in train_slices:
        highindex = lwoindex + k
        ori_train_pre.append(np.mean(train_pre[lwoindex:highindex]))
        lwoindex = highindex
    lwoindex = 0
    for k in test_slices:
        highindex = lwoindex + k
        ori_test_pre.append(np.mean(test_pre[lwoindex:highindex]))
        lwoindex = highindex
    # train_auc = roc_auc_score(train_label, ori_train_pre)
    # test_auc = roc_auc_score(test_label, ori_test_pre)
    trainauc, trainacc, train_sen, train_spe, testauc,\
    testacc, test_sen, test_spe, trainpre, testpre, train_trueIndex, test_trueIndex = auctoAcc(train_label, ori_train_pre, test_label, ori_test_pre, prnconfu=False)
    print(trainauc, trainacc, train_sen, train_spe, testauc, testacc, test_sen, test_spe)
    return trainauc, trainacc, train_sen, train_spe, testauc, testacc, test_sen, test_spe, trainpre, testpre, train_trueIndex, test_trueIndex

def slice_divide(train, test, trainlabel, testlabel):
    alldata = np.concatenate([train, test], axis=0)
    label = trainlabel + testlabel
    train, test, trainlabel, testlabel = train_test_split(alldata, label, shuffle=True, test_size=1 / 4, random_state=0, stratify=label)
    # print('trainlabel: %s, testlabel: %s' % (Counter(trainlabel), Counter(testlabel)))
    print(Counter(trainlabel)[1] / Counter(trainlabel)[0], Counter(testlabel)[1] / Counter(testlabel)[0])
    return train, test, trainlabel, testlabel

def write_classfier(xtrain, xtest):
    train_pat = xtrain.iloc[train_trueIndex, ].name
    test_pat = xtest.iloc[test_trueIndex, ].name
    train_wrong_pat = np.setdiff1d(xtrain.name, train_pat).tolist()
    test_wrong_pat = np.setdiff1d(xtest.name, test_pat).tolist()
    classfied_ID = pd.DataFrame([train_pat.tolist(), test_pat.tolist(), train_wrong_pat, test_wrong_pat]).T
    classfied_ID.columns = ['train_true', 'test_true', 'train_wrong', 'test_wrong']
    classfied_ID.to_csv(r'H:\Wei\liaoxiao\Process\classify.csv', index=False)
    return train_pat, test_pat

# ori_getdata()
# train, test, trainlabel, testlabel = slice_divide(train, test, trainlabel, testlabel)
train, test, trainlabel, testlabel, ori_train_slices, ori_test_slices, ori_trainlabel, ori_testlabel = get_data(img_info_path)
print('trainlabel: %s, testlabel: %s' % (Counter(trainlabel), Counter(testlabel)))
X_train, X_val, y_train, y_val, X_train_slices, X_val_slices, X_trainlabel, X_vallabel = get_val_data(img_info_path)
# X_train, X_val, y_train, y_val = train_test_split(train, trainlabel, shuffle=True, test_size=0.2, random_state=42, stratify=trainlabel)
print('trainlabel: %s, validatelabel: %s' % (Counter(y_train), Counter(y_val)))


def fine_tune(MODEL, preprocess, height, freeze_till, lr, batch, nb_epoch, weights=None):
    x = Input(shape=(height, height, 3))
    x = Lambda(preprocess)(x)

    base_model = MODEL(include_top=False, input_tensor=x, weights='imagenet', pooling='avg')
    for layer in base_model.layers:
        layer.trainable = True
    for layer in base_model.layers[:freeze_till]:
        layer.trainable = False

    y = Dropout(0.5)(base_model.output)
    y = Dense(1, activation='sigmoid')(y)   # , kernel_initializer='he_normal'

    train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, shear_range=0.01,
                                       zoom_range=0.2, horizontal_flip=True,  # fill_mode='nearest',
                                       height_shift_range=0.2)
    validation_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, shear_range=0.01,
                                            zoom_range=0.2, horizontal_flip=True, height_shift_range=0.2)  # rescale=1./255,
    # test_datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2)
    train_generator = train_datagen.flow(x=X_train, y=y_train, batch_size=batch, shuffle=True)
                                         # save_to_dir=r'H:\Wei\liaoxiao\Process\generater_png', save_prefix='train', save_format='png')
    validation_generator = validation_datagen.flow(x=X_val, y=y_val, batch_size=batch, shuffle=True)
    # test_generator = test_datagen.flow(x=test, y=np.array(testlabel), batch_size=batch, shuffle=False)
    model = Model(inputs=base_model.input, outputs=y, name='Transfer_Learning')
    sgd = SGD(lr=lr, decay=5e-4, momentum=0.9, nesterov=True)  # 1e-6  5e-4
    # adam = Adam(lr=lr)
    # rmsprop = RMSprop(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])  # 'accuracy'  'binary_crossentropy',
    # trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    # non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    # print('Trainable: %d, Non-Trainable: %d' % (trainable_count, non_trainable_count))
    model.summary()
    if weights is not None:
        model.load_weights(weights)

    # Prepare Callbacks for Model Checkpoint, Early Stopping and Tensorboard.
    log_name = '/REICST-EP{epoch:02d}-LOSS{val_loss:.4f}.h5-AUC{roc_auc_val:.4f}'  #
    log_dir = datetime.now().strftime('transfer_vgg19_model_%Y%m%d_%H%M')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    rc = roc_auc_callback(training_data=(X_train, y_train), validation_data=(X_val, y_val))
    es = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    cl = CSVLogger('keras-5fold-run-01-v1-epochs.log', separator=',', append=False)
    mc = ModelCheckpoint(log_dir + log_name, monitor='val_loss', save_best_only=True, verbose=0, mode='min')  # val_loss
    tb = TensorBoard(log_dir=log_dir)

    history = model.fit_generator(train_generator, epochs=nb_epoch, verbose=2, callbacks=[rc, es, cl, mc, tb],
                                  validation_data=validation_generator, class_weight='auto', shuffle=True)
    with open('log_txt', 'w') as f:
        f.write(str(history.history))
    plt_score(history)
    plt_aucScore(history)
    files = os.listdir(log_dir)
    model.load_weights(os.path.join(log_dir, files[-1]))
    if print_txt == True:
        f_result = open('./clf_log.txt', 'a')
        sys.stdout = f_result
    print('\n', os.path.join(log_dir, files[-1]))

    X_trainacc = model.evaluate(x=X_train, y=y_train, verbose=0)
    X_valacc = model.evaluate(x=X_val, y=y_val, verbose=0)
    X_trainauc = roc_auc_score(y_train, model.predict(X_train))
    X_valauc = roc_auc_score(y_val, model.predict(X_val))

    trainacc = model.evaluate(x=train, y=trainlabel, verbose=0)
    testacc = model.evaluate(x=test, y=testlabel, verbose=0)
    # testacc = model.evaluate_generator(test_generator, steps=len(test_generator) * 10)
    trainscore = roc_auc_score(trainlabel, model.predict(train))
    testscore = roc_auc_score(testlabel, model.predict(test))
    # test_output = model.predict_generator(test_generator, steps=len(test_generator) * 10)
    # test_output = test_output.reshape((test.shape[0], 10))
    # test_pre = np.mean(test_output, axis=1)
    # testscore = roc_auc_score(testlabel, test_pre)
    train_pre = model.predict(x=train, batch_size=batch)
    test_pre = model.predict(x=test, batch_size=batch)
    print('X_trainauc: %.4f, X_valauc: %.4f' % (X_trainauc, X_valauc))
    print('X_trainacc: %.4f, X_valacc: %.4f' % (X_trainacc[1], X_valacc[1]))
    print('trainauc: %.4f, testauc: %.4f' % (trainscore, testscore))
    print('trainacc: %.4f, testacc: %.4f' % (trainacc[1], testacc[1]))
    return train_pre, test_pre

def fine_dense(MODEL, preprocess, height, freeze_till, lr, batch, nb_epoch, weights=None):
    x = Input(shape=(height, height, 3))
    x = Lambda(preprocess)(x)

    base_model = MODEL(include_top=False, input_tensor=x, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = True
    for layer in base_model.layers[:freeze_till]:
        layer.trainable = False

    y = GaussianDropout(0.5)(base_model.output)
    y = GlobalAveragePooling2D()(y)
    y = Dense(256, activation='relu')(y)  # , kernel_initializer='he_normal' , 'selu'
    y = GaussianDropout(0.5)(y)
    y = Dense(1, activation='sigmoid')(y)   # , kernel_initializer='he_normal'

    train_datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, shear_range=0.2,
                                       zoom_range=0.2, horizontal_flip=True, fill_mode='nearest',
                                       height_shift_range=0.2)
    validation_datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, shear_range=0.2,
                                            zoom_range=0.2, horizontal_flip=True)  # rescale=1./255,
    # test_datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2)
    train_generator = train_datagen.flow(x=X_train, y=y_train, batch_size=batch, shuffle=True)
                                         # save_to_dir=r'G:\Radiomics\Py_program\generator_png', save_prefix='train', save_format='png')
    validation_generator = validation_datagen.flow(x=X_val, y=y_val, batch_size=batch, shuffle=True)
    # test_generator = test_datagen.flow(x=test, y=np.array(testlabel), batch_size=batch, shuffle=False)

    model = Model(inputs=base_model.input, outputs=y, name='Transfer_Learning')
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # print('Trainable: %d, Non-Trainable: %d' % get_params_count(model))
    model.summary()
    if weights is not None:
        model.load_weights(weights)
    # Prepare Callbacks for Model Checkpoint, Early Stopping and Tensorboard.
    log_name = '/REICST-EP{epoch:02d}-LOSS{val_loss:.4f}-AUC{roc_auc_val:.4f}.h5'  #
    log_dir = datetime.now().strftime('transfer_resnet50_model_%Y%m%d_%H%M')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    rc = roc_auc_callback(training_data=(X_train, y_train), validation_data=(X_val, y_val))
    es = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    cl = CSVLogger('keras-5fold-run-01-v1-epochs.log', separator=',', append=False)
    mc = ModelCheckpoint(log_dir + log_name, monitor='val_loss', save_best_only=True, verbose=0, mode='min')  # val_loss
    tb = TensorBoard(log_dir=log_dir)

    history = model.fit_generator(generator=train_generator, epochs=nb_epoch, verbose=2, class_weight='auto',
                                  validation_data=validation_generator, callbacks=[rc, es, cl, mc, tb])
    with open('log_txt', 'w') as f:
        f.write(str(history.history))
    plt_score(history)
    plt_aucScore(history)
    files = os.listdir(log_dir)
    model.load_weights(os.path.join(log_dir, files[-1]))
    if print_txt == True:
        f_result = open('./clf_log.txt', 'a')
        sys.stdout = f_result
    print('\n', os.path.join(log_dir, files[-1]))

    X_trainacc = model.evaluate(x=X_train, y=y_train, verbose=0)
    X_valacc = model.evaluate(x=X_val, y=y_val, verbose=0)
    X_trainauc = roc_auc_score(y_train, model.predict(X_train))
    X_valauc = roc_auc_score(y_val, model.predict(X_val))

    trainacc = model.evaluate(x=train, y=trainlabel, verbose=0)
    testacc = model.evaluate(x=test, y=testlabel, verbose=0)
    # testacc = model.evaluate_generator(test_generator, steps=len(test_generator) * 10)
    trainscore = roc_auc_score(trainlabel, model.predict(train))
    testscore = roc_auc_score(testlabel, model.predict(test))
    # test_output = model.predict_generator(test_generator, steps=len(test_generator) * 10)
    # test_output = test_output.reshape((test.shape[0], 10))
    # test_pre = np.mean(test_output, axis=1)
    # testscore = roc_auc_score(testlabel, test_pre)
    train_pre = model.predict(x=train, batch_size=batch)
    test_pre = model.predict(x=test, batch_size=batch)
    print('X_trainauc: %.4f, X_valauc: %.4f' % (X_trainauc, X_valauc))
    print('X_trainacc: %.4f, X_valacc: %.4f' % (X_trainacc[1], X_valacc[1]))
    print('trainauc: %.4f, testauc: %.4f' % (trainscore, testscore))
    print('trainacc: %.4f, testacc: %.4f' % (trainacc[1], testacc[1]))
    return train_pre, test_pre

if __name__ == "__main__":
    train_pre, test_pre = fine_tune(Xception, xception.preprocess_input, height, freeze_till=116, lr=0.0001, batch=64, nb_epoch=150)
    # train_pre, test_pre = fine_tune(InceptionV3, inception_v3.preprocess_input, height, freeze_till=0, lr=0.001, batch=64, nb_epoch=100)
    # train_pre, test_pre = fine_tune(ResNet50, preprocess=resnet50.preprocess_input, height=height, freeze_till=179, lr=0.001, batch=64, nb_epoch=100, weights=None)  # 156
    # train_pre, test_pre = fine_tune(VGG19, preprocess=vgg19.preprocess_input, height=height, freeze_till=20, lr=0.001, batch=64, nb_epoch=150)
    # train_pre, test_pre = fine_tune(DenseNet121, preprocess=densenet.preprocess_input, height=height, freeze_till=120, lr=0.001, batch=64, nb_epoch=100)
    # train_pre, test_pre = fine_tune(MobileNet, preprocess=mobilenet.preprocess_input, height=height, freeze_till=87, lr=0.001, batch=64, nb_epoch=100)

    # log = './transfer_vgg19_model_20181115_1652/REICST-EP10-LOSS0.6045.h5-AUC0.5874'
    # train_pre, test_pre = fine_tune(InceptionV3, inception_v3.preprocess_input, height, freeze_till=0, lr=0.001, batch=64, nb_epoch=100, weights=log)
    # train_pre, test_pre = fine_tune(Xception, xception.preprocess_input, height, freeze_till=116, lr=0.001, batch=32, nb_epoch=100, weights=log)
    # train_pre, test_pre = fine_tune(ResNet50, preprocess=resnet50.preprocess_input, height=height, freeze_till=178, lr=0.01, batch=64, nb_epoch=100, weights=log)
    # train_pre, test_pre = fine_tune(DenseNet121, preprocess=densenet.preprocess_input, height=height, freeze_till=0, lr=0.001, batch=64, nb_epoch=1, weights=log)
    # train_pre, test_pre = fine_tune(MobileNet, preprocess=mobilenet.preprocess_input, height=height, freeze_till=75, lr=0.0002, batch=64, nb_epoch=100, weights=log)
    # train_pre, test_pre = fine_tune(VGG19, preprocess=vgg19.preprocess_input, height=height, freeze_till=25, lr=0.0001, batch=32, nb_epoch=100, weights=log)

    # log = r'./transfer_model_20181106_1628/REICST-EP63-LOSS0.5606-AUC0.6782.h5'
    # train_pre, test_pre = fine_tune(ResNet50, preprocess=resnet50.preprocess_input, height=height, freeze_till=100, lr=0.001, batch=32, nb_epoch=50, weights=log)

    # fine_dense(VGG19, vgg19.preprocess_input, height, freeze_till=25, lr=0.005, batch=32, nb_epoch=100, weights=None)
    # log = './transfer_model_20181031_1557/Differentiation-EP07-LOSS0.4430.h5'
    # fine_dense(VGG19, vgg19.preprocess_input, height, freeze_till=20, lr=0.001, batch=32, nb_epoch=50, weights=log)
    xtrain, xtest, train_info, val_info = read_metainfo(img_info_path=img_info_path)
    trainauc, trainacc, train_sen, train_spe, testauc, testacc, test_sen, test_spe, \
    ori_trainpre, ori_testpre, train_trueIndex, test_trueIndex = true_performance(ori_train_slices, ori_test_slices, ori_trainlabel, ori_testlabel, train_pre, test_pre)
    train_pat, test_pat = write_classfier(xtrain, xtest)
    print('training accuracy: {:d}/{:d}, {:.3f}'.format(len(train_pat), len(ori_trainlabel), len(train_pat)/len(ori_trainlabel)))
    print('test accuracy: {:d}/{:d}, {:.3f}'.format(len(test_pat), len(ori_testlabel), len(test_pat)/len(ori_testlabel)))
