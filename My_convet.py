from keras import Sequential
from keras.layers import Flatten, Dropout, Dense, MaxPooling2D, Conv2D, BatchNormalization, Activation
from DL_augment_training import *

def scale_input(train, test):
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)  # samplewise_center=True, samplewise_std_normalization=True rescale=1./255
    datagen.fit(train.astype(dtype=np.float64), augment=True, seed=1)
    scale_train = datagen.standardize(train.astype(dtype=np.float64))
    scale_test = datagen.standardize(test.astype(dtype=np.float64))
    # scale_train_generator = datagen.flow(x=train, y=trainlabel, shuffle=False)
    # scale_test_generator = datagen.flow(x=test, y=testlabel, shuffle=False)
    return scale_train, scale_test

def my_convet(X_train, y_train, X_val, y_val, lr, batch, nb_epoch, weights=None):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=(height, height, 3)))  # activation='relu',
    model.add(BatchNormalization(mode=0, axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3)))  # , activation='relu'
    model.add(BatchNormalization(mode=0, axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3)))   # , activation='relu'
    model.add(BatchNormalization(mode=0, axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128))  # , activation='relu'
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))    # , kernel_initializer='he_normal'
    model.summary()
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])  # optimizers.RMSprop(lr=1e-4)
    train_datagen = ImageDataGenerator(
        # samplewise_center=True, samplewise_std_normalization=True,
        featurewise_center=True, featurewise_std_normalization=True,
        # rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        # fill_mode='nearest'
    )
    #             rotation_range=40,
    #             width_shift_range=0.2,
    #             height_shift_range=0.2,
    validation_datagen = ImageDataGenerator(# samplewise_center=True, samplewise_std_normalization=True
                                            featurewise_center=True, featurewise_std_normalization=True
                                            # rescale=1./255
                                            # shear_range=0.2,
                                            # zoom_range=0.2,
                                            # horizontal_flip=True
                                            )
    train_datagen.fit(X_train, augment=True, seed=1)
    validation_datagen.fit(X_val, augment=False, seed=1)
    train_generator = train_datagen.flow(x=X_train, y=y_train, batch_size=batch, shuffle=True)
    validation_generator = validation_datagen.flow(x=X_val, y=y_val, batch_size=batch, shuffle=True)

    if weights is not None:
        model.load_weights(weights)

    # Prepare Callbacks for Model Checkpoint, Early Stopping and Tensorboard.
    log_name = '/REICST-EP{epoch:02d}-LOSS{val_loss:.4f}-AUC{roc_auc_val:.4f}.h5'  # -AUC{roc_auc_val:.4f}
    log_dir = datetime.now().strftime('myconvet_model_%Y%m%d_%H%M')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    rc = roc_auc_callback(training_data=(X_train, y_train), validation_data=(X_val, y_val))
    es = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
    cl = CSVLogger('keras-5fold-run-01-v1-epochs.log', separator=',', append=False)
    mc = ModelCheckpoint(log_dir + log_name, monitor='val_loss', save_best_only=True, verbose=0,
                         mode='min')  # val_loss
    tb = TensorBoard(log_dir=log_dir)

    # history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
    #                     class_weight={0: 1, 1: 1}, callbacks=callback_list)  # validation_split=0.1,
    history = model.fit_generator(train_generator, epochs=nb_epoch, class_weight={0: 2, 1: 1}, verbose=2,
                                  validation_data=validation_generator, callbacks=[rc, es, cl, mc, tb], shuffle=True)
    with open('log_txt', 'w') as f:
        f.write(str(history.history))
    plt_score(history)
    plt_aucScore(history)
    files = os.listdir(log_dir)
    model.load_weights(os.path.join(log_dir, files[-1]))
    # model.load_weights('./myconvet_model_20181105_2258/REICST-EP13-LOSS0.7347-AUC0.5836.h5')
    scale_train, scale_test = scale_input(train, test)
    # scale_train, scale_test = train, test
    trainacc = model.evaluate(x=scale_train, y=trainlabel, verbose=0)
    testacc = model.evaluate(x=scale_test, y=testlabel, verbose=0)
    trainscore = roc_auc_score(trainlabel, model.predict(scale_train))
    testscore = roc_auc_score(testlabel, model.predict(scale_test))
    train_pre = model.predict(x=scale_train, batch_size=batch)
    test_pre = model.predict(x=scale_test, batch_size=batch)
    print('trainauc: %.4f, testauc: %.4f' % (trainscore, testscore))
    print('trainacc: %.4f, testacc: %.4f' % (trainacc[1], testacc[1]))
    return train_pre, test_pre

train_pre, test_pre = my_convet(X_train, y_train, X_val, y_val, lr=0.5, batch=128, nb_epoch=100, weights=None)

# log = './myconvet_model_20181105_1753/REICST-EP95-LOSS0.4723.h5'
# train_pre, test_pre = my_convet(X_train, y_train, X_val, y_val, lr=0.001, batch=256, nb_epoch=1, weights=log)

trainauc, trainacc, train_sen, train_spe, testauc, testacc, test_sen, test_spe, \
ori_trainpre, ori_testpre = true_performance(ori_train_slices, ori_test_slices, ori_trainlabel, ori_testlabel, train_pre, test_pre)

