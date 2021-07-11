import os
import matplotlib.pyplot as plt
import sklearn
import random
import time
import tensorflow as tf

from sklearn.model_selection import ShuffleSplit
from Classifier import *
from ConfusionMatrix import plot_confusion_matrix

k = 1
cross_validation = 5
data_set = 'DataSetHockeyFights/'

split_f1_Train, split_f1_Test, split_f2_Train, split_f2_Test = list(), list(), list(), list()
f1x, f2x = list(), list()

for file in os.listdir(data_set + 'Data'):
    if file.split(' ')[0] == 'NV':
        f1x.append(file)
    else:
        f2x.append(file)

rs = ShuffleSplit(n_splits=cross_validation, test_size=0.2, train_size=0.8, random_state=1)
for train_index_f1, test_index_f1 in rs.split(f1x):
    split_f1_Train.append([f1x[i] for i in train_index_f1])
    split_f1_Test.append([f1x[i] for i in test_index_f1])

for train_index_f2, test_index_f2 in rs.split(f2x):
    split_f2_Train.append([f2x[i] for i in train_index_f2])
    split_f2_Test.append([f2x[i] for i in test_index_f2])

for split in range(cross_validation):

    xTrain = split_f1_Train[split] + split_f2_Train[split]
    xTest = split_f1_Test[split] + split_f2_Test[split]

    yTrain = [1 if 'NV' in k else 0 for k in xTrain]
    yTest = [1 if 'NV' in k else 0 for k in xTest]

    trainXY = list(zip(xTrain, yTrain))
    testXY = list(zip(xTest, yTest))

    random.shuffle(trainXY)
    random.shuffle(testXY)

    xTrain, yTrain = zip(*trainXY)
    xTest, yTest = zip(*testXY)

    classifier = ClassifierHAR3D()
    classifier.channels = 3
    classifier.width = 224
    classifier.height = 224
    classifier.time = 50
    classifier.batch_size_train = 12
    classifier.batch_size_test = 1
    classifier.labels = [0, 1]
    classifier.pathTrain = data_set + 'Data'
    classifier.pathTest = data_set + 'Data'
    classifier.ftr = xTrain
    classifier.fts = xTest
    classifier.ftr_labels = yTrain
    classifier.fts_labels = yTest

    model = classifier.model_dense((classifier.time, classifier.height, classifier.width, classifier.channels))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print(tf.config.list_physical_devices('GPU'))

    history = model.fit(classifier.generatorTrain3D(),
                                epochs=100,
                                verbose=2,
                                steps_per_epoch=int(len(classifier.ftr) / classifier.batch_size_train))

    model.save_weights(data_set + 'Results/model_weights' + str(k) + '.h5')
    model.save(data_set + 'Results/model' + str(k) + '.h5')

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(data_set + 'Results/accuracy' + str(k) + '.png')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(data_set + 'Results/loss' + str(k) + '.png')
    plt.show()

    start = time.time()
    classifier.predictions = model.predict(classifier.generatorTest3D(),
                                                    steps=len(classifier.fts) / classifier.batch_size_test,
                                                    callbacks=None,
                                                    max_queue_size=10,
                                                    workers=1,
                                                    use_multiprocessing=False,
                                                    verbose=2)
    end = time.time()
    print('Inference time: ' + str((end - start)/len(classifier.fts)))

    classifier.predictions = np.argmax(classifier.predictions, axis=1)
    cmatrix = sklearn.metrics.confusion_matrix(classifier.fts_labels, classifier.predictions, labels=classifier.labels)
    test_accuracy = sklearn.metrics.accuracy_score(classifier.fts_labels, classifier.predictions, normalize=True)
    print(test_accuracy)
    k += 1
