from keras import Model
from Utils import *
from keras.layers import *
from keras.utils import np_utils

class ClassifierHAR3D(object):

    def __init__(self):

        self.width = None
        self.height = None
        self.channels = None
        self.time = None
        self.batch_size_train = None
        self.batch_size_validation = None
        self.batch_size_test = None
        self.labels = None
        self.pathTrain = None
        self.pathTest = None
        self.pathValidation = None
        self.ftr = None
        self.ftv = None
        self.fts = None
        self.ftr_labels = None
        self.ftv_labels = None
        self.fts_labels = None
        self.predictions = None

    def generatorTrain3D(self):

        while True:

            for count in range(int(len(self.ftr) / self.batch_size_train)):

                batch_start = self.batch_size_train * count
                batch_stop = self.batch_size_train + (self.batch_size_train * count)

                lx1 = list()
                ly = list()

                for i in range(batch_start, batch_stop):

                    if self.ftr[i] != '.ipynb_checkpoints':

                        ly.append(self.ftr_labels[i])

                        optical_flow = extract_videos3D_optical_flow(self.pathTrain + self.ftr[i], self.height,
                                                                     self.width)

                        if len(optical_flow) < self.time:
                            while len(optical_flow) < self.time:
                                optical_flow.append(optical_flow[-1])
                        else:
                            optical_flow = optical_flow[0:self.time]

                        lx1.append(optical_flow)

                x1 = np.array(lx1)
                x1 = x1.astype('float32')
                x1 /= 255
                x1 = x1.reshape(x1.shape[0], self.time, self.height, self.width, self.channels)

                y = np.array(ly)
                y = np_utils.to_categorical(y, len(self.labels))

                yield x1, y

    def generatorTest3D(self):

        while True:

            for count in range(int(len(self.fts) / self.batch_size_test)):

                batch_start = self.batch_size_test * count
                batch_stop = self.batch_size_test + (self.batch_size_test * count)

                lx1 = list()

                for i in range(batch_start, batch_stop):

                    if self.fts[i] != '.ipynb_checkpoints':

                        optical_flow = extract_videos3D_optical_flow(self.pathTest + self.fts[i], self.height,
                                                                     self.width)

                        if len(optical_flow) < self.time:
                            while len(optical_flow) < self.time:
                                optical_flow.append(optical_flow[-1])
                        else:
                            optical_flow = optical_flow[0:self.time]

                        lx1.append(optical_flow)

                x1 = np.array(lx1)
                x1 = x1.astype('float32')
                x1 /= 255
                x1 = x1.reshape(x1.shape[0], self.time, self.height, self.width, self.channels)

                yield x1

    # Models

    def modelHAR3DDenseOp(self, input_shape):

        input1 = Input(shape=input_shape)

        net = Conv3D(filters=32, kernel_size=(7, 7, 7), strides=(1, 2, 2), activation='relu', padding='same')(input1)
        net = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(net)

        net1 = BatchNormalization()(net)
        net1 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net1)
        net1 = BatchNormalization()(net1)
        net1 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net1)

        net2 = BatchNormalization()(net1)
        net2 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net2)
        net2 = BatchNormalization()(net2)
        net2 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net2)

        c1 = Concatenate()([net1, net2])

        net3 = BatchNormalization()(c1)
        net3 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net3)
        net3 = BatchNormalization()(net3)
        net3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net3)

        c2 = Concatenate()([net1, net2, net3])

        net4 = BatchNormalization()(c2)
        net4 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net4)
        net4 = BatchNormalization()(net4)
        net4 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net4)

        c3 = Concatenate()([net1, net2, net3, net4])

        net5 = BatchNormalization()(c3)
        net5 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net5)
        net5 = BatchNormalization()(net5)
        net5 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net5)

        c4 = Concatenate()([net1, net2, net3, net4, net5])

        net6 = BatchNormalization()(c4)
        net6 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net6)
        net6 = BatchNormalization()(net6)
        net6 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net6)

        t1 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net6)
        t1 = AveragePooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(t1)

        net7 = BatchNormalization()(t1)
        net7 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net7)
        net7 = BatchNormalization()(net7)
        net7 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net7)

        net8 = BatchNormalization()(net7)
        net8 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net8)
        net8 = BatchNormalization()(net8)
        net8 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net8)

        c5 = Concatenate()([net7, net8])

        net9 = BatchNormalization()(c5)
        net9 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net9)
        net9 = BatchNormalization()(net9)
        net9 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net9)

        c6 = Concatenate()([net7, net8, net9])

        net10 = BatchNormalization()(c6)
        net10 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net10)
        net10 = BatchNormalization()(net10)
        net10 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net10)

        c7 = Concatenate()([net7, net8, net9, net10])

        net11 = BatchNormalization()(c7)
        net11 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net11)
        net11 = BatchNormalization()(net11)
        net11 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net11)

        c8 = Concatenate()([net7, net8, net9, net10, net11])

        net12 = BatchNormalization()(c8)
        net12 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net12)
        net12 = BatchNormalization()(net12)
        net12 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net12)

        c9 = Concatenate()([net7, net8, net9, net10, net11, net12])

        net13 = BatchNormalization()(c9)
        net13 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net13)
        net13 = BatchNormalization()(net13)
        net13 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net13)

        c10 = Concatenate()([net7, net8, net9, net10, net11, net12, net13])

        net14 = BatchNormalization()(c10)
        net14 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net14)
        net14 = BatchNormalization()(net14)
        net14 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net14)

        c11 = Concatenate()([net7, net8, net9, net10, net11, net12, net13, net14])

        net15 = BatchNormalization()(c11)
        net15 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net15)
        net15 = BatchNormalization()(net15)
        net15 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net15)

        c12 = Concatenate()([net7, net8, net9, net10, net11, net12, net13, net14, net15])

        net16 = BatchNormalization()(c12)
        net16 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net16)
        net16 = BatchNormalization()(net16)
        net16 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net16)

        c13 = Concatenate()([net7, net8, net9, net10, net11, net12, net13, net14, net15, net16])

        net17 = BatchNormalization()(c13)
        net17 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net17)
        net17 = BatchNormalization()(net17)
        net17 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net17)

        c14 = Concatenate()([net7, net8, net9, net10, net11, net12, net13, net14, net15, net16, net17])

        net18 = BatchNormalization()(c14)
        net18 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net18)
        net18 = BatchNormalization()(net18)
        net18 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net18)

        t2 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net18)
        t2 = AveragePooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(t2)

        net19 = BatchNormalization()(t2)
        net19 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net19)
        net19 = BatchNormalization()(net19)
        net19 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net19)

        net20 = BatchNormalization()(net19)
        net20 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net20)
        net20 = BatchNormalization()(net20)
        net20 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net20)

        c15 = Concatenate()([net19, net20])

        net21 = BatchNormalization()(c15)
        net21 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net21)
        net21 = BatchNormalization()(net21)
        net21 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net21)

        c16 = Concatenate()([net19, net20, net21])

        net22 = BatchNormalization()(c16)
        net22 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net22)
        net22 = BatchNormalization()(net22)
        net22 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net22)

        c17 = Concatenate()([net19, net20, net21, net22])

        net23 = BatchNormalization()(c17)
        net23 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net23)
        net23 = BatchNormalization()(net23)
        net23 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net23)

        c18 = Concatenate()([net19, net20, net21, net22, net23])

        net24 = BatchNormalization()(c18)
        net24 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net24)
        net24 = BatchNormalization()(net24)
        net24 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net24)

        c19 = Concatenate()([net19, net20, net21, net22, net23, net24])

        net25 = BatchNormalization()(c19)
        net25 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net25)
        net25 = BatchNormalization()(net25)
        net25 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net25)

        c20 = Concatenate()([net19, net20, net21, net22, net23, net24, net25])

        net26 = BatchNormalization()(c20)
        net26 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net26)
        net26 = BatchNormalization()(net26)
        net26 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net26)

        c21 = Concatenate()([net19, net20, net21, net22, net23, net24, net25, net26])

        net27 = BatchNormalization()(c21)
        net27 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net27)
        net27 = BatchNormalization()(net27)
        net27 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net27)

        c22 = Concatenate()([net19, net20, net21, net22, net23, net24, net25, net26, net27])

        net28 = BatchNormalization()(c22)
        net28 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net28)
        net28 = BatchNormalization()(net28)
        net28 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net28)

        c23 = Concatenate()([net19, net20, net21, net22, net23, net24, net25, net26, net27, net28])

        net29 = BatchNormalization()(c23)
        net29 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net29)
        net29 = BatchNormalization()(net29)
        net29 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net29)

        c24 = Concatenate()([net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29])

        net30 = BatchNormalization()(c24)
        net30 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net30)
        net30 = BatchNormalization()(net30)
        net30 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net30)

        c25 = Concatenate()([net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30])

        net31 = BatchNormalization()(c25)
        net31 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net31)
        net31 = BatchNormalization()(net31)
        net31 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net31)

        c26 = Concatenate()([net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31])

        net32 = BatchNormalization()(c26)
        net32 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net32)
        net32 = BatchNormalization()(net32)
        net32 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net32)

        c27 = Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32])

        net33 = BatchNormalization()(c27)
        net33 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net33)
        net33 = BatchNormalization()(net33)
        net33 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net33)

        c28 = Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33])

        net34 = BatchNormalization()(c28)
        net34 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net34)
        net34 = BatchNormalization()(net34)
        net34 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net34)

        c29 = Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34])

        net35 = BatchNormalization()(c29)
        net35 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net35)
        net35 = BatchNormalization()(net35)
        net35 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net35)

        c30 = Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35])

        net36 = BatchNormalization()(c30)
        net36 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net36)
        net36 = BatchNormalization()(net36)
        net36 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net36)

        c31 = Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35, net36])

        net37 = BatchNormalization()(c31)
        net37 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net37)
        net37 = BatchNormalization()(net37)
        net37 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net37)

        c32 = Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35, net36, net37])

        net38 = BatchNormalization()(c32)
        net38 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net38)
        net38 = BatchNormalization()(net38)
        net38 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net38)

        c33 = Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35, net36, net37, net38])

        net39 = BatchNormalization()(c33)
        net39 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net39)
        net39 = BatchNormalization()(net39)
        net39 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net39)

        c34 = Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35, net36, net37, net38, net39])

        net40 = BatchNormalization()(c34)
        net40 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net40)
        net40 = BatchNormalization()(net40)
        net40 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net40)

        c35 = Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35, net36, net37, net38, net39, net40])

        net41 = BatchNormalization()(c35)
        net41 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net41)
        net41 = BatchNormalization()(net41)
        net41 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net41)

        c36 = Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35, net36, net37, net38, net39, net40, net41])

        net42 = BatchNormalization()(c36)
        net42 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net42)
        net42 = BatchNormalization()(net42)
        net42 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net42)

        t3 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net42)
        t3 = AveragePooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(t3)

        net43 = BatchNormalization()(t3)
        net43 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net43)
        net43 = BatchNormalization()(net43)
        net43 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net43)

        net44 = BatchNormalization()(net43)
        net44 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net44)
        net44 = BatchNormalization()(net44)
        net44 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net44)

        c37 = Concatenate()([net43, net44])

        net45 = BatchNormalization()(c37)
        net45 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net45)
        net45 = BatchNormalization()(net45)
        net45 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net45)

        c38 = Concatenate()([net43, net44, net45])

        net46 = BatchNormalization()(c38)
        net46 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net46)
        net46 = BatchNormalization()(net46)
        net46 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net46)

        c39 = Concatenate()([net43, net44, net45, net46])

        net47 = BatchNormalization()(c39)
        net47 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net47)
        net47 = BatchNormalization()(net47)
        net47 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net47)

        c40 = Concatenate()([net43, net44, net45, net46, net47])

        net48 = BatchNormalization()(c40)
        net48 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net48)
        net48 = BatchNormalization()(net48)
        net48 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net48)

        c41 = Concatenate()([net43, net44, net45, net46, net47, net48])

        net49 = BatchNormalization()(c41)
        net49 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net49)
        net49 = BatchNormalization()(net49)
        net49 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net49)

        c42 = Concatenate()([net43, net44, net45, net46, net47, net48, net49])

        net50 = BatchNormalization()(c42)
        net50 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net50)
        net50 = BatchNormalization()(net50)
        net50 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net50)

        c43 = Concatenate()([net43, net44, net45, net46, net47, net48, net49, net50])

        net51 = BatchNormalization()(c43)
        net51 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net51)
        net51 = BatchNormalization()(net51)
        net51 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net51)

        c44 = Concatenate()([net43, net44, net45, net46, net47, net48, net49, net50, net51])

        net52 = BatchNormalization()(c44)
        net52 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net52)
        net52 = BatchNormalization()(net52)
        net52 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net52)

        c45 = Concatenate()([net43, net44, net45, net46, net47, net48, net49, net50, net51, net52])

        net53 = BatchNormalization()(c45)
        net53 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net53)
        net53 = BatchNormalization()(net53)
        net53 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net53)

        c46 = Concatenate()([net43, net44, net45, net46, net47, net48, net49, net50, net51, net52, net53])

        net54 = BatchNormalization()(c46)
        net54 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net54)
        net54 = BatchNormalization()(net54)
        net54 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net54)

        c47 = Concatenate()([net43, net44, net45, net46, net47, net48, net49, net50, net51, net52, net53, net54])

        net55 = BatchNormalization()(c47)
        net55 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net55)
        net55 = BatchNormalization()(net55)
        net55 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net55)

        c48 = Concatenate()([net43, net44, net45, net46, net47, net48, net49, net50, net51, net52, net53, net54, net55])

        net56 = BatchNormalization()(c48)
        net56 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net56)
        net56 = BatchNormalization()(net56)
        net56 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net56)

        c49 = Concatenate()(
            [net43, net44, net45, net46, net47, net48, net49, net50, net51, net52, net53, net54, net55, net56])

        net57 = BatchNormalization()(c49)
        net57 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net57)
        net57 = BatchNormalization()(net57)
        net57 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net57)

        c50 = Concatenate()(
            [net43, net44, net45, net46, net47, net48, net49, net50, net51, net52, net53, net54, net55, net56, net57])

        net58 = BatchNormalization()(c50)
        net58 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net58)
        net58 = BatchNormalization()(net58)
        net58 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net58)

        c51 = Concatenate()(
            [net43, net44, net45, net46, net47, net48, net49, net50, net51, net52, net53, net54, net55, net56, net57,
             net58])

        net59 = BatchNormalization()(c51)
        net59 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net59)
        net59 = BatchNormalization()(net59)
        net59 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net59)

        net59 = AveragePooling3D(pool_size=(1, 7, 7))(net59)

        net59 = Bidirectional(ConvLSTM2D(filters=128, kernel_size=(1, 1), activation='relu', return_sequences=True))(net59)

        net59 = Flatten()(net59)

        net59 = Dense(128)(net59)
        net59 = Dropout(0.25)(net59)
        net59 = Dense(16)(net59)
        net59 = Dropout(0.25)(net59)
        output = Dense(len(self.labels), activation='sigmoid')(net59)

        model = Model(inputs=input1, outputs=output)

        return model
