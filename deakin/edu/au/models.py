# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021.  Mohamed Reda Bouadjenek, Deakin University                       +
#               Email:  reda.bouadjenek@deakin.edu.au                                    +
#                                                                                        +
#       Licensed under the Apache License, Version 2.0 (the "License");                  +
#       you may not use this file except in compliance with the License.                 +
#       You may obtain a copy of the License at:                                         +
#                                                                                        +
#       http://www.apache.org/licenses/LICENSE-2.0                                       +
#                                                                                        +
#       Unless required by applicable law or agreed to in writing, software              +
#       distributed under the License is distributed on an "AS IS" BASIS,                +
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         +
#       See the License for the specific language governing permissions and              +
#       limitations under the License.                                                   +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Activation, Lambda, Conv2D, MaxPool2D, Reshape,GlobalAveragePooling2D, Multiply, Concatenate, \
experimental,Layer,BatchNormalization,MaxPooling2D
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.applications import VGG19, VGG16, ResNet50, Xception
from deakin.edu.au.data import get_Marine_dataset
import deakin.edu.au.metrics as metrics
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from prettytable import PrettyTable
import tensorflow.keras.backend as K
import numpy
from tensorflow.keras.initializers import Constant
from tensorflow.keras.losses import Loss
from tensorflow.keras.initializers import TruncatedNormal
from keras.regularizers import l2


import tensorflow_addons as tfa


from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                         WEIGHTS_PATH,
                         cache_subdir='models')


#Functions needed to calculate bilinear models

def outer_product_2(x):
    return tf.einsum('abi,abj,abk->aijk',x[0],x[1],x[2])/x[0].get_shape().as_list()[1]
def outer_product(x):
    return K.batch_dot(x[0], x[1], axes = [1, 1]) / x[0].get_shape().as_list()[1]
def signed_sqrt(x):
    return K.sign(x)*K.sqrt(K.abs(x) + 1e-9)
def l2_normalize(x, axis = -1):
    return K.l2_normalize(x, axis=axis)



def get_mout_model(num_classes: list,
                   image_size,
                   conv_base='vgg19',
                   learning_rate=1e-5,
                   loss_weights=[],
                   lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = layers.Input(shape=image_size, name='main_input')
    conv_base = get_conv_base(conv_base, regularizer=regularizer)(in_layer)
    conv_base = layers.Flatten()(conv_base)
    # create output layers
    out_layers = []
    for idx, v in enumerate(num_classes):
        out_layers.append(layers.Dense(v, activation="softmax", name='out_level_' + str(idx))(conv_base))

    # Build the model
    model = Model(name='mout_model',
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


def get_BCNN2(num_classes: list,
              image_size,
              in_layer,
              reverse=False,
              conv_base='vgg19',
              learning_rate=1e-5,
              loss_weights=[],
              lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = in_layer
    conv_base = get_conv_base(conv_base, regularizer=regularizer)(in_layer)
    conv_base = layers.Flatten()(conv_base)
    # create output layers
    logits_layers = []
    out_layers = []
    if reverse:
        num_classes = list(reversed(num_classes))
    for idx, v in enumerate(num_classes):
        if reverse:
            idx = len(num_classes) - idx - 1
        if len(logits_layers) == 0:
            logits = layers.Dense(v, name='logits_level_' + str(idx))(conv_base)
            out_layers.append(layers.Activation(keras.activations.softmax, name='out_level_' + str(idx))(logits))
            logits_layers.append(layers.Activation(keras.activations.relu)(logits))
        else:
            logits = layers.Dense(v, name='logits_level_' + str(idx))(logits)
            out_layers.append(layers.Activation(keras.activations.softmax, name='out_level_' + str(idx))(logits))
            logits_layers.append(layers.Activation(keras.activations.relu)(logits))

    if reverse:
        out_layers = list(reversed(out_layers))
    # Build the model
    if reverse:
        name = 'BCNN2_reversed_model'
    else:
        name = 'BCNN2_model'
    model = Model(name=name,
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


def get_mnets(num_classes: list,
              image_size,
              conv_base='vgg19',
              learning_rate=1e-5,
              loss_weights=[],
              lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = layers.Input(shape=image_size, name='main_input')
    conv_base_list = [get_conv_base(conv_base, regularizer=regularizer) for x in num_classes]
    out_layers = []
    for i in range(len(conv_base_list)):
        conv_base_list[i]._name = 'conv_base' + str(i)
        conv_base_list[i] = conv_base_list[i](in_layer)
        conv_base_list[i] = layers.Flatten()(conv_base_list[i])
        out_layers.append(
            layers.Dense(num_classes[i], activation="softmax", name='out_level_' + str(i))(conv_base_list[i]))

    # Build the model
    model = Model(name='mnets_model',
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


class BaselineModel(Model):
    def __init__(self, taxonomy, *args, **kwargs):
        super(BaselineModel, self).__init__(*args, **kwargs)
        self.taxonomy = taxonomy

    def predict(self, X):
        pred = super().predict(X)
        out = []
        for i in range(len(self.taxonomy) + 1):
            out.append([])
        for v in pred:
            child = np.argmax(v)
            out[-1].append(v)
            for i in reversed(range(len(self.taxonomy))):
                m = self.taxonomy[i]
                row = list(np.transpose(m)[child])
                parent = row.index(1)
                child = parent
                one_hot = np.zeros(len(row))
                one_hot[child] = 1
                # one_hot = np.random.uniform(low=1e-6, high=1e-5, size=len(row))
                # one_hot[child] = 1 - (np.sum(one_hot) - one_hot[child])
                out[i].append(one_hot)
        return out


def get_Baseline_model(num_classes: list,
                       image_size,
                       taxonomy, conv_base='vgg19',
                       learning_rate=1e-5,
                       lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = layers.Input(shape=image_size, name='main_input')
    conv_base = get_conv_base(conv_base, regularizer=regularizer)(in_layer)
    conv_base = layers.Flatten()(conv_base)
    # create output layers
    out_layer = layers.Dense(num_classes[-1], activation="softmax", name='output')(conv_base)
    # Build the model
    model = BaselineModel(name='baseline_model',
                          taxonomy=taxonomy,
                          inputs=in_layer,
                          outputs=out_layer)
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    return model


def get_Classifier_model(num_classes,
                         image_size,
                         conv_base='vgg19',
                         learning_rate=1e-5,
                         lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = layers.Input(shape=image_size)
    # rescale = experimental.preprocessing.Rescaling(1. / 255)(in_layer)
    conv_base = get_conv_base(conv_base, regularizer=regularizer)(in_layer)
    conv_base = layers.Flatten()(conv_base)
    # create output layers
    out_layer = layers.Dense(num_classes, kernel_regularizer=regularizer, activation="softmax")(conv_base)
    # Build the model
    model = Model(inputs=in_layer,
                  outputs=out_layer)
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    return model


def get_MLPH_model(num_classes: list,
                   image_size,
                   in_layer,
                   features = None,
                   learning_rate=1e-4,
                   loss_weights=[],
                   lam=0):
    # Conv base
    
    if features is None:
        
        
        in_layer = in_layer
        conv_base = tf.keras.applications.VGG16(include_top=False, weights='imagenet',input_tensor=in_layer)
        
    else:
        
        conv_base = features


    feature_model_1 = conv_base
    relu5_1_X = feature_model_1.layers[15].output
    relu5_1_X_shape = feature_model_1.layers[15].output_shape
    relu5_2_Y = feature_model_1.layers[16].output
    relu5_2_Y_shape = feature_model_1.layers[16].output_shape
    relu5_3_Z = feature_model_1.layers[17].output  
    relu5_3_Z_shape = feature_model_1.layers[17].output_shape



    relu5_1_X = Reshape([relu5_1_X_shape[1]*relu5_1_X_shape[2],relu5_1_X_shape[-1]])(relu5_1_X)

    relu5_2_Y = Reshape([relu5_2_Y_shape[1]*relu5_2_Y_shape[2],relu5_2_Y_shape[-1]])(relu5_2_Y)

    relu5_3_Z = Reshape([relu5_3_Z_shape[1]*relu5_3_Z_shape[2],relu5_3_Z_shape[-1]])(relu5_3_Z)



    UTXoVTY =  Lambda(outer_product)([relu5_1_X, relu5_2_Y])
    UTXoVTY = Reshape([relu5_1_X_shape[-1]*relu5_2_Y_shape[-1]])(UTXoVTY)
    UTXoVTY = Lambda(signed_sqrt)(UTXoVTY)
    UTXoVTY = Lambda(l2_normalize)(UTXoVTY)


    UTXoWTZ = Lambda(outer_product)([relu5_1_X, relu5_3_Z])
    UTXoWTZ = Reshape([relu5_1_X_shape[-1]*relu5_3_Z_shape[-1]])(UTXoWTZ)
    UTXoWTZ = Lambda(signed_sqrt)(UTXoWTZ)
    UTXoWTZ = Lambda(l2_normalize)(UTXoWTZ)

    VTYoWTZ =  Lambda(outer_product)([relu5_2_Y, relu5_3_Z])
    VTYoWTZ = Reshape([relu5_2_Y_shape[-1]*relu5_3_Z_shape[-1]])(VTYoWTZ)
    VTYoWTZ = Lambda(signed_sqrt)(VTYoWTZ)
    VTYoWTZ = Lambda(l2_normalize)(VTYoWTZ)



    UTXoVTYoUTXoWTZoVTYoWTZ =  Lambda(outer_product_2)([relu5_1_X,relu5_2_Y, relu5_3_Z])
    UTXoVTYoUTXoWTZoVTYoWTZ =  Reshape([relu5_1_X.shape[-1]*relu5_2_Y.shape[-1]* relu5_3_Z.shape[-1]])(UTXoVTYoUTXoWTZoVTYoWTZ)
    UTXoVTYoUTXoWTZoVTYoWTZ =  Lambda(signed_sqrt)(UTXoVTYoUTXoWTZoVTYoWTZ)
    UTXoVTYoUTXoWTZoVTYoWTZ =  Lambda(l2_normalize)(UTXoVTYoUTXoWTZoVTYoWTZ)

    #features
    concat = Concatenate()([UTXoVTY,UTXoWTZ,VTYoWTZ])
    # create output layers
    out_layers = []
    for idx, v in enumerate(num_classes):
        out_layers.append(layers.Dense(v, activation="softmax", kernel_regularizer=l2(1e-08),bias_initializer="glorot_uniform", name='out_level_' + str(idx))(concat))

    # Build the modelget_pred_indexes
    model = Model(name='MLPH_model',
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate = learning_rate,beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 1e-4, amsgrad = False)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


    
def get_BCNN_model(num_classes: list,
                   image_size,
                   loss_weights,
                   learning_rate
                   ):
    

    
    filters_1= 512
    
    filters_2 = 4096
    
    outputs = []

    
    
    in_layer = Input(shape=image_size, name='main_input')
    
    #--- block 1 ---
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(in_layer)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    #--- block 2 ---
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
    #--- coarse 1 branch ---
    c_1_bch = Flatten(name='c1_flatten')(x)
    c_1_bch = Dense(256, activation='relu', name='c1_fc_cifar10_1')(c_1_bch)
    c_1_bch = BatchNormalization()(c_1_bch)
    c_1_bch = Dropout(0.5)(c_1_bch)
    c_1_bch = Dense(256, activation='relu', name='c1_fc2')(c_1_bch)
    c_1_bch = BatchNormalization()(c_1_bch)
    c_1_bch = Dropout(0.5)(c_1_bch)
    c_1_pred = Dense(num_classes[0], activation='softmax', name='c1_predictions_cifar10')(c_1_bch)
    
    outputs.append(c_1_pred)
    
    
    
    
    for idx, v in enumerate(num_classes):
        
        
        if idx >= 2:
        
            #filters_1 = filters_1 * 2
            
            #filters_2 = filters_2 * 4
            
            #--- block 4 ---
            x = Conv2D(filters_1, (3, 3), activation='relu', padding='same', name='block4_conv1_'+ str(idx))(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters_1, (3, 3), activation='relu', padding='same', name='block4_conv2_'+ str(idx))(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters_1, (3, 3), activation='relu', padding='same', name='block4_conv3_'+ str(idx))(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_'+ str(idx))(x)
            
            
            #--- block 5 ---
            x = Conv2D(filters_1, (3, 3), activation='relu', padding='same', name='block5_conv1_'+ str(idx))(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters_1, (3, 3), activation='relu', padding='same', name='block5_conv2_'+ str(idx))(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters_1, (3, 3), activation='relu', padding='same', name='block5_conv3_'+ str(idx))(x)
            x = BatchNormalization()(x)
            
            #--- fine block ---
            c_3_bch = Flatten(name='flatten'+ str(idx))(x)
            c_3_bch = Dense(filters_2, activation='relu', name='fc_1_'+ str(idx))(c_3_bch)
            c_3_bch = BatchNormalization()(c_3_bch)
            c_3_bch = Dropout(0.5)(c_3_bch)
            c_3_bch= Dense(filters_2, activation='relu', name='fc_2_'+ str(idx))(c_3_bch)
            c_3_bch = BatchNormalization()(c_3_bch)
            c_3_bch = Dropout(0.5)(c_3_bch)
            fine_pred = Dense(num_classes[idx], activation='softmax', name='predictions_'+ str(idx))(c_3_bch)
            
            outputs.append(fine_pred)
            
        elif idx==1:
        
            
            #--- block 3 ---
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
            x = BatchNormalization()(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
            x = BatchNormalization()(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
            
            #--- coarse 2 branch ---
            c_2_bch = Flatten(name='c2_flatten')(x)
            c_2_bch = Dense(1024, activation='relu', name='c2_fc_cifar100_1')(c_2_bch)
            c_2_bch = BatchNormalization()(c_2_bch)
            c_2_bch = Dropout(0.5)(c_2_bch)
            c_2_bch = Dense(1024, activation='relu', name='c2_fc2')(c_2_bch)
            c_2_bch = BatchNormalization()(c_2_bch)
            c_2_bch = Dropout(0.5)(c_2_bch)
            c_2_pred = Dense(num_classes[idx], activation='softmax', name='c2_predictions_cifar100')(c_2_bch)
            
            outputs.append(c_2_pred)
    
        
        
        
    model = Model(in_layer, outputs, name='BCNN_C_model')
    
    #model.load_weights(weights_path, by_name=True)
    
    
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    model.compile(optimizer=sgd,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model




class nin_model(Model):

    def __init__(self):
        super(nin_model, self).__init__()

    def build(self, input_shape):
        self.Conv2D1 = Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same')
        self.Conv2D2 = Conv2D(filters=160, kernel_size=(1, 1), activation='relu', padding='same')
        self.Conv2D3 = Conv2D(filters=96, kernel_size=(1, 1), activation='relu', padding='same')

        self.Conv2D4 = Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same')
        self.Conv2D5 = Conv2D(filters=192, kernel_size=(1, 1), activation='relu', padding='same')
        self.Conv2D6 = Conv2D(filters=192, kernel_size=(1, 1), activation='relu', padding='same')

        self.Conv2D7 = Conv2D(filters=192, kernel_size=(3, 3), activation='relu', padding='same')
        self.Conv2D8 = Conv2D(filters=192, kernel_size=(1, 1), activation='relu', padding='same')
        self.Conv2D9 = Conv2D(filters=192, kernel_size=(1, 1), activation='relu', padding='same')

    def call(self, inputs):
        x = self.Conv2D1(inputs)
        x = self.Conv2D2(x)
        x = self.Conv2D3(x)
        x = MaxPool2D(2, strides=2, padding='same')(x)
        x = Dropout(0.5)(x)
        x = self.Conv2D4(x)
        x = self.Conv2D5(x)
        x = self.Conv2D6(x)
        x = MaxPool2D(2, strides=2, padding='same')(x)
        x = Dropout(0.5)(x)
        x = self.Conv2D7(x)
        x = self.Conv2D8(x)
        x = self.Conv2D9(x)
        return GlobalAveragePooling2D()(x)


def get_conv_base(conv_base, regularizer=tf.keras.regularizers.l2(0)):
    if conv_base.lower() == 'vgg19':
        conv_base = VGG19(include_top=False, weights="imagenet")
    elif conv_base.lower() == 'vgg16':
        conv_base = VGG16(include_top=False, weights="imagenet")
    elif conv_base.lower() == 'nin':
        conv_base = nin_model()
    elif conv_base.lower() == 'resnet50':
        conv_base = ResNet50(include_top=False, weights="imagenet")
    elif conv_base.lower() == 'xception':
        conv_base = Xception(include_top=False, weights="imagenet")
    elif conv_base.lower() == 'efficientnetb0':
        conv_base = EfficientNetB0(weights='imagenet', include_top=False)
    else:
        return None
    for layer in conv_base.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    return conv_base





if __name__ == '__main__':
    dataset = get_Marine_dataset(output_level='all',image_size=(64,64),batch_size=32)
    num_classes = [dataset.num_classes_l0, dataset.num_classes_l1, dataset.num_classes_l2,dataset.num_classes_l3,dataset.num_classes_l4]
    model = get_mnets(num_classes, dataset.image_size)
    model.summary()
