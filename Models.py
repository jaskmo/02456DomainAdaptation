
# coding: utf-8

# In[2]:


from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input, Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy as acc
from keras.applications.vgg16 import VGG16
from keras.losses import categorical_crossentropy
from extras.flip_gradient import flip_gradient
from numpy import floor_divide
import numpy as np
#from ourUtils import 


# ### lable modle without DA

# In[ ]:


def lable_model(l2_reg = 0.01, do_rate = 0, vgg_train = True, nrUnits = [2048, 1024]):
    # Load the convolutional part of the VGG16 network 
    vgg16Conv = VGG16(weights='imagenet', include_top=False)

    # Input to network
    vggInput = Input(shape=(224, 224, 3), name='image_input')
    # Output of convolutional part
    output_vgg16Conv = vgg16Conv(vggInput)
    # Stack lable layers
    preDns = Flatten(name='preLp')(output_vgg16Conv)
    preDnsBN = BatchNormalization()(preDns)
    preDnsDo = Dropout(rate=do_rate, seed=42, name='preDnsDo')(preDnsBN)
    #dns1 = Dense(nrUnits[0], activation='relu', kernel_initializer='glorot_normal', 
    #             bias_initializer='glorot_normal', kernel_regularizer=l2(l=l2_reg), name='lpl1')(preDnsDo)
    dns1 = Dense(nrUnits[0], kernel_initializer='glorot_normal', 
                 bias_initializer='glorot_normal', kernel_regularizer=l2(l=l2_reg), name='lpl1')(preDnsDo)
    dns1BN = BatchNormalization()(dns1)
    dns1ACT = Activation('relu')(dns1BN)
    dns1Do = Dropout(rate=do_rate, seed=42, name='lpl1Do')(dns1ACT)
    dns2 = Dense(nrUnits[1], kernel_initializer='glorot_normal', 
                 bias_initializer='glorot_normal', kernel_regularizer=l2(l=l2_reg), name='lpl2')(dns1Do)
    dns2BN = BatchNormalization()(dns2)
    dns2ACT = Activation('relu')(dns2BN)
    modelOut = Dense(5, activation='softmax', kernel_initializer='glorot_normal', name='lplOut')(dns2ACT)

    vggConvSleep = Model(inputs=vggInput, outputs=modelOut)

    if not vgg_train:
        for layer in vggConvSleep.layers[1].layers[:-2]:
            layer.trainable = False
        vggConvSleep.layers[1].layers[-2].kernel_regularizer = vggConvSleep.layers[-4].kernel_regularizer

    # Optimizer
    optimize = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # Compile the model
    vggConvSleep.compile(loss='categorical_crossentropy', optimizer=optimize, metrics=['categorical_accuracy'])

    # Get model summary
    vggConvSleep.summary()
    
    return vggConvSleep


# ### DAnet

# In[2]:


def DA_model(lamFunk, l2_reg = 0.01, do_rate_dpl = 0, do_rate_lpl = 0, vgg_train = True, nrUnits = [2048, 1024]):
    
    # Load the convolutional part of the VGG16 network 
    vgg16Conv = VGG16(weights='imagenet', include_top=False)

    # Input to network
    vggInput = Input(shape=(224, 224, 3), name='image_input')
    # Output of convolutional part
    output_vggConv = vgg16Conv(vggInput)
    # pre Dence layer
    preDns = Flatten(name='preDa')(output_vggConv)
    # create the shared part as a model instance
    sharedVGG16 = Model(inputs=vggInput, outputs=preDns)
    
    #Create lable predictive model
    lpl_input = Input(shape=(224,224,3), name='lplInput')
    # run lpl input through the shared part of the network
    lpl_vgg_out = sharedVGG16(lpl_input)
    lpl_vgg_out_BN = BatchNormalization()(lpl_vgg_out)
    lpl_vgg_outDo = Dropout(rate=do_rate_lpl, seed=42, name='lpl_vgg_outDo')(lpl_vgg_out_BN)
    lpl1 = Dense(nrUnits[0], kernel_initializer='glorot_normal', 
                 kernel_regularizer=l2(l=l2_reg), name='lpl1')(lpl_vgg_outDo)
    lpl1BN = BatchNormalization()(lpl1)
    lpl1ACT = Activation('relu')(lpl1BN)
    lpl1Do = Dropout(rate=do_rate_lpl, seed=42, name='lpl1Do')(lpl1ACT)
    lpl2 = Dense(nrUnits[1], kernel_initializer='glorot_normal', 
                 kernel_regularizer=l2(l=l2_reg), name='lpl2')(lpl1Do)
    lpl2BN = BatchNormalization()(lpl2)
    lpl2ACT = Activation('relu')(lpl2BN)
    lplOut = Dense(5, activation='softmax', kernel_initializer='glorot_normal', name='lplOut')(lpl2ACT)
    
    #Create domain predictive model 
    dpl_input = Input(shape=(224,224,3), name='dplInput')
    # run dpl input through the shared part of the network
    dpl_vgg_out = sharedVGG16(dpl_input)
    #lambdalayer for the flip gradient
    flipGrad = Lambda(lambda x: flip_gradient(x,lamFunk),name='flipGrad')(dpl_vgg_out)
    dpl1 = Dense(nrUnits[0], activation='relu', kernel_initializer='glorot_normal', 
                 kernel_regularizer=l2(l=l2_reg), name='dpl1')(flipGrad)
    dpl1Do = Dropout(rate=do_rate_dpl, seed=42, name='dpl1Do')(dpl1)
    dpl2 = Dense(nrUnits[1], activation='relu', kernel_initializer='glorot_normal', 
                 kernel_regularizer=l2(l=l2_reg), name='dpl2')(dpl1Do)
    dplOut = Dense(2, activation='softmax', kernel_initializer='glorot_normal', name='dplOut')(dpl2)
    
    #stitch modle together
    DAnetwork = Model(inputs=[lpl_input, dpl_input], outputs=[lplOut, dplOut]) 
    
    if not vgg_train:
        for layer in DAnetwork.layers[1].layers[1].layers[:-2]:
            layer.trainable = False
        DAnetwork.layers[1].layers[1].layers[-2].kernel_regularizer = DAnetwork.layers[-8].kernel_regularizer    
    
    # Optimizer
    optimize = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    # Compile the model
    DAnetwork.compile(optimizer=optimize, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Get model summary
    DAnetwork.summary()
    
    return DAnetwork

