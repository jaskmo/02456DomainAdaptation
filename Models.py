
# coding: utf-8

# In[ ]:


from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy as acc
from keras.applications.vgg16 import VGG16
from keras.losses import categorical_crossentropy
from extras.flip_gradient import flip_gradient
from keras.backend import in_test_phase, learning_phase
from numpy import floor_divide
import tensorflow as tf
#from ourUtils import 


# ### lable modle without DA

# In[ ]:


def lable_model(l2_reg = 0.01, do_rate = 0, vgg_train = True):
    # Load the convolutional part of the VGG16 network 
    vgg16Conv = VGG16(weights='imagenet', include_top=False)

    # Input to network
    vggInput = Input(shape=(224, 224, 3), name='image_input')
    # Output of convolutional part
    output_vgg16Conv = vgg16Conv(vggInput)
    # Stack lable layers
    preDns = Flatten(name='preLp')(output_vgg16Conv)
    dns1 = Dense(2048, activation='relu', kernel_initializer='glorot_normal', 
                 bias_initializer='glorot_normal', kernel_regularizer=l2(l=l2_reg), name='lpl1')(preDns)
    dns1Do = Dropout(rate=do_rate, seed=42, name='lpl1Do')(dns1)
    dns2 = Dense(1024, activation='relu', kernel_initializer='glorot_normal', 
                 bias_initializer='glorot_normal', kernel_regularizer=l2(l=l2_reg), name='lpl2')(dns1Do)
    modelOut = Dense(5, activation='softmax', kernel_initializer='glorot_normal', name='lplOut')(dns2)

    vggConvSleep = Model(inputs=vggInput, outputs=modelOut)

    if not vgg_train:
        for layer in vggConvSleep.layers[:2]:
            layer.trainable = False
    
    # Optimizer
    optimize = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # Compile the model
    vggConvSleep.compile(loss='categorical_crossentropy', optimizer=optimize, metrics=['categorical_accuracy'])

    # Get model summary
    vggConvSleep.summary()
    
    return vggConvSleep


# ### DAnet

# In[ ]:


def DA_model(batch_size, l2_reg = 0.01, do_rate = 0, vgg_train = True):
    #init
    lam_slice = floor_divide(batch_size,2,dtype='int8')
    # Load the convolutional part of the VGG16 network 
    vgg16Conv = VGG16(weights='imagenet', include_top=False)

    # Input to network
    vggInput = Input(shape=(224, 224, 3), name='image_input')
    # Output of convolutional part
    output_vggConv = vgg16Conv(vggInput)
    # pre Dence layer
    preDns = Flatten(name='preDa')(output_vggConv)
    # Stack lable layers
    lpl1 = Dense(2048, activation='relu', kernel_initializer='glorot_normal', 
                 bias_initializer='glorot_normal', kernel_regularizer=l2(l=l2_reg), name='lpl1')(preDns) #(lplSlice)
    lpl1Do = Dropout(rate=do_rate, seed=42, name='lpl1Do')(lpl1)
    lpl2 = Dense(1024, activation='relu', kernel_initializer='glorot_normal', 
                 bias_initializer='glorot_normal', kernel_regularizer=l2(l=l2_reg), name='lpl2')(lpl1Do)
    lplOut = Dense(5, activation='softmax', kernel_initializer='glorot_normal', name='lplOut')(lpl2)
    # Stack domain layers
    flipGrad = Lambda(lambda x: flip_gradient(x,1),name='flipGrad')(preDns)
    dpl1 = Dense(2048, activation='relu', kernel_initializer='glorot_normal', 
                 bias_initializer='glorot_normal', kernel_regularizer=l2(l=l2_reg), name='dpl1')(flipGrad)
    dpl1Do = Dropout(rate=do_rate, seed=42, name='dpl1Do')(dpl1)
    dpl2 = Dense(1024, activation='relu', kernel_initializer='glorot_normal', 
                 bias_initializer='glorot_normal', kernel_regularizer=l2(l=l2_reg), name='dpl2')(dpl1Do)
    dplOut = Dense(2, activation='softmax', kernel_initializer='glorot_normal', name='dplOut')(dpl2)
    
    lplOut._uses_learning_phase = True
    
    #stitch modle together
    vggConvSleep = Model() 
    
    if not vgg_train:
        for layer in vggConvSleep.layers[:2]:
            layer.trainable = False
    
    # Optimizer
    optimize = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    # Compile the model
    vggConvSleep.compile()

    # Get model summary
    vggConvSleep.summary()
    
    return vggConvSleep


# In[ ]:




