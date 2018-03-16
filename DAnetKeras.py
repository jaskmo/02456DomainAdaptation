
# coding: utf-8

# ### Imports

# In[1]:


from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, Callback
from datetime import datetime
import keras.backend as K
import extras.ourUtils as utils
import numpy as np
import Models
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from extras.ourUtils import reverse_data_split, create_data_split


# ### Init

# In[2]:


batch_size = 64
nrEpochs = 25
full_train = True
path = '/home/jaskmo/Documents/programering/02456DomainAdaptation/'
source_data = path + 'newTaperImg/Physionet/'
target_data = path + 'newTaperImg/Hospital/'
stdout_cell = sys.stdout
MIQ = ['DA', 'target', 'source']
kf = KFold(n_splits = 10)
n_subjects_phys = np.arange(1,21);
n_subjects_hosp = np.arange(1,35);

class FlipControle(Callback):
    def __init__(self, alphaIn):
        self.alpha = alphaIn
        print(K.get_value(lamFunk))
        
    def on_epoch_end(self, epoch, logs={}):
        p = (epoch+1)/nrEpochs
        K.set_value(self.alpha, (2/(1+np.exp(-10*p)))-1)
        print(K.get_value(lamFunk))


# ## Get data as generators

# In[3]:


datagen = ImageDataGenerator(rescale=1./255)

# make a data generator for dplInput
def train_gen_DAnet(source, target, batch_size):
    half = batch_size//2
    while True:
        source_data, source_lable = source.next()
        target_data, target_lable = target.next()
        if len(source_lable) != batch_size or len(target_lable) != batch_size:
            continue
        dpl_data = np.concatenate((source_data[:half,...],target_data[:half,...]),axis=0)
               
        domain_tmp = np.ones(batch_size, dtype='int8')
        domain_tmp[half:] = domain_tmp[half:] * 0
        dpl_lable = np.concatenate((domain_tmp.reshape(batch_size,1),
                                       np.flip(domain_tmp,0).reshape(batch_size,1)),1)

        yield({'lplInput':source_data,'dplInput':dpl_data}, {'lplOut':source_lable,'dplOut':dpl_lable})
        
def test_gen_DAnet(source, target, batch_size):
    half = batch_size//2
    while True:
        source_data, source_lable = source.next()
        target_data, target_lable = target.next()
        if len(source_lable) != batch_size or len(target_lable) != batch_size:
            continue
        dpl_data = np.concatenate((source_data[:half,...],target_data[:half,...]),axis=0)
               
        domain_tmp = np.ones(batch_size, dtype='int8')
        domain_tmp[half:] = domain_tmp[half:] * 0
        dpl_lable = np.concatenate((domain_tmp.reshape(batch_size,1),
                                       np.flip(domain_tmp,0).reshape(batch_size,1)),1)

        yield({'lplInput':target_data,'dplInput':dpl_data}, {'lplOut':target_lable,'dplOut':dpl_lable})


# In[4]:


whitelist = dir()
whitelist.append('whitelist')
whitelist.append('phys_split')
whitelist.append('hosp_split')
whitelist.append('this')


# In[5]:


for phys_split, hosp_split in zip(kf.split(n_subjects_phys),kf.split(n_subjects_hosp)):

    # memory management
    this = sys.modules[__name__]
    for n in dir():
        if n not in whitelist: delattr(this, n)
    K.clear_session()

    #Define current time
    now = datetime.now()
    #print("PHYS:\n" + "Train:" + str(phys_split[0]) + '\n' + 'Test:' + str(phys_split[1]) + "\n")
    #print("Hosp:\n" + "Train:" + str(hosp_split[0]) + '\n' + 'Test:' + str(hosp_split[1]) + "\n")
    
    reverse_data_split(source_data)
    reverse_data_split(target_data)
    
    #Data split for physionet
    create_data_split(source_data,phys_split[1]+1)
    #Data split for target
    create_data_split(target_data,hosp_split[1]+1)
    
    
    #### Train data


    train_gen_source = datagen.flow_from_directory(source_data + '/train', target_size=(224, 224), 
                                                   batch_size=batch_size, class_mode='categorical', shuffle=True)

    train_gen_target = datagen.flow_from_directory(target_data + '/train', target_size=(224, 224), 
                                                   batch_size=batch_size, class_mode='categorical', shuffle=True)

    train_gen_DA = train_gen_DAnet(train_gen_source, train_gen_target, batch_size)

    train_stepE = np.floor_divide(train_gen_source.n, batch_size)
    train_stepE_target = np.floor_divide(train_gen_target.n, batch_size)
    #### validation data

    valid_gen_source = datagen.flow_from_directory(source_data + '/validation', target_size=(224, 224), 
                                                   batch_size=batch_size, class_mode='categorical', shuffle=True)
    valid_gen_target = datagen.flow_from_directory(target_data + '/validation', target_size=(224, 224), 
                                                   batch_size=batch_size, class_mode='categorical', shuffle=True)

    valid_gen_DA = test_gen_DAnet(valid_gen_source, valid_gen_target, batch_size)

    valid_stepE = np.floor_divide(valid_gen_source.n, batch_size)
    valid_stepE_target = np.floor_divide(valid_gen_target.n, batch_size)

    #### test data

    test_gen_source = datagen.flow_from_directory(source_data + '/test', target_size=(224, 224), 
                                                   batch_size=batch_size, class_mode='categorical', shuffle=True)
    test_gen_target = datagen.flow_from_directory(target_data + '/test', target_size=(224, 224), 
                                                   batch_size=batch_size, class_mode='categorical', shuffle=True)

    test_gen_DA = test_gen_DAnet(test_gen_source, test_gen_target, batch_size)

    test_stepE = np.floor_divide(test_gen_source.n, batch_size)
    test_stepE_target = np.floor_divide(test_gen_target.n, batch_size)   
    
    
    #Define lamFunk
    lamFunk = K.variable(0.0)
    
    for item in MIQ:
        if item=="DA":         
            current_model = Models.DA_model(lamFunk,do_rate_dpl=0, do_rate_lpl=0.5, vgg_train=False, nrUnits=[512,512])
        else:
            current_model = Models.lable_model(do_rate=0.5, vgg_train=False, nrUnits=[512,512])
        
        csv_logger = CSVLogger('/media/jaskmo/ELEK/bme/Project02456/trainingLog/' + item + 
                        str(now.day) + '-' + str(now.month) + '-' + str(now.year) + '_' + 
                        str(now.hour) + str(now.minute) + '.log')
        if item=='DA':
            current_model.fit_generator(train_gen_DA, train_stepE, epochs=nrEpochs, verbose=1, validation_data=valid_gen_DA, 
                            validation_steps=valid_stepE, callbacks=[csv_logger,FlipControle(lamFunk)], initial_epoch=0,
                            max_queue_size=5, class_weight = {'lplOut':[1,0.33,0.33,0.33,0.5],'dplOut':[1,1]})
        elif item=='source':
            current_model.fit_generator(train_gen_source, train_stepE, epochs=nrEpochs, verbose=1, validation_data=valid_gen_source, 
                            validation_steps=valid_stepE, callbacks=[csv_logger], initial_epoch=0,
                            max_queue_size=5, class_weight = [1,0.33,0.33,0.33,0.5])
        else:
            current_model.fit_generator(train_gen_target, train_stepE_target, epochs=nrEpochs, verbose=1, validation_data=valid_gen_target, 
                            validation_steps=valid_stepE_target, callbacks=[csv_logger], initial_epoch=0,
                            max_queue_size=5, class_weight = [1,0.33,0.33,0.33,0.5])
                     
        if item == "DA":
            DAlpm = utils.dissect_DAlpm(current_model)
            current_model = DAlpm
            
#         #Save the model
#         current_model.save(filepath=path + 'models/'+ item + str(now.day) + '-' + str(now.month) + '-' + str(now.year) + '_' + 
#                        str(now.hour) + str(now.minute) + '.h5')
        
        #TEST
        DTD_dic = {'Physionet':test_gen_source,'Hospital':test_gen_target}
        # save to file 
        test_file = '/media/jaskmo/ELEK/bme/Project02456/testLog/'+ item + str(now.day) + '-' + str(now.month) + '-' + str(now.year) + '_' + str(now.hour) + str(now.minute) + '.log'
        sys.stdout = open(test_file, 'w')
        
        for dom in ['Physionet', 'Hospital']:
            test_img, test_lable = DTD_dic[dom].next()
            for count in range(DTD_dic[dom].n//batch_size):
                tmp_img, tmp_lable = DTD_dic[dom].next()
                test_img = np.concatenate((test_img, tmp_img), axis=0)
                test_lable = np.concatenate((test_lable, tmp_lable),axis=0)

            # Compute the test metrecis 
            inv_map = {v: k for k, v in DTD_dic[dom].class_indices.items()}
            target_names = list(inv_map.values())

            targets_test_int = [np.where(r == 1)[0][0] for r in test_lable]
            y_pred = current_model.predict(test_img)
            y_pred2 = np.argmax(y_pred, axis = 1)
            # Test accuracy:
            acc = accuracy_score(targets_test_int, y_pred2)
            
            conf_mat = confusion_matrix(targets_test_int, y_pred2)
            # Per class metrics
            class_report = classification_report(targets_test_int, y_pred2, target_names=target_names)

            print('Accuracy on ' + dom + ' data = ' + str(acc) +'\n \n' + 
                  'Confution matric on ' + dom + ' data: \n' + str(conf_mat) + '\n\n' + 
                  'Class report on ' + dom + ' data: \n' + class_report + '\n\n\n\n')

        sys.stdout = stdout_cell


# In[ ]:





# In[ ]:




