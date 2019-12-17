print("in py")

# In[ ]:

import os

import pickle
import pandas as pd
import numpy as np
import sklearn as sk
import xgboost as xgb
from keras.utils import to_categorical, multi_gpu_model
import tensorflow as tf
import collections
import random

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Activation, Embedding, concatenate, Input, Dropout, CuDNNLSTM, Bidirectional, Conv1D
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras import regularizers, metrics
import frogress
import keras.callbacks, keras.optimizers
import keras

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

import keras.backend as ke
import time

class Timer:
    def __init__(self):
        self.last = time.time()
    
    def elapsed(self):
        new = time.time()
        duration = new - self.last
        self.last = new
        return duration
    
    def report_milestone(self,title):
        print(title,':',self.elapsed(),'s')
        

# In[238]:


#from clineage.utils.groups
import itertools
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip(*args)

# In[1]:

NN_DIR = "/home/dcsoft/s/nn_project/"
#DATASET_PATH = "df_m_sr_panel_bi_2019_08_26.pickle"
#NN_DIR = "/home/dcsoft/s/Ofir/"
# DATASET_PATH = "df_m_sr_panel_bi_2019_11_27.pickle"
DATASET_PATH = "df_m_sr_panel_mono_pseudo_2019_12_04_merged_ac_only_filtered30.pickle"
PARAMS_DIR = NN_DIR + "nn_params/"

print("Started reading pickle")
timer = Timer()
df_full = pd.read_pickle(NN_DIR + DATASET_PATH)
timer.report_milestone("Reading pickle")

# In[21]:


#Remove length 3,4 and all nan sites

filt = lambda x: ~np.isnan(x) and x >= 0
minus5 = lambda x: x-5
df_full['raw_labels'] = pd.Series([list(filter(filt, map(minus5,row))) for row in df_full[['bi_allele1','bi_allele2']].values.tolist()], df_full.index)
df_full = df_full[df_full.raw_labels.apply(lambda x: len(x) > 0)]

#df_full['raw_labels'] = pd.Series([list(filter(filt, map(minus5,row))) for row in df_full[['pseudo_allele1','pseudo_allele2']].values.tolist()], df_full.index)

# remove all rows with no calling label




# In[212]:


#Partition Train-test. don't merge groups as this may allow overfitting by learning experiment specific traits (i.e. exact readnum->length)


dd = df_full.groupby(["msid","individual_id"])
lst = list(dd.groups)
random.shuffle(lst)
train_groups = lst[int(len(lst)/20):]
test_groups = lst[:int(len(lst)/20)]
train_data = pd.concat( [ dd.get_group(group) for group in train_groups]).reset_index(drop=False)
test_data = pd.concat( [ dd.get_group(group) for group in test_groups]).reset_index(drop=False)
train_groupby = train_data.groupby(["msid","individual_id"])


# In[213]:


#Format data

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(train_data.raw_labels)
y_train = multilabel_binarizer.transform(train_data.raw_labels)
y_test = multilabel_binarizer.transform(test_data.raw_labels)
train_rfe = train_data[list(range(3,30)) + ["read_num"]]
test_rfe = test_data[list(range(3,30)) + ["read_num"]]
n_calsses = len(multilabel_binarizer.classes_)



ind_ids = {ind_id:i for i,ind_id in enumerate(df_full.individual_id.unique())}
ind_id_binarizer = LabelBinarizer()
#for efficiency we use our version of binarizer
ind_id_binarizer.classes_ = np.array(list(ind_ids.keys()))
# In[214]:
timer.report_milestone("Preprocessing")


#        ------------Model-----------

#Drop out level
DPLVL = 0.02

DENSE_RELU_LAYERS = 6

#Dense layer size
DNSS = 256

#Amount of groups [msid,individual_id (i.e. patient)] per batch
BATCH_G_S = 300

#Define inputs
input_hists_read_num = Input(shape=(train_rfe.shape[1] +  len(ind_id_binarizer.classes_),), dtype='float32')
input_group_id_onehot = Input(shape=(BATCH_G_S,), dtype='float')

#Build compute
x = input_hists_read_num
x = Dropout(DPLVL, noise_shape=None, seed=None)(Dense(DNSS, activation='linear')(x))
for _ in range(DENSE_RELU_LAYERS):
    x = Dropout(DPLVL, noise_shape=None, seed=None)(Dense(DNSS, activation='relu')(x))
main_output = Dense(n_calsses, activation='sigmoid', name='main_output')(x)


# In[239]:


def custom_loss(grp_enc,alpha=1,beta=0.2, gamma=0.15,delta=0.6):
#def custom_loss():
    def my_loss(y_true,y_pred):
        # b - main batch idx
        # g - group
        # l - label (repeat length)
        grp_avg_pred = tf.einsum('bg,bl->lg',grp_enc, y_pred) / ke.sum(grp_enc,axis=0)
        exp_grp_avg_pred = tf.einsum('bg,lg->bl', grp_enc, grp_avg_pred) 
        exp_M = exp_grp_avg_pred #for shorthand

        ret =  alpha * ke.mean(ke.binary_crossentropy(y_true,y_pred)) + \
        beta  * ke.mean(y_pred[:,:-1]*y_pred[:,1:]) + \
        gamma * ke.mean(ke.mean(y_pred)) + \
        delta * ke.mean(y_pred[:,:-1]*exp_M[:,1:] + y_pred[:,1:]*exp_M[:,:-1])
        #ret =  alpha * ke.mean(ke.binary_crossentropy(y_true,y_pred))
        return ret
    return my_loss

def onehot_encoding(X, i, bins=None):
    if bins == None:
        bins = BATCH_G_S
    OHE = np.zeros((X.shape[0],bins), dtype='float32')
    OHE[:,i] = 1.0
    return OHE
    
def preprocess_groupby(groupby_obj, df_full):
    ind_ids = {ind_id:i for i,ind_id in enumerate(df_full.individual_id.unique())}
    group_lst = [groupby_obj.get_group(x) for x in  groupby_obj.groups]
    group_X__Y = [(np.concatenate([np.array(x[list(range(3,30)) + ["read_num"]],dtype='float32'), \
                        onehot_encoding(x,ind_ids[x.individual_id.iloc[0]],bins=len(ind_ids))],axis=1), \
                         np.array(multilabel_binarizer.transform(x.raw_labels),dtype='float32')) \
                        for x in  group_lst if len(x) > 1]
    return group_X__Y

def my_generator(group_X__Y):
    while 1:
        random.shuffle(group_X__Y)
        for chunk in filter(None, grouper(group_X__Y, BATCH_G_S)):
            onehot_groupid = [onehot_encoding(x[0],i) for i,x in enumerate(chunk)]
            yield [np.concatenate([x[0] for x in chunk]), np.concatenate(onehot_groupid)],np.concatenate([x[1] for x in chunk])
            
# In[216]:


#Model
model = multi_gpu_model(Model(inputs=[input_hists_read_num,input_group_id_onehot], outputs=main_output),gpus=2)
#model = Model(inputs=[input_hists_read_num,input_group_id_onehot], outputs=main_output)

# In[224]:


model.compile(optimizer=keras.optimizers.Adam(), loss=custom_loss(input_group_id_onehot),metrics=[metrics.binary_accuracy])
#model.compile(optimizer=keras.optimizers.Adam(), loss=custom_loss(),metrics=[metrics.binary_accuracy])


# In[218]:


model.summary()


# In[225]:


es_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      ## if min_delta is not acheived between epochs it's not an improvement
                                      patience=100,  ## number of epochs to allow without improvement
                                      verbose=0, mode='auto')
                                      
#for now save all, if we'll run in to space issues, we'll change
# mc_cb = keras.callbacks.ModelCheckpoint(PARAMS_DIR+'NN_model_new_data.h5', verbose=1, save_best_only=False) 

PERIOD = 1000
periodic_save_cb = keras.callbacks.ModelCheckpoint(PARAMS_DIR+'NN_model_epoch{epoch:08d}______params___1__0.5__0__10.h5', 
                                     save_weights_only=False, period=PERIOD)
#monitor='val_loss', mode='min', verbose=1, save_best_only=True)


# In[226]:


EPOCHS = 15000


# In[26]:


#model.fit([train_rfe], y_train, epochs=EPOCHS,
#         validation_data=([test_rfe],y_test), batch_size=int(10240),
#         callbacks=[es_cb, mc_cb])


# In[227]:


import math
def batches_in_epoch(train_groups,groups_by_batch):
        return int(math.ceil(len(train_groups)/groups_by_batch))


# In[241]:

preprocessed_groups = preprocess_groupby(train_groupby, df_full)

timer.report_milestone("Before 1st epoch")
model.fit_generator(my_generator(preprocessed_groups), epochs=EPOCHS,
            steps_per_epoch=batches_in_epoch(train_groups,BATCH_G_S),callbacks=[periodic_save_cb])
timer.report_milestone("Finished training")


df_pred = np.concatenate([df_full[list(range(3,30)) + ["read_num"]],ind_id_binarizer.transform(df_full.individual_id.astype(int))],axis=1)            
df_x = df_full
model_pred = Model(inputs=[input_hists_read_num], outputs=main_output)


preds = model_pred.predict(df_pred)
for i in range(25):
    df_x["p" + str(i)] = preds[:,i]
    


#new_labels = []
#for j,pred in frogress.bar(enumerate(preds)):
#    new_labels.append(sorted(enumerate(pred), key=lambda x:x[1],reverse=True)[:2])
    

#df_x["plabels"] = new_labels
pickle.dump(df_x[['read_num'] + ["p" + str(i) for i in range(25)]],open(NN_DIR + "preds____1__0.5__0__10__15kepochs.pkl","wb"))

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cudnn-v7.0-for-cuda-9.0/lib64/
#server n96

#import sys
#sys.path.append('/home/dcsoft/clineage/')
#import clineage.wsgi
#from sequencing.calling.hist import Histogram
#Histogram({i+6:v for i,v in enumerate(df_x.loc[(31414857,75012,16818)][[i for i in range(3,40)]])})
