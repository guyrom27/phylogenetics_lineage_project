#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import pandas as pd
import numpy as np
import sklearn as sk
import xgboost as xgb
from keras.utils import to_categorical
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


# In[238]:


#from clineage.utils.groups
import itertools
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip(*args)

# In[1]:


df_mono = pd.read_pickle("/home/dcsoft/s/Ofir/df_m_sr_panel_bi_2019_08_26.pickle")


# In[21]:


#Remove length 3,4 and all nan sites

filt = lambda x: ~np.isnan(x) and x >= 0
minus5 = lambda x: x-5
df_mono['raw_labels'] = pd.Series([list(filter(filt, map(minus5,row))) for row in df_mono[['bi_allele1','bi_allele2']].values.tolist()], df_mono.index)
#df_mono['raw_labels'] = pd.Series([list(filter(filt, map(minus5,row))) for row in df_mono[['pseudo_allele1','pseudo_allele2']].values.tolist()], df_mono.index)

# remove all rows with no calling label




# In[212]:


#Partition Train-test. don't merge groups as this may allow overfitting by learning experiment specific traits (i.e. exact readnum->length)


dd = df_mono.groupby(["msid","individual_id"])
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


ind_id_binarizer = LabelBinarizer()
ind_id_binarizer.fit(df_mono.individual_id.astype(str))

# In[214]:



#        ------------Model-----------

#Drop out level
DPLVL = 0.03

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



def custom_loss(grp_enc):
#def custom_loss():
    def my_loss(y_true,y_pred,alpha=1,beta=0.2, gamma=0.0,delta=0.6):
        # b - main batch idx
        # g - group
        # l - label (repeat length)
        grp_avg_pred = tf.einsum('bg,bl->lg',grp_enc, y_pred) / ke.sum(grp_enc,axis=0)
        exp_grp_avg_pred = tf.einsum('bg,lg->bl', grp_enc, grp_avg_pred) 
        exp_M = exp_grp_avg_pred #for shorthand

        ret =  alpha * ke.mean(ke.binary_crossentropy(y_true,y_pred)) +
		beta  * ke.mean(y_pred[:,:-1]*y_pred[:,1:]) + 
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
	
def preprocess_groupby(groupby_obj, df_mono):
    ind_ids = {ind_id:i for i,ind_id in enumerate(df_mono.individual_id.unique())}
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
model = Model(inputs=[input_hists_read_num,input_group_id_onehot], outputs=main_output)
#model = Model(inputs=[input_hists_read_num], outputs=main_output)


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
mc_cb = keras.callbacks.ModelCheckpoint('sim2_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)


# In[226]:


EPOCHS = 20000


# In[26]:


#model.fit([train_rfe], y_train, epochs=EPOCHS,
#         validation_data=([test_rfe],y_test), batch_size=int(10240),
#         callbacks=[es_cb, mc_cb])


# In[227]:


import math
def batches_in_epoch(train_groups,groups_by_batch):
        return int(math.ceil(len(train_groups)/groups_by_batch))


# In[241]:

preprocessed_groups = preprocess_groupby(train_groupby, df_mono)

model.fit_generator(my_generator(preprocessed_groups), epochs=EPOCHS,
            steps_per_epoch=batches_in_epoch(train_groups,BATCH_G_S))


# In[ ]:

