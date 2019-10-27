#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import pandas as pd
import numpy as np
import sklearn as sk
import xgboost as xgb
from keras.utils import to_categorical
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

from sklearn.preprocessing import MultiLabelBinarizer

import keras.backend as ke


# In[238]:


#from clineage.utils.groups
import itertools
def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# In[1]:


df_mono = pd.read_pickle("/home/dcsoft/s/Ofir/df_m_sr_panel_bi_2019_08_26.pickle")		


# In[21]:


#Remove length 3,4 and all nan sites

filt = lambda x: ~np.isnan(x) and x >= 0
minus5 = lambda x: x-5
df_mono['raw_labels'] = pd.Series([list(filter(filt, map(minus5,row))) for row in df_mono[['bi_allele1','bi_allele2']].values.tolist()], df_mono.index)


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
input_hists_read_num = Input(shape=(test_rfe.shape[1],), dtype='float32')
input_group_id_onehot = Input(shape=(BATCH_G_S,), dtype='float')

#Build compute
x = input_hists_read_num
x = Dropout(DPLVL, noise_shape=None, seed=None)(Dense(DNSS, activation='linear')(x))
for _ in range(DENSE_RELU_LAYERS):
    x = Dropout(DPLVL, noise_shape=None, seed=None)(Dense(DNSS, activation='relu')(x))
main_output = Dense(n_calsses, activation='sigmoid', name='main_output')(x)


# In[239]:



def custom_loss(grp_enc):
    def my_loss(y_true,y_pred,alpha=1,beta=1.5, gamma=0.0,delta=0.08):
        # b - main batch idx
        # g - group
        # l - label (repeat length)
        grp_avg_pred = tf.einsum('bg,bl->lg',grp_enc, y_pred) / ke.sum(grp_enc,axis=0)
        exp_grp_avg_pred = tf.einsum('bg,lg->bl', grp_enc, grp_avg_pred) 
        exp_M = exp_grp_avg_pred #for shorthand

        ret =  alpha * ke.mean(ke.binary_crossentropy(y_true,y_pred)) +                beta  * ke.mean(y_pred[:,:-1]*y_pred[:,1:]) +                delta * ke.mean(y_pred[:,:-1]*exp_M[:,1:] + y_pred[:,1:]*exp_M[:,:-1])
        return ret
    return my_loss

def onehot_encoding(X, i):
    OHE = np.zeros((X.shape[0],BATCH_G_S), dtype='float32')
    OHE[:,i] = 1.0
    return OHE

def my_generator(groupby_obj):
    while 1:
        non_singleton_groups = filter(lambda y: y.shape[0] > 1, 
                                      map(lambda x: groupby_obj.get_group(x), groupby_obj.groups))
        for chunk in filter(None, grouper(BATCH_G_S, non_singleton_groups)):
            X = [np.array(g[list(range(3,30)) + ["read_num"]],dtype='float32') for g in chunk]
            Y = [np.array(multilabel_binarizer.transform(g.raw_labels),dtype='float32') for g in chunk]
            onehot_groupid = [onehot_encoding(x,i) for i,x in enumerate(X)]
            yield [np.concatenate(X), np.concatenate(onehot_groupid)],np.concatenate(Y)


# In[216]:


#Model
model = Model(inputs=[input_hists_read_num,input_group_id_onehot], outputs=main_output)


# In[224]:


model.compile(optimizer=keras.optimizers.Adam(), loss=custom_loss(input_group_id_onehot),metrics=[metrics.categorical_accuracy])


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


EPOCHS = 2


# In[26]:


model.fit([train_rfe], y_train, epochs=EPOCHS,
         validation_data=([test_rfe],y_test), batch_size=int(10240),
         callbacks=[es_cb, mc_cb])


# In[227]:


import math
def batches_in_epoch(train_groups,groups_by_batch):
        return int(math.ceil(len(train_groups)/groups_by_batch))


# In[241]:


model.fit_generator(my_generator(train_groupby), epochs=EPOCHS,
            steps_per_epoch=batches_in_epoch(train_groups,BATCH_G_S))


# In[ ]:


filter


# In[ ]:




