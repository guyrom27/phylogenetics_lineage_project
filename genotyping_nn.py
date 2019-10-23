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


# In[27]:


import keras.backend as ke

def my_loss(y_true,y_pred,alpha=1,beta=1.5, gamma=0.0,delta=0.08):
	avg_pred = ke.mean(y_pred,axis=0)
	return alpha*ke.mean(ke.binary_crossentropy(y_true,y_pred)) + beta*ke.mean(y_pred[:,:24]*y_pred[:,1:]) + delta*ke.mean(y_pred[:,:24]*avg_pred[1:] + y_pred[:,1:]*avg_pred[:24])
		

def my_generator(data):
	dd = data.groupby(["msid","individual_id"])
	while 1:
		for group in dd.groups:
			g = dd.get_group(group)
			if (g.shape[0] == 1):
				continue
			X = np.array(g[list(range(3,30)) + ["read_num"]],dtype='float32')
			Y = np.array(multilabel_binarizer.transform(g.raw_labels),dtype='float32')
			yield [X],Y


# In[1]:


df_mono = pd.read_pickle("/home/dcsoft/s/Ofir/df_m_sr_panel_bi_2019_08_26.pickle")		


# In[21]:


filt = lambda x: ~np.isnan(x) and x >= 0
minus5 = lambda x: x-5
df_mono['raw_labels'] = pd.Series([list(filter(filt, map(minus5,row))) for row in df_mono[['bi_allele1','bi_allele2']].values.tolist()], df_mono.index)


# In[23]:


dd = df_mono.groupby(["msid","individual_id"])


lst = list(dd.groups)
random.shuffle(lst)
train_groups = lst[int(len(lst)/20):]
test_groups = lst[:int(len(lst)/20)]
train_data = pd.concat( [ dd.get_group(group) for group in train_groups]).reset_index(drop=False)
test_data = pd.concat( [ dd.get_group(group) for group in test_groups]).reset_index(drop=False)


# In[24]:


multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(train_data.raw_labels)
y_train = multilabel_binarizer.transform(train_data.raw_labels)
y_test = multilabel_binarizer.transform(test_data.raw_labels)


# In[ ]:


train_rfe = train_data[list(range(3,30)) + ["read_num"]]
test_rfe = test_data[list(range(3,30)) + ["read_num"]]

DPLVL = 0.03
DNSS = 256
input2 = Input(shape=(test_rfe.shape[1],), dtype='float32')
x = input2

x = Dropout(DPLVL, noise_shape=None, seed=None)(Dense(DNSS, activation='linear')(x))
for _ in range(6):
    x = Dropout(DPLVL, noise_shape=None, seed=None)(Dense(DNSS, activation='relu')(x))

main_output = Dense(len(multilabel_binarizer.classes_), activation='sigmoid', name='main_output')(x)
model = Model([input2], main_output)

model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy',metrics=[metrics.categorical_accuracy])

model.summary()


# In[25]:


es_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      ## if min_delta is not acheived between epochs it's not an improvement
                                      patience=100,  ## number of epochs to allow without improvement
                                      verbose=0, mode='auto')
mc_cb = keras.callbacks.ModelCheckpoint('sim2_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)


# In[26]:


model.fit([train_rfe], y_train, epochs=int(2),
         validation_data=([test_rfe],y_test), batch_size=int(10240),
         callbacks=[es_cb, mc_cb])


# In[32]:


#model.fit_generator(my_generator(train_data), epochs=int(2),
#            validation_data=([test_rfe],y_test), steps_per_epoch=len(train_groups),
#            callbacks=[es_cb, mc_cb])


# In[ ]:




