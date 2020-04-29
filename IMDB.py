#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ##  1.讀入深度學習套件

# In[4]:


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb


# ## 2.分析數據集

# In[5]:


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=7788)


# In[6]:


len(x_train)


# In[7]:


len(x_test)


# In[8]:


len(x_train[0])


# In[9]:


len(x_train[1])


# In[9]:


y_train[0]


# In[10]:


y_train[1]


# In[11]:


print(len(x_train[66]))
y_train[66]
#第67則評論為383個字，且為正評


#  ## 3.資料處理

# In[12]:


x_train = sequence.pad_sequences(x_train, maxlen=300)
x_test = sequence.pad_sequences(x_test, maxlen=300) 
#改成300字


# ## 4. step 01: 打造一個函數學習機¶

# In[13]:


model = Sequential()


# In[14]:


model.add(Embedding(7788, 128))
#將7788維壓縮成128維


# In[15]:


model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3))


# In[16]:


model.add(Dense(1, activation='sigmoid'))


# In[17]:


model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
#分類型的問題用binary_crossentropy


# In[18]:


model.summary()


# In[19]:


(128+128+1)*4*128


# ## 5. step 02: 訓練

# In[ ]:


model.fit(x_train, y_train, batch_size=30, epochs=3,
         validation_data=(x_test, y_test))


# In[ ]:


#減少印出正確率經過的訓練次數,減少epochs(從5減少成3)
#觀察是否有持續進步來決定要不要繼續訓練,不然若是5會花費太久的時間
#validation: 用測試資料算誤差但不參與訓練#減少印出正確率經過的訓練次數,減少epochs(從5減少成3)


# In[ ]:




