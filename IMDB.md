```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

##  1.讀入深度學習套件


```python
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb
```

## 2.分析數據集


```python
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=7788)
```


```python
len(x_train)
```




    25000




```python
len(x_test)
```




    25000




```python
len(x_train[0])
```




    218




```python
len(x_train[1])
```




    189




```python
y_train[0]
```




    1




```python
y_train[1]
```




    0




```python
print(len(x_train[66]))
y_train[66]
#第67則評論為383個字，且為正評
```

    383





    1



 ## 3.資料處理


```python
x_train = sequence.pad_sequences(x_train, maxlen=300)
x_test = sequence.pad_sequences(x_test, maxlen=300) 
#改成300字
```

## 4. step 01: 打造一個函數學習機¶


```python
model = Sequential()
```


```python
model.add(Embedding(7788, 128))
#將7788維壓縮成128維
```


```python
model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3))
```


```python
model.add(Dense(1, activation='sigmoid'))
```


```python
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
#分類型的問題用binary_crossentropy
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, None, 128)         996864    
    _________________________________________________________________
    lstm (LSTM)                  (None, 128)               131584    
    _________________________________________________________________
    dense (Dense)                (None, 1)                 129       
    =================================================================
    Total params: 1,128,577
    Trainable params: 1,128,577
    Non-trainable params: 0
    _________________________________________________________________



```python
(128+128+1)*4*128
```




    131584



## 5. step 02: 訓練


```python
model.fit(x_train, y_train, batch_size=30, epochs=3,
         validation_data=(x_test, y_test))
```

    Train on 25000 samples, validate on 25000 samples
    Epoch 1/3
    25000/25000 [==============================] - 1625s 65ms/sample - loss: 0.4997 - accuracy: 0.7577 - val_loss: 0.4465 - val_accuracy: 0.8023
    Epoch 2/3
    25000/25000 [==============================] - 1539s 62ms/sample - loss: 0.4061 - accuracy: 0.8222 - val_loss: 0.4057 - val_accuracy: 0.8230
    Epoch 3/3
    25000/25000 [==============================] - 1527s 61ms/sample - loss: 0.3442 - accuracy: 0.8548 - val_loss: 0.4348 - val_accuracy: 0.8099





    <tensorflow.python.keras.callbacks.History at 0x63b5dd710>




```python
#減少印出正確率經過的訓練次數,減少epochs(從5減少成3)
#觀察是否有持續進步來決定要不要繼續訓練,不然若是5會花費太久的時間
#validation: 用測試資料算誤差但不參與訓練#減少印出正確率經過的訓練次數,減少epochs(從5減少成3)
```


```python
model.fit(x_train, y_train, batch_size=30, epochs=3,
         validation_data=(x_test, y_test))
```

    Train on 25000 samples, validate on 25000 samples
    Epoch 1/3
    25000/25000 [==============================] - 1688s 68ms/sample - loss: 0.3539 - accuracy: 0.8478 - val_loss: 0.6237 - val_accuracy: 0.6521
    Epoch 2/3
    25000/25000 [==============================] - 1632s 65ms/sample - loss: 0.3049 - accuracy: 0.8696 - val_loss: 0.3283 - val_accuracy: 0.8624
    Epoch 3/3
    25000/25000 [==============================] - 69891s 3s/sample - loss: 0.2051 - accuracy: 0.9218 - val_loss: 0.3261 - val_accuracy: 0.8690





    <tensorflow.python.keras.callbacks.History at 0x64a84b590>




```python

```
