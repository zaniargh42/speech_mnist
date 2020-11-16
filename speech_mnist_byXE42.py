# first import any library we may need

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from PIL import Image
from numpy import asarray
import os
import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , LeakyReLU
from keras.layers import Conv2D, Dense, Activation, Dropout, MaxPool2D, Flatten, LeakyReLU

# import data from the folder and theire labels
labels=[]
data=[]
for i in os.listdir(path of the file+'/speech mnist/recordings'):
  adsress=path of the file+/speech mnist/recordings/'+i
  data.append(adsress)
  labels.append(i[0])

# checking the imbalancement of the data
a=np.unique(labels)
a=a.tolist()
for i in a:
  print('count',i, labels.count(i))
 
# it's time to convert sounds to spectrogram 

audio_fpath = path of the file/speech mnist/recordings/'
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))

from datetime import datetime
e=0
for i in range(0,2500):
  path=audio_fpath+audio_clips[i]
  a,b=librosa.load(path, sr=48000)
  X = librosa.stft(a)
  Xdb = librosa.amplitude_to_db(abs(X))
  plt.figure(figsize=(5, 5))
  librosa.display.specshow(Xdb, sr=b, x_axis='time', y_axis='log')
  plt.savefig(path of the file/speech mnist/spec image/'+labels[e]+'_'+'number_'+str(e)+'.png')
  print(labels[e]+'_'+'number_'+str(e)+'.png')
  e+=1
  plt.close()
# this may take a while to be done!

# now convert spectrograms to nparrays

from PIL import Image
data=[]
labels=[]
path=path of the file'/speech mnist/spec image/'

a=os.listdir(path)

for i in range(0,2500):
  img=Image.open(path+a[i])
  img=img.crop((45,43,325,317))
  img=np.asarray(img)
  data.append(img)
  labels.append(a[i][0])
  print(i)
data=np.asarray(data)
labels=np.asarray(labels)

# train test spliting
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42,stratify=labels)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#convert class vectors to binary class matrices
y_train_onehot = keras.utils.to_categorical(y_train,num_classes=10)
y_test_onehot = keras.utils.to_categorical(y_test,num_classes=10)

print(y_train_onehot.shape)


# our model to train 

model = Sequential()
model.add(Conv2D(128, [3,3], strides = [2,2], padding = 'SAME', input_shape=(274,280,4)))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPool2D(padding = 'SAME'))
model.add(Conv2D(256, [5,5], padding = 'SAME'))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPool2D(padding = 'SAME'))
model.add(Conv2D(256, [1,1], padding = 'SAME'))
model.add(Conv2D(256, [3,3], padding = 'SAME'))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPool2D(padding = 'SAME'))
model.add(Conv2D(512, [1,1], padding = 'SAME'))
model.add(Conv2D(512, [3,3], padding = 'SAME',activation = 'relu'))
model.add(Conv2D(512, [1,1], padding = 'SAME'))
model.add(Conv2D(512, [3,3], padding = 'SAME', activation = 'relu'))
model.add(MaxPool2D(padding = 'SAME'))
model.add(Flatten())
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
model.compile(optimizer =  opt , loss = 'categorical_crossentropy', metrics = ['acc'])
cnn=model.fit(x_train, y_train_onehot ,batch_size=64,epochs=60,validation_data=(x_test,y_test_onehot))

# to visualize the acc and loss of the train and test set

import matplotlib.pyplot as plt
loss=cnn.history['loss']
val_loss=cnn.history['val_loss']
acc=cnn.history['acc']
val_acc=cnn.history['val_acc']

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(1,2,2)
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#finish
#coding by xe42
