# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 04:04:07 2019

@author: user
"""

from keras.callbacks import History
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,BatchNormalization,GaussianNoise
from tensorflow.keras.layers import MaxPooling2D,ZeroPadding2D
from tensorflow.keras.layers import Flatten,Activation
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
from keras import backend as K
import numpy as np

batchsize=32

#1st conv layer
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=90)

test_datagen = ImageDataGenerator(rescale=1./255)

valid_datagen = ImageDataGenerator(rescale=1./255)

seed=0


path_test=r'F:\dataset\breast-ultrasound-image\us-dataset\originals\predict'
path_train=r'F:\dataset\breast-ultrasound-image\us-dataset\originals\train'
path_valid=r'F:\dataset\breast-ultrasound-image\us-dataset\originals\validation'

train_set = train_datagen.flow_from_directory(path_train,
                                                 target_size=(90,70),color_mode = "grayscale",
                                                 class_mode='binary',
                                                 batch_size=batchsize,seed=seed)
                                            
valid_set = valid_datagen.flow_from_directory(path_valid,
                                            target_size=(90,70),color_mode = "grayscale",
                                            batch_size=batchsize,
                                            class_mode='binary',seed=seed)


test_set = test_datagen.flow_from_directory(path_test,color_mode = "grayscale",
                                            target_size=(90,70),
                                            batch_size=batchsize,
                                            class_mode='binary',seed=seed)


model=Sequential()
model.add(GaussianNoise(0.05))
model.add(Convolution2D(8,kernel_size=(3,3),
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.0001),
                        input_shape=(90,70,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Convolution2D(8,kernel_size=(3,3),
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)))

model.add(BatchNormalization())
model.add(Dropout(0.8))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['acc'])
va = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)


output=model.fit_generator(train_set,
                           steps_per_epoch =200/batchsize ,
                           epochs = 60,
                           validation_data=valid_set,
                           validation_steps = 94/batchsize)


plt.figure(figsize=(6,3))
plt.plot(output.history['acc'])
plt.plot(output.history['val_acc'])
plt.title('classifier accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.figure(figsize=(6,3))
plt.plot(output.history['loss'])
plt.plot(output.history['val_loss'])
plt.title('classifier loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_loss, test_acc=model.evaluate_generator(test_set)

print('test accuracy:'+str(test_acc))
print('test loss: '+str(test_loss))

from glob import glob 

zeros=np.zeros((10,1))
ones=np.ones((15,1))

y_true=np.concatenate((ones,zeros), axis=0)
y_hat=[]
path_test_png=r'F:\dataset\breast-ultrasound-image\us-dataset\originals\predict\*.bmp'
from keras.preprocessing import image
pngss=glob(path_test_png)
for i in range(len(pngss)):
    test_image = image.load_img(pngss[i] ,target_size= (90,70),color_mode = "grayscale")
    arr = np.array(test_image)
    arr = np.true_divide(arr,[255.0],out=None)


# Changing the input of the size...
    test_image = image.img_to_array(arr)

# Adding a new dimension (the placement of the image in the batchsize)
    test_image = np.expand_dims(test_image, axis=0)

    predic_classes = model.predict_classes(test_image)
    y_hat.append(predic_classes[0])

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_true,y_hat)
print(cm)

from sklearn.metrics import auc,roc_curve
classi_report=classification_report(y_true, y_hat)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_hat)
auc_curve=auc(fpr_keras, tpr_keras)

print('auc score is:'+str(auc_curve))

test_accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[1,0]+cm[0,1])
print('predict accuracy:'+str(test_accuracy))


plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='AUC = {:.3f}'.format(auc_curve))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#model.save_weights(r"F:/model_brain_ultra_auc_.883.h5")





