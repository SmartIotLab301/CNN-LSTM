# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 00:35:18 2024

@author: user
"""

import glob
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, MaxPooling1D, Dense, Conv1D, Flatten, GRU, LSTM, GlobalAveragePooling1D, BatchNormalization,Bidirectional
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler, Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import itertools
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report

def inter(data,o_x,n_x):
    xp = np.arange(0, len(data),o_x/n_x)
    XXX = interp1d(np.arange(len(data)), data,  bounds_error = False,axis=0)
    XXX(xp)
    return XXX(xp)

name1 = '西南沿海'
name = name1
db ='西南'
Hz_list = '0Hz'
OOFSK_files = glob.glob('path_to_your_file') 

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(sess)

##########################原處理方式#############################
DATA = np.load(OOFSK_files[0], allow_pickle=True)
print(DATA.shape)
XX1 = []
y = []
X1 = []

for i in range(DATA.shape[0]):
    XX1.append(np.real(DATA[i][1]))
    y.append(DATA[i][2])
XX1 = np.array(XX1)
y = np.array(y)

XX=[]
XX = np.array(XX1)   
XX = XX[:,:500000]

X = np.reshape(XX,(-1,500,1))
y = np.reshape(y,(-1,1))
print(X.shape)
print(y.shape)

path = 'C:/Users/user/Desktop/'+name

file_name = 'CNN_LSTM'
model_name = file_name
if not os.path.isdir(path + file_name):
    os.makedirs(path + file_name)    


X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=0.2, random_state=5, stratify = y)
X_test, X_val, y_test, y_val = train_test_split(X_test1, y_test1, test_size=0.5, random_state=5, stratify = y_test1)
print(len(y_train))
print(list(y_train).count(0), list(y_train).count(1))
print(len(y_test)) 
print(list(y_test).count(0), list(y_test).count(1))
print(len(y_val))
print(list(y_val).count(0), list(y_val).count(1))

Scaler = StandardScaler()
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1]*X_val.shape[2]))

Scaler.fit(X_train)
X_train = Scaler.transform(X_train)
X_val = Scaler.transform(X_val)
X_test = Scaler.transform(X_test)

X_train = np.reshape(X_train, (X_train.shape[0],  X.shape[1], X.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0],  X.shape[1], X.shape[2]))
X_val = np.reshape(X_val, (X_val.shape[0],  X.shape[1], X.shape[2]))

X_test.shape
X_val.shape
X_train.shape
y_test.shape
y_train.shape
y_val.shape

step_num = 70

inputs = Input(shape = (X.shape[1], X.shape[2]))
layer = Conv1D(32, 3, padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
layer = BatchNormalization()(layer)
layer = Dropout(0.5)(layer)
layer = Conv1D(64, 3, padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(layer)
layer = BatchNormalization()(layer)
layer = Dropout(0.5)(layer)
layer = LSTM(128, return_sequences = True, dropout = 0.5, kernel_regularizer=regularizers.l2(0.01))(layer)
layer = BatchNormalization()(layer)
layer = LSTM(256, dropout = 0.5, kernel_regularizer=regularizers.l2(0.01))(layer)
layer = BatchNormalization()(layer)
output = Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.01))(layer)
model = Model(inputs, output)
model.summary()

def binary_focal_loss_fixed(y_true, y_pred):
    """
    y_true shape need be (None,1)ˊ
    y_pred need be compute after sigmoid
    
    """
    gamma = tf.constant(2, dtype=tf.float32)
    alpha = tf.constant(0.75, dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
    return K.mean(focal_loss)

count = 0

class auroc(Callback):
    def __init__(self, monitor = 'val_loss'):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.max_auroc = 0
    
    def on_epoch_end(self, epoch, logs={}):
        global X_test, y_test, model, count #, X_test_statstics
        print(X_test.shape)
        print(y_test.shape)
        current = logs.get(self.monitor)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict(X_test).ravel())
        # fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict([X_test, X_test_statstics]).ravel())
        print(epoch+1, ':', auc(fpr, tpr))
        if auc(fpr, tpr) > self.max_auroc:
            print('current epoch :', epoch+1, 'from', self.max_auroc, 'to', auc(fpr, tpr))
            self.max_auroc = auc(fpr, tpr)
            model.save(path + file_name + '/' + model_name + '_best.h5')
            count = 0
        else:
            count += 1
            # temp_count += 1
        if count >= step_num:
            print('ok')
            self.model.stop_training = True

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 50, min_lr = 0.0000001, verbose = 1)

opt = Adam(learning_rate = 0.00001)
model.compile(loss = binary_focal_loss_fixed, optimizer = opt, metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size = 128, epochs = 1000, validation_data = (X_val, y_val), shuffle = True, callbacks = [reduce_lr, auroc(monitor = 'val_loss')])

model.save(path + file_name + '/' + model_name + '_last.h5')

plt.title('Training and Validation Accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], '--', color='orange')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
plt.savefig(path + file_name + '/' + model_name + '_Accuracy.tif', dpi = 300)
plt.show()

plt.title('Training and Validation Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], '--', color='orange')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
plt.savefig(path + file_name + '/' + model_name + '_Loss.tif' ,dpi = 300)
plt.show()


print(model.evaluate(X_test, y_test))
model_predict = model.predict(X_test)

y_pred = []
for i in range(len(model_predict)):
    if model_predict[i] > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

vegetables = ["0", "1"]
farmers = ["0", "1"]
data = np.array(pd.crosstab(y_test[:,0], np.array(y_pred), rownames=['Actual'], colnames=['Prediction']))
harvest = data
np.average(harvest)
fig, ax = plt.subplots()
plt.imshow(harvest,cmap=plt.cm.Blues)
plt.colorbar()
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j], ha="center", va="center", color="white" if np.average(harvest)+2<harvest[i, j] else 'black')
fig.tight_layout()
plt.savefig(path + file_name + '/' + model_name + '_混淆矩陣_last.tif' ,dpi = 300)
plt.show()

fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict(X_test).ravel())

auc = auc(fpr, tpr)
print("AUC : ", auc)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='AUROC = {:.3f}'.format(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='best')
plt.savefig(path + file_name + '/' + model_name + '_AUROC_last.tif' ,dpi = 300)
plt.show()    

precision, recall, thresholds = precision_recall_curve(y_test, model.predict(X_test).ravel())

auc = auc(recall, precision)
print("AUC : ", auc)
plt.figure()
plt.plot([0, 1], [len(y_test[y_test==1]) / len(y_test), len(y_test[y_test==1]) / len(y_test)], 'k--')
plt.plot(recall, precision, label='AUPRC = {:.3f}'.format(auc))
plt.xlabel('Recall') 
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.legend(loc='best')
plt.savefig(path + file_name + '/' + model_name + '_AUPRC_last.tif' ,dpi = 300)
plt.show()    

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')
print(classification_report(y_test, y_pred))