#LIBRERIAS  DE KERAS Y TENSORFLOW
import numpy as np 
from scipy import misc  
from PIL import Image  
import glob  
import matplotlib.pyplot as plt  
import scipy.misc  
from matplotlib.pyplot import imshow 
from IPython.display import SVG  
import cv2  
import seaborn as sn  
import pandas as pd  
import pickle  
from tensorflow.keras import layers  
from tensorflow.keras.layers import Flatten, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout  
from tensorflow.keras.models import Sequential, Model, load_model  
from tensorflow.keras.preprocessing import image  
from tensorflow.keras.preprocessing.image import load_img  
from tensorflow.keras.preprocessing.image import img_to_array  
from tensorflow.keras.applications.imagenet_utils import decode_predictions  
from tensorflow.python.keras.utils import layer_utils, np_utils  
from tensorflow.python.keras.utils.data_utils import get_file  
from tensorflow.keras.applications.imagenet_utils import preprocess_input  
from tensorflow.python.keras.utils.vis_utils import model_to_dot  
from tensorflow.keras.utils import plot_model  
from tensorflow.keras.initializers import glorot_uniform  
from tensorflow.keras import losses  
import tensorflow.keras.backend as K  
from tensorflow.keras.callbacks import ModelCheckpoint  
from sklearn.metrics import confusion_matrix, classification_report  
import tensorflow as tf  
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import vgg16


from sklearn.datasets import make_classification  
from sklearn.preprocessing import label_binarize  
from scipy import interp  
from itertools import cycle
from sklearn.metrics import roc_curve, auc


#Función para cambiar el tamaño de la imagen
def resize_data(data):  
    data_upscaled = np.zeros((data.shape[0], 48, 48, 3))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    return data_upscaled

  
#Crear el modelo VGG19
def create_vgg19():  
  model = tf.keras.applications.vgg19.VGG19(include_top=True, weights=None, input_tensor=None, input_shape=(48,48,3), pooling=None, classes=100)
  return model



#Separación de los datos de entrenamiento y validación
(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar100.load_data(label_mode='fine')
print(y_train_original)

y_train = np_utils.to_categorical(y_train_original, 100)
y_test = np_utils.to_categorical(y_test_original, 100)
print(x_train_original)

imgplot = plt.imshow(x_train_original[3])
plt.show()

#Normalización de los datos
x_train = x_train_original/255  
x_test = x_test_original/255  

K.set_image_data_format('channels_last')  
#K.set_learning_phase(1)  

#Datos con cambio de pixeles para VGG19
x_train_resized = resize_data(x_train_original)  
x_test_resized = resize_data(x_test_original)  
x_train_resized = x_train_resized / 255  
x_test_resized = x_test_resized / 255

#Creación del modelo y compilación
vgg19_model = create_vgg19()  
vgg19_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', 'mse'])  
vgg19_model.summary()



#Entrenamiento del modelo
vgg19 = vgg19_model.fit(x=x_train_resized, y=y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test_resized, y_test), shuffle=True)  


#Creación de las gráficas de exactitud y error
plt.figure(0)  
plt.plot(vgg19.history['acc'],'r')  
plt.plot(vgg19.history['val_acc'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])

plt.figure(1)  
plt.plot(vgg19.history['loss'],'r')  
plt.plot(vgg19.history['val_loss'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])

plt.show() 

#Matriz de confusión
vgg19_pred = vgg19_model.predict(x_test_resized, batch_size=32, verbose=1)  
vgg19_predicted = np.argmax(vgg19_pred, axis=1)



#Creamos la matriz de confusión
vgg19_cm = confusion_matrix(np.argmax(y_test, axis=1), vgg19_predicted)

# Visualizar la matriz de confusión
vgg19_df_cm = pd.DataFrame(vgg19_cm, range(100), range(100))  
plt.figure(figsize = (20,14))  
sn.set(font_scale=1.4) #for label size  
sn.heatmap(vgg19_df_cm, annot=True, annot_kws={"size": 12}) # font size  
plt.show()  


vgg19_report = classification_report(np.argmax(y_test, axis=1), vgg19_predicted)  
print(vgg19_report)

###CURVAS DE ROC

n_classes = 100
lw = 2

# Crear curva ROC y área de ROC para cada clase
fpr = dict()  
tpr = dict()  
roc_auc = dict()  
for i in range(n_classes):  
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], vgg19_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Crear micro-average curva ROC y área ROC 
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), vgg19_pred.ravel())  
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Agregar todos los falsos positivos
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Interpolar curvas ROC
mean_tpr = np.zeros_like(all_fpr)  
for i in range(n_classes):  
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Calcular el promedio y crear AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr  
tpr["macro"] = mean_tpr  
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Mostrar curvas ROC
plt.figure(1)  
plt.plot(fpr["micro"], tpr["micro"],  
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],  
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])  
for i, color in zip(range(n_classes-97), colors):  
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Some extension of Receiver operating characteristic to multi-class')  
plt.legend(loc="lower right")  
plt.show()


# Ver a detalle las curvas ROC 
plt.figure(2)  
plt.xlim(0, 0.2)  
plt.ylim(0.8, 1)  
plt.plot(fpr["micro"], tpr["micro"],  
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],  
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])  
for i, color in zip(range(3), colors):  
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Some extension of Receiver operating characteristic to multi-class')  
plt.legend(loc="lower right")  
plt.show()  


