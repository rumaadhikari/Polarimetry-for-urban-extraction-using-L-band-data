import glob
import numpy as np
import os
import shutil
np.random.seed(30)

files = glob.glob('/shared/RUMA/A_T_C_D/train/*')

urban_files = [fn for fn in files if 'URBAN' in fn]
nonurban_files = [fn for fn in files if 'NONURBAN' in fn]
len(urban_files), len(nonurban_files)

urban_train = np.random.choice(urban_files, size=1000, replace=False)
nonurban_train = np.random.choice(nonurban_files, size=1000, replace=False)
urban_files = list(set(urban_files) - set(urban_train))
nonurban_files = list(set(nonurban_files) - set(nonurban_train))

urban_val = np.random.choice(urban_files, size=50, replace=False)
nonurban_val = np.random.choice(nonurban_files, size=50, replace=False)
urban_files = list(set(urban_files) - set(urban_val))
nonurban_files = list(set(nonurban_files) - set(nonurban_val))

urban_test = np.random.choice(urban_files, size=50, replace=False)
nonurban_test = np.random.choice(nonurban_files, size=50, replace=False)

print('urban datasets:', urban_train.shape, urban_val.shape, urban_test.shape)
print('nonurban datasets:', nonurban_train.shape, nonurban_val.shape, nonurban_test.shape)

# WRITING DATA TO DIRECTORY

train_dir = 'trainingdata'
val_dir = 'validationdata'
test_dir = 'testdata'

train_files = np.concatenate([urban_train, nonurban_train])
validate_files = np.concatenate([urban_val, nonurban_val])
test_files = np.concatenate([urban_test, nonurban_test])

os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
os.mkdir(val_dir) if not os.path.isdir(val_dir) else None
os.mkdir(test_dir) if not os.path.isdir(test_dir) else None

for fn in train_files:
    shutil.copy(fn, train_dir)

for fn in validate_files:
    shutil.copy(fn, val_dir)
    
for fn in test_files:
    shutil.copy(fn, test_dir)
    
    #PREPARING DATASET
    
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
# %matplotlib inline


IMG_DIM = (150, 150)

train_files = glob.glob('/shared/RUMA/A_T_C_D/trainingdata/*')
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
train_labels = [fn.split('/shared/RUMA/A_T_C_D/trainingdata/')[1].split('.')[0].strip() for fn in train_files]

validation_files = glob.glob('/shared/RUMA/A_T_C_D/validationdata/*')
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
validation_labels = [fn.split('/shared/RUMA/A_T_C_D/validationdata/')[1].split('.')[0].strip() for fn in validation_files]
print('Train dataset shape:', train_imgs.shape, 
     '\tValidation dataset shape:', validation_imgs.shape)

# SCALING FROM 0 TO 255

train_imgs_scaled = train_imgs.astype('float32')
validation_imgs_scaled  = validation_imgs.astype('float32')
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

print(train_imgs[0].shape)
array_to_img(train_imgs[0])

#ENCODING 0 OR 1


batch_size = 10
num_classes = 2
epochs = 20
input_shape = (150, 150, 3)

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validation_labels_enc = le.transform(validation_labels)

print(train_labels[24:30], train_labels_enc[24:30])



#RESNET 50

from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
import pandas as pd

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(150,150,3))

output = resnet.layers[-1].output
output = keras.layers.Flatten()(output)
resnet = Model(resnet.input, output=output)


#finetune

resnet.trainable = True
set_trainable = False
for layer in resnet.layers:
    if layer.name in ['res5c_branch2b', 'res5c_branch2c', 'activation_97']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
layers = [(layer, layer.name, layer.trainable) for layer in resnet.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])



"""

#Pre-trained CNN model with Fine-tuning and Image Augmentation

from keras.applications import vgg16
from keras.models import Model
import keras
import pandas as pd

#downloading vgg model

vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                     input_shape=input_shape)

output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

#fine tuning

vgg_model.trainable = True

set_trainable = False
for layer in vgg_model.layers:
    if layer.name in ['block5_conv1', 'block4_conv1', 'block3_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

"""

# train data

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=5,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=10)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=10)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(resnet)

#model.add(vgg_model)

#model.add(Flatten())

model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])

       

history = model.fit_generator(train_generator, steps_per_epoch=20, epochs=20,
                              validation_data=val_generator, validation_steps=5, 
                              verbose=1) 



model.save("urban_nonurban_finetune_A_T_C_D_cnn.h5")   

#PLOTTING GRAPH

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(0,20))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 20, 20))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 20, 20))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

#testing

# load dependencies
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model
import model_evaluation_utils as meu
#matplotlib inline

# load saved model

tl_img_aug_finetune_cnn = load_model('urban_nonurban_finetune_A_T_C_D_cnn.h5')

# load other configurations
IMG_DIM = (150, 150)
input_shape = (150, 150, 3)
num2class_label_transformer = lambda l: ['urban' if x == 0 else 'nonurban' for x in l]
class2num_label_transformer = lambda l: [0 if x == 'urban' else 1 for x in l]


#final test
    
IMG_DIM = (150, 150)

test_files = glob.glob('/shared/RUMA/A_T_C_D/testdata/*')
test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in test_files]
test_imgs = np.array(test_imgs)
test_labels = [fn.split('/shared/RUMA/A_T_C_D/testdata/')[1].split('.')[0].strip() for fn in test_files]

test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255
test_labels_enc = class2num_label_transformer(test_labels)

print('Test dataset shape:', test_imgs.shape)
#print(test_labels[11:16], test_labels_enc[11:16])


batch_size = 10
num_classes = 2
epochs = 20
input_shape = (150, 150, 3)

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(test_labels)
test_labels_enc = le.transform(test_labels)
#validation_labels_enc = le.transform(validation_labels)

print(test_labels[30:40], test_labels_enc[30:40])

string_list = test_labels

for i in range(len(string_list)):
#Iterate through string_list

    string_list[i] = string_list[i].lower()
#Convert each string to lowercase




test_labels =  string_list
#performance


predictions = tl_img_aug_finetune_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)
meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions, 
                                              classes=list(set(test_labels)))
