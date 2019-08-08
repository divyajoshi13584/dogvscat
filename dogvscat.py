import cv2
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from random import shuffle
import tflearn 
Train_Dir=r"C:\Users\DIVYA JOSHI\Desktop\ic\train"
Test_Dir=r"C:\Users\DIVYA JOSHI\Desktop\ic\test"
Img_size=50
Lr=0.0001
model_name="Dogvscat-{}-{}.model".format(Lr,"6conv-basic")
def label_img(img):
    word_label=img.split(".")[-3]
    if word_label=="cat":
        return [1,0]
    elif word_label=="dog":
        return [0,1]

def create_train_data():
    training_data=[]
    for img in tqdm(os.listdir(Train_Dir)):
        label=label_img(img)
        Path=os.path.join(Train_Dir,img)
        img=cv2.imread(Path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(Img_size,Img_size))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save("train_data.npy",training_data)
    return training_data

def create_test_data():
    testing_data=[]
    for img in tqdm(os.listdir(Test_Dir)):
        img_n=img.split(".")[0]
        Path=os.path.join(Test_Dir,img)
        img=cv2.imread(Path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(Img_size,Img_size))
        testing_data.append([np.array(img),img_n])
    shuffle(testing_data)
    np.save("test_data.npy",testing_data)
    return testing_data

train_data=create_train_data()
test_data=create_test_data()

from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
tf.reset_default_graph()
convnet=input_data(shape=[None,Img_size,Img_size,1],name="input")
convnet=conv_2d(convnet,32,5,activation="relu")
connvnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,64,5,activation="relu")
convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,128,5,activation="relu")
convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,64,5,activation="relu")
convnet=max_pool_2d(convnet,5)
covnet=conv_2d(convnet,32,5,activation="relu")
convnet=max_pool_2d(convnet,5)
convnet=fully_connected(convnet,1024,activation="relu")
convnet=dropout(convnet,0.8)
convnet=fully_connected(convnet,2,activation="softmax")
convnet=regression(convnet,optimizer="adam",learning_rate=Lr,loss="categorical_crossentropy",name="targets")

model=tflearn.DNN(convnet,tensorboard_dir="log")
train=train_data[:-500]
test=train_data[-500:]
X=np.array([i[0] for i in train]).reshape(-1,Img_size,Img_size,1)
Y=[i[1] for i in train]
test_x=np.array([i[0] for i in test]).reshape(-1,Img_size,Img_size,1)
test_y=[i[1] for i in test]

model.fit({"input":X},{"targets":Y},n_epoch=5,validation_set=({"input":test_x},{"targets":test_y}),show_metric=True,snapshot_step=200,run_id=model_name)
model.save(model_name)
import matplotlib.pyplot as plt
test_data=np.load("test_data.npy")
fig=plt.figure()
for num,data in enumerate(test_data[:20]):
    image_num=data[1]
    image_data=data[0]
    y=fig.add_subplot(4,5,num+1)
    orig=image_data
    data=image_data.reshape(Img_size,Img_size,1)
    model_out=model.predict([data])[0]
    if np.argmax(model_out)==1:
        str_label="Dog"
    else:
        str_label="Cat"
    y.imshow(orig,cmap="gray")
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()






               
        
