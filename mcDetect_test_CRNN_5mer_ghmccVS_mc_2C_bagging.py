""" LSTM Network.
A RNN Network (LSTM) implementation example using Keras.
This example is using the MNIST handwritten digits dataset (http://yann.lecun.com/exdb/mnist/)

Ressouces:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

Repository: https://github.com/ar-ms/lstm-mnist
"""

# Imports
import sys,os,datetime
from tqdm import tqdm
import random
#from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential,Model
from keras.layers import Input,LSTM, Dense,Dropout,Flatten
from keras.models import load_model
import numpy as np
from keras.layers import Input, BatchNormalization, LSTM,Bidirectional,Conv1D,MaxPooling1D,Activation,add,GlobalAveragePooling1D
from keras.layers.recurrent import GRU
import argparse
class MnistLSTMClassifier(object):
    def __init__(self):
        # Classifier
        self.time_steps=1 # timesteps to unroll
        self.n_units=128 # hidden LSTM units
        self.n_inputs=155 # rows of 28 pixels (an mnist img is 28x28)
        self.n_sigLenth=150 # rows of 28 pixels (an mnist img is 28x28)
        self.n_classes=2 # mnist classes/labels (0-9)
        self.batch_size=5000 # Size of each batch
        self.n_epochs=100
        # Internal
        self._train_data_loaded = False
        self._test_data_loaded = False
        self._trained = False
        self.x_train=[]
        self.l_train=[]
        self.x_test=[]
        self.f_test=[]
    
    #建立深度模型
    def __create_model(self):
        #self.model = Sequential()
        #self.model.add(LSTM(128, input_shape=(self.time_steps, self.n_inputs)))
        #self.model.add(GRU(512, input_shape=(self.time_steps, self.n_inputs)))
        #self.model.add(GRU(512,activation='relu', return_sequences=True,input_shape=(self.time_steps, self.n_inputs)))
        #self.model.add(GRU(512, activation='relu',return_sequences=True))
        #self.model.add(GRU(512, activation='relu'))
        #self.model.add(LSTM(128, input_shape=(self.time_steps, self.n_inputs)))
        #self.model.add(Conv1D(32,1,activation='relu', input_shape=(self.time_steps, self.n_inputs)))
        #self.model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid"))
        #self.model.add(Conv1D(64, 3, use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal'))
        #self.model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid"))
        #self.model.add(Conv1D(128, 3, use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal'))
        #self.model.add(Conv1D(256, 3, use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal'))
        #self.model.add(Dense(512))
        #self.model.add(Dropout(0.2))
        #self.model.add(Bidirectional(GRU(512, return_sequences=True, kernel_initializer='he_normal'),merge_mode='sum', input_shape=(self.time_steps, self.n_inputs)))
        '''self.model.add(Dense(512))
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(GRU(512, return_sequences=True, kernel_initializer='he_normal'),merge_mode='sum'))
        self.model.add(Dense(512))
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(GRU(512, return_sequences=True, kernel_initializer='he_normal'),merge_mode='sum'))'''
        #self.model.add(Dense(self.n_classes, activation='softmax'))
        
        self.inputs = Input(name='the_inputs', shape=(self.n_inputs,self.time_steps))
        '''x=res(64,self.inputs)
        x=res(128,x)
        x=res(128,x)'''
        x = cnn_cell(64, self.inputs,pool=True)
        x = cnn_cell(128, x,pool=True)
        x = cnn_cell(256, x, pool=True)
        #x = cnn_cell(256, x, pool=False)
        #x = cnn_cell(512,x, pool=False)
        #x = dense(512, x)
        #x = bidir_gru(256, self.inputs)
        x=Bidirectional(GRU(256, return_sequences=True, kernel_initializer='he_normal'),merge_mode='sum')(x)
        #x=Bidirectional(GRU(512, return_sequences=True, kernel_initializer='he_normal'),merge_mode='sum')(x)
        #x=Bidirectional(GRU(512, return_sequences=True, kernel_initializer='he_normal'),merge_mode='sum')(x)
        #x = bidir_gru(128, x)
        #x = bidir_gru(256, x)
        #x = bidir_gru(256, x)
        #x = bidir_gru(256, x)
        #x=Flatten()(x)
        #x=res(64,self.inputs)
        #x=res(128,x)
        #x=res(128,x)
        '''x= GRU(256, return_sequences=True,kernel_initializer='he_normal')(self.inputs)
        x=GRU(256, return_sequences=True,kernel_initializer='he_normal')(x)
        x=GRU(256, return_sequences=True,kernel_initializer='he_normal')(x)
        x=GRU(64, kernel_initializer='he_normal')(x)'''

        x = GlobalAveragePooling1D()(x)

        #x=GRU(128)(x)
        self.outputs = Dense(self.n_classes, activation='softmax')(x)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        
    '''def __create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
        self.model.add(Dense(self.n_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])'''
    def __load_train_data(self):
        #self.mnist = input_data.read_data_sets("mnist", one_hot=True)
        trainSet=[]
        labelSet=[]
        #羟甲基化
        print('Loading 5ghmC data...')
        time01=datetime.datetime.now()
        #with open("5hmc_guppy_202008.11bases.final.tsv.2000000", 'r', encoding='utf8') as f:
        with open("/BIGDATA1/sysu_szmed_xli/xuhan/01.5mer/01.CRNN/5mc_202008.5bases.tsv.2000000", 'r', encoding='utf8') as f:
            data = f.readlines()[:2000000]
            resamp=random.sample(data,500000)
            for line in resamp:
                line=line.rstrip('\n')
                line=line.rstrip('\t')
                sigItem=line.split('\t')
                seqline=sigItem[4]#基因序列
                #seqline=line[2:8]
                sigline=sigItem[5]#信号
                seqlist=[]
                for x in seqline:
                    if x=='A':
                        seqlist.append(float(1))
                    elif x=='C':
                        seqlist.append(float(2))
                    elif x=='G':
                        seqlist.append(float(3))
                    elif x=='T':
                        seqlist.append(float(4))
                #print([x for x in sigline.split('\t')])
                newline = [float(x) for x in sigline.split(',')]
                #补零
                if len(newline) < self.n_sigLenth:
                    for i in range(self.n_sigLenth-len(newline)):
                        newline.append(0)
                #print(np.array(seqlist+newline).shape)
                if np.array(seqlist+newline).shape[0]==self.n_inputs:
                    trainSet.append(np.array(seqlist+newline))
                    labelSet.append([1,0])
                    
                    
                
        
        time02=datetime.datetime.now()
        print('读取数据时间:',time02-time01)
        print(len(trainSet))

        #甲基化
        print('Loading 5mC data...')
        time01=datetime.datetime.now()
        with open("/BIGDATA1/sysu_szmed_xli/xuhan/01.5mer/01.CRNN/5mc_202008.5bases.tsv.2000000", 'r', encoding='utf8') as f:
            data = f.readlines()[:2000000]
            resamp=random.sample(data,500000)
            for line in resamp:
                line=line.rstrip('\n')
                line=line.rstrip('\t')
                
                sigItem=line.split('\t')
                seqline=sigItem[4]#基因序列
                #seqline=line[2:8]
                sigline=sigItem[5]#信号
                seqlist=[]
                for x in seqline:
                    if x=='A':
                        seqlist.append(float(1))
                    elif x=='C':
                        seqlist.append(float(2))
                    elif x=='G':
                        seqlist.append(float(3))
                    elif x=='T':
                        seqlist.append(float(4))
                #print([x for x in sigline.split('\t')])
                newline = [float(x) for x in sigline.split(',')]
                #补零
                if len(newline) < self.n_sigLenth:
                    for i in range(self.n_sigLenth-len(newline)):
                        newline.append(0)
                #print(np.array(seqlist+newline).shape)
                if np.array(seqlist+newline).shape[0]==self.n_inputs:
                    trainSet.append(np.array(seqlist+newline))
                    labelSet.append([1,0])
                    
                    
        
        time02=datetime.datetime.now()
        print('读取数据时间:',time02-time01)
        print(len(trainSet))

        #非甲基化
        print('Loading C data...')
        time01=datetime.datetime.now()
        with open("/BIGDATA1/sysu_szmed_xli/xuhan/01.5mer/01.CRNN/c_202011.5bases.tsv", 'r', encoding='utf8') as f:
            data = f.readlines()[:2000000]
            resamp=random.sample(data,1000000)
            for line in resamp:
                line=line.rstrip('\n')
                line=line.rstrip('\t')
                
                sigItem=line.split('\t')
                seqline=sigItem[4]#基因序列
                #seqline=line[2:8]
                sigline=sigItem[5]#信号
                seqlist=[]
                for x in seqline:
                    if x=='A':
                        seqlist.append(float(1))
                    elif x=='C':
                        seqlist.append(float(2))
                    elif x=='G':
                        seqlist.append(float(3))
                    elif x=='T':
                        seqlist.append(float(4))
                #print(sigline.split('\t'))
                #补零
                newline = [float(x) for x in sigline.split(',')]
                if len(newline) < self.n_sigLenth:
                    for i in range(self.n_sigLenth-len(newline)):
                        newline.append(0)
                #print(np.array(seqlist+newline).shape)
                if np.array(seqlist+newline).shape[0]==self.n_inputs:
                    trainSet.append(np.array(seqlist+newline))
                    labelSet.append([0,1])
                    
                        
        time02=datetime.datetime.now()
        print('读取数据时间:',time02-time01)
        #np.random.shuffle(trainSet)
        #fbank=np.array([t[:-1] for t in trainSet])
        #补零
        #new_matrix = list(map(lambda l:l + [0.0]*(self.n_inputs - len(l)), self.x_train))
        #fbank = np.array(new_matrix)
        #b=fbank.reshape(-1,1,300)
        #print(b.shape)
        #self.x_train = [x.reshape(-1, self.time_steps, self.n_inputs) for x in trainSet]
        print('样本：',np.array(trainSet).shape)
        self.x_train = np.array(trainSet).reshape((-1, self.n_inputs,self.time_steps))
        print('样本结构',self.x_train.shape)
        #print('样本数据',self.x_train)
        
        #label
        print('Loading labels...')
        '''for i in tqdm(range(len(trainSet))):            	
            if(i<len(trainSet)//2):
                newline=[1,0]
            else:
                newline=[0,1]
            
            #print(newline)
            self.l_train.append(newline)'''   
        self.l_train = np.array(labelSet)
        print('标签结构',self.l_train.shape)
        #print('标签数据',self.l_train)
        #self._train_data_loaded=True
        
    def __load_test_data(self, testfile):
        print('Loading test data...')
        with open(testfile, 'r', encoding='utf8') as f:
            data = f.readlines()
            #for line in tqdm(data[2000000:2010000]):
            for line in tqdm(data):
                line=line.rstrip('\n')
                line=line.rstrip('\t')
                if len(line)>0:
                #if line.find('TGGGCGTGGG')>=0:
                    sigItem=line.split('\t')
                    self.f_test.append(sigItem[0]+'\t'+sigItem[1]+'\t'+sigItem[2]+'\t'+sigItem[3]+'\t'+sigItem[4])
                    seqline=sigItem[4]#基因序列
                    #seqline=line[2:8]
                    sigline=sigItem[5]#信号
                    seqlist=[]
                    for x in seqline:
                        if x=='A':
                            seqlist.append(float(1))
                        elif x=='C':
                            seqlist.append(float(2))
                        elif x=='G':
                            seqlist.append(float(3))
                        elif x=='T':
                            seqlist.append(float(4))

                    newline = [float(x) for x in sigline.split(',')]
                    if len(newline) > self.n_sigLenth:
                        newline=newline[:self.n_sigLenth]
                    self.x_test.append(seqlist+newline)
                    #self.x_test.append(newline)
        #补零
        new_matrix = list(map(lambda l:l + [0.0]*(self.n_inputs - len(l)), self.x_test))
        fbank = np.array(new_matrix)
        self.x_test = [np.array(x).reshape((-1, self.n_inputs, self.time_steps)) for x in fbank]
        self.x_test = np.array(self.x_test).reshape((-1, self.n_inputs, self.time_steps))
        #print(self.x_test[0])
        #print('测试样本结构',self.x_test.shape)
        #print('测试样本数据',self.x_test)       
          
        self._data_loaded = True

    def train(self, save_model=False,seqnum=0):
        self.__create_model()
        if self._train_data_loaded == False:
            self.__load_train_data()
        
        print(self.x_train.shape)
        for i in range(self.n_epochs//50):
            self.model.fit(self.x_train, self.l_train,
                      batch_size=self.batch_size, epochs=50, shuffle=True)           
            
            self.model.save("./saved_model/bagging_"+str(seqnum)+"_"+str(i)+".h5")
        self._trained = True

    def evaluate(self, modelname=None,testfile=None,resultfile=None):
        if self._test_data_loaded == False:
            self.__load_test_data(testfile)
        result=[]
        for i in range(7):
            model = load_model(modelname+"bagging_"+str(i)+"_1.h5")
            if i==0:
                result=model.predict(self.x_test)
            else:
                result=result+model.predict(self.x_test)
        mcNum=0
        cNum=0
        for i in range(result.shape[0]):
            self.f_test[i]=self.f_test[i]+'\t'+str(result[i,0])+'\t'+str(result[i,1])
            if result[i,0]>=result[i,1]:
                mcNum+=1
                self.f_test[i]=self.f_test[i]+'\tM'
            else:
                cNum+=1
                self.f_test[i]=self.f_test[i]+'\tC'
            #print(np.round(result,3)[i])
        print('甲基化：',mcNum)
        print('非甲基化:',cNum)
        filename=resultfile
        if os.path.exists(filename):
            os.remove(filename)
            os.mknod(filename)
            f = open(filename, 'w')
            f.writelines([line+'\r\n' for line in self.f_test])
        else:
            os.mknod(filename)
            f = open(filename, 'w')
            f.writelines([line+'\r\n' for line in self.f_test])

# ============================模型组件=================================
def bi_gru(units, x):
    x = Dropout(0.2)(x)
    y1 = GRU(units, return_sequences=True,
        kernel_initializer='he_normal')(x)
    y2 = GRU(units, return_sequences=True, go_backwards=True,
        kernel_initializer='he_normal')(x)
    y = add([y1, y2])
    return y

def bidir_gru(units, x):
    y=Bidirectional(GRU(units, return_sequences=True,
        kernel_initializer='he_normal'),merge_mode='sum')(x)
    return y

def sidir_gru(units, x):
    y=GRU(units, return_sequences=True)(x)
    #y=GRU(units, return_sequences=True,
    #    kernel_initializer='he_normal')(x)
    return y
def conv1d(size):
    return Conv1D(size, 5, use_bias=True, activation='relu',
        padding='same', kernel_initializer='he_normal')

def norm(x):
    return BatchNormalization(axis=-1)(x)
def maxpool(x):
    return MaxPooling1D(pool_size=2, strides=None, padding="valid")(x)

def cnn_cell(size, x, pool=True):
    x = norm(conv1d(size)(x))
    if pool:
        x = maxpool(x)
    return x

def dense(units, x, activation="relu"):
    x = Dropout(0.2)(x)
    y = Dense(units, activation=activation, use_bias=True,
        kernel_initializer='he_normal')(x)
    return y
def resMaxPool(units, x):
    conv_x = Conv1D(filters=units, kernel_size=8, padding='same')(x)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = MaxPooling1D(pool_size=2, strides=None, padding="valid")(conv_x)
    
    conv_y = Conv1D(filters=units, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    

    conv_z = Conv1D(filters=units, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)
    

    # expand channels for the sum
    shortcut_y = Conv1D(filters=units, kernel_size=1, padding='same')(x)
    shortcut_y = BatchNormalization()(shortcut_y)
    shortcut_y = MaxPooling1D(pool_size=2, strides=None, padding="valid")(shortcut_y)

    block = add([shortcut_y, conv_z])
    block = Activation('relu')(block)
    return block

def res(units, x):
    conv_x = Conv1D(filters=units, kernel_size=8, padding='same')(x)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    
    conv_y = Conv1D(filters=units, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    

    conv_z = Conv1D(filters=units, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)
    

    # expand channels for the sum
    shortcut_y = Conv1D(filters=units, kernel_size=1, padding='same')(x)
    shortcut_y = BatchNormalization()(shortcut_y)
    
    block = add([shortcut_y, conv_z])
    block = Activation('relu')(block)
    return block




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description='please enter the parameter input file ...'
    parser.add_argument("-i", "--inputF", help="this is parameter input file", dest="argI", type=str, default="")
    parser.add_argument("-o", "--outputF", help="this is parameter output file", dest="argO", type=str, default="")
    args = parser.parse_args()
    if args.argI!='' and args.argO!='':
        lstm_classifier = MnistLSTMClassifier()
        '''for i in range(7):
            lstm_classifier.train(save_model=True,seqnum=i)'''
        
        #lstm_classifier.train(save_model=True)
        #lstm_classifier.evaluate(testfile=args.argI, resultfile=args.argO)
        # Load a trained model.
        #lstm_classifier.evaluate(model="./saved_model/ratio99.8.h5", testfile=args.argI, resultfile=args.argO)
        #lstm_classifier.evaluate(model="./saved_model/lstm-model.h5", testfile=args.argI, resultfile=args.argO)
        #lstm_classifier.evaluate(model="./saved_model/ceshi2NewCRNN_ghmcmcVSc.h5", testfile=args.argI, resultfile=args.argO)#ghmc+mc VS. c
        #lstm_classifier.evaluate(model="./saved_model/ceshi1NewCRNN_mccVSghmc.h5", testfile=args.argI, resultfile=args.argO)#mc+c VS. ghmc
        #lstm_classifier.evaluate(modelname="./bagging_model/", testfile=args.argI, resultfile=args.argO)
        lstm_classifier.evaluate(modelname="./saved_model/", testfile=args.argI, resultfile=args.argO)
    else:
        print('please enter the parameter input file ...')
