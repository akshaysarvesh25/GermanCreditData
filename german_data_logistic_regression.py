import requests
import numpy as np
import random
from operator import itemgetter
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class DataGet:
    def __init__(self,sample_train,sample_test):
        self.x_train_path = sample_train
        self.x_test_path  = sample_test

    def GetSamples(self):
        x_train = np.genfromtxt(self.x_train_path,delimiter=',')
        x_train = np.array(x_train)
        x_test = np.genfromtxt(self.x_test_path,delimiter=',')
        x_test = np.array(x_test)
        return x_train,x_test

    def CollateData(self):
        x_train,x_test = self.GetSamples()
        y_train =  x_train[:,(x_train.shape[1] - 1)]
        y_test = x_test[:,(x_test.shape[1] - 1)]
        x_train_ip = x_train[:,0:(x_train.shape[1] - 1)]
        x_test_ip  = x_test[:,0:(x_test.shape[1] - 1)]
        y_test = np.array(y_test).reshape(-1,1)
        y_train = np.array(y_train).reshape(-1,1)
        new_col = x_test_ip.sum(1)[...,None]
        x_test_ip = np.hstack((x_test_ip, new_col))
        return x_train_ip,y_train,x_test_ip,y_test


class OnlineDataGet:

    def __init__(self,url1):
        self.url = url1

    def GetOnlineURLData(self):
        HTTPObj = requests.get(self.url)
        with open("German_Data_numeric.txt",'wb') as f:
            f.write(HTTPObj.content)
        return np.array(np.loadtxt('German_Data_numeric.txt'))

    def PartitionData(self):
        data = self.GetOnlineURLData()
        data_1 = np.zeros(data.shape[1])
        data_2 = np.zeros(data.shape[1])
        for i in range(0,data.shape[0]):
            if(data[i][data.shape[1]-1] == 1):
                data_1 = np.vstack((data_1,data[i]))
            elif(data[i][data.shape[1]-1] == 2):
                data_2 = np.vstack((data_2,data[i]))
            else:
                print('invalid block')
        partition_threshold = 70
        data_1_part = int((data_1.shape[0]-1)*partition_threshold*0.01)
        data_2_part = int((data_2.shape[0]-1)*partition_threshold*0.01)
        return data_1[1:data_1_part,:],data_2[1:data_2_part,:],data_1[(data_1_part+1):(data_1.shape[0]),:],data_2[(data_2_part+1):(data_2.shape[0]),:]


def RandomizedTrainingDataGenerator(data_1_train1,data_2_train1,NumbSamples):
    Samples_1_train_idx = np.array(random.sample(range(0, len(data_1_train1)), NumbSamples))
    Samples_2_train_idx = np.array(random.sample(range(0, len(data_2_train1)), NumbSamples))
    Samples_1_train_dat = np.array(itemgetter(*Samples_1_train_idx)(data_1_train1))
    Samples_2_train_dat = np.array(itemgetter(*Samples_2_train_idx)(data_2_train1))
    Samples_train_ip = np.vstack((Samples_1_train_dat,Samples_2_train_dat))
    return Samples_train_ip[:,0:(Samples_train_ip.shape[1]-2)],Samples_train_ip[:,(Samples_train_ip.shape[1]-1)] #x_samples,np.array(itemgetter(*Samples)(y_train_ip))

def GeneratePredictor(x_train_rand1,y_train_rand1):
    return LogisticRegression().fit(x_train_rand1, y_train_rand1)

def PredictOutput(reg1,x_test1):
    y_predicted = reg1.predict(x_test1)
    return y_predicted

def OutputError(y_predict1,y_test1):
    y_predict1 = np.array(y_predict1).reshape(-1,1)
    y_test1 = np.array(y_test1).reshape(-1,1)
    return (y_predict1 != y_test1).sum()



def main():
    InputPath = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
    DataObj = OnlineDataGet(InputPath)
    data_1_train,data_2_train,data_1_test,data_2_test = DataObj.PartitionData()
    x_test_ip = np.vstack((data_1_test,data_2_test))
    y_test = x_test_ip[:,(x_test_ip.shape[1]-1)]
    Train_data_numbers = [20, 50, 75, 100,150,200]
    error =  np.zeros(len(Train_data_numbers))
    error_ = []
    iterations = 1000

    for iters in range(1,iterations):

        for Train_data_numbers_ in Train_data_numbers:

            x_train_rand,y_train_rand = RandomizedTrainingDataGenerator(data_1_train,data_2_train,Train_data_numbers_)
            reg = GeneratePredictor(x_train_rand,y_train_rand)
            y_predict = PredictOutput(reg,x_test_ip[:,0:(x_test_ip.shape[1]-2)])
            error_.append(OutputError(y_predict,y_test))


        error = np.vstack((error,error_))
        error_ = []

    error = np.array(error)
    error = np.sum(error, axis=0)
    error = error/iterations

    plot1,= plt.plot(Train_data_numbers,error,'b.',label='Logistic regression error')
    plt.grid()
    plt.xlabel('Samples')
    plt.ylabel('Training error')
    plt.title('Training error for Logistic Regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
