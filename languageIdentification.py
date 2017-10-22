
#####################################
# Author : Daniel D'souza
# Email : ddsouza@umich.edu
#####################################


#####################################
######## Library Definitions ########
#####################################
import numpy as np
import scipy
import pandas as pd
import sklearn
import codecs
import sys
import io
import math
import csv
from sklearn.metrics import accuracy_score
from collections import Counter
## Set Random Seed for Consistency
np.random.seed(0)

##############################################################################################
################################ EECS 595 : ASSIGNMENT 2 #####################################
###################### LANGUAGE IDENTIFICATION USING NEURAL NETS #############################
##############################################################################################

## Main Function ##
def main():
    # Paths to Data
    train_path = sys.argv[1]
    dev_path = sys.argv[2]
    test_path = sys.argv[3]
    output_path = './languageIdentificationPart1.output'

    EPOCH = 3

    ####### PART 1 ##########################################################################

    # HyperParameters

    hidden_Size = 100
    Eta = 0.1
    # Creating a Model Object X
    X = Model(train_path,dev_path,test_path,output_path,hidden_Size,Eta)

    # Train the Model
    X.print_seperator()
    X.train(EPOCH,1)

    #Generate the Test Solutions
    print "Predicting Test Solutions..."
    X.generate_test_output(True)
    X.print_seperator()
    ###########################################################################################


    ################# UNCOMMENT BELOW SECTION FOR PART 2 #######################################


    # ####### PART 2 ############################################################################
    # hidden_size_vals = [50,100,150,200]
    # eta_vals = [0.01, 0.1, 0.25, 0.5]
    #
    # for i in range(len(eta_vals)):
    #     hidden_Size = hidden_size_vals[i]
    #     Eta = eta_vals[i]
    #     # Creating a Model Object X
    #     X = Model(train_path,dev_path,test_path,output_path,hidden_Size,Eta)
    #
    #     # Train the Model
    #     X.print_seperator()
    #     X.train(EPOCH,1)
    #
    #     #Generate the Test Solutions
    #     print "Predicting Test Solutions..."
    #     X.generate_test_output(True)
    #     X.print_seperator()
    # ############################################################################################

###########################################################################################################################################################################################
    # X.print_mah_solutions()
## Classes :##############
##          1. Data_Ops
##          2. Model ( derives from Data_Ops)
##########################

##############################################################################################
## A Data_Ops class object pertains to data and the vector operations and cleanup
##############################################################################################
class Data_Ops(object):

    def charvecs_and_labels(self,path,Keys,Vocab):
        Data_old,Labels_temp_old = self.load_data(path,Keys)
        Data,Labels_temp = self.scramble_data(Data_old,Labels_temp_old)
        char_vex = self.make_char_embed(Data,Vocab)
        Labels = self.make_ycap(Labels_temp)
        return char_vex,Labels

    def load_data(self,path,Keys):
        Texts = []
        Labels = []
        alltextchars = []
        f = open(path, 'r').read().lower()
        lines = []
        for line in f.split('\n'):
                lines.append(line)
                label,chunks,letter_splits = self.break_labels_text((line.lower()),Keys)
                Texts+=chunks
                Labels+=label
        return Texts,Labels


    def scramble_data(self,data,label):
        length = len(data)
        a = np.arange(length)
        np.random.shuffle(a)
        a_new = list(a)
        data_new = [data[i] for i in a_new]
        label_new = [label[i] for i in a_new]
        return data_new,label_new


    def get_from_dict(self,letter,V):
        if letter in V:
            return V[letter]
        else:
            return 0

    def make_char_embed(self,chunks,V):
        n = len(V)
        X = np.zeros((len(chunks),5*len(V)))
        alpha_to_num_vex = [[self.get_from_dict((j.lower()),V) for j in i] for i in chunks]
        for j in range(len(alpha_to_num_vex)):
            ind = alpha_to_num_vex[j]
            IND = [ind[i]+(n*i) for i in range(len(ind))]
            X[j][IND] = 1
        return X

    def break_labels_text(self,line,Keys):
        # print(line)
        lsplit = line.split(' ')
        letter_splits = list(" ".join(lsplit[1:]))
        CHUNKS = [letter_splits[i:i+5] for i in range(0,len(letter_splits)-4)]
        label_num = self.get_from_dict(lsplit[0],Keys)
        return [label_num]*len(CHUNKS),CHUNKS,letter_splits


    def make_set_to_list(self,word_set):
        vocab = {}
        num = 0
        for i in word_set:
            vocab[(i.lower())]=num
            num+=1
        return vocab

    def make_ycap(self,train_Labels):
        t = [0,0,0]
        ycap = []
        for i in range(len(train_Labels)):
            t = [0,0,0]
            t[train_Labels[i]] = 1
            ycap.append(t)
        return ycap

    def print_seperator(self):
        print '\n'
        print "*"*100
        print "~"*100
        print "*"*100
        print '\n'

    def siggy(self,x):
        return np.asarray([1/(1+math.exp(-x[i])) for i in range(len(x))])

    def softma(self,x):
        temp = [math.exp(x[i]) for i in range(len(x))]
        return np.asarray([temp[i]/sum(temp) for i in range(len(temp))])

    def create_vocab(self,train_path,keys):
        VOCABULARY = {}
        alltextchars = []
        f = open(train_path, 'r').read().lower()
        for line in f.split('\n'):
                label,chunks,letter_splits = self.break_labels_text(line,keys)
                alltextchars+=letter_splits
        VOCABULARY = self.make_set_to_list(set(alltextchars))
        return VOCABULARY

    def softmax_derive(self,dLdy,y):
        dLdy_d = []
        N = range(len(y))
        for j in N:
            temp = 0
            for i in N:
                temp+=(dLdy[i]*(y[i]*((1 if i==j else 0)-y[j])))
            dLdy_d.append(temp)
        return np.asarray(dLdy_d)


    def test_eval(self,test_preds):
        data_file = open(self.test_sol_Path, 'r').read().lower()
        lines =[]
        for line in data_file.split('\n'):
            lines.append(line.split(' ',1))
        n=0
        for i in range((len(lines)-1)):
            if((lines[i][1])==test_preds[i]):
                n+=1
        print "Accuracy on Test Set is :", (n*100)/len(test_preds)," %"

    def generate_test_output(self,eval_on_test=True):
            test_lines,test_predictions = self.get_test_data(self.test_Path)
            if eval_on_test==True:
                self.test_eval(test_predictions)
            op_file = open(self.output_Path,'w+')
            for i in range(len(test_lines)):
                curr_line = test_lines[i]
                curr_pred = test_predictions[i]
                per_line = str(curr_line)+" "+str(curr_pred)+"\n"
                op_file.write(per_line)
            op_file.close()

            print "\n  Generated languageIdentificationPart1.output ! :) "

    def get_test_data(self,test_data_path):
        lines = []
        Data = []
        test_preds = []
        f = open(test_data_path, 'r').read().lower()
        for line in f.split('\n'):
                lines.append(line)
                clean_line = line.lower()
                lsplit = clean_line.split(' ')
                letter_splits = list(" ".join(lsplit[1:]))
                CHUNKS = [letter_splits[i:i+5] for i in range(0,len(letter_splits)-4)]
                char_vex = self.make_char_embed(CHUNKS,self.Vocab)
                test_preds_per_line = self.model_predict(char_vex)
                if(len(test_preds_per_line)>1):
                    curr_prediction_select = Counter(test_preds_per_line).most_common(1)[0][0]
                else:
                    curr_prediction_select = test_preds_per_line
                test_preds.append(curr_prediction_select)
        return lines[:-1],test_preds[:-1]

    def print_the_solutions(self):
        f = open(self.output_Path, 'r').read().lower()
        for line in f.split('\n'):
            print line

    def make_graphs(self,epoch,train,dev):
        epoch_X = [i for i in range(epoch+1)]
        y1 = train
        y2 = dev
        import matplotlib.pyplot as plt

        line1 = plt.plot(epoch_X,y1,'-ob',label = 'Training_Accuracy')
        line2 = plt.plot(epoch_X,y2,'-or',label = 'Dev_Accuracy')

        plt.ylabel('Accuracy')
        plt.xlabel('EPOCHS')

        plt.legend(loc='upper left')
        plt.savefig('accuracy.png')
##############################################################################################
## A Model class object contains the model which can be trained,optimized and tested
##############################################################################################

class Model(Data_Ops):

    def __init__(self,train_Path,dev_Path,test_Path,output_Path,hidden_size,eta):
        # Paths to Data
        self.train_Path = train_Path
        self.dev_Path = dev_Path
        self.test_Path = test_Path
        self.output_Path = output_Path
        self.test_sol_Path = './languageIdentification.data/test_solutions'

        # Output Keys
        self.Keys = {'english':0,'french':1,'italian':2}
        self.rev_Keys = {0:'english', 1:'french', 2:'italian'}

        ## HyperParameters
        # Size of Hidden Layer : h =100
        self.hidden_size = hidden_size
        # Learning Rate : eta = 0.1
        self.eta = eta
        print "Hidden Size : ",self.hidden_size
        print "eta :",self.eta

    def __init_weights__(self,n,N):
        self.n = n
        self.N = N
        #Weights W1 and b1
        self.W1t = 2*np.random.rand(self.hidden_size,(self.n)*5)-1
        self.b1 = 2*np.random.rand(self.hidden_size)-1
        #Weights W2 and b2
        self.W2t = 2*np.random.rand(self.N,self.hidden_size)-1
        self.b2 = 2*np.random.rand(self.N)-1

    def run_and_train(self,charvec,ycap,W1t,b1,W2t,b2):
        loss=0

        inp = np.asarray(charvec)
        h_dash = np.add(np.matmul(W1t,inp),b1)
        h = self.siggy(h_dash)
        y_dash = np.add(np.matmul(W2t,h),b2)
        y = self.softma(y_dash)
        pred_lab = sorted(zip(y,[0,1,2]),reverse=True)[:3][0][1]

        ## Loss : Squared Loss
        loss= (1/2)*np.transpose(y-ycap).dot(y-ycap)

        ## delL/dely = y-ycap
        dLdy = y-ycap
        #print(dLdy," = ",y,"-",ycap)

        ## Softmax Layer Derivative
        dLdy_dash = self.softmax_derive(dLdy,y)

        ## W2 and b2 Derivatives
        ht = np.asarray(h)
        ht = ht.reshape(1,self.hidden_size)
        dLdy_dash_temp = dLdy_dash.reshape(len(dLdy_dash),1)

        dLdW2 = np.dot(dLdy_dash_temp,ht)
        dLdb2 = dLdy_dash

        # h Derivative

        dLdh = np.dot(np.transpose(W2t),dLdy_dash)

        # h_dash Derivative
        dLdh_dash = dLdh*(h*(1-h))

        ## W1 and b1 Derivatives
        charvec_t = np.asarray(charvec)
        charvec_t = charvec_t.reshape(1,self.n*5)
        dLdh_dash_temp = dLdh_dash.reshape(len(dLdh_dash),1)

        dLdW1 = np.dot(dLdh_dash_temp,charvec_t)
        dLdb1 = dLdh_dash

        W1t-=(self.eta*dLdW1)
        b1-=(self.eta*dLdb1)
        W2t-=(self.eta*dLdW2)
        b2-=(self.eta*dLdb2)

        return W1t,b1,W2t,b2

    def line_accuracy(self,data_path):
        Texts = []
        Labels = []
        alltextchars = []
        per_line_truth = []
        per_line_pred = []

        f = open(data_path, 'r').read().lower()
        for line in f.split('\n'):
            if line == '':
                continue
            if len(line.split()) > 1:
                label,chunks,letter_splits = self.break_labels_text((line),self.Keys)
                char_vex = self.make_char_embed(chunks,self.Vocab)
                # label gives a vector of size of number of 5 character chunks with the same label in number form
                # chunks gives us an NP array of individual 5cX1 vectors
                curr_prediction = self.model_predict(char_vex,words=True)
                curr_prediction_nums = [self.Keys[name] for name in curr_prediction]
                curr_prediction_select = Counter(curr_prediction_nums).most_common(1)[0][0]
                per_line_truth.append(label[0])
                per_line_pred.append(curr_prediction_select)
        line_acc = sklearn.metrics.accuracy_score(per_line_truth,per_line_pred)
        print "Lines Accuracy =",line_acc*100," %"



    def evaluate_on_set(self,charvec_mat,ycap_mat,W1t,b1,W2t,b2):
        loss=0
        PRED = []
        TRUE = []
        for i in range(len(charvec_mat)):
            charvec = charvec_mat[i]
            ycap = ycap_mat[i]

            inp = np.asarray(charvec)
            h_dash = np.add(np.matmul(W1t,inp),b1)
            h = self.siggy(h_dash)
            y_dash = np.add(np.matmul(W2t,h),b2)
            y = self.softma(y_dash)
            pred_lab = sorted(zip(y,[0,1,2]),reverse=True)[:3][0][1]
            tru_lab = sorted(zip(ycap,[0,1,2]),reverse=True)[:3][0][1]
            PRED.append(pred_lab)
            TRUE.append(tru_lab)
            loss+= (1/2)*np.transpose(y-ycap).dot(y-ycap)
        acc = sklearn.metrics.accuracy_score(TRUE,PRED)

        print "Accuracy =",acc*100," %"
        # print("Loss on Prediction = ",loss)
        return acc*100,loss


    def model_predict(self,charvec_mat,words=True):
        PRED = []
        pred = []
        t= [[1,0,0],[0,1,0],[0,0,1]]
        for i in range(len(charvec_mat)):
            charvec = charvec_mat[i]
            inp = np.asarray(charvec)
            h_dash = np.add(np.matmul(self.W1t,inp),self.b1)
            h = self.siggy(h_dash)
            y_dash = np.add(np.matmul(self.W2t,h),self.b2)
            y = self.softma(y_dash)
            pred_lab = sorted(zip(y,[0,1,2]),reverse=True)[:3][0][1]
            pred.append(pred_lab)
            PRED.append(self.rev_Keys[pred_lab])
        if words==True:
            return PRED
        else:
            return [t[i] for i in pred]

    def train_weights(self,all_char_Vec,all_char_Labels,W1t,b1,W2t,b2):
        tempW1t = W1t
        tempW2t = W2t
        tempb1 = b1
        tempb2 = b2
        for i in range(len(all_char_Vec)):
            tempW1t,tempb1,tempW2t,tempb2 = self.run_and_train(all_char_Vec[i],all_char_Labels[i],tempW1t,tempb1,tempW2t,tempb2)
        return tempW1t,tempb1,tempW2t,tempb2

    def train(self,EPOCH=1,OPTIMIZE=0):
        #Create Vocabulary
        self.Vocab = self.create_vocab(self.train_Path,self.Keys)
        print " Vocabulary Created!\n"
        ## Load the Training and Development Data
        print " Loading Data..."
        self.dev_char_vex,self.dev_Labels = self.charvecs_and_labels(self.dev_Path,self.Keys,self.Vocab)
        self.train_char_vex,self.train_Labels = self.charvecs_and_labels(self.train_Path,self.Keys,self.Vocab)
        print " Data Loaded!"

        print " Initializing Weights and Biases!"
        ## Weights and Biases Initialization
        self.__init_weights__(len(self.Vocab),len(self.Keys))

        trained_W1t=self.W1t
        trained_b1=self.b1
        trained_W2t=self.W2t
        trained_b2=self.b2

        train_acc_vals = []
        dev_acc_vals = []
        curr_epoch_train_acc = 0
        curr_epoch_dev_acc = 0

        train_loss_vals = []
        dev_loss_vals = []
        curr_epoch_train_loss = 0
        curr_epoch_dev_loss = 0
        print "Initilization Complete!\n "


        print "Loss and Accuracy BEFORE OPTIMIZATION"

        print "Training Data:"
        curr_epoch_train_acc,curr_epoch_train_loss = self.evaluate_on_set(self.train_char_vex,self.train_Labels,trained_W1t,trained_b1,trained_W2t,trained_b2)
        self.line_accuracy(self.train_Path)
        print "Dev Data:"
        curr_epoch_dev_acc,curr_epoch_dev_loss = self.evaluate_on_set(self.dev_char_vex,self.dev_Labels,trained_W1t,trained_b1,trained_W2t,trained_b2)
        self.line_accuracy(self.dev_Path)

        train_acc_vals.append(curr_epoch_train_acc)
        train_loss_vals.append(curr_epoch_train_loss)

        dev_acc_vals.append(curr_epoch_dev_acc)
        dev_loss_vals.append(curr_epoch_dev_loss)

        if OPTIMIZE:
            print "^^^"*30
            print "*~*"*30
            print "\t\t\t\t Start Training"
            print "*~*"*30
            print "^^^"*30
            print "\n"

            for i in range(EPOCH):
                print "\n EPOCH :\t",i+1,"\n"
                trained_W1t,trained_b1,trained_W2t,trained_b2 =self.train_weights(self.train_char_vex,self.train_Labels,trained_W1t,trained_b1,trained_W2t,trained_b2)

                print "Training Data:"
                curr_epoch_train_acc,curr_epoch_train_loss = self.evaluate_on_set(self.train_char_vex,self.train_Labels,trained_W1t,trained_b1,trained_W2t,trained_b2)
                self.line_accuracy(self.train_Path)

                print "Dev Data:"
                curr_epoch_dev_acc,curr_epoch_dev_loss = self.evaluate_on_set(self.dev_char_vex,self.dev_Labels,trained_W1t,trained_b1,trained_W2t,trained_b2)
                self.line_accuracy(self.dev_Path)

                train_acc_vals.append(curr_epoch_train_acc)
                train_loss_vals.append(curr_epoch_train_loss)

                dev_acc_vals.append(curr_epoch_dev_acc)
                dev_loss_vals.append(curr_epoch_dev_loss)

                print "*~*"*20

            self.train_acco = train_acc_vals
            self.train_losso = train_loss_vals

            self.dev_acco = dev_acc_vals
            self.dev_losso = dev_loss_vals

            self.make_graphs(EPOCH,train_acc_vals,dev_acc_vals)

            print "\n"
            print "^^^"*30
            print "*~*"*30
            print "\t\t\t\t Training Complete"
            print "*~*"*30
            print "^^^"*30


####################################################
### Calling the Main Function
####################################################
main()
