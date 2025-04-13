import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import tensorflow as tf
from sklearn.utils import shuffle  
# for modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
# set random seed
my_seed = 80
np.random.seed(my_seed)
import random 
random.seed(my_seed)
tf.random.set_seed(my_seed)
# package for checking results
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score, roc_curve

class nnClassifier():
    def __init__(self, train_address = './train.csv', test_address = './test.csv'):
        self.lam = None
        self.lr = None
        self.train_address = train_address
        self.test_address = test_address
        self.dataLoad()

    def dataLoad(self):
        self.origin_train = pd.read_csv(self.train_address)
        self.origin_train = shuffle(self.origin_train)  
        self.origin_test = pd.read_csv(self.test_address)

    def nnFitting(self,N_HIDDEN = 80, activate = "tanh", optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001)):
        '''
        input: hidden units, activation function and optimizer
                 choose activation function from [relu, sigmoid, softplus, softmax, softsign, tanh, selu, elu]
                 choose optimizers from [Adam, Nadam, Adagrad, Adamax, Ftrl]
        output: the fitted model
        You can adjust the layers and units by yourself
        '''
        y = self.origin_train['class']
        x = self.origin_train.drop(['class'], axis = 1)
        # x_norm = x.apply(lambda t:(t - np.min(t)) / (np.max(t) - np.min(t)))
        # x_norm = x.apply(lambda t:((t - np.mean(t)) / np.std(t)))
        x_norm = x
        X = np.array(x_norm)
        self.model = Sequential()
        # choose activation function from [relu, sigmoid, softplus, softmax, softsign, tanh, selu, elu]

        activate = activate

        # add layers in the nn model
        self.model.add(Dense(N_HIDDEN, input_shape=(X.shape[1],), activation=activate)) # Add an input shape (features,)
        # model.add(Dropout(.3))
        self.model.add(Dense(N_HIDDEN, activation=activate))
        # model.add(Dropout(.3))
        self.model.add(Dense(N_HIDDEN, activation=activate))
        # model.add(Dropout(.3))
        self.model.add(Dense(N_HIDDEN, activation=activate))
        # model.add(Dropout(.3))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary() 

        # compile the model 
        # choose optimizers from [Adam, Nadam, Adagrad, Adamax, Ftrl]
        opt = optimizer
        self.model.compile(optimizer = opt, 
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        # callbacks
        es_val_loss = EarlyStopping(monitor='val_loss',patience = 2)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                        patience=2, verbose=1, mode='auto',
                                        min_delta=0.0001, cooldown=0, min_lr=1e-8)

        # for unblanced dataset, set weights for class_i by (1 / number of class_i)
        counts = np.bincount(y)
        weight_for_0 = 1.0 / counts[0]
        weight_for_1 = 1.0 / counts[1]
        class_weight = {0: weight_for_0, 1: weight_for_1}

        # now we just update our model fit call
        self.history = self.model.fit(X,
                                y,
                                callbacks=[reduce_lr, es_val_loss],
                                epochs=100, # you can set this to a big number
                                batch_size=128,
                                validation_split = 0.25,
                                shuffle=True,
                                class_weight = class_weight,
                                verbose=1)
    
    def nnFittingVisualization(self):
        '''
        give the result on training and validation set
        output: two graphs, one for loss and one for accuracy
        '''
        history_dict = self.history.history
        # Learning curve(Loss)
        # let's see the training and validation loss by epoch

        # loss
        loss_values = history_dict['loss'] # you can change this
        val_loss_values = history_dict['val_loss'] # you can also change this

        # range of X (no. of epochs)
        epochs = range(1, len(loss_values) + 1) 

        # plot
        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        # range of X (no. of epochs)
        epochs = range(1, len(acc) + 1)

        # plot
        # "bo" is for "blue dot"
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        # orange is for "orange"
        plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def testResult(self):  
        '''
        give the results on test set
        '''
        test_y = self.origin_test['class']
        test_x = self.origin_test.drop(['class'], axis = 1)
        # you can choose to standard the dataset if you haven't done it in the preprocession
        # test_norm = test_x.apply(lambda t:(t - np.min(t)) / (np.max(t) - np.min(t)))
        # test_norm = test_x.apply(lambda t:((t - np.mean(t)) / np.std(t)))
        test_norm = test_x
        test_x = np.array(test_norm)
        predicition = self.model.predict(test_x)
        preds = np.round(self.model.predict(test_x),0)
        confu_matrix = confusion_matrix(test_y, preds)
        result_report = classification_report(test_y, preds)
        AUC = roc_auc_score(test_y, predicition)
        print(confu_matrix)
        print(result_report)
        print(AUC)

        score = self.model.predict(test_x)
        auc = roc_auc_score(test_y, score)
        fpr, tpr, _ = roc_curve(test_y, score)
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(fpr, tpr, color='darkorange',
                label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for testing set')
        plt.legend(loc="lower right")

        train_x = self.origin_train.drop(['class'], axis = 1)
        train_y = self.origin_train['class']
        score = self.model.predict(train_x)
        auc = roc_auc_score(train_y, score)
        fpr, tpr, _ = roc_curve(train_y, score)
        plt.subplot(122)
        plt.plot(fpr, tpr, color='darkorange',
                label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for training set')
        plt.legend(loc="lower right")
        plt.show()
        return preds

if __name__=="__main__":
    p = nnClassifier()
    p.nnFitting()
    p.nnFittingVisualization()
    p.testResult()