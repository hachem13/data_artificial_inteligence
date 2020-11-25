from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import scipy.optimize as opt
import pandas as pd
import numpy as np


query = pd.read_csv("/home/hachem/Documents/git/data_artificial_inteligence/brief_projet/regression_logistique/data.csv")
query['Gender'] = pd.get_dummies(query['Gender'])

#x = np.matrix([np.ones(query.shape[0]),query["Age"].values,query["EstimatedSalary"].values])
x = np.matrix([np.ones(query.shape[0]),query["Gender"].values,query["Age"].values,query["EstimatedSalary"].values])

y = np.matrix(query["Purchased"])

class model_regression_logistic():

    def sigmoid(self, Z):
        """sigmoid """
        self.Z = Z
        proba = np.exp(Z)/(1+np.exp(Z))
        return proba
    def cost(self, x, y, theta):
        """Error function"""
        self.x = x
        self.y = y
        self.theta = theta

        m = x.shape
        #h = theta.dot(x) # produit (1, 3) (3, m)
        #h = x.dot(theta) # produit (m, 3) (3, 1) 
        #j_theta = -1/m(-y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1 -h)))
        #grad = 1/m (h - y).dot(x)
        j_theta = -1/m*(np.log(self.sigmoid(theta.dot(x))).dot(y.T)+np.log(1-self.sigmoid(theta.dot(x))).dot((1-y).T))
        grad = (1/m)*(self.sigmoid(theta.dot(x))-y).dot(x.T)
        return [j_theta, grad]
    
    def predict(self,x, theta):
        self.x = x
        self. theta = theta
        m = x.shape
        proba = self.sigmoid(theta.dot(x))
        pred = np.matrix([np.zeros(m)])
        for i in range(0,m):
            if proba[0, i] > 0.5:
                pred[0, i] = 1
            else:
                pred[0, i] = 0
        return pred
    
theta = np.matrix([0, 0, 0])
logistic_reg = model_regression_logistic()
fuction_cost = logistic_reg.cost(theta, x,y)
print("la fonction de coût :", fuction_cost)
