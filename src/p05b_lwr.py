import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel

def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training and testing set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    lwr = LocallyWeightedLinearRegression(tau)
    lwr.fit(x_train,y_train)
    predict_train = lwr.predict(x_train)
    predict_eval = lwr.predict(x_eval)

    # Plot validation predictions on top of training set
    plot(x_train,y_train,predict_train,save_path='output/p05b_x_train.png')
    #Testing the predictions on the evaluation/valid data set
    plot(x_eval,y_eval,predict_eval,save_path='output/p05b_x_eval.png')

    # Get MeanSquaredError value on the validation set
    mse = np.mean((predict_eval - y_eval)**2)
    print(f'MSE={mse}')
    # *** END CODE HERE ***

class LocallyWeightedLinearRegression(LinearModel):

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.
        """
        # *** START CODE HERE ***
        self.x = x #Our training dataset inputs have shape (300,1)
        self.y = y #Our training dataset outputs have shape (300,)
        m,n = self.x.shape
        #Initializing theta to a zero vector. This theta vector will contain all the thetas for every point calculated from LWR
        self.theta = np.zeros((n,1))
        # *** END CODE HERE ***

    def predict(self, x):
        # *** START CODE HERE ***
         
        m = x.shape[0] #Our evaluating dataset inputs have shape (200,1)
        n = self.x.shape[0]
        w = np.mat(np.eye(n))  # Initialising W as a identity matrix with shape (n,n)
        predictions = np.zeros((m,1)) #The outputs for each set of parameters will be saved here

        for i in range(m):
            #The local point that will be taken into consideration will be from the new dataset. This point will be weighted with the points from the training dataset.
            xi = x[i]
            #This nested loop will populate w with the weights of each point w.r.t every other point.
            for j in range(n): 
                xj = self.x[j]
                denominator = 2 * self.tau * self.tau
                w[j,j] = np.exp(-(np.dot((xi-xj), (xi-xj).T))/denominator) #Gaussian kernel
            #After the completion of every inner loop, the value of theta for the specific xi will be calculated and saved in the matrix of all thetas
            self.y = self.y.reshape((n,1))
            self.theta = (np.linalg.pinv(self.x.T @ w @ self.x) @ (self.x.T @ w @ self.y))
            y_predict = np.dot(self.theta.T, xi) #Shape will be (1,)
            predictions[i] = y_predict
        return predictions
        # *** END CODE HERE ***
def plot(x_values, y_values, pred_values,save_path):
    plt.figure()
    plt.plot(x_values, y_values, 'bx', linewidth=2)
    plt.plot(x_values, pred_values, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(save_path)