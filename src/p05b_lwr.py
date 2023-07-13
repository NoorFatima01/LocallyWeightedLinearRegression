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
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    lwr = LocallyWeightedLinearRegression(tau)
    lwr.fit(x_train,y_train)
    predict = lwr.predict(x_train)
    # lwr.plot(x_train,y_train,predict,"Training set")

    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_train, predict, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05b.png')

    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.
        """
        # *** START CODE HERE ***
        m,n = x.shape
        self.theta = np.zeros((m,1,n)) #Initializing theta to a zero vector. This theta vector will contain all the thetas for every point calculated from LWR
        self.x = x
        self.y = y

        # Initialising W as a identity matrix (diagonal) with shape (m,m)
        w = np.mat(np.eye(m))  

        #This nested loop will populate w with the weights of each point w.r.t every other point.
        for i in range(m):
            xi = self.x[i]
            for j in range(m):
                xj = self.x[j]
                denominator = 2 * self.tau * self.tau
                w[j,j] = np.exp(-(np.dot((xi-xj), (xi-xj).T))/denominator) #Gaussian kernel

            #After the completion of every inner loop, the value of theta for the specific xi will be calculated and saved in the matrix of all thetas
            self.y = self.y.reshape((m,1))
            temp = w@self.x 
            temp1 = np.linalg.pinv(self.x.T@temp)
            temp2 = (w @ self.y)
            temp3 = self.x.T @ temp2
            self.theta[i] = (temp1 @ temp3).T

        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.
        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        predictions = []
        #Calculating y_predict for every point using its respective theta evaluated from weighted regression
        for i in range(m):
            y_predict = np.dot(self.theta[i], x[i]) #Shape will be (1,)
            predictions.append(y_predict)
        return predictions
        # *** END CODE HERE ***