import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
        
    def fit(self, X, y):

        
        """
        # Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================

        self.weights = np.zeros(X.shape[1])
        self.bias = 0


        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias # Formel for y-pred

            m = X.shape[0] # Antall elementer

            # Mean squared error
            mse = np.square(np.subtract(y, y_pred)).mean()

            #Finn grad w og grad b
            grad_w = (2/m) * np.dot(X.T, y_pred-y)
            grad_bias = (2/m) * np.sum(y_pred - y)

            #Oppdater
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_bias


            #Lagre loss
            self.loss_history.append(mse)

        return self
        #raise NotImplementedError("LinearRegression.fit is not implemented yet.")
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================

        return np.dot(X, self.weights) + self.bias
        # raise NotImplementedError("LinearRegression.predict is not implemented yet.")


        # LOGISTIC REGRESSION
    
class LogisticRegression():

    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
    
    def fit(self, X, y):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================

        m = X.shape[1] # Total observasjoner
        n = X.shape[0] # features

        self.weights = np.zeros((n,1)) # Datapunkter
        self.bias = 0

        cost_list = []


        # Gradient Descent
        for i in range(self.n_iterations):

            z = np.dot(self.weights.T, X) + self.bias
            A = self.sigmoid(z)

            cost = -(1/m)*np.sum( y*np.log(A) + (1-y)*np.log(1-A))

            dW = (1/m)*np.dot(A-y, X.T)
            dB = (1/m)*np.sum(A - y)

            self.weights = self.weights - self.lr * dW.T
            self.bias = self.bias - self.lr * dB

            cost_list.append(cost)



        return self.weights, self.bias, cost_list

    """  
            y_pred = self.predict(X) # Formel for y-pred

            m = X.shape[0] # Antall elementer

            # Mean squared error
            mse = np.square(np.subtract(y, y_pred)).mean()

            #Finn grad w og grad b
            grad_w = (2/m) * np.dot(X.T, y_pred-y)
            grad_bias = (2/m) * np.sum(y_pred - y)

            #Oppdater
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_bias


            #Lagre loss
            self.loss_history.append(mse)


        
"""





        # raise NotImplementedError("LogisticRegression.fit is not implemented yet.")
    
    # Gir selve modellen sin rå output. En sannsynlighet for at hver rad i X hører til den positive klassen
    def predict_proba(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        z = X*self.weights + self.bias

        return self.sigmoid(z)
    

        

        # raise NotImplementedError("LogisticRegression.predict_proba is not implemented yet.")

    # Den endelige diskre besluttningen - 0 eller 1, ja eller nei
    def predict(self, X):
        
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================

        lin_model = np.matmul(X, self.weights) + self.bias
        y_pred = self._sigmoid(lin_model)
        return [1 if _y > 0.5 else 0 for _y in y_pred]
        
        # raise NotImplementedError("LogisticRegression.predict is not implemented yet.")
    
    def sigmoid(self, z):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================

        return 1/(1+np.exp(-z))
        # raise NotImplementedError("LogisticRegression.sigmoid is not implemented yet.")