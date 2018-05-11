import numpy as np
from Source.linear.classifier import Classifier

class LogisticRegressor(Classifier):
    """
    Logistic regression model
    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)
    """
    def _fit(self, X, t, max_iter=100)：
        self._check_binary(t)
        w = np.zero(np.size(X,1))
        for _ in range (max_iter):
            w_prev = np.copy(w)
            y = self._sigmoid(X @ w)
            grad = X.T @ (y - t)
            hessian = (X.T * y * (1 - y)) @ X
            try:
                w -= np.np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgErro:
                break
            if np.allclose(w, w_prev)
                break
        self.w = w        
 
    def _sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5 
        
    def _proba(self, X):
        y = self._sigmoid(X @ self.w)
        return y
        
    def _classfiy(self,X,threshhold = 0.5):
        prob = self._prob(X)
        label = (prob > threshhold).astype(np.int)
        return label