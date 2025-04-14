from sklearn.base import BaseEstimator , ClassifierMixin
from sklearn.compose import TransformedTargetRegressor

from Function_TargetTransformation import TargetTransformation , NumericalTarget_Function , NumericalTarget_InverseFunction

from typing import Callable

class RegressionCategorical(BaseEstimator,ClassifierMixin):
    def __init__(self,regressor,Transformation:Callable=TargetTransformation,FunctionTransformation:Callable=NumericalTarget_Function,FunctionInverseTransformation:Callable=NumericalTarget_InverseFunction):
        """
            Estimator based on using the prediction of 
            some regressor and transforming the resulting values 
            into categorical or discrete values under some 
            transformation.

            -- regressor :: Regressor on which the model prediction is based

            -- Transformation : Callable :: Transformation applied to predicted values

            -- FunctionTransformation : Callable :: Function to be applied to training values before using the fit method

            -- FunctionInverseTransformation : Callable :: Function to be applied to the values after using the predict method
        """
        self.FunctionTransformation = FunctionTransformation
        self.FunctionInverseTransformation = FunctionInverseTransformation
        
        self.regressor = regressor
        self.__regressor = TransformedTargetRegressor(regressor,func=FunctionTransformation,inverse_func=FunctionInverseTransformation)
        
        self.Transformation = Transformation

    def fit(self,X,y):
        self.__regressor.fit(X,y)
        self.is_fitted_ = True
        return self
    
    def predict(self,X,y=None):
        return self.Transformation(self.__regressor.predict(X))