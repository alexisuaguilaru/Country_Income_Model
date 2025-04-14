import numpy as np
from sklearn.metrics import make_scorer

from Function_TargetTransformation import TargetTransformation

from typing import Callable

def AppliedTransformationBeforeScoreFunction(ScoreFunction:Callable,*args_score_function:tuple,Transformation:Callable=TargetTransformation,**kwargs_score_function:dict) -> Callable:
    """
        Function to wrap some scoring function 
        in order to apply a transformation to the 
        values of y_true
        
        -- ScoreFunction : Callable :: Scoring function being applied or wrapped

        -- args_score_function : tuple :: Arguments for scoring function

        -- Transformation : Callable :: Transformation applied to the values of y_true

        -- kwargs_score_function : dict :: Keyword arguments for scoring function
    """
    def ScoringFunction(y_true:np.ndarray,y_pred:np.ndarray) -> float:
        y_true_trans = Transformation(y_true)
        return ScoreFunction(y_true_trans,y_pred,*args_score_function,**kwargs_score_function)
    return make_scorer(ScoringFunction)