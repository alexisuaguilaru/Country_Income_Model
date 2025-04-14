import numpy as np

def NumericalTarget_Function(y_value:np.ndarray) -> np.ndarray: 
    """
        Function to applied log_10 
        over a array of values

        -- y_value : np.ndarray :: Values to which the log_10 is applied

        Return a array of transformed 
        values 
    """
    return np.log10(y_value)

def NumericalTarget_InverseFunction(y_value:np.ndarray) -> np.ndarray:
    """
        Function to applied 10^y_value
        over a array of values

        -- y_value : np.ndarray :: Values to which the 10^y_value is applied

        Return a array of transformed 
        values 
    """
    return np.power(10,y_value)

def TargetTransformation(NumericalValues:np.ndarray) -> np.ndarray:
    """
        Function to transform numerical target into 
        a categorical target based on a 
        fixed decision rule 

        -- NumericalValues : np.ndarray :: Values of target on which the transformation is applied

        Return a array with the transformed 
        values (categories)
    """
    decision_values = np.array([5000,25000])
    border_names = np.array([border_name+'-income' for border_name in ['lower','average','high']])

    border_index = decision_values.searchsorted(NumericalValues)
    return border_names[border_index]