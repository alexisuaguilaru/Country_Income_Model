import pandas as pd
import numpy as np

from typing import Callable

def MissingValuesByFeatures(Dataset:pd.DataFrame,Percent:bool=False) -> pd.Series:
    """
        Function to get the number or percent 
        of missing values by feature

        -- Dataset : pd.DataFrame :: Dataset where is got their missing values

        -- Percent : bool :: Whether the absolute or relative count of missing values is returned
        
        Return a series with the amount of 
        missing values
    """
    percent_scalar = 100/Dataset.shape[0] if Percent else 1
    return percent_scalar*Dataset.isna().sum()

def FiltersValuesBasedOnDataType(Dataset:pd.DataFrame,Attribute:str,Datatype) -> pd.Series:
    """
        Function to filter a dataframe based on datatype 
        of a feature or variable

        -- Dataset : pd.DataFrame :: Dataset where filtering is applied

        -- Attribute : str :: Feature to be filtered

        -- Datatype :: Data type to filter

        Return a series of boolean values where the 
        feature value's is equal with the data type
    """
    return Dataset[Attribute].apply(lambda value : type(value) is Datatype)

def SplittingFeaturesBasedDataType(Dataset:pd.DataFrame,Features:list[str]) -> list[list[str]]:
    """
        Function to split features of a dataset based 
        on their data types

        -- Dataset : pd.DataFrame :: Dataset where feature splitting is applied

        -- Features : list[str] :: Feature to be splitted

        Return a list of categorical, discrete numerical 
        and continuous numerical features
    """
    categorical_features , discrete_features , continuous_features = [] , [] , []

    for feature in Features:
        if (dtype_feature:=Dataset[feature].dtype) == 'object':
            categorical_features.append(feature)
        elif dtype_feature == 'int':
            discrete_features.append(feature)
        else:
            continuous_features.append(feature)
    
    return categorical_features , discrete_features , continuous_features

def ImputationMissingValuesUsingMedian(Dataset:pd.DataFrame,GroupingFeatures:list[str],NumericalFeatures:list[str]) -> Callable:
    """
        Function to impute missing values based on 
        stratified medians by a group of features

        -- Dataset : pd.DataFrame :: Dataset where imputation is applied

        -- GroupingFeatures : list[str] :: List of features for grouping

        -- NumericalFeatures : list[str] :: List of numerical features to get their medians

        Return a function for applying imputation 
        on a feature
    """
    if GroupingFeatures:
        stratified_medians = Dataset.groupby(GroupingFeatures)[NumericalFeatures].median()
    else:
        stratified_medians = Dataset[NumericalFeatures].median()

    def ImputationFeatureValues(Feature:str) -> Callable:
        """
            Function to apply imputation of 
            missing values on a feature 

            -- Feature : str :: Feature where imputation is applied

            Return a function for returning stratified 
            medians based on a group of values (allowed 
            values in grouping features)
        """

        def ApplyImputation(GroupValues:pd.Series) -> pd.DataFrame|pd.Series:
            """
                Function to return values for imputation 
                of missing values of a feature based on 
                grouping features 

                -- GroupValues : pd.Series :: Series of grouping values

                Return the imputation value for each 
                series' instance 
            """
            if GroupingFeatures:
                return stratified_medians.loc[GroupValues,Feature]
            else:
                return stratified_medians.loc[Feature]
        
        return ApplyImputation

    return ImputationFeatureValues

def TargetTransformation(Dataset:pd.DataFrame,Target:str) -> np.ndarray:
    """
        Function to transform the target into 
        a categorical target based on a 
        decision rule 

        -- Dataset : pd.DataFrame :: Dataset where transformation is applied

        -- Target : str :: Attribute on which the transformation is applied

        Return a array with the transformed 
        values 
    """
    decision_values = np.array([5000,25000])
    decision_border_names = np.array([border_name+'-income' for border_name in ['lower','average','high']])

    border_indexes = decision_values.searchsorted(Dataset[Target])
    return decision_border_names[border_indexes]