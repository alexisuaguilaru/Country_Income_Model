from sklearn.metrics import accuracy_score , precision_recall_fscore_support
import matplotlib.pyplot as plt

import pandas as pd 
import numpy as np

def EvaluateModels(ModelNames:list[str],Models:list,EvaluationFeatureSet_X:pd.DataFrame|np.ndarray,EvaluationTargetSet_y:pd.Series|np.ndarray) -> pd.DataFrame:
    """
        Function to evaluate a set of models 
        using certain metrics on an evaluation 
        dataset

        -- ModelNames : list[str] :: Models's names 

        -- Models : list :: Models to be evaluated

        -- EvaluationFeatureSet_X : pd.DataFrame|np.ndarray :: Set X for evaluation 

        -- EvaluationTargetSet_y : pd.Series|np.ndarray :: Set y for evaluation

        Returns a dataframe with the results 
        obtained by each model.
    """
    evaluation_results = {}
    for model_name , model in zip(ModelNames,Models):
        PredictTarget_y = model.predict(EvaluationFeatureSet_X)
        evaluation_results[f'Model {model_name}'] = [accuracy_score(EvaluationTargetSet_y,PredictTarget_y)] + list(precision_recall_fscore_support(EvaluationTargetSet_y,PredictTarget_y,average='macro')[:3])

    return pd.DataFrame(evaluation_results,index=['Accuracy','Precision','Recall','F1'])

def PlotEvaluationResults(EvaluationResults:pd.DataFrame,ColorPalette:list[str]|dict[str,str]=None) -> None:
    """
        Function to plot obtained results 
        for each model

        -- EvaluationResults : pd.DataFrame :: Results obtained by each model

        -- ColorPalette : list[str]|dict[str,str] :: Color palette to represent each model
    """
    fig , axes = plt.subplots()
    axes.set_title('Evaluation Results of each\nModel with Different Metrics',size=18)

    EvaluationResults.plot.bar(rot=0,ax=axes,color=ColorPalette,legend=False)

    axes.set_xlabel('Metrics',size=12)
    axes.set_ylabel('Score',size=12)
    fig.legend(title='Models',bbox_to_anchor=(1.15,0.6))