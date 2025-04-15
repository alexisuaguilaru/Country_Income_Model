import pickle

from typing import Iterable

def DumpSaveModels(Models:list,ModelNames:list[str]) -> None:
    """
        Function to save a trained, fine-tuned model 
        like a .pkl file

        -- Models : list :: Models to be saved

        -- ModelNames : list[str] :: Models's names 

    """
    for model_name , model in zip(ModelNames,Models):
        with open(f'model_{model_name}.pkl','wb') as model_file:
            pickle.dump(model,model_file,protocol=5)

def LoadModels(ModelNames:list[str]) -> Iterable:
    """
        Function to load a saved model from a 
        .pkl file

        -- ModelNames : list[str] :: Model's name to be loaded or recovered

        Yield each loaded model 
    """
    for model_name in ModelNames:
        with open(f'model_{model_name}.pkl','rb') as model_file:
            yield pickle.load(model_file)