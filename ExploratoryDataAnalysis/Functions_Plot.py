import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def PlotsDistributionFeatureByTarget(Dataset:pd.DataFrame,Feature:str,NumericalTarget:str,EncodedTarget:str,Color:str,PalleteColors:dict[str,str]) -> None:
    """
        Function to plot feature distribution and 
        its scatter plot using the target

        -- Dataset : pd.DataFrame :: Dataset whose data is to be plotted

        -- Feature : str :: Attribute used in the plots

        -- NumericalTarget : str :: Attribute target represented in numerical form

        -- EncodedTarget : str :: Attribute target represented in categorical form

        -- Color : str :: Color for data

        -- PalleteColors : dict[str,str] :: Color palette for categorical data
    """
    fig , axes = plt.subplots(2,2,figsize=(10,10),layout='constrained')
    
    sns.histplot(Dataset,x=Feature,ax=axes[0,0],stat='probability',color=Color)
    axes[0,0].set_title(f'Distribution of\n{Feature}')

    sns.boxplot(Dataset,x=Feature,hue=EncodedTarget,ax=axes[0,1],legend=False,palette=PalleteColors)
    axes[0,1].set_title(f'Distribution of\n{Feature}\nby {EncodedTarget}')

    sns.regplot(Dataset,x=Feature,y=NumericalTarget,ci=None,ax=axes[1,0],color=Color)
    axes[1,0].set_title(f'Relation between\n{Feature}\nand {NumericalTarget}')

    for category_value in Dataset[EncodedTarget].unique():
        index_categories = Dataset.query(f"`{EncodedTarget}` == @category_value").index
        sns.regplot(Dataset.loc[index_categories],x=Feature,y=NumericalTarget,ci=None,ax=axes[1,1],label=category_value,color=PalleteColors[category_value])
    axes[1,1].set_title(f'Relation between\n{Feature}\nand {NumericalTarget} by {EncodedTarget}')
    
    fig.legend(title='Type of Income',bbox_to_anchor=(1.18,0.508))
    
    fig.suptitle(f'Distribution and Scatter Plots of\n{Feature}',size=20)