import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp



def get_summary(df: pd.DataFrame):
    df_summary = pd.DataFrame(columns=['column_type', 'na', 'na_pct', 'top_class', 'top_class_pct', 'nunique', 'unique_values'])
    for col in df.columns:
        df_summary.at[col, 'column_type'] = str(df[col].dtype)
        df_summary.at[col, 'na'] = df[col].isna().sum()
        df_summary.at[col, 'na_pct'] = df[col].isna().sum() / len(df) * 100
        df_summary.at[col, 'top_class'] = str(df[col].value_counts().index[0])
        df_summary.at[col, 'top_class_pct'] = df[col].value_counts().values[0] / len(df) * 100
        df_summary.at[col, 'nunique'] = df[col].nunique()
        if df[col].nunique() < 10:
            df_summary.at[col, 'unique_values'] = str(df[col].unique().tolist())
        else:
            df_summary.at[col, 'unique_values'] = '...'
    return df_summary


def cdf(sample, x, sort = False):
    '''
    Return the value of the Cumulative Distribution Function, evaluated for a given sample and a value x.
    
    Args:
        sample: The list or array of observations.
        x: The value for which the numerical cdf is evaluated.
    
    Returns:
        cdf = CDF_{sample}(x)
    '''
    
    # Sorts the sample, if needed
    if sort:
        sample.sort()
    
    # Counts how many observations are below x
    cdf = sum(sample <= x)
    
    # Divides by the total number of observations
    cdf = cdf / len(sample)
    
    return cdf


def plot_hist(df: pd.DataFrame, title: str, ax: plt.Axes, color0: str = 'b', color1: str = 'g', hue:bool=True):
    if hue:
        sns.histplot(data=df, x='score', hue='y_true', kde=True, bins=50, ax=ax, common_norm=False, stat="density", palette={0: color0, 1: color1})
    else:
        sns.histplot(data=df, x='score', kde=True, bins=50, ax=ax, common_norm=False, stat="density")

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # turn on y axis grid lines with a transparency and gray color
    ax.yaxis.grid(True, linestyle='-', linewidth=0.2)
    ax.xaxis.grid(True, linestyle='-', linewidth=0.2)


# Function to calculate cumulative distribution function for a sorted array
def get_classes_cdf_array(input_array: np.array):
    cumsum_ = np.cumsum(input_array)
    output = cumsum_ / cumsum_[-1]  # Normalize by the last value to ensure the CDF goes from 0 to 1
    
    return output

# Gets the class CDFs
def get_classes_cdf(df: pd.DataFrame, proba_col: str, true_value_col: str):

    # Mask for class 0
    mask0 = (df[true_value_col] == 0)
    
    # Recover each class, sorted by probability column
    class0 = df[mask0].sort_values(proba_col)[proba_col].values
    class1 = df[~mask0].sort_values(proba_col)[proba_col].values
    
    # Calculate the cumulative distribution functions
    cdf0 = get_classes_cdf_array(np.array(class0))
    cdf1 = get_classes_cdf_array(np.array(class1))
    
    # Results
    results = {
        'cdf0': cdf0,
        'cdf1': cdf1,
        'proba0': class0,
        'proba1': class1
    }
    
    return results


# Função para plotar as CDFs e a linha de KS
def plot_cdf_ks(cdf: dict, ks: float, ax: plt.Axes, color0='b', color1='g'):
    # Plotando as CDFs
    ax.plot(cdf['proba0'], cdf['cdf0'], color='b', linewidth=2)
    ax.plot(cdf['proba1'], cdf['cdf1'], color='g', linewidth=2)

    # Interpolando para garantir que tenham o mesmo número de pontos
    interp_cdf1 = interp1d(cdf['proba1'], cdf['cdf1'], kind='linear', bounds_error=False, fill_value=(0, 1))
    # Valores interpolados de 'cdf1' na grade de 'proba0'
    cdf1_interp = interp_cdf1(cdf['proba0'])
    
    # Encontrando o ponto de KS (onde a diferença é máxima)
    ks_x = cdf['proba0'][abs(cdf['cdf0'] - cdf1_interp).argmax()]
    ks_y0 = cdf['cdf0'][abs(cdf['cdf0'] - cdf1_interp).argmax()]
    ks_y1 = cdf1_interp[abs(cdf['cdf0'] - cdf1_interp).argmax()]
    print("Pontos de interpolação KS:")
    print(f"ks_x: {ks_x}, ks_y0: {ks_y0}, ks_y1: {ks_y1}")
    print(f"proba0 range: {cdf['proba0'].min()} - {cdf['proba0'].max()}")
    print(f"proba1 range: {cdf['proba1'].min()} - {cdf['proba1'].max()}")
    # Desenhando a linha vertical no ponto de KS
    ax.vlines(x=ks_x, ymin=ks_y0, ymax=ks_y1, color='r', linestyle='--', label=f'KS = {ks:.4f}')
    
    # Configurando a legenda e o título
    ax.legend(["class 0", "class 1", f"KS line"])
    ax.set_title(f"KS: {ks:.4f}")
    
    # Ajustando a estética
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.2)
    ax.xaxis.grid(True, linestyle='-', linewidth=0.2)

