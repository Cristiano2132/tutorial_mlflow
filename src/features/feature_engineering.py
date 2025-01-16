import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def custom_cut(df, column, bins, labels=None, nan_value: any=None)->list:
    # Ordena as bins
    bins = sorted(bins)
    
    while bins[0] < df[column].min():
        bins.pop(0)
        if len(bins) == 0:
            break
        
    while bins[-1] > df[column].max():
        bins.pop()
        if len(bins) == 0:
            break
        
    if len(bins) == 0:
        raise ValueError("Bins are out of range of values in the dataframe")
    
    # Cria labels padrão se não forem fornecidas
    if labels is None:
        labels = []
        labels.append(f"<{bins[0]}")
        for i in range(len(bins) - 1):
            labels.append(f"[{bins[i]}, {bins[i+1]})")
        labels.append(f">={bins[-1]}")
        
    # Função para categorizar os valores
    def categorize(value):
        if nan_value:
            if value == nan_value:
                return "N/A"
        elif pd.isna(value):
            return "N/A"
        
        if value < bins[0]:
            return labels[0]
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i+1]:
                return labels[i+1]
        return labels[-1]
    
    # Aplica a função de categorização
    return df[column].apply(categorize)


#  Classe para categorizar variaveis continuas
class TreeCategizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def get_splits(self, target_column: str, feature_column: str, max_depth: int =2)->list:
        regr = DecisionTreeClassifier(max_depth= max_depth, random_state=1234, max_leaf_nodes = 3)
        model = regr.fit(self.df[[feature_column]], self.df[[target_column]])
        self.model = model
        splits = model.tree_.threshold
        splits = list(set(splits))
        bins = sorted(splits)
        return bins

#  classe para calcular o woe de cada categoria
class WOEEncoder:
    def __init__(self, df: pd.DataFrame, target_col: str, regularization: float=1.0):
        self.df = df
        self.target_col = target_col
        self.regularization = regularization
        self.woe_dict = {}
    def fit(self, col: str):
        # Verifica se a coluna alvo é binária
        unique = self.df[self.target_col].unique()
        if len(unique) != 2:
            raise ValueError("A coluna alvo deve ser binária.")
        
        # Calcula as estatísticas globais
        total_events = self.df[self.target_col].sum()
        total_non_events = len(self.df) - total_events
        
        # Calcula WOE para cada categoria
        woe_dict = {}
        for category in self.df[col].unique():
            category_df = self.df[self.df[col] == category]
            events = category_df[self.target_col].sum()
            non_events = len(category_df) - events
            
            # Aplica regularização para evitar divisão por zero
            event_rate = (events + self.regularization) / (total_events + 2 * self.regularization)
            non_event_rate = (non_events + self.regularization) / (total_non_events + 2 * self.regularization)
            woe = np.log(event_rate / non_event_rate)
            woe_dict[category] = woe
        self.woe_dict[col] = woe_dict
        return woe_dict
    
    def fit_transform(self, col: str):
        self.fit(col)
        return self.df[col].map(self.woe_dict.get(col))



if __name__ == "__main__":
    # exemplo usandoTree categorizer e custom_cut e WOEEncoder
    data = {'feature': [1, 3, 2, 3.4, 7, 9, 10, 9.8, np.nan, None, np.nan, None], 'class': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]}
    df = pd.DataFrame(data)
    # replace nan e none por -1
    tc = TreeCategizer(df.dropna())
    bins = tc.get_splits('class', 'feature')
    df['feature_cat'] = custom_cut(df, 'feature', bins)
    encoder = WOEEncoder(df, 'class')
    encoder.fit('feature_cat')
    df['feature_cat_woe'] = df['feature_cat'].map(encoder.woe_dict.get('feature_cat'))
    print(df)