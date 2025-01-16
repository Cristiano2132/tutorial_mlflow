import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Carregar os dados do arquivo CSV"""
    df = pd.read_csv(filepath)
    return df

def split_data(df, features, label, test_size=0.2, random_state=42):
    """Dividir os dados em treino e teste"""
    X = df[features]
    y = df[label]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test