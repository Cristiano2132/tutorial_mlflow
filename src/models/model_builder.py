from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from models.bayesian_opt import HyperparameterOptimizer
import pandas as pd

class ModelBuilder:
    """
    A class to simplify model building, training, and hyperparameter optimization.

    This class supports Logistic Regression, LightGBM, and XGBoost models. It provides
    methods to train models with default or custom parameters and optimize hyperparameters
    using Bayesian optimization.

    Attributes:
    ----------
    X : pandas.DataFrame
        Feature matrix for training the models.
    y : pandas.Series
        Target vector for training the models.
    random_state : int, optional
        Random state for reproducibility (default is 42).

    Methods:
    -------
    build_model(model_type, params={})
        Build and train a model with the specified type and parameters.
    get_optimized_model(model_type, param_space, n_trials=10)
        Optimize hyperparameters for the specified model type and train the optimized model.
    get_hyperparameters_space(model_type)
        Get the default hyperparameter search space for the specified model type.

    Example:
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification
    >>> features_list = ['feature1', 'feature2']
    >>> X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    >>> df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(X.shape[1])])
    >>> df['target'] = y
    >>> builder = ModelBuilder(df[features_list], df['target'])
    >>> params = {'C': 1.0, 'penalty': 'l2'}
    >>> model = builder.build_model('logistic', params)
    >>> print(model.coef_)
    """

    def __init__(self,X: pd.DataFrame, y: pd.Series, random_state=42):
        """
        Initialize the ModelBuilder with data and a random state.

        Parameters:
        ----------
        X : pandas.DataFrame
            Feature matrix for training the models.
        y : pandas.Series
            Target vector for training the models.
        random_state : int, optional
            Random state for reproducibility (default is 42).
        """
        self.X = X
        self.y = y
        self.random_state = random_state

    def __get_model(self, model_type):
        """
        Get an untrained model of the specified type.

        Parameters:
        ----------
        model_type : str
            Type of the model ('logistic', 'lgbm', 'xgb').

        Returns:
        -------
        object
            An instance of the specified model type.
        """
        if model_type == "logistic":
            return LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif model_type == "lgbm":
            return LGBMClassifier(random_state=self.random_state)
        elif model_type == "xgb":
            return XGBClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Modelo '{model_type}' não reconhecido.")

    def build_model(self, model_type: str, params: dict = {}):
        """
        Build and train a model with the specified type and parameters.

        Parameters:
        ----------
        model_type : str
            Type of the model ('logistic', 'lgbm', 'xgb').
        params : dict, optional
            Parameters to set for the model (default is an empty dictionary).

        Returns:
        -------
        object
            A trained instance of the specified model type.
        """
        model = self.__get_model(model_type)
        model.set_params(**params)
        model.fit(self.X, self.y)
        return model
    

    def build_optimized_model(self, model_type: str, param_space=None, n_trials: int = 10, return_best_params: bool=False):
        """
        Optimize hyperparameters for a given model type using Bayesian optimization.

        Parameters:
        ----------
        model_type : str
            Type of the model to be optimized ('logistic', 'lgbm', 'xgb').
        param_space : dict, optional
            Dictionary defining the hyperparameter search space.
        n_trials : int, optional
            Number of trials for the optimizer (default is 10).
        
        return_best_params : bool, optional
            Whether to return the best hyperparameters (default is False).
            
        Returns:
        -------
        object
            A trained model with the best hyperparameters.
        """
        model = self.__get_model(model_type)
        if not param_space:
            param_space = self.get_hyperparameters_space(model_type)
        optimizer = HyperparameterOptimizer(model=model, param_space=param_space, n_trials=n_trials, random_state=self.random_state)
        study = optimizer.optimize(X=self.X, y=self.y)
        best_params = study.best_params
        if return_best_params:
            return best_params
        return self.build_model(model_type, best_params)

    def get_hyperparameters_space(self, model_type: str):
        """
        Get the default hyperparameter search space for the specified model type.

        Parameters:
        ----------
        model_type : str
            Type of the model ('logistic', 'lgbm', 'xgb').

        Returns:x
        -------
        dict
            A dictionary representing the hyperparameter search space.
        """
        spaces = {
            "logistic": {
            'C': {'type': 'float', 'low': 0.0001, 'high': 10.0, 'log': True},
            'max_iter': {'type': 'int', 'low': 50, 'high': 300},
            'tol': {'type': 'float', 'low': 1e-6, 'high': 1e-3},
        }
,
            "lgbm": {
                "max_depth": {"type": "int", "low": 2, "high": 50},
                "n_estimators": {"type": "int", "low": 50, "high": 1000},
                "learning_rate": {"type": "float", "low": 0.01, "high": 1.0, "log": True},
                "num_leaves": {"type": "int", "low": 2, "high": 50},
                "min_child_samples": {"type": "int", "low": 1, "high": 20},
                "subsample": {"type": "float", "low": 0.05, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.1, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 1e-9, "high": 1.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-9, "high": 1.0, "log": True}
            },
            "xgb": {
                "max_depth": {"type": "int", "low": 2, "high": 50},
                "n_estimators": {"type": "int", "low": 50, "high": 1000},
                "learning_rate": {"type": "float", "low": 0.01, "high": 1.0, "log": True},
                "subsample": {"type": "float", "low": 0.05, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.1, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 1e-9, "high": 1.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-9, "high": 1.0, "log": True}
            }
        }
        if model_type not in spaces:
            raise ValueError(f"Modelo '{model_type}' não reconhecido.")
        return spaces[model_type]
    
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import pandas as pd

    # Criando um dataset de exemplo
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(X.shape[1])])
    df['target'] = y

    # Criando o builder
    builder = ModelBuilder(df.iloc[:, :-1], df['target'])

    # Otimizando e treinando o modelo Logistic Regression
    logistic_model = builder.build_optimized_model('logistic', n_trials=20)
    print("Coeficientes do modelo:", logistic_model.coef_)