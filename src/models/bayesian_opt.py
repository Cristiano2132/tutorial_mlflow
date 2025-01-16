import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from optuna import create_study
from optuna.trial import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any

def get_ks(df: pd.DataFrame, proba_col: str, true_value_col: str) -> float:
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic for the given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the true labels and predicted probabilities.
        proba_col (str): Column name for predicted probabilities.
        true_value_col (str): Column name for true labels.
    
    Returns:
        float: KS statistic.
    """
    class0 = df[df[true_value_col] == 0]
    class1 = df[df[true_value_col] == 1]
    ks = ks_2samp(class0[proba_col], class1[proba_col])
    return ks.statistic

class HyperparameterOptimizer:
    def __init__(self, model: Any, param_space: Dict[str, Any], n_folds: int = 5, n_trials: int = 10, random_state: int = 42):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            model_class (Any): The model class to be optimized.
            param_space (Dict[str, Any]): The hyperparameter space for optimization.
            n_folds (int): Number of cross-validation folds. Default is 5.
            n_trials (int): Number of optimization trials. Default is 50.
            random_state (int): Random state for reproducibility. Default is 42.
        """
        self.model = model
        self.param_space = param_space
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.random_state = random_state

    def objective(self, trial: Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            trial (Trial): A trial object to generate hyperparameters.
            X (pd.DataFrame): Training feature data.
            y (pd.Series): Training target data.
        
        Returns:
            float: Mean KS statistic across cross-validation folds.
        """
        params = {key: self._suggest_value(trial, key, value) for key, value in self.param_space.items()}

        valid_ks = []
        strat = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for train_index, valid_index in strat.split(X, y):
            x_train, x_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            clf = self.model.set_params(**params)
            clf.fit(x_train, y_train)

            preds_valid = clf.predict_proba(x_valid)[:, 1]
            df_classifier = pd.DataFrame({'y_true': y_valid, 'score': preds_valid})
            ks_ = get_ks(df=df_classifier, proba_col='score', true_value_col='y_true')
            valid_ks.append(ks_)

        return np.mean(valid_ks)

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> create_study:
        """
        Run the hyperparameter optimization process.
        
        Args:
            X (pd.DataFrame): Training feature data.
            y (pd.Series): Training target data.
        
        Returns:
            create_study: The resulting study object containing the best parameters and performance.
        """
        sampler = TPESampler(seed=self.random_state)
        study = create_study(direction='maximize', sampler=sampler)
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=self.n_trials)
        print("Melhores hiperparâmetros:", study.best_params)
        print("Melhor KS médio:", study.best_value)
        return study

    def _suggest_value(self, trial: Trial, param_name: str, param_info: Any) -> Any:
        """
        Suggest a value for a given hyperparameter.
        
        Args:
            trial (Trial): A trial object to generate hyperparameters.
            param_name (str): Name of the hyperparameter.
            param_info (Any): Information about the hyperparameter space.
        
        Returns:
            Any: Suggested value for the hyperparameter.
        """
        if param_info['type'] == 'int':
            return trial.suggest_int(param_name, param_info['low'], param_info['high'])
        elif param_info['type'] == 'float':
            return trial.suggest_float(param_name, param_info['low'], param_info['high'], log=param_info.get('log', False))
        else:
            raise ValueError(f"Unsupported parameter type for {param_name}")

