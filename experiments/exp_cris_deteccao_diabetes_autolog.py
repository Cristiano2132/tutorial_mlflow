import os
import sys
from pathlib import Path
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn

# Configurar paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(BASE_DIR / "data"))

from data.load_data import load_data
from features.feature_engineering import TreeCategizer, WOEEncoder, custom_cut
from evaluation.metrics import get_report_metrics
from utils import get_summary
from models.model_builder import ModelBuilder

def set_experiment(exp_name: str):
    """
    Set or create a new MLflow experiment.

    Args:
        exp_name (str): Name of the experiment.
    """
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        mlflow.create_experiment(exp_name)
        print(f"Experimento '{exp_name}' criado.")
    else:
        print(f"Usando experimento existente: '{exp_name}'")

    mlflow.set_experiment(exp_name)


def autolog_loader(model_name: str):
    if model_name == "logistic":
        return mlflow.sklearn.autolog
    elif model_name == "lgbm":
        return mlflow.lightgbm.autolog
    elif model_name == "xgb":
        return mlflow.xgboost.autolog
    else:
        raise ValueError(f"Modelo '{model_name}' não reconhecido.")


if __name__ == "__main__":
    # Configurar URI do MLflow
    mlflow.set_tracking_uri("http://0.0.0.0:5002/")
    
    exp_name = "bruna_diabetes_modeling"
    set_experiment(exp_name)
    
 
    data_path = BASE_DIR / "data" / "raw" / "diabetes.csv"
    df = load_data(data_path)
        # Pré-processamento
    features = df.columns[:-1]
    label = df.columns[-1]
    train_index = df.sample(frac=0.8, random_state=42).index
    test_index = df.drop(train_index).index
    print("Resumo inicial dos dados:")
    print(get_summary(df))
    
    # Feature engineering
    bins_dict = {}
    tc = TreeCategizer(df=df.loc[train_index].dropna())
    for feat in features:
        bins_dict[feat] = tc.get_splits(target_column=label, feature_column=feat)
    
    for feat in features:
        df[feat] = custom_cut(df, feat, bins_dict[feat])
    
    print("Resumo após TreeCategizer e custom_cut:")
    print(get_summary(df))
    
    encoder = WOEEncoder(df.loc[train_index], label)
    for feat in features:
        encoder.fit(feat)
        df[feat] = df[feat].map(encoder.woe_dict.get(feat))
    print("Resumo após WOE encoding:")
    print(get_summary(df))
    
    # Finalizar qualquer run ativa
    if mlflow.active_run():
        mlflow.end_run()

    models = ["logistic", "lgbm", "xgb"]
    for model_name in models:
        
        model_builder = ModelBuilder(
                X=df.loc[train_index][features], 
                y=df.loc[train_index][label], 
                random_state=42
            )
        best_params = model_builder.build_optimized_model(model_type=model_name, return_best_params=True)
        
        # Inicializando o autolog antes de cada execução
        autolog = autolog_loader(model_name)
        autolog()

        # Iniciar o run após o autolog estar configurado
        with mlflow.start_run(run_name=f"run_{model_name}_v1"):
            # Logando os artefatos de feature engineering
            mlflow.log_dict(bins_dict, "feature_engineering/bins.json")
            mlflow.log_dict(encoder.woe_dict, "feature_engineering/woe.json")
            
            # Treinamento do modelo com otimização de hiperparâmetros
            model_path = BASE_DIR / f"{model_name}_model.pkl"
            model_builder = ModelBuilder(
                X=df.loc[train_index][features], 
                y=df.loc[train_index][label], 
                random_state=42
            )
            model = model_builder.build_model(model_type=model_name, params=best_params)
            print("Modelo treinado com sucesso!")
            
            # Salvando o modelo como artefato
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(model_path), artifact_path="modelssss")
            os.remove(model_path)
            
            # Avaliação do modelo no conjunto de treino
            y_pred = model.predict(df[features])
            df_result = pd.DataFrame({'y': df[label], 'y_pred': y_pred})
            report_train = get_report_metrics(df=df_result, proba_col='y_pred', true_value_col='y', base="train")

            # Avaliação do modelo no conjunto de teste
            y_pred = model.predict(df.loc[test_index][features])
            df_result = pd.DataFrame({'y': df.loc[test_index][label], 'y_pred': y_pred})
            report_test = get_report_metrics(df=df_result, proba_col='y_pred', true_value_col='y', base="test")

            # Logando as métricas
            report = {}
            report.update(report_train)
            report.update(report_test)
            mlflow.log_metrics(report)

            mlflow.set_tag("model_type", model_name)
        
        print("Run finalizado com sucesso!")

