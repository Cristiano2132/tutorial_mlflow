import sys
import pickle
from pathlib import Path
import mlflow
import pandas as pd
import json
from mlflow.client import MlflowClient

# Configurar paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(BASE_DIR / "data"))

from data.load_data import load_data
from features.feature_engineering import custom_cut
from evaluation.metrics import get_ks
from utils import get_summary

def get_latest_model_run_id(model_name, stage="Production"):
    """Retorna o run_id da versão mais recente do modelo no estágio especificado."""
    client = MlflowClient()
    
    # Obter a versão mais recente do modelo no estágio especificado
    versions = client.get_latest_versions(model_name, stages=[stage])
    
    if not versions:
        print(f"Nenhuma versão encontrada para o modelo '{model_name}' no estágio '{stage}'.")
        return None
    
    # Retorna o run_id da versão mais recente
    return versions[0].run_id

def load_artifact_from_mlflow(run_id, artifact_path):
    """Carrega um artefato salvo no MLflow dado o run_id e o artifact_path."""
    try:
        client = MlflowClient()
        # Download artifact
        artifact_uri = client.download_artifacts(run_id, artifact_path, dst_path='/tmp')
        with open(artifact_uri, 'rb') as f:
            artifact = json.load(f)
        return artifact
    
    except Exception as e:
        print(f"Erro ao baixar o artefato '{artifact_path}': {e}")
        sys.exit(1)


def get_model_type_by_tag(run_id: str):
    client = MlflowClient()
    run_info = client.get_run(run_id)
    model_type = run_info.data.tags.get("model_type")
    if not model_type:
        raise ValueError(f"Não foi possível determinar o tipo do modelo para o run_id: {run_id}")
    return model_type

def load_model(model_name, model_type, stage:str="Production"):
    
    model_uri = f"models:/{model_name}/{stage}"
    if model_type == "logistic":
        return mlflow.sklearn.load_model(model_uri=model_uri)
    elif model_type == "lgbm":
        return mlflow.lightgbm.load_model(model_uri=model_uri)
    elif model_type == "xgb":
        return mlflow.xgboost.load_model(model_uri=model_uri)
    else:
        raise ValueError(f"Modelo '{model_name}' não reconhecido.")

if __name__ == "__main__":
    mlflow_model_name = 'modelo_02'
    # Configuração inicial
    data_path = BASE_DIR / "data" / "raw" / "diabetes.csv"
    df = load_data(data_path)
    
    features = df.columns[:-1]
    label = df.columns[-1]
    train_index = df.sample(frac=0.8, random_state=42).index
    test_index = df.drop(train_index).index
    
    # Configurar o MLflow
    mlflow.set_tracking_uri("http://0.0.0.0:5002/")
    
    run_id = get_latest_model_run_id(model_name=mlflow_model_name, stage="Production")
    model_type = get_model_type_by_tag(run_id=run_id)
    
    print("Carregando modelo...")
    # para carregar diretamente e fazer predict proba como a seguir utilize o log_personalizado ao invés do autolog
    model = load_model(model_name=mlflow_model_name, model_type=model_type, stage="Production")
    
    # Carregar artefatos feature_engineering/bins.json e feature_engineering/woe.json
    print("Carregando artefatos do MLflow...")

    bins_dict = load_artifact_from_mlflow(run_id=run_id, artifact_path="feature_engineering/bins.json")
    woe_dict = load_artifact_from_mlflow(run_id=run_id, artifact_path="feature_engineering/woe.json")
    print("Artefatos carregados com sucesso!")


    print("Modelo carregado com sucesso!")
    
    # Transformar os dados
    for feat in features:
        df[feat] = custom_cut(df, feat, bins_dict[feat])
        df[feat] = df[feat].map(woe_dict[feat])
    
    print("Dados transformados com sucesso!")
    print(get_summary(df))
    
    # Previsão com o modelo carregado
    y_pred = model.predict_proba(df[features])[:, 1]
    print(y_pred)
    
    # Avaliação do modelo
    df_result = pd.DataFrame({'y': df[label], 'y_pred': y_pred})
    ks_train = get_ks(df=df_result.loc[train_index], proba_col='y_pred', true_value_col='y')
    ks_test = get_ks(df=df_result.loc[test_index], proba_col='y_pred', true_value_col='y')
    
    print(f"KS no conjunto de treino: {ks_train}")
    print(f"KS no conjunto de teste: {ks_test}")
    print(model)
