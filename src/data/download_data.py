import os
from kaggle.api.kaggle_api_extended import KaggleApi

# os.environ["KAGGLE_CONFIG_DIR"] = "/Users/cristianooliveira/Documents/.kaggle.json"

def download_dataset(output_dir: str)->None:
    """
    Baixa o conjunto de dados Pima Indians Diabetes Database do Kaggle 
    e salva na pasta `data/raw`.
    """
    # Diretório para salvar os dados
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    # Inicializa a API do Kaggle
    api = KaggleApi()
    api.authenticate()

    # Dataset do Kaggle
    dataset = "uciml/pima-indians-diabetes-database"

    print(f"Baixando o dataset {dataset}...")
    # Baixa e extrai os arquivos
    api.dataset_download_files(dataset, path=output_dir, unzip=True)
    print(f"Conjunto de dados baixado e extraído em: {output_dir}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(BASE_DIR, '../../data/raw')
    download_dataset(output_dir=output_dir)