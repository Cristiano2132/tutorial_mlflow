
# Projeto MLflow - Detecção de Diabetes

Este projeto configura um ambiente Docker para executar o MLflow, focado na criação de um modelo de Machine Learning para prever fraude.

## Visão Geral
O objetivo deste projeto é construir um modelo preditivo de machine learning para determinar se um paciente apresenta sinais de diabetes com base em medições diagnósticas. O conjunto de dados utilizado neste projeto é originário do Instituto Nacional de Diabetes e Doenças Digestivas e Renais (National Institute of Diabetes and Digestive and Kidney Diseases). Todos os pacientes do conjunto de dados são mulheres com pelo menos 21 anos de idade e de herança Pima. Iremos registrar todos os experimentos e modelos no MLflow. Este projeto abrange:

1. Coleta e preparação de dados.
2. Treinamento de diferentes modelos de Machine Learning.
3. Avaliação e comparação de modelos.
4. Registro e deploy de modelos usando MLflow.

## Setup

### Pré-requisitos

- Docker e Docker Compose instalados
- Python e `virtualenv` instalados

### Passos para configuração

1. **Clone o repositório:**
   ```sh
   git clone <URL-do-repositório>
   cd <nome-do-diretório>
   ```

2. **Criar um ambiente virtual Python:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # Para Linux/Mac
   venv\Scripts\activate  # Para Windows
   ```

3. **Instalar dependências:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Subir os containers Docker:**
   Para detalhes sobre configurar e inicializar nosso servidor `MLflow` veja o arquivo [README-CONTAINERS.md](README-CONTAINERS.md)

## Estrutura do Projeto

Ao trabalhar com MLflow, é importante organizar o projeto de forma clara e escalável, permitindo o rastreamento fácil de experimentos, reprodutibilidade e colaboração. Abaixo está uma estrutura de pastas geralmente recomendada:

```bash
.
├── Dockerfile                    # Arquivo Docker para criar ambientes reprodutíveis
├── README-CONTAINERS.md          # Documentação sobre a configuração e uso dos containers Docker
├── README.md                     # Documentação do projeto
├── data/                         # Diretório para armazenar dados
│   └── raw                       # Dados brutos
│       ├── README.md
│       └── diabetes.csv
├── docker-compose.yml            # Arquivo Docker Compose para facilitar a execução do ambiente
├── experiments/                  # Scripts e arquivos relacionados aos experimentos
│   └── exp_model_logistica_autolog.py
├── main.py                       # Script principal para executar um modelo registrado no mlflow
├── requirements-dev.txt          # Dependências de desenvolvimento
├── requirements.in               # Dependências base do projeto
├── requirements.txt              # Dependências do projeto
└── src/                          # Código fonte principal do projeto
    ├── data/                     # Scripts para carregamento dados
    │   ├── download_data.py
    │   └── load_data.py
    ├── evaluation/               # Scripts para avaliação e métricas
    │   ├── calcule_ks.py
    │   └── metrics.py
    ├── features/                 # Scripts para engenharia de features
    │   └── feature_engineering.py
    ├── models/                   # Funcoes para serem utilizadas no treinamento de modelos
    │   ├── bayesian_opt.py
    │   ├── bayesian_opt_hyperopt.py
    │   ├── classification_lgbm.py
    │   ├── classification_xgb.py
    │   └── reg_logistica.py
    ├── training/                 # Scripts para treinamento do modelo
    └── utils.py                  # Scripts utilitários
```

### Descrição das Pastas

1. **Dockerfile**: Arquivo Docker para criar ambientes reprodutíveis.
2. **README-CONTAINERS.md**: Documentação sobre a configuração e uso dos containers Docker.
3. **README.md**: Documentação do projeto.
4. **data/**: Diretório para armazenar dados.
   - **raw/**: Dados brutos.
5. **docker-compose.yml**: Arquivo Docker Compose para facilitar a execução do ambiente.
6. **experiments/**: Scripts e arquivos relacionados aos experimentos.
7. **main.py**: Script principal para executar o projeto.
8. **requirements-dev.txt**: Dependências de desenvolvimento.
9. **requirements.in**: Dependências base do projeto.
10. **requirements.txt**: Dependências do projeto.
11. **src/**: Código fonte principal do projeto.
    - **data/**: Scripts de manipulação de dados.
    - **evaluation/**: Scripts para avaliação e métricas.
    - **features/**: Scripts para engenharia de features.
    - **models/**: Modelos criados ou treinados.
    - **training/**: Scripts para treinamento do modelo.
    - **utils.py**: Scripts utilitários.


### Comandos para Unix (Mac/Linux)
```bash
# Cria os subdiretórios a partir da pasta atual
mkdir -p data/{raw,processed} notebooks src/{data,models,features,training,evaluation} config tests scripts

# Cria os arquivos principais
touch Dockerfile requirements.txt environment.yml README.md .gitignore
touch config/{params.yaml,settings.yaml}
```

### Comandos para Windows (cmd)
```bash
REM Cria os subdiretórios a partir da pasta atual
mkdir data\raw data\processed
mkdir notebooks
mkdir src\data src\models src\features src\training src\evaluation
mkdir config
mkdir tests
mkdir scripts

REM Cria os arquivos principais
type nul > Dockerfile
type nul > requirements.txt
type nul > environment.yml
type nul > README.md
type nul > .gitignore
type nul > config\params.yaml
type nul > config\settings.yaml
```

### Notas

1. **Posicionamento no Terminal**:
   - Certifique-se de que o terminal está aberto na raiz do projeto (onde será criada a estrutura).

2. **Flexibilidade**:
   - Altere ou remova diretórios conforme necessário se a estrutura completa não for necessária.


# MLflow Server Docker Setup

Este repositório contém os arquivos necessários para configurar e rodar um servidor MLflow usando Docker e Docker Compose.

## Estrutura do Projeto

```
.
├── Dockerfile
├── docker-compose.yml
├── .env
└── README.md
```

- **Dockerfile**: Arquivo que define a imagem Docker customizada para rodar o servidor MLflow.
- **docker-compose.yml**: Arquivo Docker Compose que define os serviços e configurações do container.
- **.env**: Arquivo de variáveis de ambiente usado para definir configurações personalizadas.
- **README.md**: Este arquivo de documentação.

## Pré-requisitos

- [Docker](https://www.docker.com/get-started) instalado na sua máquina.
- [Docker Compose](https://docs.docker.com/compose/install/) instalado na sua máquina.


## Como Usar

### Construir build e up

1. **Construa e inicie os serviços** usando Docker Compose:

```bash
docker-compose up --build
```

### Acessar o servidor MLflow

Abra o navegador e acesse `http://localhost:5002` para visualizar a interface do MLflow.

### Parar o container

Para parar os serviços, use o comando:

```bash
docker-compose down
```

## Observações

- Certifique-se de que o caminho no arquivo `.env` está correto e que você tem permissões de escrita no diretório especificado.
- O container MLflow estará persistindo os dados no diretório especificado, garantindo que os dados permaneçam mesmo após o container ser encerrado ou removido.

