#!/bin/bash

# Nome do projeto
PROJECT_NAME="mlflow_project"

echo "Criando o diretório do projeto: $PROJECT_NAME"
mkdir $PROJECT_NAME
cd $PROJECT_NAME

# Criar subdiretórios
echo "Criando subdiretórios..."
mkdir data scripts models

# Criar arquivos base
echo "Criando arquivos necessários..."
touch scripts/train.py
touch requirements.in

# Criar arquivo requirements-dev.txt
echo "Criando requirements-dev.txt com dependências de desenvolvimento..."
cat <<EOL > requirements-dev.txt
ipython         # terminal
ipdb            # debugger
sdb             # debugger remoto
pip-tools       # lock de dependências
pytest          # execução de testes
pytest-order    # ordenação de testes
httpx           # requests async para testes
black           # auto formatação
flake8          # linter
EOL

# Adicionar dependências iniciais no requirements.in
echo "Adicionando dependências principais no requirements.in..."
cat <<EOL > requirements.in
mlflow
scikit-learn
pandas
EOL

# Configurar ambiente virtual
echo "Configurando ambiente virtual..."
python3 -m venv venv
source venv/bin/activate

# Atualizar pip e instalar dependências de desenvolvimento
pip install --upgrade pip
pip install -r requirements-dev.txt

# Usar pip-compile para gerar requirements.txt a partir de requirements.in
echo "Gerando requirements.txt a partir de requirements.in..."
pip-compile requirements.in

# Instalar as dependências do requirements.txt
echo "Instalando dependências do requirements.txt..."
pip install -r requirements.txt

echo "Configuração completa! Para começar, ative o ambiente virtual com: source venv/bin/activate"