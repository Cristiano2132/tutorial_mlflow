# Baseado na imagem oficial do Python
FROM python:3.10-slim

# Instalar dependências necessárias para o MLflow
RUN pip install mlflow

# Expor a porta padrão usada pelo MLflow
EXPOSE 5000

# Definir diretório de trabalho para MLflow 
WORKDIR /mlflow

# Comando para iniciar o servidor MLflow automaticamente
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]