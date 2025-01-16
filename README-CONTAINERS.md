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

## Configuração

1. **Clone este repositório** na sua máquina local:

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_DIRETORIO>
```

2. **Crie e configure o arquivo `.env`**:

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```plaintext
MLFLOW_LOCAL_VOLUME=/caminho/para/sua/pasta/mlflow
MLFLOW_PORT=5002
```

Substitua `/caminho/para/sua/pasta/mlflow` pelo caminho absoluto no seu sistema onde você deseja salvar os dados do MLflow.

## Como Usar

### Construir e rodar o container

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

