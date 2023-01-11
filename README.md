This repo use code from https://github.com/sviperm/neuro-comma
# Punctuation and orthography checker

## Prerequirements
 - Python 3.8 or higher *for training*
 - Docker-compose *for production*

## Installation
 
 - Option:
    ```shell
    pip install -U pip wheel setuptools
    ```
    ```shell
    pip install -r requirements.txt
    ```

## Production usage
 - Run `docker-compose`
    ```shell
    docker-compose up -d
    ```
   - As result - flask app on port 5000.
   - / - Hello page - get mapping
   - /req - post mapping, send here json: 
   - {\"text\":\"Текст на русском, где потребуются исправления.\"}
  - Stop container
    ```shell
    docker-compose down
    ```
## Metrics
 - Задача - исправить ошибки, пунктуационные и орфографические. При этом необходимо, чтобы число ошибок после применения модели уменьшилось. Тк применяя модель мы можем сгенерировать ошибки, необходимо, чтобы число новых ошибок не привышало число исправленных. Таким образом f1 мера позволяет определять число исправленных ошибок и новых ошибок. В зависимости от критичности их пропорций можно использовать f-меру с коэффициентами на precision и recall