# rerank-cross-encoder-service
The service is responsible for re-rank texts based on a query.

## Generate docker image

first clone this repo, enter into the folder and execute docker build command

```
docker built -t re-rank-service
```

## Run docker image
```
docker run -p 3000:5000 re-rank-service
```

## How use the service

```
curl --location --request POST 'http://localhost:8000/re-rank' \
--header 'Content-Type: application/json' \
--data-raw '{
    "query": "how your name?",
    "documents" :[
        {
            "id":"1",
            "text":"My name is Jack"
        },
        {
            "id":"2",
            "text":"I living in Koreia"
        },
        {
            "id":"3",
            "text":"How many people have"
        }
    ]
}'
```
