# NLP - PF

Link do vídeo no youtube: https://youtu.be/RijmNrMwOKs

## Setup
1. Install the necessary dependencies:
```bash
pip install -r requirements.txt
```
2. Create the Docker container with pgvector:
```bash
docker-compose up -d
```

3. Connect to the server through the `port 5432`

4. Create a db named `vector_db`, and run the command: 
```sql
CREATE EXTENSION vector
```

5. Create a .env and fill out the necessary environment variables. 

    **Note**: Do NOT change the `CONNECTION_STRING` unless you've created the Docker container with a differente user, password or if you've created a db with a different name. Format of the string: `postgresql+psycopg2://<USER>>:<PASSWORD>@localhost:5432/<DB_NAME>`
```

## Run app
To run the application: 
```bash
streamlit run app.py
```

and access the URL: http://localhost:8501
