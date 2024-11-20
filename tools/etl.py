from typing import List
from dotenv import load_dotenv
import os
import psycopg2
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def check_db_conn() -> None:
    """Check the connection to the database."""
    try:
        psycopg2.connect(
            host="localhost",
            dbname="vector_db",
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
        print("DATABASE: Connection to the database established!")
    except psycopg2.OperationalError as e:
        print(f"DATABASE: Error connecting to the database: {e}")
        raise e


class ETL:
    def __init__(self, data_dir: List):
        self.data_dir = data_dir #"raw_data/"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        self.connection_string = os.getenv("CONNECTION_STRING")
        self.collection_name = os.getenv("COLLECTION_NAME")

    def extract(self) -> List[Document]:
        """Extract documents from a directory."""
        ## LOAD FROM LOCAL ###
        doc_list = []
        for root, _, files in os.walk(os.path.abspath(self.data_dir)):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    doc = TextLoader(file_path).load()[0]
                    doc_list.append(doc)
                elif file.endswith(".pdf"):
                    doc = PyPDFLoader(file_path).load()[0]
                    doc_list.append(doc)
        print(f"PIPELINE: Loaded {len(doc_list)} documents")

        ### LOAD FROM S3 ###

        return doc_list

    def transform(self, docs: List[Document]) -> List[Document]:
        """Transform documents using a text splitter."""
        splits = self.text_splitter.split_documents(docs)
        print(f"PIPELINE: The documents were chunked into {len(splits)} documents")
        return splits

    def load(self, all_splits: List[Document]) -> None:
        """Load transformed documents into a PGVector database."""
        embeddings = OpenAIEmbeddings()
        chunks = 100
        all_splits_chunks = [
            all_splits[i : i + chunks] for i in range(0, len(all_splits), chunks)
        ]

        check_db_conn()
        print(f"PIPELINE: Loading {len(all_splits)} documents into PGVector")

        for chunk in tqdm(all_splits_chunks):
            PGVector.from_documents(
                embedding=embeddings,
                documents=chunk,
                collection_name=self.collection_name,
                connection_string=self.connection_string,
                use_jsonb=True
            )
        print("PIPELINE: Finished loading documents into PGVector")

    def run(self):
        documents = self.extract()
        chuck_list = self.transform(documents)
        self.load(chuck_list)
        return (len(documents))