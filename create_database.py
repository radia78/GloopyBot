from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
import os
import shutil

DATA_PATH = "data/books"

class JohnDoeClass:
    def __init__(self, data_path: str, chroma_path: str) -> None:
        # cache the data for use
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.loader = DirectoryLoader(data_path, glob="*.md")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 300,
            chunk_overlap = 50,
            length_function = len,
            add_start_index = True
        )

    def save_to_chroma(self, chunks: list[Document]):
        if os.path.exits(self.chroma_path):
            shutil.rmtree(self.chroma_path)

        # new DB from inputted documents
        db = Chroma.from_documents(
            chunks, ..., persist_directory=self.chroma_path
        )
        db.persist()
    
    def generate_data_store(self):
        documents = self.loader.load()
        chunks = self.text_splitter.split_documents(documents)

if __name__ == "__main__":
    main()
        