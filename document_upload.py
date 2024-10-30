import asyncio
import os

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Setting folder path to fetch documents
folder_path = "./sample_docs"

# Setting chroma path to DB
chroma_path = "chroma"

# Setting Embedding Model Name
model_name = "BAAI/bge-base-en-v1.5"  # "all-MiniLM-L6-v2"


async def main():

    # Getting Folder Contents
    folder_contents = os.listdir(folder_path)

    # Filtering only PDF's in the array
    folder_contents = [
        filename for filename in folder_contents if filename.endswith(".pdf")
    ]

    # Saving extracted PDF content to memory
    if len(folder_contents) > 0:
        for file in folder_contents:
            file_chunks = await convert_pdf_to_chunks(f"{folder_path}/{file}", file)
            await add_chunks_to_chroma(file_chunks)


async def add_chunks_to_chroma(file_chunks):

    # Initialize Local Chrome DB
    db = Chroma(
        persist_directory=chroma_path, embedding_function=get_embedding_function()
    )

    # Check Existing chunks in DB
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in file_chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def get_embedding_function():
    embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    return embeddings


async def convert_pdf_to_chunks(path, file):

    # Using PYPDF to load a PDF
    loader = PyPDFLoader(path)
    pages = loader.load()

    # Split Full text into Chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",  # You can choose your own separator
        chunk_size=800,  # Adjust the chunk size as needed
        chunk_overlap=50,  # Optional: overlap to maintain context
        length_function=len,
        is_separator_regex=False,
    )
    all_chunks = []

    # Split text for each page and attach metadata
    for page_number, doc in enumerate(pages, start=1):

        # Split the page content into chunks
        chunks = text_splitter.split_text(doc.page_content)
        for index, chunk in enumerate(chunks):
            # Create a Document objectsas
            document = Document(
                page_content=chunk,
                metadata={
                    "id": f"{file}:{page_number}:{index}",
                    "filename": file,
                    "page_number": page_number,
                    "chunk_index": index,
                },
            )
            all_chunks.append(document)

    return all_chunks


if __name__ == "__main__":
    asyncio.run(main())
