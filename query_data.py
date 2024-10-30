import argparse
import asyncio

from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

from document_upload import chroma_path, get_embedding_function

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


async def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="How the play the game?")
    args = parser.parse_args()
    query_text = args.query_text
    await answer_query(query_text)


async def answer_query(query_text):

    print("** Getting Embedding Function **")
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=10)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("** Calling the Model **")
    # Initialize the Ollama model
    ollama_model = OllamaLLM(model="gemma2:2b")

    # Use the model to generate text
    response = ollama_model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)
    return response


if __name__ == "__main__":
    asyncio.run(main())
