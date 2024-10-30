import asyncio

from fastapi import FastAPI, HTTPException
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

from document_upload import chroma_path, get_embedding_function

app = FastAPI()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


@app.post("/answer")
async def answer_query(query_text: str):
    try:
        print("** Getting Embedding Function **")
        embedding_function = get_embedding_function()
        db = Chroma(
            persist_directory=chroma_path, embedding_function=embedding_function
        )

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=10)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        print("** Calling the Model **")

        # Initialize the Ollama model
        ollama_model = OllamaLLM(model="gemma2:2b")

        # Use the model to generate text
        response = await asyncio.to_thread(ollama_model.invoke, prompt)

        sources = [
            {
                "source": doc.metadata.get("id", None),
                "score": _score,
                "chunk": doc.page_content,
            }
            for doc, _score in results
        ]
        formatted_response = {"response": response, "sources": sources}
        print("** Output Generated **")
        return formatted_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
