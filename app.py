import os
import tempfile

import streamlit as st

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Chatea con tus documentos", page_icon=":books:")
st.title("Chatea con tus documentos")


@st.cache_resource(ttl="1h")
def configure_retriever(files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("Splits: ", splits)
    print("Creando embeddings...")

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(splits, embeddings)

    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


def main():
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Por favor, introduce tu API Key de OpenAI.")
        st.stop()

    uploaded_files = st.sidebar.file_uploader(
        label="Sube archivos PDF", type=["pdf"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Por favor sube un documento para continuar.")
        st.stop()

    retriever = configure_retriever(files=uploaded_files)

    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.5, streaming=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    if len(msgs.messages) == 0 or st.sidebar.button("Limpiar historial"):
        msgs.clear()
        msgs.add_ai_message("En qué puedo ayudarte?")

    avatars = {"human": "user", "ai": "assistant"}

    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Escribe tu pregunta aquí"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])


if __name__ == "__main__":
    main()
