import os
from operator import itemgetter
from typing import List, Tuple

import pinecone  # noqa
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import AIMessage, HumanMessage, StrOutputParser, format_document
from langchain.schema.runnable import (
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain.vectorstores.pinecone import Pinecone
from pydantic import BaseModel, Field

from callback import StreamHandler

load_dotenv()

st.set_page_config(page_title="Chat constitucional", page_icon=":books:")
title = st.title("Chatea con tus documentos")

if "submit" not in st.session_state:
    st.session_state["submit"] = False

# -------------------------------------------------

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    placeholder="sk-...",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
)

selected_model = st.sidebar.selectbox(
    "Escoge el modelo",
    [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
    ],
    key="selected_model",
)

temperature = st.sidebar.slider(
    "Temperatura", min_value=0.1, max_value=2.0, value=0.1, step=0.01
)


type_process = st.sidebar.radio("Tipo de proceso", ["Constitución"], index=0)

# -------------------------------------------------

if not openai_api_key:
    st.info(
        "Por favor, introduce tu API Key de OpenAI. Puedes crea una [aquí]("
        "https://beta.openai.com/account/api-keys)."
    )
    st.stop()

# -------------------------------------------------

if type_process == "Constitución":
    avatars = {"human": "user", "ai": "assistant"}

    title.title("Chat constitucional")

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),
    )

    index_name = "constitution-idx"

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)

    # retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # retriever = vectorstore.as_retriever(
    #     search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50}
    # )

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.2}
    )

    # retriever = vectorstore.as_retriever(
    #     search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8}
    # )

    msgs = StreamlitChatMessageHistory()

    _template = """
    Dada la siguiente conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente, en su idioma original.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    template = """
    Dado el contexto proporcionado, debes responder asumiendo el rol de un experto en derecho constitucional con un inmenso conocimiento y experiencia en el campo.

    En el contexto encontrarás tres documentos los cuales empiezan con INICIO DEL DOCUMENTO seguido del TITULO y finalizan con FIN DEL DOCUMENTO.
    Debes analizar el contexto y separar bien los contenidos de los documentos y responder las preguntas basado en tu conocimiento y en la conversación previa. no inventes respuestas.
    
    Los documentos consisten en:
    El primer documento es la Constitución actual de la República de chile
    El segundo documento es la propuesta para el cambio de la constitución del 2022 y tiene como titulo propuesta de nueva constitución 2022
    El tercer documento es la propuesta para el cambio de la constitución del 2023 y tiene como titulo propuesta de nueva constitución 2023

    <context>
    {context}
    </context>"""

    ANSWER_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ]
    )

    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    # ----------------

    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    def update_chat_history(messages, user_query, ai_response):
        messages.add_user_message(user_query)
        messages.add_ai_message(ai_response)

    class ChatHistory(BaseModel):
        chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
        question: str

    # ----------------

    llm = ChatOpenAI(
        model_name=selected_model,
        openai_api_key=openai_api_key,
        temperature=temperature,
        streaming=True,
    )

    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser(),
        ),
        RunnableLambda(itemgetter("question")),
    )

    _inputs = RunnableMap(
        {
            "question": lambda x: x["question"],
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),
            "context": _search_query | retriever | _combine_documents,
        }
    ).with_types(input_type=ChatHistory)

    qa_chain = _inputs | ANSWER_PROMPT | llm | StrOutputParser()

    if len(msgs.messages) == 0 or st.sidebar.button("Limpiar historial"):
        msgs.clear()
        msgs.add_ai_message("En qué puedo ayudarte?")

    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Escribe tu pregunta aquí"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())

            chat_history = [(msg.type, msg.content) for msg in msgs.messages]
            response = ""

            for chunk in qa_chain.stream(
                input={
                    "question": user_query,
                    "chat_history": chat_history,
                }
            ):
                stream_handler.on_llm_new_token(chunk)
                response += chunk

            update_chat_history(msgs, user_query, response)


# ¿Cuántas propuestas de nueva constitución tienes?
# Busca y analiza solo en el texto de la propuesta de nueva constitución del 2023 y entrégame los puntos más importantes en materia de educación
# según tu análisis me puedes indicar entre la propuesta del 2022 y 2023 cúal aborda más temas en materia de seguridad
