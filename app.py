import os

import pinecone  # noqa
import streamlit as st

from dotenv import load_dotenv

# from langchain.agents import AgentType, create_pandas_dataframe_agent
# from langchain.agents import AgentType
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone

# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    StreamlitChatMessageHistory,
)
from callback import PrintRetrievalHandler, StreamHandler
# from config import configure_retriever, load_data
# from utils import file_formats_dataframe, files_formats_documents, clear_submit

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

# type_process = st.sidebar.radio("Tipo de proceso", ["Constitución", "Documentos", "Dataframe"], index=0) # en desarrollo
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
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    retriever = docsearch.as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.50}
    )

    msgs = StreamlitChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history", chat_memory=msgs, return_messages=True
    )

    llm = ChatOpenAI(
        model_name=selected_model,
        openai_api_key=openai_api_key,
        temperature=temperature,
        streaming=True,
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
    )

    if len(msgs.messages) == 0 or st.sidebar.button("Limpiar historial"):
        msgs.clear()
        msgs.add_ai_message("En qué puedo ayudarte?")

    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Escribe tu pregunta aquí"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])

# if type_process == "Documentos":
#     uploaded_files = st.sidebar.file_uploader(
#         label="Sube tus archivos", type=files_formats_documents, accept_multiple_files=True
#     )
#
#     if not uploaded_files:
#         st.info(
#             "Por favor sube un documento para continuar. Al cargar los archivos iniciará el proceso"
#         )
#         st.stop()
#
#     retriever = configure_retriever(files=uploaded_files, api_key=openai_api_key)
#
#     msgs = StreamlitChatMessageHistory()
#
#     memory = ConversationBufferMemory(
#         memory_key="chat_history", chat_memory=msgs, return_messages=True
#     )
#
#     llm = ChatOpenAI(
#         model_name=selected_model,
#         openai_api_key=openai_api_key,
#         temperature=temperature,
#         streaming=True,
#     )
#
#     # por aca pasamos el template
#     qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
#
#     if len(msgs.messages) == 0 or st.sidebar.button("Limpiar historial"):
#         msgs.clear()
#         msgs.add_ai_message("En qué puedo ayudarte?")
#
#     avatars = {"human": "user", "ai": "assistant"}
#
#     for msg in msgs.messages:
#         st.chat_message(avatars[msg.type]).write(msg.content)
#
#     if user_query := st.chat_input(placeholder="Escribe tu pregunta aquí"):
#         st.chat_message("user").write(user_query)
#
#         with st.chat_message("assistant"):
#             retrieval_handler = PrintRetrievalHandler(st.container())
#             stream_handler = StreamHandler(st.empty())
#             response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
#
# if type_process == "Dataframe":
#     df = None
#
#     uploaded_file = st.sidebar.file_uploader(
#         "Sube tus archivos",
#         type=list(file_formats_dataframe.keys()),
#         on_change=clear_submit,
#     )
#
#     if not uploaded_file:
#         st.info(
#             "Por favor sube un documento para continuar. Al cargar los archivos iniciará el proceso"
#         )
#         st.stop()
#
#     if uploaded_file:
#         df = load_data(uploaded_file)
#
#     if "messages" not in st.session_state or st.sidebar.button("Limpiar historial"):
#         st.session_state["messages"] = [
#             {"role": "assistant", "content": "En qué puedo ayudarte?"}
#         ]
#
#     for msg in st.session_state.messages:
#         st.chat_message(msg["role"]).write(msg["content"])
#
#     if prompt := st.chat_input(placeholder="Que quieres saber de los datos?"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         st.chat_message("user").write(prompt)
#
#         llm = ChatOpenAI(
#             temperature=0,
#             model="gpt-3.5-turbo-0613",
#             openai_api_key=openai_api_key,
#             streaming=True,
#         )
#
#         pandas_df_agent = create_pandas_dataframe_agent(
#             llm=llm,
#             df=df,
#             agent_type=AgentType.OPENAI_FUNCTIONS,
#         )
#
#         with st.chat_message("assistant"):
#             st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
#             response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
#             st.session_state.messages.append({"role": "assistant", "content": response})
#             st.write(response)
