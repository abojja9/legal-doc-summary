# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os

import base64
import gc
import random
import tempfile
import time
import uuid
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.core import DocumentSummaryIndex, get_response_synthesizer
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
import streamlit as st
from main import get_embeddings, get_llm, parse_documents


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    st.header(f"Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):

                    if os.path.exists(temp_dir):
                        docs = parse_documents(temp_dir)
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    # setup llm & embedding model
                    llm=get_llm()
                    embed_model = get_embeddings()
                    # Creating an index over loaded data
                    Settings.embed_model = embed_model
                    # index = VectorStoreIndex.from_documents(docs, show_progress=True)

                    # Create the query engine, where we use a cohere reranker on the fetched nodes
                    Settings.llm = llm
                    # query_engine = index.as_query_engine(streaming=True)

                    # create a summary index
                    response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", use_async=True)
                    doc_summary_index = DocumentSummaryIndex.from_documents(docs, response_synthesizer=response_synthesizer)
                    query_engine = doc_summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
                    
                    
                    # ====== Customise prompt template ======
                    # qa_prompt_tmpl_str = (
                    # "Context information is below.\n"
                    # "---------------------\n"
                    # "{context_str}\n"
                    # "---------------------\n"
                    # "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                    # "Query: {query_str}\n"
                    # "Answer: "
                    # )
                    # qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    # query_engine.update_prompts(
                    #     {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    # )
                    
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                # Inform the user that the file is processed and Display the PDF uploaded
                # st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Summary and Key Points of the Document", divider="rainbow")
    if uploaded_file:
        file_key = f"{session_id}-{uploaded_file.name}"
        query_engine = st.session_state.file_cache[file_key]
        summary_query = "Provide a comprehensive summary of the document, giving a complete overview of its content. The summary should cover all the main topics, key ideas, and important information presented."
        summary_response = query_engine.query(summary_query)
        st.subheader("Summary of the document:", divider="rainbow")
        st.write(summary_response.response)
        
        highlights_query = "Extract all the important highlights, key points, and crucial information from the document. Ensure that no essential details are missed. Include all the relevant facts, figures, and takeaways."
        highlights_response = query_engine.query(highlights_query)
        st.subheader("Extracted highlights:", divider="rainbow")
        st.write(highlights_response.response)

with col2:
    st.button("Clear ↺", on_click=reset_chat)
    
    
# # Initialize chat history
# if "messages" not in st.session_state:
#     reset_chat()


# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


# Accept user input
# if prompt := st.chat_input("What's up?"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
        
#         # Simulate stream of response with milliseconds delay
#         streaming_response = query_engine.query(prompt)
        
#         for chunk in streaming_response.response_gen:
#             full_response += chunk
#             message_placeholder.markdown(full_response + "▌")

#         # full_response = query_engine.query(prompt)

#         message_placeholder.markdown(full_response)
#         # st.session_state.context = ctx

#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": full_response})