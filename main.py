import os
import base64
import tempfile
import uuid
import xml.etree.ElementTree as ET

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Document
import streamlit as st

# Initialize session state if not already set
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

def parse_xml(file_path):
    """Parse the XML file and extract relevant data."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    for script in root.findall('.//script'):
        script_data = {
            "Name": script.find('name').text if script.find('name') is not None else "N/A",
            "Description": script.find('description').text if script.find('description') is not None else "N/A",
            "Type of Script": script.find('type').text if script.find('type') is not None else "N/A",
            "Sys_id": script.find('sys_id').text if script.find('sys_id') is not None else "N/A",
            "Other Details": script.find('other_details').text if script.find('other_details') is not None else "N/A",
        }
        data.append(script_data)
    return data

def generate_document(data):
    """Generate a technical document from the parsed data."""
    documents = [Document(text=str(item)) for item in data]

    # Set up LLM and embedding model
    llm = Ollama(model="llama3.1", request_timeout=120.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)

    # Create an index of data
    Settings.embed_model = embed_model
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    # Create the query engine
    Settings.llm = llm
    query_engine = index.as_query_engine()

    # Define the prompt template
    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "-----------------------------\n"
        "{context_str}\n"
        "-----------------------------\n"
        "Given the context information above, create a technical document with the following structure:\n"
        "- Section: [Name]\n"
        "- Description: [Description]\n"
        "- Scripts:\n"
        "{scripts}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )

    response = query_engine.query("Generate a technical document based on the provided data.")
    return response

def display_document(file):
    """Display the uploaded XML file in the Streamlit app."""
    st.markdown("### XML File Preview")
    base64_xml = base64.b64encode(file.read()).decode("utf-8")

    # Displaying File
    xml_display = f"""
    <iframe src="data:text/xml;base64,{base64_xml}" width="400" height="100%" type="text/xml"
            style="height:100vh; width:100%">
    </iframe>"""

    st.markdown(xml_display, unsafe_allow_html=True)

# Streamlit sidebar for file uploads
with st.sidebar:
    st.header("Upload your XML files")
    uploaded_files = st.file_uploader("Choose XML files", type="xml", accept_multiple_files=True)

    if uploaded_files:
        try:
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                    st.write(f"Processing file: {uploaded_file.name}")

                    # Parse XML and generate document
                    data = parse_xml(temp_file_path)
                    
                    if data:
                        response = generate_document(data)
                        st.write("Generated Technical Document:")
                        st.markdown(response)
                    else:
                        st.error("No valid data found in XML.")

                    display_document(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.header("Technical Document Generator")
