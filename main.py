from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import pickle


def parse_documents(temp_dir):
    if os.getenv("LLAMA_CLOUD_API_KEY") is None:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY environment variable is not set. "
            "Please set it in .env file or in your shell environment then run again!"
        )
    parser = LlamaParse(result_type="markdown", verbose=True, language="en")
    reader = SimpleDirectoryReader(
        temp_dir,
        required_exts=[".pdf"],
        recursive=True,
        file_metadata=get_meta,
        file_extractor={".pdf": parser},
    )
    documents = reader.load_data()

    return documents


def get_meta(file_path):
    fname, ext = file_path.split("/")[-1].split(".")
    metadata = {
        "title": fname,
        "file_path": file_path,
        "file_name": fname,
        "file_type": ext,
        "file_size": os.path.getsize(file_path),
        "creation_date": os.path.getctime(file_path),
        "last_modified_date": os.path.getmtime(file_path),
        "last_accessed_date": os.path.getatime(file_path),
    }
    return metadata


def get_llm():
    return Anthropic(model="claude-3-opus-20240229") # "claude-3-sonnet-20240229"


def get_embeddings():
    return HuggingFaceEmbedding( model_name="BAAI/bge-base-en-v1.5", trust_remote_code=True)
