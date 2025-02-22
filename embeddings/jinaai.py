"""Jina AI Embeddings wrapper for Langchain."""

import base64
from os.path import exists
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"
VALID_ENCODING = ["float", "ubinary", "binary"]
DEFAULT_MODEL = "jina-embeddings-v3"


def is_local(url: str) -> bool:
    """Check if a URL is a local file.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is a local file, False otherwise.
    """
    url_parsed = urlparse(url)
    if url_parsed.scheme in ("file", ""):  # Possibly a local file
        return exists(url_parsed.path)
    return False


def get_bytes_str(file_path: str) -> str:
    """Get the bytes string of a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The bytes string of the file.
    """
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class JinaEmbeddings(BaseModel, Embeddings):
    """Jina AI embedding models.

    Attributes:
        model_name (str): The model to use for embeddings
        jina_api_key (Optional[SecretStr]): The API key for Jina AI
        encoding_type (str): The encoding type for embeddings
        dimensions (Optional[int]): The output dimensions for embeddings
        late_chunking (Optional[bool]): Whether to use late chunking
        task (Optional[str]): The specific task for embeddings
    """

    session: Any  #: :meta private:
    model_name: str = DEFAULT_MODEL
    jina_api_key: Optional[SecretStr] = None
    encoding_type: str = "float"
    dimensions: Optional[int] = None
    late_chunking: Optional[bool] = False
    task: Optional[str] = None
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that auth token exists in environment."""
        try:
            jina_api_key = convert_to_secret_str(
                get_from_dict_or_env(values, "jina_api_key", "JINA_API_KEY")
            )
        except ValueError as original_exc:
            try:
                jina_api_key = convert_to_secret_str(
                    get_from_dict_or_env(values, "jina_auth_token", "JINA_AUTH_TOKEN")
                )
            except ValueError:
                raise original_exc

        values["jina_api_key"] = jina_api_key

        # Validate encoding type
        encoding_type = values.get("encoding_type", "float")
        if encoding_type not in VALID_ENCODING:
            raise ValueError(
                f"Encoding type {encoding_type} not supported. Choose from {VALID_ENCODING}"
            )

        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {jina_api_key.get_secret_value()}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )
        values["session"] = session
        return values

    def _process_embeddings(self, embeddings: List[Dict]) -> List[List[float]]:
        """Process embeddings based on encoding type."""
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])

        if self.encoding_type == "ubinary":
            return [
                np.unpackbits(np.array(result["embedding"], dtype="uint8")).tolist()
                for result in sorted_embeddings
            ]
        elif self.encoding_type == "binary":
            return [
                np.unpackbits(
                    (np.array(result["embedding"]) + 128).astype("uint8")
                ).tolist()
                for result in sorted_embeddings
            ]
        return [result["embedding"] for result in sorted_embeddings]

    def _embed(self, input: Any) -> List[List[float]]:
        """Internal method to get embeddings."""
        input_json = {
            "input": input,
            "model": self.model_name,
            "encoding_type": self.encoding_type,
            "late_chunking": self.late_chunking,
        }

        if self.dimensions is not None:
            input_json["dimensions"] = self.dimensions
        if self.task is not None:
            input_json["task"] = self.task

        resp = self.session.post(JINA_API_URL, json=input_json).json()

        if "data" not in resp:
            raise RuntimeError(resp.get("detail", "Unknown error occurred"))

        return self._process_embeddings(resp["data"])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents.

        Args:
            texts (List[str]): The list of texts to embed.

        Returns:
            List[List[float]]: List of embeddings, one for each text.
        """
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single query text.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: Embedding for the text.
        """
        return self._embed([text])[0]

    def embed_image(self, uri: str) -> List[float]:
        """Get embedding for a single image.

        Args:
            uri (str): The URI or path to the image.

        Returns:
            List[float]: Embedding for the image.
        """
        if is_local(uri):
            input_data = [{"bytes": get_bytes_str(uri)}]
        else:
            input_data = [{"url": uri}]
        return self._embed(input_data)[0]

    def embed_images(self, uris: List[str]) -> List[List[float]]:
        """Get embeddings for multiple images.

        Args:
            uris (List[str]): The list of URIs or paths to images.

        Returns:
            List[List[float]]: List of embeddings, one for each image.
        """
        input_data = []
        for uri in uris:
            if is_local(uri):
                input_data.append({"bytes": get_bytes_str(uri)})
            else:
                input_data.append({"url": uri})
        return self._embed(input_data)
