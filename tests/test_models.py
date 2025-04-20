import hashlib
from app.models import DocumentModel

def test_generate_digest():
    content = "Hello, World!"
    model = DocumentModel(page_content=content)
    expected_digest = hashlib.md5(content.encode()).hexdigest()
    assert model.generate_digest() == expected_digest