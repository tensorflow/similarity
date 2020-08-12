from fastapi.testclient import TestClient
from tensorflow_similarity.indexer.api.main import app

client = TestClient(app)

def test_lookup_embeddings():
    assert(True)


def test_lookup():
    assert(True)


def test_info():
    assert(True)


def test_metrics():
    assert(True)

def test_add():
    assert(True)


def test_delete():
    assert(True)