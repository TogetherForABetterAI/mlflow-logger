import pytest
from src.service.run_registry import RunRegistry

@pytest.fixture
def in_memory_db():
    return RunRegistry("sqlite:///:memory:")

def test_save_and_get_run_id(in_memory_db):
    in_memory_db.save_run_id("session_1", "run_xyz")
    retrieved = in_memory_db.get_run_id("session_1")
    assert retrieved == "run_xyz"

def test_update_existing_run_id(in_memory_db):
    in_memory_db.save_run_id("session_1", "run_a")
    in_memory_db.save_run_id("session_1", "run_b")
    retrieved = in_memory_db.get_run_id("session_1")
    assert retrieved == "run_b"

def test_get_missing_run_id_returns_none(in_memory_db):
    assert in_memory_db.get_run_id("abc") is None
