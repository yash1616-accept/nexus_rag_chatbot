import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Verify that the health check endpoint returns 200 OK."""
    response = client.get('/api/health')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "online"
    assert "version" in json_data

def test_export_chat_endpoint(client):
    """Verify that the export chat endpoint returns a formatted transcript."""
    response = client.post('/api/export-chat', json={"session_id": "test_export_session"})
    assert response.status_code == 200
    json_data = response.get_json()
    assert "transcript" in json_data
    assert json_data["session_id"] == "test_export_session"
