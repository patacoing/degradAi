import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def load_file(file_path: str) -> bytes:
    with open(file_path, "rb") as file:
        return file.read()

@pytest.mark.skip("Skip this test because the model is not trained")
@pytest.mark.parametrize(
    "filename, expected_classname",
    [
        ("aucun-rapport.jpg", "aucun-rapport"),
        ("degrade.jpg", "degrade"),
        ("degradant.jpg", "degradant"),
    ]
)
def test_infer_should_return_degrade(client, filename, expected_classname):
    file = load_file(f"tests/integration/data/{filename}")

    response = client.post(url="/infer/degradai", files={"image": ("filename", file, "image/jpg")})

    assert response.status_code == 200
    body = response.json()

    assert body["classname"] == expected_classname
    assert body["probability"] > 0.95

    if expected_classname == "degradant":
        assert body["mention"] == "ABERRANT"