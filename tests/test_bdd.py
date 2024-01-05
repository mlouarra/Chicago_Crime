import requests

# Test si l'URL de l'api réagit
def test_api_url():
    url = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
    response = requests.get(url)

    assert response.status_code == 200
