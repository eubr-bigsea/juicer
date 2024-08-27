import requests

url = "https://dev.lemonade.org.br/api/v1/tahiti/pipelines/36"

headers = {
    'x-auth-token': '123456',
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Erro na requisição: {response.status_code}")
    print(response.text)
