# Inovação na Inteligência Artificial: Utilizando a API OpenAI junto com o ChatGPT


Este artigo foi elaborado organizando perguntas e respostas obtidas via [ChatGPT](https://beta.openai.com/playground?mode=complete) e requisições diretas à API [OpenAI API](https://beta.openai.com/docs/api-reference/introduction) via Jupyter Notebook. 
Concluiremos com exemplos de códigos para implementação em Python de requisições à OpenAI API para obter:
- Respostas para perguntas aleatórias;
- Imagens criadas por IA com base em textos.


**ChatGPT**


[![Watch the video](img/ChatGPT.png)](https://youtu.be/ifWV4nVgxp8)


**Resumo**


Este artigo irá discutir a inovação na Inteligência Artificial através da utilização da API OpenAI junto com o ChatGPT. O ChatGPT é um serviço de chatbot baseado em aprendizado de máquina que usa a tecnologia de pré-treinamento GPT-3, desenvolvida pela OpenAI. A API OpenAI, por sua vez, fornece acesso ao GPT-3, possibilitando que os desenvolvedores criem seus próprios serviços de chatbot. O artigo discutirá as vantagens e desvantagens do uso de ChatGPT e da API OpenAI e discutirá como essas tecnologias podem ser usadas para melhorar a inteligência artificial. O artigo também abordará alguns dos desafios enfrentados na implementação dessas tecnologias.


**OpenAI e o ChatGPT**


OpenAI é uma organização de pesquisa sem fins lucrativos dedicada ao desenvolvimento de inteligência artificial (IA) de ponta para o benefício da humanidade. A OpenAI foi fundada em 2015 por Elon Musk, Sam Altman, Greg Brockman, Ilya Sutskever, Wojciech Zaremba e outros. A OpenAI tem como objetivo desenvolver tecnologias que permitam que as máquinas aprendam a realizar tarefas complexas, como jogar jogos, dirigir carros e entender línguas.

O ChatGPT é um serviço de chatbot baseado em aprendizado de máquina que usa a tecnologia de pré-treinamento GPT-3, desenvolvida pela OpenAI. GPT-3 é uma tecnologia de inteligência artificial de larga escala que foi criada para permitir que os chatbots forneçam respostas mais precisas e humanas. O ChatGPT usa GPT-3 para fornecer respostas mais naturais e autênticas para as perguntas que são feitas ao chatbot.

Além disso, a OpenAI fornece acesso à sua API, que permite que os desenvolvedores criem seus próprios serviços de chatbot usando GPT-3. A API OpenAI também oferece a possibilidade de personalizar o chatbot para atender às necessidades do usuário, como por exemplo, a linguagem usada, o contexto do diálogo e os tópicos abordados.

Ao usar a API OpenAI junto com o ChatGPT, os desenvolvedores de chatbot podem criar um chatbot mais inteligente e preciso, além de melhorar a experiência do usuário. Além disso, a API OpenAI oferece várias ferramentas para ajudar os desenvolvedores a criar o chatbot certo para as necessidades específicas do usuário.

No entanto, o uso da API OpenAI e do ChatGPT também apresenta alguns desafios. Por exemplo, é importante que os desenvolvedores possam garantir que o chatbot seja seguro e não possa ser usado para fins maliciosos. Além disso, é importante que os chatbots sejam capazes de fornecer respostas precisas, pois os usuários esperam que os chatbots sejam confiáveis.

**Como é feito o treinamento da OpenAI API?**

A API OpenAI é treinada usando um processo chamado treinamento de aprendizado profundo. Esta abordagem envolve o uso de um conjunto de dados de treinamento, que é passado através de várias camadas de aprendizado de máquina para atingir a melhor acurácia. Durante o treinamento, também se usam métricas para avaliar as métricas da API e melhorar caso necessário. Além disso, scripts personalizados podem ser usados para manipular conjuntos de dados. Por último, testes de validação são realizados para certificar que a API é treinada corretamente.

**De onde são extraídos os dados (dataset) para o treinamento da OpenAI API?**

Os conjuntos de dados para treinamento da OpenAI API são extraídos de fontes públicas e comerciais, como bases de dados, corpora de línguas naturais, modelos abertos, fontes pagas e documentos comprados por bibliotecas. Alguns conjuntos de dados também são criados em parceria com empresas parceiras da OpenAI a partir de seus dados próprios.

**Os dados (dataset) para o treinamento da OpenAI API foram extraídos do Facebook?**

Não, os dados (dataset) para o treinamento da OpenAI API não foram extraídos do Facebook. O OpenAI usa um conjunto de dados diferente comummente referido como o "Conjunto de Treinamento GPT-2". Esse conjunto de dados foi coletado a partir da Web e é composto principalmente de conteúdo publicado em meios digitais como sites de notícias, blogs, documentos governamentais, fóruns on-line etc.

**Os dados (dataset) para o treinamento da OpenAI API foram extraídos do Linkedin?**

Não, os dados para o treinamento da OpenAI API não foram extraídos do Linkedin. Eles foram extraídos de fontes de dados abertos, como o Common Crawl, o Wikipedia e o OpenWebText.

**Os dados do Linkedin estão no Google, logo são públicos e podem ser usados para treinamento do OpenAI API?**

Não, os dados do Linkedin não são adequados para treinamento do OpenAI API. O OpenAI API é projetado para trabalhar com dados estruturados, como dados de texto, imagens, áudio e vídeo. Os dados do Linkedin são principalmente informações não estruturadas, como informações de perfil, experiência profissional, educação, etc. Portanto, não são adequados para treinamento do OpenAI API.


**Conclusão**

ChatGPT e OpenAI API são ferramentas essenciais para a inovação na Inteligência Artificial. Ao usar essas tecnologias, os desenvolvedores podem criar chatbots mais precisos e humanos, o que melhora a experiência dos usuários. No entanto, é importante que os desenvolvedores tomem precauções ao usar essas tecnologias, como garantir que o chatbot seja seguro e que forneça respostas precisas. Ao usar ChatGPT e OpenAI API em conjunto, os desenvolvedores podem criar chatbots mais inovadores e avançados.


**Códigos**

```python
import requests,json,openai
```

```python
# Para gerar o token cadastre-se em (https://beta.openai.com/account/api-keys)
token = openai.api_key = "Bearer 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"
```

```python
url = "https://api.openai.com/v1/completions"
```

```python
headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer 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'
}
```

```python
payload = json.dumps({
  "model": "text-davinci-003",
  "prompt": "Escreva introdução para artigo que fala a respeito de OpenAI API e ChatGPT",
  "temperature": 0,
  "max_tokens": 2048
})
```

```python
response = requests.request("POST", url, headers=headers, data=payload)
```

```python
df = json.loads(response.text)
```

```python
df.keys()
```

```python

```

```python
url = "https://api.openai.com/v1/images/generations"
```

```python
payload = json.dumps({
  "prompt": "Desenhe um fundo de tela que represente o que significa OpenAI API e ChatGPT",
  "n": 3,
  "size": "1024x1024"
})
```

```python
response = requests.request("POST", url, headers=headers, data=payload)
```

```python
df = json.loads(response.text)
```

```python
df.keys()
```

### Desenhe uma casa branca, com uma árvore e um céu azul, com nuvens brancas e aves voando

```python
df['data']
```

[{'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-CXkuazTlxSPYzQapElW2AF4M.png?st=2023-01-09T21%3A56%3A45Z&se=2023-01-09T23%3A56%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T21%3A12%3A39Z&ske=2023-01-10T21%3A12%3A39Z&sks=b&skv=2021-08-06&sig=alPqeXAXf4wFanHH8ixuzCcas0t/FS5CUF2mMBZp52g%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-GWfAaLJDKA8N4Cxz4r9RpgeQ.png?st=2023-01-09T21%3A56%3A45Z&se=2023-01-09T23%3A56%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T21%3A12%3A39Z&ske=2023-01-10T21%3A12%3A39Z&sks=b&skv=2021-08-06&sig=Xi6Ixbw%2B6kNuESotFhgr7wQkxS3EJP97HbERAfd4bDg%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-d0w5vGAYLzOnbkr5wnbZLbRN.png?st=2023-01-09T21%3A56%3A45Z&se=2023-01-09T23%3A56%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T21%3A12%3A39Z&ske=2023-01-10T21%3A12%3A39Z&sks=b&skv=2021-08-06&sig=uOeIHXB7/oj%2B8XjKTst5aGZmBUIic0GQ5OExTed2Xto%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-humyHte5b3LuGFrpKF5nXBNz.png?st=2023-01-09T21%3A56%3A45Z&se=2023-01-09T23%3A56%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T21%3A12%3A39Z&ske=2023-01-10T21%3A12%3A39Z&sks=b&skv=2021-08-06&sig=TrKIlkuuMQeX%2Bd6VV2Nk3PSEDayLvND4KbVGPdDvvXU%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-ymT5Fu2XjhZKVFDXE5jX99G0.png?st=2023-01-09T21%3A56%3A45Z&se=2023-01-09T23%3A56%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T21%3A12%3A39Z&ske=2023-01-10T21%3A12%3A39Z&sks=b&skv=2021-08-06&sig=wPIMv9HgJfUdjmvZdtua4N/bJx1bazc4jS608Z0Alp4%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-Er7cSsNlTmJZY2DwYG9hoOLU.png?st=2023-01-09T21%3A56%3A45Z&se=2023-01-09T23%3A56%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T21%3A12%3A39Z&ske=2023-01-10T21%3A12%3A39Z&sks=b&skv=2021-08-06&sig=eX0Rzl0i4mgCfAxScY8IOPKsKqW8V3iUzcRtzOY191E%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-vZLuAFMTJFxpQuNOGN8ljsrR.png?st=2023-01-09T21%3A56%3A45Z&se=2023-01-09T23%3A56%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T21%3A12%3A39Z&ske=2023-01-10T21%3A12%3A39Z&sks=b&skv=2021-08-06&sig=1EZgaeh434NxcyUwBn3hyc61X2GYslWw93C7HoosCIw%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-9IocJsnXiDWyO4Z9w55JizhZ.png?st=2023-01-09T21%3A56%3A45Z&se=2023-01-09T23%3A56%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T21%3A12%3A39Z&ske=2023-01-10T21%3A12%3A39Z&sks=b&skv=2021-08-06&sig=o6Nm/4vp3P19GAQTJWwaj/v%2BSeD/rGPIUbDOGIeh9VA%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-DcqaB2RKgTQMYXCWmMHyweDK.png?st=2023-01-09T21%3A56%3A45Z&se=2023-01-09T23%3A56%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T21%3A12%3A39Z&ske=2023-01-10T21%3A12%3A39Z&sks=b&skv=2021-08-06&sig=aOcf9fdfv0GZdRznm65V0bus5aiU8/l1wYNL2K81kPw%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-iQZuZ2g2IlLTN65EpFV6qKVw.png?st=2023-01-09T21%3A56%3A45Z&se=2023-01-09T23%3A56%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T21%3A12%3A39Z&ske=2023-01-10T21%3A12%3A39Z&sks=b&skv=2021-08-06&sig=zhEB4sE1PJur1QSioMW8zDtidxIYjKzeRHiC8j7WETs%3D'}]


### Desenhe uma capa para um artigo que fala a respeito de OpenIA, ChatGPT e Tecnologia.

```python
df['data']
```

[{'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-OHC6kkmWkfr0enuOnNRCwAIw.png?st=2023-01-09T22%3A11%3A36Z&se=2023-01-10T00%3A11%3A36Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T19%3A24%3A36Z&ske=2023-01-10T19%3A24%3A36Z&sks=b&skv=2021-08-06&sig=ak4UfsO9JSB7xOKms1GM7tE0O7jN0/TS3uX76rsbubo%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-emBWvBbMpqp4fgzygBB9yKvL.png?st=2023-01-09T22%3A11%3A36Z&se=2023-01-10T00%3A11%3A36Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T19%3A24%3A36Z&ske=2023-01-10T19%3A24%3A36Z&sks=b&skv=2021-08-06&sig=JmNfA7uI4449ylIwOuvcJRkQd6UR%2BzVp5CA8gTqCnNU%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-SxlHTnjHi8luFbEMMzcf1d0d.png?st=2023-01-09T22%3A11%3A36Z&se=2023-01-10T00%3A11%3A36Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T19%3A24%3A36Z&ske=2023-01-10T19%3A24%3A36Z&sks=b&skv=2021-08-06&sig=UG/S61nbH2NVMlzS5SKhG7ycDJQqBLj584VnGhz6bPU%3D'}]


### Desenhe um fundo de tela que represente o que significa OpenAI API e ChatGPT

```python
df['data']
```

[{'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-5mXepDkcMpKXbjXsJiM000Sd.png?st=2023-01-09T22%3A15%3A36Z&se=2023-01-10T00%3A15%3A36Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T20%3A32%3A30Z&ske=2023-01-10T20%3A32%3A30Z&sks=b&skv=2021-08-06&sig=E7lqdV2DFjdlKIQ0bzyRIos2ri2dz2gGgBwGDYuTrLc%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-yfCeoD0UxVMbOXwAnW5RGvAD.png?st=2023-01-09T22%3A15%3A36Z&se=2023-01-10T00%3A15%3A36Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T20%3A32%3A30Z&ske=2023-01-10T20%3A32%3A30Z&sks=b&skv=2021-08-06&sig=PWAh%2BWLom2FX1Uy6ACFCA1cq%2Bbxo9qJBD/2coAHZEzE%3D'},
 {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-7FoyzXPtgPrSoxwHakbEG667/user-a0FVNqzRRLgdzcyK0uncb4hy/img-9jgwrowiiQzpc0G77hrEFA82.png?st=2023-01-09T22%3A15%3A36Z&se=2023-01-10T00%3A15%3A36Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-01-09T20%3A32%3A30Z&ske=2023-01-10T20%3A32%3A30Z&sks=b&skv=2021-08-06&sig=WNq19uVaXdzAhGFpvNcIJtROVzFrnrd7oez00pk2APw%3D'}]

```python

```
