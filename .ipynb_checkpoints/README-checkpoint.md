# Inovação na Inteligência Artificial: Utilizando a API OpenAI junto com o ChatGPT


Este artigo foi elaborado organizando perguntas e respostas obtidas via [ChatGPT](https://beta.openai.com/playground?mode=complete) e requisições diretas à API [OpenAI API](https://beta.openai.com/docs/api-reference/introduction) via Jupyter Notebook. 
Concluiremos com exemplos de códigos para implementação em Python de requisições à OpenAI API para obter:
- Respostas para perguntas aleatórias;
- Imagens criadas por IA a partir de descrições.


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


**Código exemplo para envio de "perguntas"**

```python
# Instale os pacotes requeridos.
# !pip install requests,json,openai
import requests,json,openai

# Para gerar o token cadastre-se em (https://beta.openai.com/account/api-keys)
# Token de acesso
# Atenção: Não permita que o seu token seja descoberto, neste caso esse token
# já foi revogado (cancelado)
token = openai.api_key = "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"

# Endpoint do OpenAI API a cada requisição enviada uma resposta é retornada.
# https://beta.openai.com/docs/api-reference/completions
url = "https://api.openai.com/v1/completions"

# Cabeçalho com formato e o token de acesso. O Token é passado via Header no POST
headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer ' + str(token)
}

# OpenAI API aceita as requisições com atributos formatados em JSON
# Atributos: Model: text-davinci-003 é o ID do modelo a ser usado.
#            Prompt: Entrada de dados, neste atributo você envia a sua "pergunta".
# Você pode usar a modelos de lista API de para ver todos os seus 
# modelos disponíveis ou consultar nossa visão geral do modelo para obter 
# as descrições deles. 
# https://beta.openai.com/docs/models/overview

payload = json.dumps({
  "model": "text-davinci-003",
  "prompt": "Defina fórmula de Bhaskara e exemplifique.",
  "temperature": 0,
  "max_tokens": 2048
})

# Envio da requisição
response = requests.request("POST", url, headers=headers, data=payload)
# Formatando a resposta
resposta = json.loads(response.text)
```

**Resposta da API OpenAI API**

```python
# resposta.keys()
# dict_keys(['id', 'object', 'created', 'model', 'choices', 'usage'])
# resposta['choices'][0].keys()
# dict_keys(['text', 'index', 'logprobs', 'finish_reason'])

print(resposta['choices'][0]['text'])
```

![](img/bhaskara.png)


**Código exemplo para geração de imagens com base em uma descrição textual**

```python
# Instale os pacotes requeridos.
import requests,json,openai

# Para gerar o token cadastre-se em (https://beta.openai.com/account/api-keys)
# Token de acesso
# Atenção: Não permita que o seu token seja descoberto, neste caso esse token
# já foi revogado (cancelado)
token = openai.api_key = "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"

# Endpoint do OpenAI API a cada requisição enviada uma resposta é retornada.
# https://beta.openai.com/docs/api-reference/images
url = "https://api.openai.com/v1/images/generations"

# Cabeçalho com formato e o token de acesso. O Token é passado via Header no POST
headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer ' + str(token)
}

# OpenAI API aceita as requisições com atributos formatados em JSON
# Atributos: Prompt: Entrada de dados, descreva o que deseja que a IA desenhe".
#                 n: Quantidade de repetições
#              size: Tamanho das imagens (1024, 512 ou 256)
# https://beta.openai.com/docs/models/overview

payload = json.dumps({
  "prompt": "Desenhe um fundo de tela que represente o que significa OpenAI API e ChatGPT",
  "n": 3,
  "size": "1024x1024"
})

response = requests.request("POST", url, headers=headers, data=payload)
resposta = json.loads(response.text)
```

```python
# resposta.keys()
# resposta['data']
```

A resposta será uma "lista" com "URLs" das imagens solicitadas.


**Solicitações**:


1. Desenhe uma casa branca, com uma árvore e um céu azul, com nuvens brancas e aves voando;

**Resposta**

![](img/1.png)


2. Desenhe uma ilustração para um artigo que fala a respeito de OpenIA, ChatGPT e Tecnologia;

**Resposta**

![](img/2.png)


3. Desenhe um fundo de tela que represente o que significa OpenAI API e ChatGPT.

**Resposta**

![](img/3.png)


**Conclusão**

Somente para registro, para escrever este artigo eu gastei \$0.75 dos \$18.00 que ganhei para teste da API. São ferramente muito interessantes e certamente permitirão "insight" importantes para o desenvolvimento de todos.

Gratidão

[Carlos Eugênio](https://github.com/carlosemsantana)


**Referências:**<p>
[1] [https://beta.openai.com/](https://beta.openai.com/) 
