# 🌾 Changelog e Atualizações do Sistema — 03/11/2026

Registro consolidado das melhorias, correções e novas funcionalidades implementadas no **AgroAlerta** na data de hoje.

---

## 📌 Sumário das Entregas
1. [Transcrição de Voz com Resposta em Texto](#1-transcrição-de-voz-com-resposta-em-texto)
2. [Nova Persona do Assistente (Agrônomo Parceiro de Campo)](#2-nova-persona-do-assistente-agrônomo-parceiro-de-campo)
3. [Integração Resiliente com Gemini e Eliminação de Respostas Incompletas](#3-integração-resiliente-com-gemini-e-eliminação-de-respostas-incompletas)
4. [Higienizador Automático de Markdown para Telegram](#4-higienizador-automático-de-markdown-para-telegram)
5. [Injeção de Microclima em Tempo Real (Rio Claro - SP)](#5-injeção-de-microclima-em-tempo-real-rio-claro---sp)
6. [Tabela de Commits do Dia](#6-tabela-de-commits-do-dia)

---

## 1. Transcrição de Voz com Resposta em Texto
- **Problema anterior:** Áudios do produtor geravam notas de voz de retorno via Text-To-Speech (TTS), cuja qualidade e velocidade nem sempre eram satisfatórias para o uso no campo, além de faltar codecs de áudio no ambiente virtual.
- **Melhoria implementada:**
  - Instalação dos pacotes de áudio (`soundfile`, `SpeechRecognition`, codecs compatíveis).
  - O bot agora realiza o download do áudio `.ogg`/Opus do Telegram, converte internamente e transcreve a fala para texto.
  - A resposta do agrônomo agora é retornada **sempre em formato de texto limpo**, exibindo o que foi compreendido (`🎙️ Entendi: "..."`) seguido da orientação técnica completa.

---

## 2. Nova Persona do Assistente (Agrônomo Parceiro de Campo)
- **Novo Prompt do Sistema (`PROMPT_SISTEMA_AGRONOMO`):**
  - **Tom de voz:** Engenheiro agrônomo experiente, caloroso e paciente, que conversa direto ao ponto no WhatsApp/Telegram.
  - **Respeito ao conhecimento do produtor:** Complementa a vivência do campo com os dados da terra, sem lições de moral nem introduções desnecessárias.
  - **Tamanho adaptativo:**
    - Pergunta simples: 2 a 4 linhas diretas.
    - Pergunta de decisão: resposta direta + até 4 tópicos com o essencial + próximo passo.
    - Pergunta de detalhes: até 8 tópicos organizados em blocos.
  - **Regras estritas para orgânico:** Recomendação de insumos permitidos (calda bordalesa, *Bacillus thuringiensis*, extratos), lembrete de EPI e cuidados práticos.

---

## 3. Integração Resiliente com Gemini e Eliminação de Respostas Incompletas
- **Problema anterior:** A chamada à API do Gemini utilizava um modelo indisponível/sobrecarregado com timeout de apenas 1,8s, forçando quedas frequentes em um fallback baseado em palavras-chave que respondia tópicos incorretos (ex.: falar sobre cotação de alface quando o usuário perguntava sobre pH).
- **Melhorias implementadas:**
  - Configuração de cascata de modelos de alta velocidade do Google:
    1. `gemini-3-flash-preview`
    2. `gemini-3.1-flash-lite`
    3. `gemini-3.5-flash-lite`
  - Expansão do teto para **1.500 tokens** e timeout de **12 segundos**.
  - Validação de integridade: se a resposta vier truncada (`MAX_TOKENS`, corte seco ou asterisco aberto no final), o sistema descarta e tenta o próximo modelo ou aciona o motor agronômico local completo.

---

## 4. Higienizador Automático de Markdown para Telegram
- **Problema anterior:** O parser de Markdown do Telegram quebrava (`400 Bad Request: can't parse entities`) sempre que a IA gerava marcadores de lista iniciados por `* ` ou utilizava negrito duplo `**`.
- **Melhoria implementada:**
  - Criada a função `sanitizar_markdown_telegram` em `backend/bot.py`.
  - Converte marcadores soltos `* ` ou `- ` em bullets nativos `• `.
  - Converte negrito duplo `**palavra**` em negrito simples `*palavra*`.
  - Fallback gracioso: caso o parser do Telegram encontre alguma entidade inválida, a mensagem é entregue em texto puro instantaneamente, garantindo entrega 100% confiável.

---

## 5. Injeção de Microclima em Tempo Real (Rio Claro - SP)
- **Problema anterior:** Perguntas sobre a previsão do tempo vinham incompletas porque o modelo não recebia dados meteorológicos específicos no prompt.
- **Melhoria implementada:**
  - Injeção das variáveis climáticas reais do Open-Meteo no prompt: temperatura atual do ar, umidade relativa, probabilidade de precipitação nas próximas 6 horas e velocidade do vento.
  - Adicionado ramo específico de clima e previsão do tempo no motor de fallback agronômico local.

---

## 6. Tabela de Commits do Dia

| Commit | Mensagem do Commit | Arquivos Principais |
| :--- | :--- | :--- |
| **`07efbf2`** | `feat(bot): responder por texto para audios e integrar Gemini LLM com fallback robusto` | `backend/bot.py`, `backend/llm_adapter.py`, `Rede Neural/crop_recommendation.py`, `.gitignore` |
| **`065ac6b`** | `fix(telegram): higienizacao de markdown para evitar quebra de parse de entidades` | `backend/bot.py`, `backend/llm_adapter.py` |
| **`41b43d1`** | `feat(prompt): atualizar PROMPT_SISTEMA_AGRONOMO com persona de campo experiente e nova estrutura de contexto` | `backend/llm_adapter.py` |
| **`f18f321`** | `fix(weather): injetar dados meteorologicos no prompt, aumentar tokens e evitar resposta truncada` | `backend/bot.py`, `backend/llm_adapter.py` |
