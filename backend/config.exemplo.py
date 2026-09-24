"""
config.exemplo.py
=================
Modelo de configuração para o AgroAlerta.
Copie este arquivo para 'backend/config.py' e preencha com suas credenciais reais.
"""

import os

# Token do Bot Telegram (gerado via @BotFather)
TOKEN = os.environ.get("TELEGRAM_TOKEN", "SEU_TOKEN_TELEGRAM_AQUI")

# Chat ID padrão do produtor no Telegram para envio de notificações
DEFAULT_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7858612258")

# Chave de API do Google Gemini (obtenha em https://aistudio.google.com/)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "SUA_CHAVE_GEMINI_AQUI")

# Coordenadas geográficas de referência padrão (Rio Claro - SP)
RIO_CLARO_LAT = -22.4114
RIO_CLARO_LON = -47.5614
