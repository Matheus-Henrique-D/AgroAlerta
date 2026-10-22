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

# Credenciais do Supabase Cloud (Dashboard Supabase -> Project Settings -> API)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://seu-projeto.supabase.co")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "sb_secret_sua_chave_secreta")
SUPABASE_PUBLISHABLE_KEY = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_sua_chave_publica")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:senha@aws-0-us-west-2.pooler.supabase.com:5432/postgres")

# Coordenadas geográficas de referência padrão (Rio Claro - SP)
RIO_CLARO_LAT = -22.4114
RIO_CLARO_LON = -47.5614
