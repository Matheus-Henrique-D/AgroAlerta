"""
bot.py
======
Bot do Telegram do AgroAlerta — Assistente Agronômico Pessoal do Produtor.
Atende o produtor via Telegram com linguagem humanizada por IA Generativa (LLM),
trazendo relatórios de irrigação, alertas fitossanitários, sugestões de plantio
para Rio Claro - SP e tirando dúvidas com base no histórico do talhão.
"""

import sys
import os
import json
import logging
from typing import Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Adiciona o diretório raiz e 'Rede Neural' ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Rede Neural"))

try:
    from config import TOKEN, DEFAULT_CHAT_ID, GEMINI_API_KEY, RIO_CLARO_LAT, RIO_CLARO_LON
except ImportError:
    TOKEN = os.environ.get("TELEGRAM_TOKEN", "mock_token")
    DEFAULT_CHAT_ID = "7858612258"
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    RIO_CLARO_LAT = -22.4114
    RIO_CLARO_LON = -47.5614

from db_manager import db_manager
from llm_adapter import (
    formatar_relatorio_para_produtor,
    responder_duvida_produtor,
    extrair_configuracao_produtor_linguagem_natural
)
from crop_recommendation import recomendar_culturas_para_produtor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgroBot")


def processar_mensagem_produtor(chat_id: str, texto: str, nome_usuario: Optional[str] = None) -> str:
    """
    Processa a mensagem do produtor em Linguagem Natural 100% humanizada,
    permitindo que ele converse normalmente, configure sua lavoura (cultura, solo, orgânico)
    e tire dúvidas baseadas nos dados reais salvos no Supabase / SQLite.
    """
    texto_strip = texto.strip()
    cid = str(chat_id)

    # 1. Recupera ou cria o cadastro do produtor
    produtor = db_manager.obter_ou_criar_produtor(cid, nome=nome_usuario)

    # 2. Registra a mensagem de entrada do produtor
    db_manager.salvar_mensagem_telegram(
        chat_id=cid,
        direcao="entrada",
        mensagem_texto=texto_strip,
        tipo_mensagem="texto_livre"
    )

    cmd_lower = texto_strip.lower()

    # 3. Verificação de Configuração em Linguagem Natural (Cultura, Solo, Orgânico)
    config_mudancas = extrair_configuracao_produtor_linguagem_natural(texto_strip)
    if config_mudancas:
        produtor_atualizado = db_manager.atualizar_configuracao_produtor(cid, **config_mudancas)
        cult_nome = str(produtor_atualizado.get("cultura_ativa", "cultura")).capitalize()
        solo_nome = str(produtor_atualizado.get("tipo_solo_ativo", "latossolo vermelho")).replace("_", " ").capitalize()
        org_status = "Cultivo Orgânico Certificado 🌱" if produtor_atualizado.get("is_organico_ativo") else "Cultivo Convencional 🚜"
        area_status = f"{produtor_atualizado.get('area_ha_ativa', 1.0):.2f} ha"

        # Se a mensagem foi puramente uma configuração ou declaração da lavoura
        palavras_pergunta = ["preciso", "quando", "quanto", "como", "devo", "qual", "tem", "por que", "porque", "?"]
        if not any(p in cmd_lower for p in palavras_pergunta):
            resposta_config = (
                f"✅ *Perfeito! Atualizei as informações da sua lavoura:*\n\n"
                f"• *Cultura Ativa:* {cult_nome}\n"
                f"• *Tipo de Solo:* {solo_nome}\n"
                f"• *Manejo:* {org_status}\n"
                f"• *Área de Referência:* ~{area_status}\n\n"
                f"Todas as próximas recomendações de água, pragas e insumos serão calculadas para essa condição!\n"
                f"Se quiser saber a situação hídrica ou se precisa regar, é só me perguntar aqui."
            )
            db_manager.salvar_mensagem_telegram(cid, "saida", resposta_config, "configuracao")
            return resposta_config

    # 4. Saudação Inicial (/start ou "Oi")
    if cmd_lower in ["/start", "oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "ajuda"]:
        talhoes = db_manager.listar_talhoes(cid)
        qtd = len(talhoes)
        cult_atual = str(produtor.get("cultura_ativa", "citros")).capitalize()
        solo_atual = str(produtor.get("tipo_solo_ativo", "latossolo vermelho")).replace("_", " ")
        org_atual = "Orgânico 🌱" if produtor.get("is_organico_ativo") else "Convencional 🚜"

        msg = (
            f"🌱 *Olá! Bem-vindo ao AgroAlerta.* 👨‍🌾\n\n"
            f"Eu sou seu consultor de campo digital, conectado aos sensores da sua terra e ao clima de Rio Claro.\n\n"
            f"🌾 *Sua Configuração Atual:*\n"
            f"• Cultura: *{cult_atual}* ({org_atual})\n"
            f"• Solo: *{solo_atual}* | Talhões mapeados: *{qtd}*\n\n"
            f"Você pode conversar comigo em *linguagem natural normal*, sem precisar de comandos! Exemplos:\n"
            f"• *'Preciso ligar a irrigação hoje?'*\n"
            f"• *'Tem perigo de pragas na minha roça?'*\n"
            f"• *'Estou plantando tomate orgânico em solo argiloso'* (para mudar sua lavoura)\n"
            f"• *'O que é bom plantar agora em outubro?'*\n\n"
            f"Como posso te orientar no campo hoje?"
        )
        db_manager.salvar_mensagem_telegram(cid, "saida", msg, "saudacao")
        return msg

    # 5. Dúvida Livre / Consulta em Linguagem Natural com Contexto Completo
    talhoes = db_manager.listar_talhoes(cid)
    historico_completo = db_manager.listar_historico_completo(cid, limit=5)
    sugestoes = recomendar_culturas_para_produtor(lat=RIO_CLARO_LAT, lon=RIO_CLARO_LON, is_organico=produtor.get("is_organico_ativo", True))

    resposta = responder_duvida_produtor(
        pergunta=texto_strip,
        contexto_talhoes=talhoes,
        ultimas_leituras=historico_completo,
        sugestoes_plantio=sugestoes,
        api_key_llm=GEMINI_API_KEY,
        produtor=produtor,
        historico_completo=historico_completo
    )

    # 6. Registra a resposta enviada
    db_manager.salvar_mensagem_telegram(cid, "saida", resposta, "resposta_llm")
    return resposta


async def enviar_mensagem_telegram(chat_id: str, mensagem: str):
    """Envia mensagem assíncrona para o produtor no Telegram via Bot API."""
    if not TOKEN or TOKEN == "mock_token":
        logger.info(f"[SIMULAÇÃO TELEGRAM para {chat_id}]:\n{mensagem}")
        return True

    try:
        from telegram import Bot
        bot = Bot(token=TOKEN)
        await bot.send_message(chat_id=chat_id, text=mensagem, parse_mode="Markdown")
        return True
    except Exception as e:
        logger.error(f"Erro ao enviar mensagem Telegram: {e}")
        return False


def notificar_analise_leitura(chat_id: str, relatorio_ia: Dict[str, Any]):
    """
    Formata e envia a análise de uma nova leitura para o Telegram do produtor.
    """
    mensagem_amigavel = formatar_relatorio_para_produtor(relatorio_ia, api_key_llm=GEMINI_API_KEY)
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(enviar_mensagem_telegram(chat_id, mensagem_amigavel))


def rodar_polling():
    """Inicia o bot em modo polling contínuo para receber perguntas dos produtores."""
    if not TOKEN or "mock" in TOKEN:
        print("Token do Telegram não configurado ou mock. Configure TELEGRAM_TOKEN para conectar à API oficial.")
        return

    async def handle_cmd(update, context):
        cid = str(update.effective_chat.id)
        user = update.effective_user
        nome_usuario = user.first_name if user else "Produtor"
        txt = update.message.text or ""
        resp = processar_mensagem_produtor(cid, txt, nome_usuario=nome_usuario)
        await update.message.reply_text(resp, parse_mode="Markdown")

    async def handle_location(update, context):
        cid = str(update.effective_chat.id)
        loc = update.message.location
        if loc:
            lat, lon = loc.latitude, loc.longitude
            talhoes = db_manager.listar_talhoes(cid)
            from geo_engine import identificar_talhao
            talhao = identificar_talhao(lat, lon, talhoes)
            nome_t = talhao["nome_talhao"] if talhao else "Área de Plantio"
            diag = db_manager.obter_ultimo_diagnostico_talhao(cid, talhao["id"] if talhao else None)
            if diag:
                resp = (
                    f"📍 *Localização Recebida no Campo ({lat:.4f}, {lon:.4f})*\n"
                    f"Área Identificada: *{nome_t}* ({str(diag.get('cultura', '')).capitalize()})\n\n"
                    f"💧 *Situação Hídrica:* {diag.get('status_irrigacao')}\n"
                    f"Pode me perguntar qualquer dúvida sobre essa lavoura agora!"
                )
            else:
                resp = (
                    f"📍 *Coordenadas Recebidas:* {lat:.4f}, {lon:.4f}\n"
                    f"Você está próximo ao talhão: *{nome_t}*.\n"
                    f"Pode me perguntar: *'Preciso regar aqui hoje?'* ou *'Tem risco de pragas?'*."
                )
            await update.message.reply_text(resp, parse_mode="Markdown")

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_cmd))
    app.add_handler(MessageHandler(filters.COMMAND, handle_cmd))
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))

    print(f"Bot AgroAlerta iniciado com sucesso! Escutando mensagens...")
    app.run_polling()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--poll":
        rodar_polling()
    else:
        # Teste interativo CLI local
        print("=== Testando AgroBot em Modo Local ===")
        resposta = processar_mensagem_produtor(DEFAULT_CHAT_ID, "O que devo plantar agora em outubro?")
        print("\nResposta Gerada:")
        print(resposta)