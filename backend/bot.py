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
from llm_adapter import formatar_relatorio_para_produtor, responder_duvida_produtor
from crop_recommendation import recomendar_culturas_para_produtor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgroBot")


def processar_mensagem_produtor(chat_id: str, texto: str) -> str:
    """
    Processa a mensagem do produtor (comando ou dúvida livre) e retorna a resposta formatada pelo LLM.
    """
    texto_strip = texto.strip()
    cmd = texto_strip.lower()

    # 1. Comando /start
    if cmd in ["/start", "oi", "olá", "ola", "bom dia", "boa tarde"]:
        talhoes = db_manager.listar_talhoes(chat_id)
        qtd = len(talhoes)
        msg = (
            f"🌱 *Olá, amigo produtor! Bem-vindo ao AgroAlerta.* 👨‍🌾\n\n"
            f"Eu sou seu assistente técnico no campo. Estou conectado aos seus sensores e ao tempo de Rio Claro e região.\n\n"
            f"📱 *Seu ID de Conexão:* `{chat_id}`\n"
            f"📍 *Talhões Cadastrados:* {qtd}\n\n"
            f"Comandos rápidos que você pode usar:\n"
            f"• */irrigar* — Ver se precisa ligar a água hoje\n"
            f"• */plantar* — Sugestões do que e como plantar neste mês\n"
            f"• */talhoes* — Ver as áreas cadastradas na sua propriedade\n"
            f"• */status* — Resumo do solo e clima recente\n\n"
            f"Ou simplesmente me mande uma pergunta aqui como: *'Vai chover hoje?'* ou *'O que coloco de adubo no meu tomate?'*!"
        )
        return msg

    # 2. Comando /talhoes
    if cmd == "/talhoes":
        talhoes = db_manager.listar_talhoes(chat_id)
        if not talhoes:
            return (
                "📍 *Nenhum talhão cadastrado ainda!*\n\n"
                "Abra o AgroAlerta no celular, clique em 'Marcar Talhão' e caminhe nos 3 ou 4 cantos da sua roça para delimitar a área."
            )
        linhas = ["🚜 *Seus Talhões Cadastrados:*\n"]
        for t in talhoes:
            org = "Orgânico 🌱" if t["is_organico"] else "Convencional 🚜"
            area = f" (~{t['area_ha']:.2f} ha)" if t.get("area_ha") else ""
            linhas.append(f"• *{t['nome_talhao']}*{area}: Cultura *{t['cultura'].capitalize()}* ({org})")
        linhas.append("\n_Para coletar dados, basta espetar o sensor no chão do talhão e apertar 'Coletar' no celular._")
        return "\n".join(linhas)

    # 3. Comando /irrigar
    if cmd in ["/irrigar", "/rega", "/agua"]:
        leituras = db_manager.listar_ultimas_leituras(chat_id, limit=3)
        if not leituras:
            return (
                "💧 *Ainda não temos leituras recentes de solo para hoje.*\n\n"
                "Faça uma leitura com o sensor no talhão pelo celular que eu te digo na hora se precisa regar e quantos milímetros!"
            )
        ult = leituras[0]
        talhao_nome = ult.get("talhao_nome") or "seu talhão"
        umid = ult.get("umidade", 0.0)
        return (
            f"💧 *Situação de Irrigação — {talhao_nome}:*\n\n"
            f"A última medição registrou *{umid:.1f}% de umidade* no solo.\n"
            f"Consulte o relatório que enviei logo após a coleta para ver a dosagem exata de água recomendada."
        )

    # 4. Comando /plantar ou /safra
    if any(cmd.startswith(x) for x in ["/plantar", "/safra", "o que plantar", "quando plantar"]):
        rec = recomendar_culturas_para_produtor(
            lat=RIO_CLARO_LAT,
            lon=RIO_CLARO_LON,
            is_organico=True  # padrão orientativo orgânico
        )
        top = rec["top_recomendacoes"]
        linhas = [f"📅 *Planejamento de Safra — Rio Claro e Região (Mês de {rec['mes_referencia']})*\n"]
        linhas.append("Baseado no clima do ZARC e nas melhores cotações do CEAGESP:\n")
        for i, item in enumerate(top, 1):
            linhas.append(f"*{i}º Lugar: {item['nome']}* (Nota {item['score_viabilidade']}/100)")
            linhas.append(f"  👉 {item['justificativas'][0]}")
            linhas.append(f"  🗓️ Colheita prevista para: *{item['previsao_colheita']}*")
            linhas.append(f"  🌱 Espaçamento: _{item['como_plantar']['espacamento']}_")
            linhas.append(f"  🌾 Adubação: _{item['como_plantar']['adubacao_recomendada']}_\n")
        return "\n".join(linhas)

    # 5. Dúvida livre do Produtor (Enviada ao LLM com Contexto Real)
    talhoes = db_manager.listar_talhoes(chat_id)
    leituras = db_manager.listar_ultimas_leituras(chat_id, limit=3)
    sugestoes = recomendar_culturas_para_produtor(lat=RIO_CLARO_LAT, lon=RIO_CLARO_LON, is_organico=True)

    resposta = responder_duvida_produtor(
        pergunta=texto_strip,
        contexto_talhoes=talhoes,
        ultimas_leituras=leituras,
        sugestoes_plantio=sugestoes,
        api_key_llm=GEMINI_API_KEY
    )
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

    from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters

    async def handle_cmd(update, context):
        cid = str(update.effective_chat.id)
        txt = update.message.text
        resp = processar_mensagem_produtor(cid, txt)
        await update.message.reply_text(resp, parse_mode="Markdown")

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT, handle_cmd))

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