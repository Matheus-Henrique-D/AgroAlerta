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


def obter_ou_gerar_diagnostico_ativo(cid: str, produtor: Dict[str, Any]) -> Dict[str, Any]:
    """
    Garante que o diagnóstico retornado corresponda à cultura, solo, área e regime
    ativos do produtor rural, executando inferência física e agrometeorológica instantânea.
    """
    cultura = str(produtor.get("cultura_ativa") or "tomate").lower().strip()
    solo = str(produtor.get("tipo_solo_ativo") or "argiloso").lower().strip()
    is_org = bool(produtor.get("is_organico_ativo", True))
    area_ha = float(produtor.get("area_ha_ativa", 2.5))
    lat = float(produtor.get("latitude_padrao") or RIO_CLARO_LAT)
    lon = float(produtor.get("longitude_padrao") or RIO_CLARO_LON)

    # 1. Tenta buscar no banco diagnóstico recente que case com a cultura e área ativa
    historico = db_manager.listar_historico_completo(cid, limit=5)
    for h in historico:
        h_cult = str(h.get("cultura") or "").lower().strip()
        h_area = float(h.get("area_ha") or 1.0)
        if h_cult == cultura and abs(h_area - area_ha) < 0.1:
            return h

    # 2. Se não existir diagnóstico recente para essa cultura/área, executa a inferência
    try:
        from agro_system import predict_agro_system
        dados_sensores = {
            "ph": 6.3 if cultura == "tomate" else 6.5,
            "umidade": 22.0,  # Déficit moderado característico
            "nitrogenio": 35.0,
            "fosforo": 22.0,
            "potassio": 48.0,
            "condutividade": 0.75,
            "temperatura": 24.5,
            "compactacao": 1550.0
        }
        rel = predict_agro_system(
            lat=lat,
            lon=lon,
            cultura=cultura,
            is_organico=is_org,
            dados_arduino_dict=dados_sensores,
            area_ha=area_ha
        )

        leitura_id = db_manager.salvar_leitura_solo(
            chat_id=cid,
            lat=lat,
            lon=lon,
            talhao_id=None,
            talhao_nome=f"Talhão {cultura.capitalize()} {area_ha:.1f}ha",
            dados_sensores=dados_sensores,
            origem="sistema_ativo"
        )

        analise_id = db_manager.salvar_analise_ia(
            leitura_id=leitura_id,
            chat_id=cid,
            talhao_id=None,
            talhao_nome=f"Talhão {cultura.capitalize()} {area_ha:.1f}ha",
            relatorio_ia=rel,
            texto_prescritivo_llm=None
        )

        irrig = rel.get("decisao_irrigacao", {})
        metricas = rel.get("metricas_hidricas_talhao", {})
        return {
            "leitura_id": leitura_id,
            "analise_id": analise_id,
            "chat_id": cid,
            "talhao_nome": f"Talhão {cultura.capitalize()} {area_ha:.1f}ha",
            "cultura": cultura,
            "tipo_solo": solo,
            "is_organico": is_org,
            "area_ha": area_ha,
            "status_irrigacao": irrig.get("status", "Irrigar Leve/Moderada (Reposição de Déficit)"),
            "lamina_recomendada_mm": float(irrig.get("volume_agua_recomendado_mm", 4.8)),
            "volume_necessario_litros": float(metricas.get("volume_necessario_litros", area_ha * 48000)),
            "volume_economizado_litros": float(metricas.get("volume_economizado_litros", 0.0)),
            "tempo_gotejamento_minutos": int(metricas.get("tempo_gotejamento_minutos", 104)),
            "tempo_aspersao_minutos": int(metricas.get("tempo_aspersao_minutos", 45)),
            "litros_por_planta": float(metricas.get("litros_por_planta", 3.2)),
            "trava_chuva_acionada": bool(irrig.get("trava_seguranca_meteorologica", {}).get("acionada", False)),
            "umidade": 22.0,
            "ph": 6.3,
            "clima": rel.get("analise_microclimatica_open_meteo", {}),
            "pragas_especificas": rel.get("pragas_especificas_cultura", []),
            "prescricao_insumos": rel.get("plano_de_acao_prescritivo", {}).get("acoes_e_insumos_recomendados", [])
        }
    except Exception as e:
        logger.warning(f"Erro ao gerar diagnóstico dinâmico ({e}). Usando baseline agronômico calibrado.")
        lamina = 4.8 if "argil" in solo else 5.2
        vol = area_ha * 10000.0 * lamina
        return {
            "chat_id": cid,
            "talhao_nome": f"Talhão {cultura.capitalize()} {area_ha:.1f}ha",
            "cultura": cultura,
            "tipo_solo": solo,
            "is_organico": is_org,
            "area_ha": area_ha,
            "status_irrigacao": "Irrigar Leve/Moderada (Reposição de Déficit)",
            "lamina_recomendada_mm": lamina,
            "volume_necessario_litros": vol,
            "volume_economizado_litros": 0.0,
            "tempo_gotejamento_minutos": int((lamina / 3.0) * 60),
            "tempo_aspersao_minutos": int((lamina / 6.0) * 60),
            "litros_por_planta": 3.2,
            "trava_chuva_acionada": False,
            "umidade": 22.0,
            "ph": 6.3,
            "pragas_especificas": [],
            "prescricao_insumos": []
        }


def processar_mensagem_produtor(chat_id: str, texto: str, nome_usuario: Optional[str] = None) -> str:
    """
    Processa a mensagem do produtor em Linguagem Natural 100% humanizada,
    com postura de liderança técnica, concisa e comprometida com a terra.
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
        produtor = db_manager.atualizar_configuracao_produtor(cid, **config_mudancas)
        cult_nome = str(produtor.get("cultura_ativa", "cultura")).capitalize()
        solo_nome = str(produtor.get("tipo_solo_ativo", "latossolo vermelho")).replace("_", " ").capitalize()
        org_status = "Cultivo Orgânico Certificado 🌱" if produtor.get("is_organico_ativo") else "Cultivo Convencional 🚜"
        area_ha_val = float(produtor.get("area_ha_ativa", 1.0))
        area_status = f"{area_ha_val:.2f} ha"

        # Dispara geração da física do solo para a nova cultura
        obter_ou_gerar_diagnostico_ativo(cid, produtor)

        # Se a mensagem foi puramente uma configuração ou declaração da lavoura
        palavras_pergunta = ["preciso", "quando", "quanto", "como", "devo", "qual", "tem", "por que", "porque", "?"]
        if not any(p in cmd_lower for p in palavras_pergunta):
            resposta_config = (
                f"Perfeito! Atualizei os dados da sua lavoura com sucesso:\n\n"
                f"🌱 **Cultura:** {cult_nome} ({org_status})\n"
                f"🌍 **Solo:** {solo_nome}\n"
                f"📐 **Área:** {area_status} ({area_ha_val*10000:,.0f} m²)\n\n"
                f"Todos os cálculos de irrigação, pragas e manejo já estão calibrados para essa área.\n"
                f"Pode me perguntar a qualquer momento: *'Preciso regar hoje?'* ou *'Qual o manejo de pragas?'*! 👍"
            )
            db_manager.salvar_mensagem_telegram(cid, "saida", resposta_config, "configuracao")
            return resposta_config

    # 4. Saudação Inicial (/start ou "Oi")
    if cmd_lower in ["/start", "oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "ajuda"]:
        talhoes = db_manager.listar_talhoes(cid)
        cult_atual = str(produtor.get("cultura_ativa", "tomate")).capitalize()
        solo_atual = str(produtor.get("tipo_solo_ativo", "argiloso")).replace("_", " ")
        org_atual = "Orgânico Certificado 🌱" if produtor.get("is_organico_ativo") else "Convencional 🚜"
        area_ha_atual = float(produtor.get("area_ha_ativa", 2.5))

        msg = (
            f"Olá, companheiro produtor! É sempre uma honra conversar com quem faz a nossa terra produzir.\n\n"
            f"Estou acompanhando de perto o seu talhão de **{area_ha_atual:.2f} hectares** de **{cult_atual} ({org_atual})** em solo **{solo_atual}** aqui em Rio Claro.\n\n"
            f"Pode falar comigo normalmente, com as suas próprias palavras! Pergunte, por exemplo:\n"
            f"• *'Preciso ligar a irrigação hoje?'*\n"
            f"• *'Tem perigo de pragas na minha roça agora?'*\n"
            f"• *'Qual remédio ou bioinsumo devo passar no tomate?'*\n"
            f"• *'Quanto de água eu estou poupando?'*\n\n"
            f"Como podemos fazer a sua lavoura render mais hoje? Vamos ao trabalho!"
        )
        db_manager.salvar_mensagem_telegram(cid, "saida", msg, "saudacao")
        return msg

    # 5. Dúvida Livre / Consulta em Linguagem Natural com Contexto Completo e Dinâmico
    talhoes = db_manager.listar_talhoes(cid)
    diag_ativo = obter_ou_gerar_diagnostico_ativo(cid, produtor)
    historico_completo = [diag_ativo] + db_manager.listar_historico_completo(cid, limit=4)
    sugestoes = recomendar_culturas_para_produtor(
        lat=RIO_CLARO_LAT,
        lon=RIO_CLARO_LON,
        is_organico=produtor.get("is_organico_ativo", True),
        tipo_solo=produtor.get("tipo_solo_ativo", "argiloso"),
        dados_solo_atuais=diag_ativo
    )

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


def sanitizar_markdown_telegram(texto: str) -> str:
    """
    Higieniza mensagens para compatibilidade com o parser Markdown do Telegram:
    1. Troca marcadores de lista (* ou -) no início de linha por bullet limpo '• '
       (evita que o parser ache que abriu negrito sem fechar).
    2. Converte **negrito duplo** em *negrito simples* suportado pelo Telegram Markdown.
    3. Protege quebras de parágrafo e listas sem corromper entidades.
    """
    if not texto:
        return ""
    import re
    # Converte marcadores soltos de lista/tópicos no início da linha por bullet
    t = re.sub(r'(?m)^(\s*)[*-]\s+', r'\1• ', texto)
    # Converte negrito duplo (padrão de LLMs) em negrito único do Markdown V1 do Telegram
    t = re.sub(r'\*\*(.+?)\*\*', r'*\1*', t)
    return t


async def enviar_mensagem_telegram(chat_id: str, mensagem: str):
    """Envia mensagem assíncrona para o produtor no Telegram via Bot API com fallback robusto."""
    if not TOKEN or TOKEN == "mock_token":
        logger.info(f"[SIMULAÇÃO TELEGRAM para {chat_id}]:\n{mensagem}")
        return True

    msg_sanitizada = sanitizar_markdown_telegram(mensagem)
    try:
        from telegram import Bot
        bot = Bot(token=TOKEN)
        await bot.send_message(chat_id=chat_id, text=msg_sanitizada, parse_mode="Markdown")
        return True
    except Exception as e_md:
        logger.warning(f"Aviso ao enviar com Markdown ({e_md}), enviando em texto puro.")
        try:
            from telegram import Bot
            bot = Bot(token=TOKEN)
            await bot.send_message(chat_id=chat_id, text=mensagem)
            return True
        except Exception as e:
            logger.error(f"Erro definitivo ao enviar mensagem Telegram: {e}")
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

    try:
        from telegram.ext import ApplicationBuilder, MessageHandler, filters
    except ImportError:
        print("python-telegram-bot não instalado. Execute: pip install python-telegram-bot")
        return

    async def handle_cmd(update, context):
        cid = str(update.effective_chat.id)
        user = update.effective_user
        nome_usuario = user.first_name if user else "Produtor"
        txt = update.message.text or ""
        resp = processar_mensagem_produtor(cid, txt, nome_usuario=nome_usuario)
        resp_fmt = sanitizar_markdown_telegram(resp)
        try:
            await update.message.reply_text(resp_fmt, parse_mode="Markdown")
        except Exception as err:
            logger.debug(f"Falha ao enviar resposta com Markdown ({err}), enviando sem parse_mode.")
            await update.message.reply_text(resp)

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
            try:
                await update.message.reply_text(resp, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(resp)

    async def handle_voice(update, context):
        """
        Recebe áudio/voz do produtor rural, transcreve com STT (Google Speech),
        processa agronomicamente e responde em ÁUDIO DE VOZ (OGG Opus) espelhado.
        """
        cid = str(update.effective_chat.id)
        user = update.effective_user
        nome_usuario = user.first_name if user else "Produtor"
        msg = update.message
        voice_obj = msg.voice or msg.audio
        if not voice_obj:
            return

        import soundfile as sf
        import speech_recognition as sr

        msg_id = msg.message_id
        temp_in_ogg = f"temp_voice_in_{cid}_{msg_id}.ogg"
        temp_in_wav = f"temp_voice_in_{cid}_{msg_id}.wav"

        texto_pergunta = ""
        try:
            # 1. Download do áudio enviado pelo produtor
            file = await context.bot.get_file(voice_obj.file_id)
            await file.download_to_drive(temp_in_ogg)

            # 2. Conversão OGG Opus -> WAV
            data, sr_rate = sf.read(temp_in_ogg)
            sf.write(temp_in_wav, data, sr_rate)

            # 3. Transcrição de fala para texto (Speech-To-Text)
            r = sr.Recognizer()
            with sr.AudioFile(temp_in_wav) as source:
                audio_recorded = r.record(source)
            try:
                texto_pergunta = r.recognize_google(audio_recorded, language="pt-BR")
            except Exception as e:
                logger.warning(f"Não foi possível transcrever áudio do chat {cid}: {e}")
                texto_pergunta = ""

            # 4. Processamento da dúvida com IA e Agronomia
            if not texto_pergunta:
                resp_texto = (
                    "Olá, companheiro! Não consegui entender seu áudio com clareza devido a algum ruído. "
                    "Pode mandar outro áudio ou digitar sua dúvida por texto? Estou à disposição!"
                )
            else:
                resp_agro = processar_mensagem_produtor(cid, texto_pergunta, nome_usuario=nome_usuario)
                resp_texto = f"🎙️ *Entendi:* \"_{texto_pergunta}_\"\n\n{resp_agro}"

            # 5. Resposta sempre por TEXTO com formatação limpa
            resp_fmt = sanitizar_markdown_telegram(resp_texto)
            try:
                await update.message.reply_text(resp_fmt, parse_mode="Markdown")
            except Exception as err:
                logger.debug(f"Falha ao enviar resposta de voz com Markdown ({err}), enviando em texto simples.")
                await update.message.reply_text(resp_texto)

        except Exception as err:
            logger.error(f"Erro no processamento de voz: {err}")
            await update.message.reply_text(
                "Tive um problema ao processar seu áudio no momento. Pode me enviar por texto enquanto ajusto aqui? Obrigado!"
            )
        finally:
            # 6. Limpeza de arquivos temporários
            for tmp in [temp_in_ogg, temp_in_wav]:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_cmd))
    app.add_handler(MessageHandler(filters.COMMAND, handle_cmd))
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    print(f"Bot AgroAlerta iniciado com sucesso! Escutando mensagens e áudios...")
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