"""
server.py
=========
Servidor API REST do AgroAlerta.
Conecta o frontend mobile/localhost aos módulos de georreferenciamento,
banco de dados, Rede Neural Multitarefa e Notificações do Telegram.
"""

import sys
import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Adiciona caminhos dos módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Rede Neural"))

from db_manager import db_manager
from geo_engine import ponto_no_poligono, calcular_centroide_e_area_hectares, identificar_talhao
from crop_recommendation import recomendar_culturas_para_produtor
from bot import notificar_analise_leitura, processar_mensagem_produtor

# Importa o sistema de IA da Rede Neural
try:
    from agro_system import predict_agro_system
    HAS_AI = True
except Exception as e:
    HAS_AI = False
    print(f"Aviso: Não foi possível carregar o pipeline completo de IA: {e}")

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)


@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({"status": "online", "modulo": "AgroAlerta Backend"})


@app.route("/api/talhoes", methods=["GET", "POST"])
def gerenciar_talhoes():
    if request.method == "POST":
        dados = request.json or {}
        chat_id = str(dados.get("chat_id", "7858612258"))
        nome = dados.get("nome_talhao", "Talhão sem nome")
        ptipo = dados.get("poligono_tipo", "quadrilatero")
        vertices = dados.get("vertices", [])
        cultura = dados.get("cultura", "milho").lower()
        is_org = bool(dados.get("is_organico", False))
        tipo_solo = dados.get("tipo_solo", "latossolo_vermelho")

        _, area_ha = calcular_centroide_e_area_hectares(vertices)

        salvo = db_manager.salvar_talhao(
            chat_id=chat_id,
            nome_talhao=nome,
            poligono_tipo=ptipo,
            vertices=vertices,
            cultura=cultura,
            is_organico=is_org,
            tipo_solo=tipo_solo,
            area_ha=area_ha
        )
        return jsonify({"sucesso": True, "talhao": salvo}), 201

    chat_id = request.args.get("chat_id")
    talhoes = db_manager.listar_talhoes(chat_id)
    return jsonify(talhoes)


@app.route("/api/identificar_talhao", methods=["POST"])
def api_identificar_talhao():
    """Identifica automaticamente em qual talhão uma coordenada GPS se encontra."""
    dados = request.json or {}
    lat = float(dados.get("latitude", 0.0))
    lon = float(dados.get("longitude", 0.0))
    chat_id = dados.get("chat_id")

    talhoes = db_manager.listar_talhoes(chat_id)
    talhao_encontrado = identificar_talhao(lat, lon, talhoes)

    if talhao_encontrado:
        return jsonify({"encontrado": True, "talhao": talhao_encontrado})
    return jsonify({"encontrado": False, "mensagem": "Coordenada fora dos talhões cadastrados"})


@app.route("/api/leituras", methods=["POST"])
def registrar_leitura():
    """
    Recebe leitura com GPS, identifica talhão, executa inferência da IA e notifica o Telegram.
    """
    dados = request.json or {}
    chat_id = str(dados.get("chat_id", "7858612258"))
    lat = float(dados.get("latitude", -22.4114))
    lon = float(dados.get("longitude", -47.5614))

    # Identifica o talhão se não foi explicitamente fornecido
    talhoes = db_manager.listar_talhoes(chat_id)
    talhao = identificar_talhao(lat, lon, talhoes)

    cultura = talhao["cultura"] if talhao else dados.get("cultura", "milho")
    is_organico = talhao["is_organico"] if talhao else bool(dados.get("is_organico", False))
    talhao_id = talhao["id"] if talhao else None
    talhao_nome = talhao["nome_talhao"] if talhao else "Área Não Delimitada"

    # Salva no banco de dados
    leitura_id = db_manager.salvar_leitura_solo(
        chat_id=chat_id,
        lat=lat,
        lon=lon,
        talhao_id=talhao_id,
        talhao_nome=talhao_nome,
        dados_sensores=dados
    )

    # Executa a inferência pela Rede Neural Multitarefa
    relatorio_ia = None
    if HAS_AI:
        try:
            relatorio_ia = predict_agro_system(
                lat=lat,
                lon=lon,
                cultura=cultura,
                is_organico=is_organico,
                dados_arduino_dict=dados
            )
            # Notifica o produtor via Telegram com linguagem humanizada
            notificar_analise_leitura(chat_id, relatorio_ia)
        except Exception as e:
            print(f"Erro na inferência da IA: {e}")

    return jsonify({
        "sucesso": True,
        "leitura_id": leitura_id,
        "talhao_nome": talhao_nome,
        "cultura": cultura,
        "is_organico": is_organico,
        "relatorio_ia": relatorio_ia
    }), 201


@app.route("/api/sincronizar_lote", methods=["POST"])
def sincronizar_lote():
    """
    Recebe todo o pacote offline do celular (talhões e leituras que estavam acumulados na memória).
    """
    lote = request.json or {}
    talhoes_pendentes = lote.get("talhoes", [])
    leituras_pendentes = lote.get("leituras", [])

    talhoes_salvos = 0
    for t in talhoes_pendentes:
        try:
            _, area_ha = calcular_centroide_e_area_hectares(t.get("vertices", []))
            db_manager.salvar_talhao(
                chat_id=str(t.get("chat_id", "7858612258")),
                nome_talhao=t.get("nome_talhao", "Talhão"),
                poligono_tipo=t.get("poligono_tipo", "quadrilatero"),
                vertices=t.get("vertices", []),
                cultura=t.get("cultura", "milho"),
                is_organico=bool(t.get("is_organico", False)),
                tipo_solo=t.get("tipo_solo", "latossolo_vermelho"),
                area_ha=area_ha
            )
            talhoes_salvos += 1
        except Exception:
            pass

    leituras_salvas = 0
    for l in leituras_pendentes:
        try:
            chat_id = str(l.get("chat_id", "7858612258"))
            lat = float(l.get("latitude", -22.4114))
            lon = float(l.get("longitude", -47.5614))
            talhao = identificar_talhao(lat, lon, db_manager.listar_talhoes(chat_id))
            cultura = talhao["cultura"] if talhao else l.get("cultura", "milho")
            is_organico = talhao["is_organico"] if talhao else bool(l.get("is_organico", False))

            db_manager.salvar_leitura_solo(
                chat_id=chat_id,
                lat=lat,
                lon=lon,
                talhao_id=talhao["id"] if talhao else None,
                talhao_nome=talhao["nome_talhao"] if talhao else None,
                dados_sensores=l
            )

            if HAS_AI:
                rel = predict_agro_system(lat, lon, cultura, is_organico, l)
                notificar_analise_leitura(chat_id, rel)

            leituras_salvas += 1
        except Exception:
            pass

    return jsonify({
        "sucesso": True,
        "talhoes_processados": talhoes_salvos,
        "leituras_processadas": leituras_salvas,
        "mensagem": "Sincronização offline concluída com sucesso!"
    })


@app.route("/api/recomendacao_safra", methods=["GET"])
def api_recomendacao_safra():
    mes = request.args.get("mes", type=int)
    is_org = request.args.get("is_organico", default="true").lower() == "true"
    resultado = recomendar_culturas_para_produtor(mes_atual=mes, is_organico=is_org)
    return jsonify(resultado)


if __name__ == "__main__":
    print("Iniciando servidor REST AgroAlerta na porta 5001...")
    app.run(host="0.0.0.0", port=5001, debug=False)
