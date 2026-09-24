"""
llm_adapter.py
==============
Adaptador de Inteligência Artificial Generativa (LLM) para o AgroAlerta.
Humaniza diagnósticos técnicos, relatórios de irrigação e responde perguntas livres
do produtor rural via Telegram com linguagem acolhedora, clara e adaptada ao homem do campo.
Suporta Google Gemini API, OpenAI ou gerador local resiliente.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger("AgroLLM")


PROMPT_SISTEMA_AGRONOMO = """Você é o consultor agronômico digital do AgroAlerta, especialista na agricultura do estado de São Paulo (região de Rio Claro, Araras, Limeira e Piracicaba).
Sua missão é traduzir diagnósticos de solo, previsões meteorológicas e decisões de irrigação para uma conversa simples, prática, respeitosa e acolhedora com o produtor rural.

Diretrizes essenciais de comunicação:
1. Use tom coloquial e acolhedor (como um agrônomo de confiança conversando na porteira).
2. NUNCA use termos de programação ou estatística fria (nada de "modelo multitarefa", "logits", "cross-entropy", "dataset"). Diga "analisamos a umidade da terra e a previsão do tempo".
3. Destaque logo no início a ação prática: "Ligar a irrigação ou não", "Quantos milímetros / tempo aproximado", "Qual remédio caseiro ou biológico usar se for orgânico".
4. Se o produtor for ORGÂNICO, NUNCA recomende veneno químico ou fertilizante sintético. Recomende bioinsumos (Bacillus thuringiensis, calda bordalesa, sabão potássico, esterco, pó de rocha).
5. Se for convite para novo plantio, recomende com base na época certa do ano em Rio Claro e no valor de venda no mercado.
6. Responda em parágrafos curtos com emojis de plantas e clima para facilitar a leitura no celular.
"""


def _chamar_gemini_api(prompt_completo: str, api_key: str) -> Optional[str]:
    """Chama a API do Google Gemini via REST."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt_completo}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 800
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            candidatos = data.get("candidates", [])
            if candidatos:
                return candidatos[0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        logger.warning(f"Erro ao chamar API Gemini: {e}")
    return None


def formatar_relatorio_para_produtor(relatorio_ia: Dict[str, Any], api_key_llm: Optional[str] = None) -> str:
    """
    Transforma o relatório JSON da Rede Neural em uma mensagem amigável para o Telegram.
    Se houver chave de LLM disponível, usa o LLM para humanizar; caso contrário, usa o gerador nativo empático.
    """
    meta = relatorio_ia.get("meta", {})
    irrig = relatorio_ia.get("decisao_irrigacao", {})
    clima = relatorio_ia.get("analise_microclimatica_open_meteo", {})
    pragas = relatorio_ia.get("alertas_risco_patogenos", [])
    acoes = relatorio_ia.get("plano_de_acao_prescritivo", {}).get("acoes_e_insumos_recomendados", [])

    cultura = meta.get("cultura", "lavoura").capitalize()
    is_organico = "Orgânico" in meta.get("regime_cultivo", "")
    regime_str = "Cultivo Orgânico 🌱" if is_organico else "Cultivo Convencional 🚜"

    prompt_llm = f"""{PROMPT_SISTEMA_AGRONOMO}

O sistema acabou de analisar uma leitura de solo no campo. Resuma as informações abaixo em uma mensagem direta e amigável para enviar pelo Telegram ao produtor:

- Talhão / Cultura: {cultura} ({regime_str})
- Decisão sobre Irrigação: {irrig.get('status')}
- Lâmina de água necessária: {irrig.get('volume_agua_recomendado_mm', 0)} mm
- Trava de Chuva acionada?: {irrig.get('trava_seguranca_meteorologica', {}).get('acionada')} (Motivo: {irrig.get('trava_seguranca_meteorologica', {}).get('justificativa')})
- Temperatura atual: {clima.get('temperatura_2m_celsius')} °C, Umidade do Ar: {clima.get('umidade_relativa_2m_pct')}%, Chuva prevista 12h: {clima.get('chuva_prevista_12h_mm')} mm
- Alertas de Pragas/Doenças: {json.dumps(pragas, ensure_ascii=False)}
- Recomendações e Insumos: {json.dumps(acoes, ensure_ascii=False)}
"""

    if not api_key_llm:
        api_key_llm = os.environ.get("GEMINI_API_KEY", "")

    if api_key_llm:
        resposta_llm = _chamar_gemini_api(prompt_llm, api_key_llm)
        if resposta_llm:
            return resposta_llm

    # Gerador Local Nativo Empático (Fallback robusto e humanizado)
    linhas = []
    linhas.append(f"👨‍🌾 *Boletim do Campo — AgroAlerta*")
    linhas.append(f"📍 *Cultura:* {cultura} | {regime_str}\n")

    # Bloco de Irrigação
    vol_mm = irrig.get("volume_agua_recomendado_mm", 0.0)
    trava = irrig.get("trava_seguranca_meteorologica", {}).get("acionada", False)

    if trava:
        linhas.append(f"🌧️ *Irrigação: NÃO REGAR HOJE!*")
        linhas.append(f"👉 {irrig.get('trava_seguranca_meteorologica', {}).get('justificativa')}")
    elif vol_mm == 0.0:
        linhas.append(f"💧 *Irrigação: Solo bem abastecido!*")
        linhas.append("A umidade atual da terra está suficiente para as raízes. Não precisa ligar a irrigação agora para economizar água e energia.")
    else:
        linhas.append(f"🚿 *Irrigação Recomendada: {vol_mm:.1f} mm de água*")
        linhas.append(f"A terra tá pedindo uma reposição hídrica leve para a cultura do {cultura} continuar desenvolvendo bem.")

    # Bloco de Clima em Rio Claro
    temp = clima.get("temperatura_2m_celsius", "--")
    ur = clima.get("umidade_relativa_2m_pct", "--")
    chuva12 = clima.get("chuva_prevista_12h_mm", 0.0)
    linhas.append(f"\n🌤️ *Tempo na Região:* {temp}°C | Umidade do ar: {ur}% | Chuva prevista: {chuva12} mm")

    # Bloco de Pragas
    pragas_alertas = [p for p in pragas if p.get("nivel_alerta") in ["Alto", "Moderado"]]
    if pragas_alertas:
        linhas.append("\n⚠️ *Atenção fitossanitária no talhão:*")
        for p in pragas_alertas:
            linhas.append(f"• {p['patogeno_estresse']} (Risco {p['nivel_alerta']} - {p['probabilidade']}%)")
    else:
        linhas.append("\n✅ *Fitossanidade:* Nenhuma praga com risco crítico no momento.")

    # Recomendações
    if acoes:
        linhas.append("\n📋 *O que fazer agora:*")
        for acao in acoes[:3]:
            linhas.append(f"• {acao}")

    return "\n".join(linhas)


def responder_duvida_produtor(
    pergunta: str,
    contexto_talhoes: List[Dict[str, Any]],
    ultimas_leituras: List[Dict[str, Any]],
    sugestoes_plantio: Optional[Dict[str, Any]] = None,
    api_key_llm: Optional[str] = None
) -> str:
    """
    Responde perguntas livres do produtor rural pelo Telegram com base nos dados reais salvos.
    """
    prompt = f"""{PROMPT_SISTEMA_AGRONOMO}

O produtor enviou a seguinte pergunta pelo Telegram:
"{pergunta}"

Dados reais disponíveis da propriedade dele em Rio Claro - SP:
1. Talhões cadastrados:
{json.dumps(contexto_talhoes, ensure_ascii=False, indent=2)}

2. Últimas medições de solo e análises:
{json.dumps(ultimas_leituras, ensure_ascii=False, indent=2)}

3. Sugestões de Plantio e Mercado (ZARC Rio Claro / CEAGESP):
{json.dumps(sugestoes_plantio, ensure_ascii=False, indent=2) if sugestoes_plantio else "Não consultado"}

Responda diretamente à dúvida do produtor, citando os dados reais dos talhões dele se for relevante, com tom prestativo, simples e orientativo.
"""

    if not api_key_llm:
        api_key_llm = os.environ.get("GEMINI_API_KEY", "")

    if api_key_llm:
        resposta_llm = _chamar_gemini_api(prompt, api_key_llm)
        if resposta_llm:
            return resposta_llm

    # Respostas inteligentes locais baseadas em intenção
    p_lower = pergunta.lower()
    if any(palavra in p_lower for palavra in ["plantar", "safra", "epoca", "mes", "semente", "o que"]):
        if sugestoes_plantio and "top_recomendacoes" in sugestoes_plantio:
            top = sugestoes_plantio["top_recomendacoes"][0]
            resp = (
                f"🌱 *Sugestão de Plantio para Rio Claro neste mês:*\n\n"
                f"A cultura mais recomendada agora é o *{top['nome']}* (Pontuação {top['score_viabilidade']}/100).\n"
                f"• *Por que plantar agora?* {top['justificativas'][0]}.\n"
                f"• *Colheita estimada:* {top['previsao_colheita']}.\n"
                f"• *Como plantar:* Espaçamento {top['como_plantar']['espacamento']}.\n"
                f"• *Adubação:* {top['como_plantar']['adubacao_recomendada']}."
            )
            return resp

    if any(palavra in p_lower for palavra in ["regar", "irrigar", "agua", "chuva", "seco"]):
        if ultimas_leituras:
            ult = ultimas_leituras[0]
            talhao = ult.get("talhao_nome", "seu talhão")
            umid = ult.get("umidade", 0.0)
            return (
                f"💧 *Situação de Água do {talhao}:*\n\n"
                f"A umidade medida recentemente no solo foi de {umid:.1f}%.\n"
                f"Se o dia estiver quente e sem previsão de chuva forte, acompanhe o boletim diário para saber se é hora de aplicar a lâmina leve."
            )

    # Resumo geral dos talhões
    qtd_talhoes = len(contexto_talhoes)
    return (
        f"Olá, amigo produtor! 👨‍🌾\n\n"
        f"Vi sua mensagem sobre: *\"{pergunta}\"*.\n"
        f"Você tem atualmente *{qtd_talhoes} talhão(ões)* cadastrado(s) no sistema.\n"
        f"Você pode me perguntar a qualquer momento:\n"
        f"• *'Preciso regar hoje?'*\n"
        f"• *'O que é melhor plantar agora?'*\n"
        f"• *'Como está a terra do talhão 1?'*\n\n"
        f"Estou aqui para ajudar no dia a dia da roça!"
    )
