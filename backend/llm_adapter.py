"""
llm_adapter.py
==============
Adaptador de Inteligência Artificial Generativa (LLM) para o AgroAlerta.
Gera a síntese prescritiva completa de campo ("O que o produtor deve fazer hoje"),
integrando as saídas analíticas da Rede Neural, os tempos operacionais de rega,
pragas específicas da cultura e as receitas de insumos (Orgânico vs Convencional).
Suporta Google Gemini API, OpenAI ou gerador local empático nativo.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger("AgroLLM")


PROMPT_SISTEMA_AGRONOMO = """Você é o consultor agronômico digital do AgroAlerta, especialista na agricultura do estado de São Paulo (região de Rio Claro, Araras, Limeira e Piracicaba).
Sua missão é transformar diagnósticos técnicos de solo, clima e rede neural em um guia prescritivo direto, prático, acolhedor e humanizado sobre "O QUE O PRODUTOR DEVE FAZER HOJE NO TALHÃO".

Diretrizes essenciais de comunicação:
1. Use tom coloquial e acolhedor (como um agrônomo parceiro de confiança conversando na lavoura).
2. NUNCA use jargões de programação ou estatística fria (nada de "modelo multitarefa", "logits", "cross-entropy", "dataset"). Diga "analisamos a umidade da terra e a previsão do tempo".
3. Divida sua orientação em 4 blocos claros:
   - 🚀 AÇÃO IMEDIATA (Ligar ou não a irrigação, tempo exato de bomba e melhor horário)
   - 🔍 O QUE VISTORIAR NA PLANTA (Cite as pragas específicas da cultura e os sintomas nas folhas/frutos)
   - 🌿 PRESCRIÇÃO DE INSUMOS (Se for orgânico, cite estritamente bioinsumos do MAPA, caldas e horários para evitar sol)
   - 💧 SUSTENTABILIDADE (Água economizada ou hidratação necessária)
4. Use emojis pertinentes para facilitar a leitura rápida no celular e no Telegram.
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
            "temperature": 0.35,
            "maxOutputTokens": 1000
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


def gerar_texto_prescritivo_llm(relatorio_ia: Dict[str, Any], api_key_llm: Optional[str] = None) -> str:
    """
    Consome o relatório analítico enxuto da Rede Neural e gera a prescrição
    humanizada de manejo completa via LLM, integrando o tamanho real do talhão (ha),
    a quantidade exata de água necessária ou a economia em litros.
    """
    meta = relatorio_ia.get("meta", {})
    irrig = relatorio_ia.get("decisao_irrigacao", {})
    operacional = relatorio_ia.get("orientacoes_operacionais", {})
    metricas_hidricas = relatorio_ia.get("metricas_hidricas_talhao", {})
    clima = relatorio_ia.get("analise_microclimatica_open_meteo", {})
    pragas_cultura = relatorio_ia.get("pragas_especificas_cultura", [])
    plano_acao = relatorio_ia.get("plano_de_acao_prescritivo", {})

    cultura = meta.get("cultura", "lavoura").capitalize()
    regime_raw = meta.get("regime_cultivo", "").lower()
    is_organico = any(k in regime_raw for k in ["orgânic", "organic"])
    regime_str = "Cultivo Orgânico Certificado 🌱" if is_organico else "Cultivo Convencional 🚜"

    area_ha = float(meta.get("area_ha") or metricas_hidricas.get("area_ha", 1.0))
    vol_necessario = float(metricas_hidricas.get("volume_necessario_litros") or operacional.get("volume_necessario_litros", 0.0))
    vol_economizado = float(metricas_hidricas.get("volume_economizado_litros") or operacional.get("volume_economizado_litros", 0.0))
    vol_mm = float(irrig.get("volume_agua_recomendado_mm", 0.0))
    trava_chuva = irrig.get("trava_seguranca_meteorologica", {}).get("acionada", False)

    if not api_key_llm:
        api_key_llm = os.environ.get("GEMINI_API_KEY", "")

    # Monta prompt rico para o LLM
    prompt_llm = f"""{PROMPT_SISTEMA_AGRONOMO}

Dados consolidados do talhão (Rio Claro - SP):
- Cultura: {cultura} ({regime_str}) | Solo: {meta.get('tipo_solo')}
- Área Total Delimitada: {area_ha:.2f} hectares
- Decisão de Irrigação da Rede: {irrig.get('status')}
- Lâmina líquida calculada: {vol_mm:.1f} mm
- Volume Total de Água Necessário para o Talhão: {vol_necessario:,.0f} Litros
- Volume de Água Economizado se Não Regar: {vol_economizado:,.0f} Litros
- Tempo Estimado de Gotejamento: {operacional.get('tempo_gotejamento')}
- Tempo Estimado de Aspersão: {operacional.get('tempo_aspersao')}
- Litros por Planta: {operacional.get('litros_por_planta')} L
- Janela Ideal de Rega: {operacional.get('melhor_horario_irrigacao')}
- Janela Ideal de Pulverização: {operacional.get('melhor_horario_pulverizacao')}
- Trava de Chuva Iminente: {trava_chuva} (Chuva prevista 6h: {clima.get('chuva_prevista_6h_mm', 0)} mm, Prob: {clima.get('probabilidade_chuva_6h_pct', 0)}%)
- Clima Atual: {clima.get('temperatura_2m_celsius')}°C, UR: {clima.get('umidade_relativa_2m_pct')}%, Vento: {clima.get('velocidade_vento_10m_kmh')} km/h
- Pragas Específicas com Alerta: {json.dumps(pragas_cultura, ensure_ascii=False)}
- Prescrição Técnica de Insumos: {json.dumps(plano_acao.get('acoes_e_insumos_recomendados', []), ensure_ascii=False)}

Instruções:
- Explique ao produtor em bom português exatamente o que fazer hoje no campo.
- Destaque o tamanho do talhão ({area_ha:.2f} ha) e o cálculo exato em litros de água (quanto precisa aplicar ou quanto está economizando).
- Indique o tempo de motobomba e o melhor horário.
- Oriente a inspeção visual das pragas citadas e forneça as receitas de insumos permitidos para o regime ({regime_str}).
"""

    if api_key_llm:
        resposta_llm = _chamar_gemini_api(prompt_llm, api_key_llm)
        if resposta_llm:
            return resposta_llm

    # =========================================================================
    # GERADOR LOCAL NATIVO ESTRUTURADO (Fallback de Altíssima Qualidade)
    # =========================================================================
    linhas = []
    linhas.append(f"👨‍🌾 *Prescrição do Campo — AgroAlerta*")
    linhas.append(f"📍 *Talhão de {cultura}* (~{area_ha:.2f} ha) | {regime_str}\n")

    # 1. Bloco de Ação Imediata na Irrigação
    linhas.append("🚀 *1. O QUE FAZER NA IRRIGAÇÃO AGORA:*")
    if trava_chuva:
        chuva_prev = clima.get("chuva_prevista_6h_mm", 0)
        prob_chuva = clima.get("probabilidade_chuva_6h_pct", 0)
        linhas.append(f"• *BOMBA DESLIGADA:* Alerta de chuva iminente ({chuva_prev:.1f} mm com {prob_chuva:.0f}% de chance nas próximas 6h)!")
        linhas.append("  👉 A terra receberá água natural da chuva. Ligar a bomba agora causaria encharcamento e desperdício.")
        linhas.append(f"  💧 *Economia Real no Talhão:* ~{vol_economizado:,.0f} litros de água e energia poupados hoje.")
    elif vol_mm <= 0.0:
        linhas.append("• *BOMBA DESLIGADA:* A umidade da terra está no ponto ideal para as raízes.")
        linhas.append("  👉 Solo bem suprido. Não há necessidade de ligar os motores hoje.")
        linhas.append(f"  💧 *Economia Real no Talhão:* ~{vol_economizado:,.0f} litros de água conservados.")
    else:
        linhas.append(f"• *LIGAR IRRIGAÇÃO:* Aplicar lâmina de *{vol_mm:.1f} mm* de reposição hídrica.")
        linhas.append(f"  💧 *Volume Total da Área:* ~{vol_necessario:,.0f} litros de água para os {area_ha:.2f} hectares.")
        linhas.append(f"  ⏱️ *Tempo de Gotejamento:* {operacional.get('tempo_gotejamento')}")
        linhas.append(f"  ⏱️ *Tempo de Aspersão:* {operacional.get('tempo_aspersao')}")
        if operacional.get("litros_por_planta", 0) > 0:
            linhas.append(f"  🥤 *Dose:* ~{operacional.get('litros_por_planta')} litros por planta.")
        linhas.append(f"  ⏰ *Melhor Horário:* {operacional.get('melhor_horario_irrigacao')}")

    linhas.append("")

    # 2. Bloco do Que Vistoriar na Planta
    linhas.append(f"🔍 *2. O QUE VISTORIAR NO SEU {cultura.upper()}:*")
    pragas_relevantes = [g for g in pragas_cultura if g.get("relevante")]
    if pragas_relevantes:
        for grupo in pragas_relevantes:
            linhas.append(f"• *Alerta de {grupo['grupo']} (Risco {grupo['nivel_risco']} - {grupo['probabilidade_pct']}%):*")
            for p in grupo.get("especies_principais", [])[:2]:
                linhas.append(f"  ⚠️ *{p['nome']}:* Olhar nas folhas para ver se há: _{p['sintoma']}_.")
    else:
        linhas.append("• ✅ *Lavoura com sanidade estável!* Faça apenas a vistoria visual de rotina nas bordas do talhão.")

    linhas.append("")

    # 3. Bloco de Prescrição e Receita de Insumos
    acoes = plano_acao.get("acoes_e_insumos_recomendados", [])
    linhas.append(f"🌿 *3. RECEITA E MANEJO PRÁTICO ({'ORGÂNICO' if is_organico else 'CONVENCIONAL'}):*")
    linhas.append(f"⏰ *Janela de Aplicação:* {operacional.get('melhor_horario_pulverizacao')}")
    if acoes:
        for acao in acoes[:3]:
            linhas.append(f"• {acao}")
    else:
        linhas.append("• Nenhuma aplicação química ou biológica necessária hoje.")

    # 4. Síntese Explicativa formulada pelo LLM
    linhas.append("\n💡 *Por que tomamos essa decisão:*")
    if trava_chuva:
        linhas.append(
            f"Previsão de precipitação convectiva de {clima.get('chuva_prevista_6h_mm', 0):.1f} mm "
            f"com {clima.get('probabilidade_chuva_6h_pct', 0):.0f}% de certeza meteorológica nas próximas 6 horas em Rio Claro. "
            f"Segurar a irrigação previne asfixia do sistema radicular e preserva a estrutura biológica do solo."
        )
    elif vol_mm <= 0.0:
        linhas.append(
            f"A reserva de água útil do solo ({meta.get('tipo_solo')}) está atendendo plenamente "
            f"à demanda evapotranspirativa diária da cultura ({cultura}), sem risco de estresse hídrico."
        )
    else:
        linhas.append(
            f"O déficit hídrico calculado para a área de {area_ha:.2f} ha é de {vol_mm:.1f} mm, "
            f"considerando a demanda evaporativa do ar de {clima.get('evapotranspiracao_et0_12h_mm', 0):.1f} mm "
            f"e a fase fenológica da planta."
        )

    return "\n".join(linhas)


def formatar_relatorio_para_produtor(relatorio_ia: Dict[str, Any], api_key_llm: Optional[str] = None) -> str:
    """Wrapper para compatibilidade retroativa que chama a síntese prescritiva do LLM."""
    return gerar_texto_prescritivo_llm(relatorio_ia, api_key_llm)


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

    p_lower = pergunta.lower()
    if any(palavra in p_lower for palavra in ["plantar", "safra", "epoca", "mes", "semente", "o que"]):
        if sugestoes_plantio and "top_recomendacoes" in sugestoes_plantio:
            top = sugestoes_plantio["top_recomendacoes"][0]
            return (
                f"🌱 *Sugestão de Plantio para Rio Claro neste mês:*\n\n"
                f"A cultura mais recomendada agora é o *{top['nome']}* (Pontuação {top['score_viabilidade']}/100).\n"
                f"• *Por que plantar agora?* {top['justificativas'][0]}.\n"
                f"• *Colheita estimada:* {top['previsao_colheita']}.\n"
                f"• *Como plantar:* Espaçamento {top['como_plantar']['espacamento']}.\n"
                f"• *Adubação:* {top['como_plantar']['adubacao_recomendada']}."
            )

    if any(palavra in p_lower for palavra in ["regar", "irrigar", "agua", "chuva", "seco"]):
        if ultimas_leituras:
            ult = ultimas_leituras[0]
            talhao = ult.get("talhao_nome", "seu talhão")
            umid = ult.get("umidade", 0.0)
            return (
                f"💧 *Situação de Água do {talhao}:*\n\n"
                f"A umidade medida recentemente no solo foi de {umid:.1f}%.\n"
                f"Se o dia estiver quente e sem previsão de chuva forte, consulte o boletim diário com o tempo exato de bomba."
            )

    qtd_talhoes = len(contexto_talhoes)
    return (
        f"Olá, amigo produtor! 👨‍🌾\n\n"
        f"Vi sua mensagem sobre: *\"{pergunta}\"*.\n"
        f"Você tem atualmente *{qtd_talhoes} talhão(ões)* cadastrado(s) no sistema.\n"
        f"Pode me perguntar a qualquer momento:\n"
        f"• *'Preciso regar hoje?'*\n"
        f"• *'O que é melhor plantar agora?'*\n"
        f"• *'Tem perigo de praga no meu milho?'*\n\n"
        f"Estou aqui para ajudar no dia a dia da roça!"
    )
