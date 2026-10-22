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


def extrair_configuracao_produtor_linguagem_natural(texto: str) -> Dict[str, Any]:
    """
    Identifica se a mensagem do produtor no Telegram contém instruções
    de configuração da lavoura (cultura, tipo de solo e regime de cultivo).
    Exemplos:
    - 'planto citros orgânico em latossolo vermelho'
    - 'meu talhão é de milho convencional'
    - 'aqui é terra roxa e cultivo tomate orgânico em 3 hectares'
    """
    t = texto.lower()
    config = {}

    # 1. Detecção da Cultura
    mapa_culturas = {
        "citros": ["citros", "laranja", "tangerina", "limão", "limao", "pomar"],
        "tomate": ["tomate", "tomateiro"],
        "milho": ["milho", "milharal"],
        "cana": ["cana", "canavial", "cana-de-açúcar", "cana de acucar"],
        "alface": ["alface", "hortaliça", "horta"],
        "cafe": ["café", "cafe", "cafeeiro"],
        "soja": ["soja", "sojicultor"]
    }
    for cult_key, palavras in mapa_culturas.items():
        if any(p in t for p in palavras):
            config["cultura"] = cult_key
            break

    # 2. Detecção do Tipo de Solo
    if any(s in t for s in ["latossolo", "terra roxa", "vermelho", "terra vermelha"]):
        config["tipo_solo"] = "latossolo_vermelho"
    elif any(s in t for s in ["argiloso", "argila", "pesado", "terra preta"]):
        config["tipo_solo"] = "argiloso"
    elif any(s in t for s in ["arenoso", "areia", "leve"]):
        config["tipo_solo"] = "arenoso"
    elif any(s in t for s in ["solo ideal", "franco", "médio", "medio"]):
        config["tipo_solo"] = "ideal"

    # 3. Detecção do Regime de Cultivo (Orgânico vs Convencional)
    if any(r in t for r in ["orgânico", "organico", "biológico", "biologico", "sem veneno", "agroecológico", "agroecologico"]):
        config["is_organico"] = True
    elif any(r in t for r in ["convencional", "químico", "quimico", "defensivo tradicional"]):
        config["is_organico"] = False

    # 4. Detecção de Área em Hectares
    import re
    match_area = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ha|hectare|hectares)", t)
    if match_area:
        try:
            config["area_ha"] = float(match_area.group(1).replace(",", "."))
        except Exception:
            pass

    return config


def formatar_contexto_historico_para_llm(
    historico: List[Dict[str, Any]],
    produtor: Optional[Dict[str, Any]] = None
) -> str:
    """
    Transforma o histórico do Supabase em um resumo semântico e cronológico
    para que a LLM tenha memória precisa de campo.
    """
    linhas = []
    if produtor:
        org_str = "Orgânico 🌱" if produtor.get("is_organico_ativo") else "Convencional 🚜"
        linhas.append(f"• Configuração Ativa do Produtor: Cultura {str(produtor.get('cultura_ativa')).capitalize()} ({org_str}), Solo {produtor.get('tipo_solo_ativo')}, Área ~{produtor.get('area_ha_ativa', 1.0)} ha.")

    if not historico:
        linhas.append("• Ainda não há medições anteriores gravadas no banco de dados.")
        return "\n".join(linhas)

    linhas.append("• Histórico Recente de Diagnósticos e Leituras de Solo:")
    for i, h in enumerate(historico[:3], 1):
        dt = h.get("leitura_data", "Data recente")
        talhao = h.get("talhao_nome") or "Talhão Principal"
        cult = str(h.get("cultura", "lavoura")).capitalize()
        status_irrig = h.get("status_irrigacao", "Status não registrado")
        litros_nec = h.get("volume_necessario_litros", 0.0)
        litros_econ = h.get("volume_economizado_litros", 0.0)
        umid = h.get("umidade", 0.0)
        ph = h.get("ph", 6.5)

        detalhes_hidricos = ""
        if litros_nec > 0:
            detalhes_hidricos = f"Volume Recomendado: ~{litros_nec:,.0f} L"
        elif litros_econ > 0:
            detalhes_hidricos = f"Economia Real: ~{litros_econ:,.0f} L poupados"

        pragas = h.get("alertas_pragas", [])
        pragas_txt = ", ".join([p.get("patogeno_estresse", "") for p in pragas[:2] if isinstance(p, dict)]) if pragas else "Nenhum alerta crítico"

        linhas.append(f"  {i}. [{dt}] {talhao} ({cult}): Umidade={umid:.1f}%, pH={ph:.1f}. Decisão: {status_irrig}. {detalhes_hidricos}. Pragas: {pragas_txt}.")

    return "\n".join(linhas)


def responder_duvida_produtor(
    pergunta: str,
    contexto_talhoes: List[Dict[str, Any]],
    ultimas_leituras: List[Dict[str, Any]],
    sugestoes_plantio: Optional[Dict[str, Any]] = None,
    api_key_llm: Optional[str] = None,
    produtor: Optional[Dict[str, Any]] = None,
    historico_completo: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Responde perguntas livres do produtor rural pelo Telegram com base na memória
    e nos dados reais registrados no Supabase / SQLite.
    Garante linguagem 100% natural, clara, amigável e direta.
    """
    hist = historico_completo if historico_completo is not None else ultimas_leituras
    resumo_historico = formatar_contexto_historico_para_llm(hist, produtor)

    prompt = f"""{PROMPT_SISTEMA_AGRONOMO}

O produtor enviou uma mensagem em linguagem natural no Telegram:
"{pergunta}"

Memória e Contexto Factual da Propriedade (Rio Claro - SP):
1. Configuração e Histórico de Campo:
{resumo_historico}

2. Talhões Mapeados na Fazenda:
{json.dumps(contexto_talhoes, ensure_ascii=False, indent=2) if contexto_talhoes else "Nenhum talhão formalmente delimitado"}

3. Recomendações de Mercado e Safra (ZARC / CEAGESP):
{json.dumps(sugestoes_plantio, ensure_ascii=False, indent=2) if sugestoes_plantio else "Não consultado"}

Diretrizes de Resposta:
- Seja extremamente claro, acolhedor e direto ao ponto.
- Responda como um engenheiro agrônomo parceiro de campo no WhatsApp/Telegram.
- Se a pergunta for sobre irrigação, cite os litros de água ou a economia exata do talhão dele.
- Se for sobre pragas, cite as pragas específicas da cultura dele e a receita do que passar.
- NUNCA use termos de programação, nomes de tabelas ou jargões técnicos de machine learning.
"""

    if not api_key_llm:
        api_key_llm = os.environ.get("GEMINI_API_KEY", "")

    if api_key_llm:
        resposta_llm = _chamar_gemini_api(prompt, api_key_llm)
        if resposta_llm:
            return resposta_llm

    # =========================================================================
    # FALLBACK LOCAL ESTRUTURADO INTELIGENTE (Sem Depender de API Externa)
    # =========================================================================
    p_lower = pergunta.lower()

    # 1. Dúvidas sobre Irrigação / Água / Rega
    if any(palavra in p_lower for palavra in ["regar", "irrigar", "agua", "chuva", "seco", "bomba", "molhar"]):
        if hist:
            ult = hist[0]
            talhao = ult.get("talhao_nome") or "seu talhão"
            cultura = str(ult.get("cultura") or (produtor.get("cultura_ativa") if produtor else "cultura")).capitalize()
            status_irrig = ult.get("status_irrigacao", "")
            vol_nec = ult.get("volume_necessario_litros", 0.0)
            vol_econ = ult.get("volume_economizado_litros", 0.0)
            tempo_got = ult.get("tempo_gotejamento_minutos", 0)
            tempo_asp = ult.get("tempo_aspersao_minutos", 0)
            trava = ult.get("trava_chuva_acionada", False)
            umid = ult.get("umidade", 0.0)

            if trava or "suspender" in status_irrig.lower():
                return (
                    f"💧 *Situação de Irrigação — {talhao} ({cultura}):*\n\n"
                    f"• *Bomba Desligada:* Há alerta de chuva iminente para a região de Rio Claro!\n"
                    f"• *Economia de Água:* Você está poupando aproximadamente *{vol_econ:,.0f} litros* neste talhão hoje.\n"
                    f"• A umidade da terra está em {umid:.1f}%. Deixe a água da chuva hidratar o solo naturalmente."
                )
            elif vol_nec > 0:
                h_got, m_got = divmod(tempo_got, 60)
                return (
                    f"💧 *Situação de Irrigação — {talhao} ({cultura}):*\n\n"
                    f"• *Ligar Irrigação:* O solo apresentou déficit hídrico (umidade em {umid:.1f}%).\n"
                    f"• *Volume Necessário:* Aplicar cerca de *{vol_nec:,.0f} litros de água* em toda a área.\n"
                    f"• *Tempo de Bomba:* {h_got}h {m_got:02d}min no gotejamento (ou {tempo_asp} min na aspersão).\n"
                    f"• *Melhor Horário:* Início da manhã (06:00 às 08:30) ou fim da tarde para evitar perda por evaporação."
                )
            else:
                return (
                    f"💧 *Situação de Irrigação — {talhao} ({cultura}):*\n\n"
                    f"• *Bomba Desligada:* A umidade da terra está boa ({umid:.1f}%), suprindo bem as raízes.\n"
                    f"• Não há necessidade de ligar os motores hoje. Você economiza energia elétrica e conserva a água."
                )

    # 2. Dúvidas sobre Pragas / Doenças / Insumos
    if any(palavra in p_lower for palavra in ["praga", "bicho", "doença", "doenca", "fungo", "lagarta", "veneno", "remedio", "bioinsumo", "passar"]):
        if hist:
            ult = hist[0]
            talhao = ult.get("talhao_nome") or "talhão"
            cultura = str(ult.get("cultura") or (produtor.get("cultura_ativa") if produtor else "cultura")).capitalize()
            is_org = ult.get("is_organico", True)
            regime = "Orgânico 🌱" if is_org else "Convencional 🚜"
            pragas = ult.get("pragas_especificas", [])
            prescricao = ult.get("prescricao_insumos", [])

            linhas_resp = [f"🌿 *Manejo Fitossanitário — {talhao} ({cultura} - {regime}):*\n"]
            if pragas:
                linhas_resp.append("🔍 *Atenção aos seguintes sintomas nas folhas:*")
                for p in pragas[:2]:
                    if isinstance(p, dict) and "especies_principais" in p:
                        for esp in p["especies_principais"][:1]:
                            linhas_resp.append(f"• *{esp.get('nome')}:* _{esp.get('sintoma')}_")
                linhas_resp.append("")

            if prescricao:
                linhas_resp.append("📋 *O que aplicar:*")
                for acao in prescricao[:2]:
                    linhas_resp.append(f"• {acao}")
                linhas_resp.append(f"\n⏰ *Horário ideal:* Final da tarde (após as 16h30) para melhor absorção.")
            else:
                linhas_resp.append("✅ *A lavoura está com sanidade equilibrada!* Mantenha apenas a inspeção visual rotineira.")

            return "\n".join(linhas_resp)

    # 3. Dúvidas sobre O que Plantar / Safra / Época
    if any(palavra in p_lower for palavra in ["plantar", "safra", "epoca", "mes", "semente", "o que"]):
        if sugestoes_plantio and "top_recomendacoes" in sugestoes_plantio:
            top = sugestoes_plantio["top_recomendacoes"][0]
            return (
                f"🌱 *Sugestão de Plantio para Rio Claro neste mês:*\n\n"
                f"A cultura mais recomendada agora é o *{top['nome']}* (Pontuação {top['score_viabilidade']}/100).\n"
                f"• *Por que plantar agora?* {top['justificativas'][0]}.\n"
                f"• *Colheita estimada:* {top['previsao_colheita']}.\n"
                f"• *Espaçamento:* {top['como_plantar']['espacamento']}.\n"
                f"• *Adubação:* {top['como_plantar']['adubacao_recomendada']}."
            )

    # 4. Saudação ou Dúvida Geral
    nome_prod = produtor.get("nome_produtor", "amigo produtor") if produtor else "amigo produtor"
    cultura_ativa = (produtor.get("cultura_ativa") if produtor else "sua lavoura").capitalize()
    solo_ativo = (produtor.get("tipo_solo_ativo") if produtor else "latossolo vermelho").replace("_", " ")

    return (
        f"Olá, {nome_prod}! 👨‍🌾\n\n"
        f"Estou acompanhando sua plantação de *{cultura_ativa}* em *{solo_ativo}*.\n\n"
        f"Pode conversar comigo naturalmente! Pergunte, por exemplo:\n"
        f"• *'Preciso ligar a irrigação hoje?'*\n"
        f"• *'Tem perigo de pragas na minha roça?'*\n"
        f"• *'Mudei meu talhão para tomate orgânico'*\n"
        f"• *'Qual o melhor horário para pulverizar?'*\n\n"
        f"Como posso te ajudar no campo hoje?"
    )
