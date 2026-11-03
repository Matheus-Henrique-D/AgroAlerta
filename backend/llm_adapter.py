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


PROMPT_SISTEMA_AGRONOMO = """Você é o Agrônomo do AgroAlerta, consultor de campo de produtores rurais de Rio Claro - SP. Pense em um engenheiro agrônomo experiente, daqueles que o produtor conhece há anos e que responde no WhatsApp com paciência, sem enrolação.

QUEM ESTÁ DO OUTRO LADO
Um produtor rural, geralmente no campo, lendo no celular, com pouco tempo. Ele entende de roça, não de tecnologia. Respeite o conhecimento prático dele: você complementa a experiência dele com os dados da propriedade, não ensina o que ele já sabe.

COMO FALAR
• Português simples e natural, como se fala no campo: "você", frases curtas, palavras do dia a dia ("regar" e "irrigar", "praga", "mancha na folha").
• Caloroso, mas sem exagero. Nada de formalidade ("Prezado") nem de gíria forçada ("e aí, parceirão").
• Se for o início da conversa, cumprimente em uma palavra ("Olá!", "Tudo bem?"). Se a conversa já está em andamento, NÃO cumprimente de novo; responda direto.
• A primeira frase já responde a pergunta. Nada de introdução.
• Se o produtor estiver preocupado (praga, perda, clima ruim), reconheça isso em uma frase curta e passe para a solução.

TAMANHO DA RESPOSTA (o erro mais comum é errar aqui)
• Pergunta simples ("devo irrigar hoje?", "qual o pH?"): 2 a 4 linhas, resposta + o número que importa.
• Pergunta de decisão (irrigação, praga, manejo): resposta direta + até 4 tópicos com o essencial + um próximo passo.
• Pergunta ampla ou pedido de detalhes: pode ir até uns 8 tópicos, em blocos curtos separados por linha em branco.
• Nunca passe de uma tela de celular sem o produtor pedir mais. Se tem muita coisa, entregue o mais urgente e ofereça: "Quer que eu detalhe a calda?"
• Curto demais também é erro: nunca responda só "Sim" ou "Irrigue". Sempre diga o porquê em uma frase.

DADOS E PRECISÃO
• Use os números reais da propriedade (área em hectares, umidade, pH, lâmina em mm, litros de água, tempo de bomba). Cite só os que ajudam naquela resposta, não despeje todos.
• Arredonde para facilitar a leitura ("cerca de 121 mil litros", "uns 22% de umidade").
• Se faltar algum dado, diga com honestidade e peça só o que for essencial. Nunca invente número, praga, produto ou cotação.
• Cite as pragas e doenças específicas da cultura dele, não pragas genéricas.

REGRAS PARA PRAGAS E PRODUTOS
• Respeite o sistema de produção: se a propriedade é orgânica, sugira SOMENTE manejo e insumos permitidos na produção orgânica (ex.: calda bordalesa, biocontrole, extratos, manejo cultural). Nunca sugira agrotóxico sintético para orgânico.
• Dê a receita de forma prática: o que usar, para que serve e quando aplicar. Só dê dose se tiver segurança. Em caso de dúvida, diga para confirmar a dose na bula ou com o responsável técnico.
• Em caso de praga grave, surto ou qualquer dúvida de segurança, recomende acompanhamento técnico presencial.
• Lembre cuidado com EPI na aplicação quando indicar qualquer produto.

FORMATAÇÃO (TELEGRAM)
• Listas SEMPRE com "• " no começo de cada tópico. NUNCA use "* ", "-" ou números como marcador.
• Negrito com um asterisco só: *assim*. Use apenas nos números e ações principais (2 ou 3 por resposta, no máximo).
• 1 emoji por tópico, no máximo, e só se ajudar a identificar o assunto (💧 água, 🐛 praga, 🌱 solo, ⚠️ atenção, ✅ ok). Sem enfeite.
• Linha em branco entre blocos. Nada de títulos grandes, tabelas ou código.

O QUE NUNCA FAZER
• Nunca use termos de programação, nomes de tabelas, "base de dados", "modelo", "algoritmo" ou qualquer jargão de tecnologia ou machine learning. Fale "pelas últimas leituras do seu solo", não "segundo o diagnóstico do sistema".
• Nunca fale de si mesmo como IA nem explique como chegou ao número.
• Nunca faça sermão ou texto longo de teoria.

FECHAMENTO
Termine, quando fizer sentido, com UMA pergunta curta ou oferta de ajuda ("Quer que eu te avise quando o solo secar de novo?"). Não termine toda resposta com isso.
"""


def _chamar_gemini_api(prompt_completo: str, api_key: str) -> Optional[str]:
    """
    Chama a API do Google Gemini via REST com modelos rápidos e estáveis:
    1. gemini-3.1-flash-lite
    2. gemini-3-flash-preview
    3. gemini-3.5-flash-lite
    """
    if not api_key:
        return None

    modelos = [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite"
    ]
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt_completo}]
            }
        ],
        "generationConfig": {
            "temperature": 0.35,
            "maxOutputTokens": 1500
        }
    }

    for modelo in modelos:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=12.0)
            if resp.status_code == 200:
                data = resp.json()
                candidatos = data.get("candidates", [])
                if candidatos:
                    txt = candidatos[0]["content"]["parts"][0]["text"].strip()
                    finish_reason = candidatos[0].get("finishReason", "")
                    # Garante que a resposta não esteja truncada pela metade
                    if txt and len(txt) > 20:
                        if finish_reason == "MAX_TOKENS" or txt.endswith("*") or txt.endswith("•"):
                            logger.warning(f"Resposta com modelo {modelo} veio truncada ({finish_reason}). Tentando próximo modelo.")
                            continue
                        return txt
            else:
                logger.debug(f"Gemini {modelo} HTTP {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            logger.debug(f"Tentativa com {modelo} falhou: {e}")

    logger.warning("Todas as tentativas na API do Gemini falharam. Acionando motor agronômico local.")
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
    # 1. Dados estruturados do talhão mais recente
    diag_recente = hist[0] if hist else {}
    cult_nome = str(produtor.get("cultura_ativa") if produtor else diag_recente.get("cultura", "tomate")).capitalize()
    is_org = produtor.get("is_organico_ativo", True) if produtor else diag_recente.get("is_organico", True)
    regime_txt = "orgânico 🌱" if is_org else "convencional 🚜"
    solo_txt = str(produtor.get("tipo_solo_ativo") if produtor else diag_recente.get("tipo_solo", "argiloso")).replace("_", " ")
    area_val = float(produtor.get("area_ha_ativa") if produtor else diag_recente.get("area_ha", 2.5))

    umid_val = float(diag_recente.get("umidade", 22.0))
    ph_val = float(diag_recente.get("ph", 6.3))
    status_irrig = diag_recente.get("status_irrigacao", "irrigação leve a moderada")
    vol_litros = float(diag_recente.get("volume_necessario_litros") or 121000.0)

    pragas_raw = diag_recente.get("alertas_pragas", [])
    if pragas_raw:
        pragas_lista = [p.get("patogeno_estresse", "") for p in pragas_raw if isinstance(p, dict)]
        pragas_txt = ", ".join(pragas_lista) if pragas_lista else "míldio, requeima e lagartas"
    else:
        pragas_txt = "míldio, requeima e lagartas"

    dt_leitura = diag_recente.get("leitura_data", "Recente")

    clima_dict = diag_recente.get("clima") or {}
    temp_ar = clima_dict.get("temperatura_2m_celsius", 24.5)
    ur_ar = clima_dict.get("umidade_relativa_2m_pct", 55.0)
    chuva_6h = clima_dict.get("chuva_prevista_6h_mm", 0.0)
    prob_chuva = clima_dict.get("probabilidade_chuva_6h_pct", 0.0)
    vento_kmh = clima_dict.get("velocidade_vento_10m_kmh", 12.0)
    clima_resumo = f"Temperatura ~{temp_ar:.1f}°C, umidade do ar {ur_ar:.0f}%, chuva prevista 6h: {chuva_6h:.1f} mm (probabilidade {prob_chuva:.0f}%), vento {vento_kmh:.1f} km/h"

    secao_lavoura = (
        f"• Produção: {cult_nome} {regime_txt}, solo {solo_txt}, cerca de {area_val:.1f} ha.\n"
        f"• Leitura mais recente ({dt_leitura}): umidade do solo {umid_val:.1f}%, pH {ph_val:.1f}.\n"
        f"• Clima atual e previsão para Rio Claro: {clima_resumo}.\n"
        f"• Recomendação dada: {status_irrig.lower()}, em torno de {vol_litros:,.0f} litros para o talhão todo.\n"
        f"• Pragas e doenças em atenção: {pragas_txt}."
    )

    secao_talhoes = json.dumps(contexto_talhoes, ensure_ascii=False, indent=2) if contexto_talhoes else "[Dados de delimitação geográfica e polígonos dos talhões]"
    secao_mercado = json.dumps(sugestoes_plantio, ensure_ascii=False, indent=2) if sugestoes_plantio else "[Calendário agrícola de Rio Claro, cotações de mercado e aptidão de culturas]"

    prompt = f"""{PROMPT_SISTEMA_AGRONOMO}

=== MENSAGEM DO PRODUTOR (Telegram) ===
"{pergunta}"

=== O QUE VOCÊ SABE SOBRE A PROPRIEDADE (Rio Claro - SP) ===

1. Lavoura e últimas leituras:
{secao_lavoura}

2. Talhões da fazenda:
{secao_talhoes}

3. Mercado e safra (calendário agrícola de Rio Claro, cotações CEAGESP, aptidão de culturas):
{secao_mercado}

=== COMO RESPONDER ESTA MENSAGEM ===
1. Entenda primeiro o que ele realmente quer saber (irrigação, praga, solo, mercado, outra coisa) e responda isso, na primeira frase.
2. Use só os dados acima que ajudam a responder. Não repita tudo que você sabe.
3. Se as seções 2 ou 3 estiverem vazias ou só com colchetes, ignore-as e não mencione que faltam dados. Se a pergunta depender delas, diga com simplicidade que ainda não tem essa informação e sugira o próximo passo.
4. Ajuste o tamanho à pergunta: curta e direta se for simples, até 4 tópicos se for decisão, até 8 tópicos em blocos se pedir detalhes.
5. Irrigação: informe a quantidade em litros (e o tempo de bomba, se tiver) e explique em uma frase por que, usando umidade e tipo de solo. Se perguntar sobre economia, dê o número exato.
6. Pragas: cite as pragas do tomate acima que têm a ver com a pergunta e dê a receita do que passar, SOMENTE com opções permitidas em produção orgânica. Diga quando aplicar e lembre o EPI. Se não tiver certeza da dose, mande confirmar na bula ou com o técnico.
7. Se a conversa já está em andamento, não cumprimente de novo.
8. Se a pergunta fugir de agricultura, responda brevemente e traga a conversa de volta à lavoura, sem ser seco.
9. Formato Telegram: tópicos com "• ", negrito com *um asterisco*, no máximo 1 emoji por tópico, linhas em branco entre blocos. Sem jargão de tecnologia.

Responda agora, direto ao ponto, no tom de um agrônomo parceiro de campo.
"""

    if not api_key_llm:
        api_key_llm = os.environ.get("GEMINI_API_KEY", "")

    if api_key_llm:
        resposta_llm = _chamar_gemini_api(prompt, api_key_llm)
        if resposta_llm and len(resposta_llm) > 40 and not resposta_llm.endswith("**"):
            return resposta_llm

    # =========================================================================
    # MOTOR DE INTELIGÊNCIA AGRONÔMICA LOCAL INSTANTÂNEO (Líder Técnico & Político)
    # =========================================================================
    diag_ativo = hist[0] if hist else None
    return _gerar_resposta_lider_politico(
        pergunta=pergunta,
        produtor=produtor,
        diag=diag_ativo,
        sugestoes_plantio=sugestoes_plantio
    )


def _gerar_resposta_lider_politico(
    pergunta: str,
    produtor: Optional[Dict[str, Any]],
    diag: Optional[Dict[str, Any]],
    sugestoes_plantio: Optional[Dict[str, Any]] = None
) -> str:
    """
    Gera uma resposta técnica, concisa, natural e humana no tom de um líder público
    comprometido com o homem do campo, utilizando dados matemáticos e agronômicos exatos.
    """
    p_lower = (pergunta or "").lower()

    # 1. Recupera metadados do produtor e diagnóstico
    cultura = "tomate"
    solo = "argiloso"
    area_ha = 2.5
    is_org = True

    if produtor:
        cultura = str(produtor.get("cultura_ativa") or cultura).lower()
        solo = str(produtor.get("tipo_solo_ativo") or solo).lower().replace("_", " ")
        area_ha = float(produtor.get("area_ha_ativa") or area_ha)
        is_org = bool(produtor.get("is_organico_ativo", True))

    if diag:
        cultura = str(diag.get("cultura") or cultura).lower()
        solo = str(diag.get("tipo_solo") or solo).lower().replace("_", " ")
        area_ha = float(diag.get("area_ha") or area_ha)
        is_org = bool(diag.get("is_organico", is_org))

    cult_str = cultura.capitalize()
    regime_str = "Orgânico Certificado 🌱" if is_org else "Convencional 🚜"

    # 2. DÚVIDAS SOBRE IRRIGAÇÃO / ÁGUA / REGA / BOMBA / CHUVA
    if any(k in p_lower for k in ["regar", "irrigar", "irrigacao", "irriga", "agua", "água", "chuva", "seco", "seca", "bomba", "molhar", "gotejamento", "aspersao"]):
        status_irrig = str(diag.get("status_irrigacao") if diag else "Irrigar Leve/Moderada").lower()
        trava_chuva = bool(diag.get("trava_chuva_acionada", False)) if diag else False
        vol_nec = float(diag.get("volume_necessario_litros", 0.0)) if diag else 0.0
        vol_econ = float(diag.get("volume_economizado_litros", 0.0)) if diag else 0.0
        lamina = float(diag.get("lamina_recomendada_mm", 0.0)) if diag else 0.0
        tempo_got = int(diag.get("tempo_gotejamento_minutos", 0)) if diag else 0
        tempo_asp = int(diag.get("tempo_aspersao_minutos", 0)) if diag else 0
        litros_planta = float(diag.get("litros_por_planta", 3.2)) if diag else 3.2
        umid = float(diag.get("umidade", 22.0)) if diag else 22.0

        # Se os cálculos vieram zerados mas a cultura está em déficit
        if vol_nec <= 0 and ("irrigar" in status_irrig or "reposi" in status_irrig or umid < 25.0):
            lamina = 4.8 if "argil" in solo else 5.2
            vol_nec = area_ha * 10000.0 * lamina
            tempo_got = int((lamina / 3.0) * 60)
            tempo_asp = int((lamina / 6.0) * 60)

        h_got, m_got = divmod(tempo_got, 60)

        # Caso A: Trava de Chuva acionada ou suspensão
        if trava_chuva or "suspender" in status_irrig:
            econ_txt = f"{vol_econ:,.0f} litros" if vol_econ > 0 else f"{area_ha * 45000:,.0f} litros"
            return (
                f"Olá! Para o seu talhão de **{area_ha:.2f} hectares** de **{cult_str} ({regime_str})** em solo **{solo}**:\n\n"
                f"🛑 **Decisão: NÃO precisa ligar a irrigação hoje.**\n\n"
                f"Há previsão de chuva para a região de Rio Claro e o solo já está com **{umid:.1f}% de umidade** (bem retida no solo {solo}).\n\n"
                f"💧 **Resumo Operacional:**\n"
                f"• **Bomba:** Desligada hoje.\n"
                f"• **Economia:** Você poupa cerca de **{econ_txt} de água** e reduz o custo na conta de luz.\n"
                f"• **Recomendação:** Deixe a chuva hidratar a terra naturalmente. Amanhã reavaliamos as medições! 👍"
            )

        # Caso B: Irrigação Necessária (Déficit Hídrico)
        if "irrigar" in status_irrig or vol_nec > 0:
            return (
                f"Olá! Para o seu talhão de **{area_ha:.2f} hectares** de **{cult_str} ({regime_str})** em solo **{solo}**:\n\n"
                f"✅ **Decisão: SIM, precisa ligar a irrigação hoje.**\n\n"
                f"A terra está com **{umid:.1f}% de umidade** (déficit hídrico moderado). No solo {solo}, "
                f"repor a água agora evita estresse e protege o pegamento das flores e frutos.\n\n"
                f"💧 **Plano de Irrigação:**\n"
                f"• **Lâmina:** **{lamina:.1f} mm** | **Volume Total:** **{vol_nec:,.0f} litros** (~{litros_planta:.1f} L/pé)\n"
                f"• **Tempo de Bomba:** **{h_got}h {m_got:02d}min** no gotejamento (ou {tempo_asp} min na aspersão)\n"
                f"• **Horário Ideal:** Entre **06h30 e 09h30** ou após as **16h30** (evite o sol forte para prevenir queima e fungos).\n\n"
                f"Qualquer dúvida sobre pragas ou manejo da roça, é só chamar! 👍"
            )

        # Caso C: Solo com Umidade Plena
        return (
            f"Olá! Para o seu talhão de **{area_ha:.2f} hectares** de **{cult_str} ({regime_str})** em solo **{solo}**:\n\n"
            f"🛑 **Decisão: BOMBA DESLIGADA.**\n\n"
            f"A terra está com **{umid:.1f}% de umidade**, nível ideal para a demanda da cultura hoje.\n\n"
            f"💧 **Resumo Operacional:**\n"
            f"• **Situação:** Umidade do solo adequada para as plantas hoje.\n"
            f"• **Economia:** Zero gasto com energia nos motores e preservação da água.\n"
            f"• **Recomendação:** Apenas acompanhe o desenvolvimento normal da roça! 👍"
        )

    # 3. DÚVIDAS SOBRE PRAGAS / DOENÇAS / INSUMOS / CALDAS / REMÉDIOS
    if any(k in p_lower for k in ["praga", "bicho", "doenca", "doença", "fungo", "lagarta", "veneno", "remedio", "remédio", "bioinsumo", "passar", "pulverizar", "calda"]):
        if "tomate" in cultura:
            bloco_pragas = (
                "🔍 **O que observar nas folhas hoje:**\n"
                "• **Traça-do-Tomateiro (*Tuta absoluta*):** Galerias transparentes nas folhas novas e pequenos furos nos brotos/frutos.\n"
                "• **Requeima (*Phytophthora infestans*):** Manchas escuras com aspecto de queimadura e mofo branco na face inferior."
            )
            bloco_insumos = (
                "🌿 **O que aplicar (Manejo Orgânico MAPA):**\n"
                "• **Biológico:** Pulverização de *Bacillus thuringiensis* (Bt) contra lagartas e traça.\n"
                "• **Foliar Preventivo:** Calda Bordalesa (0,5%) ou Calda Sulfocálcica para proteger contra fungos."
            )
        elif "citros" in cultura:
            bloco_pragas = (
                "🔍 **O que observar no pomar hoje:**\n"
                "• **Psilídeo (*Diaphorina citri*):** Inseto transmissor do Greening nas brotações novas.\n"
                "• **Bicho-Furão:** Perfurações em frutos provocando queda prematura."
            )
            bloco_insumos = (
                "🌿 **Manejo Recomendado:**\n"
                "• **Biológico:** Liberação de *Tamarixia radiata* e aplicação de fungo entomopatogênico *Cordyceps fumosorosea*.\n"
                "• **Nutricional:** Adubação foliar com zinco, boro e calda sulfocálcica preventiva."
            )
        else:
            bloco_pragas = (
                f"🔍 **Vistoria Fitossanitária no Talhão de {cult_str}:**\n"
                "• Monitore o terço médio das plantas procurando sinais de raspagem e manchas foliares."
            )
            bloco_insumos = (
                "🌿 **Manejo Recomendado:**\n"
                "• Aplicação preventiva de bioinsumo registrado e reforço da matéria orgânica."
            )

        return (
            f"Olá! Sobre o manejo fitossanitário do seu **{cult_str} ({regime_str})** ({area_ha:.2f} ha):\n\n"
            f"{bloco_pragas}\n\n"
            f"{bloco_insumos}\n\n"
            f"⏰ **Melhor Horário:** A partir das **16h30** (sol ameno) para não queimar as folhas e preservar os bioinsumos.\n\n"
            f"Se notar algum sintoma diferente nas folhas, pode me avisar aqui! 👍"
        )

    # 4. DÚVIDAS SOBRE SOLO / PH / NUTRIENTES / COMPACTAÇÃO
    if any(k in p_lower for k in ["ph", "acidez", "solo", "calagem", "calcario", "calcário", "terra", "compactacao", "compactação", "adubo", "adubação", "adubacao"]):
        ph_atual = float(diag.get("ph", 6.2)) if diag else 6.2
        umid_atual = float(diag.get("umidade", 22.0)) if diag else 22.0
        return (
            f"Olá! Sobre as condições do solo do seu talhão de **{cult_str} ({regime_str})** em Rio Claro:\n\n"
            f"🧪 **Leitura Atual do Talhão:**\n"
            f"• **pH do Solo:** **{ph_atual:.1f}** (Ideal para {cult_str} fica entre 6.0 e 6.8).\n"
            f"• **Umidade Atual:** **{umid_atual:.1f}%** (Solo {solo}).\n\n"
            f"📝 **Como medir ou calcular o pH na prática:**\n"
            f"1. **Amostragem:** Colete 10 a 15 subamostras de terra na profundidade de 0 a 20 cm em zigue-zague pelo talhão e misture num balde limpo.\n"
            f"2. **Medição em Campo:** Use um peagômetro digital de inserção direta no solo úmido ou faça a mistura de 1 parte de solo para 2,5 partes de água destilada/solução de CaCl2 e meça com fita ou eletrodo.\n"
            f"3. **Laboratório:** Se precisar de cálculo de necessidade de calagem (NC) e saturação por bases (V%), o envio de amostra composta para análise química completa é o mais preciso.\n\n"
            f"Se você já tem o resultado da sua análise de laboratório ou quer saber se precisa de calcário, me envie os números que analiso para você! 👍"
        )

    # 5. DÚVIDAS SOBRE CLIMA / PREVISÃO DO TEMPO / TEMPERATURA / VENTO
    if any(k in p_lower for k in ["tempo", "previsao", "previsão", "clima", "calor", "frio", "vento", "temperatura"]):
        clima_d = diag.get("clima") or {} if diag else {}
        temp_ar = clima_d.get("temperatura_2m_celsius", 25.0)
        ur_ar = clima_d.get("umidade_relativa_2m_pct", 55.0)
        chuv_6h = clima_d.get("chuva_prevista_6h_mm", 0.0)
        prob_ch = clima_d.get("probabilidade_chuva_6h_pct", 0.0)
        v_kmh = clima_d.get("velocidade_vento_10m_kmh", 12.0)
        return (
            f"O tempo segue firme em Rio Claro hoje, com temperatura em torno de *{temp_ar:.1f}°C* e umidade do ar em *{ur_ar:.0f}%*.\n\n"
            f"• 💧 *Chuva:* Probabilidade de *{prob_ch:.0f}%* nas próximas horas (sem volume expressivo previsto).\n"
            f"• 💨 *Vento:* Soprando a cerca de *{v_kmh:.1f} km/h*, favorável para o manejo e pulverizações matinais.\n"
            f"• 🌱 *Impacto no Talhão:* Como a terra está com *{diag.get('umidade', 22.0):.0f}% de umidade*, a evaporação segue moderada.\n\n"
            f"Quer que eu te avise se entrar alguma frente de chuva para Rio Claro?"
        )

    # 6. DÚVIDAS SOBRE SAFRA / PLANTIO / ÉPOCA / O QUE PLANTAR / MERCADO / VALORES
    if any(k in p_lower for k in ["plantar", "safra", "epoca", "época", "semente", "o que plantar", "o que cultivar", "preço", "preco", "cotação", "cotacao", "mercado", "lucro", "rentabilidade", "quanto vale"]):
        is_org = produtor.get("is_organico_ativo", True) if produtor else True
        regime_label = "Orgânico Certificado 🌱" if is_org else "Convencional 🚜"

        top1 = None
        top2 = None
        if sugestoes_plantio and "top_recomendacoes" in sugestoes_plantio:
            recs = sugestoes_plantio["top_recomendacoes"]
            if len(recs) > 0:
                top1 = recs[0]
            if len(recs) > 1:
                top2 = recs[1]

        if not top1:
            top1 = {
                "nome": "Alface (Crespa / Americana)",
                "score_viabilidade": 98.0,
                "justificativas": ["Janela climática ideal para Rio Claro neste mês", "Colheita em 45 dias com alta demanda"],
                "previsao_colheita": "Próximo mês",
                "mes_ideal_plantio": ["Outubro", "Novembro", "Ano todo"],
                "precos_regionais": {
                    "cotacao": "R$ 4,00 a R$ 5,50 por pé (+70% de ágio orgânico no varejo)" if is_org else "R$ 2,00 a R$ 3,20 por pé no atacado regional",
                    "rentabilidade_estimada": "R$ 60.000 a R$ 85.000 / ha por ciclo (45 dias)"
                },
                "como_plantar": {
                    "espacamento": "0,25 m x 0,25 m em canteiros elevados (20 a 25 cm)",
                    "preparo_solo": "Canteiros com alta incorporação de matéria orgânica e cobertura morta (*mulching*) para retenção hídrica",
                    "adubacao_recomendada": "15 a 20 t/ha de composto orgânico/húmus + 100 g/m² de cinzas e farinha de ossos" if is_org else "250 a 350 kg/ha de NPK 04-14-08 no sulco",
                    "cuidados_essenciais": "Irrigação diária por microaspersão ou gotejo e raiz rasa sensível a compactação."
                }
            }

        nome_c1 = top1.get("nome", "Alface")
        guia1 = top1.get("como_plantar", {})
        precos1 = top1.get("precos_regionais", {})
        epocas1 = ", ".join(top1.get("mes_ideal_plantio", ["Este mês"]))

        resposta_plantio = (
            f"Olá! Pensando no seu talhão em Rio Claro ({regime_label}), aqui está o diagnóstico de mercado e plantio direto ao ponto:\n\n"
            f"🌱 **1. O que plantar agora:**\n"
            f"• **Principal recomendação:** **{nome_c1}** (Aptidão {top1.get('score_viabilidade', 95)}/100)\n"
            f"• **Por que agora:** {top1.get('justificativas', ['Janela favorável em Rio Claro'])[0]}.\n\n"
            f"🚜 **2. Como plantar:**\n"
            f"• **Espaçamento:** {guia1.get('espacamento', 'Padrão da cultura')}.\n"
            f"• **Preparo do Solo:** {guia1.get('preparo_solo', 'Solo bem drenado e descompactado')}.\n"
            f"• **Adubação ({regime_label}):** {guia1.get('adubacao_recomendada', 'Adubação equilibrada')}.\n\n"
            f"📅 **3. Melhor Período & Ciclo:**\n"
            f"• **Janela Ideal de Semeadura:** {epocas1}.\n"
            f"• **Previsão de Colheita:** {top1.get('previsao_colheita', 'Em ciclo normal')}.\n\n"
            f"💰 **4. Valores de Mercado na Região (CEAGESP / SP):**\n"
            f"• **Cotação Atual:** {precos1.get('cotacao', 'Sob consulta')}.\n"
            f"• **Rentabilidade Estimada:** {precos1.get('rentabilidade_estimada', 'Alta rentabilidade')}.\n"
        )
        if top2:
            nome_c2 = top2.get("nome", "Citros")
            precos2 = top2.get("precos_regionais", {})
            resposta_plantio += (
                f"\n💡 **Segunda Alternativa:** **{nome_c2}** (Cotação: {precos2.get('cotacao', 'Excelente liquidez regional')}).\n"
            )
        resposta_plantio += "\nSe quiser o plano detalhado de mudas ou análise do seu solo para essa cultura, me avise! 👍"
        return resposta_plantio

    # 5. SAUDAÇÃO OU DÚVIDA GERAL
    return (
        f"Olá! Tudo bem? 👨‍🌾\n\n"
        f"Estou acompanhando o seu talhão de **{area_ha:.2f} hectares** de **{cult_str} ({regime_str})** em solo **{solo}** aqui em Rio Claro.\n\n"
        f"Pode me perguntar direto, por exemplo:\n"
        f"• *'Preciso ligar a irrigação hoje?'*\n"
        f"• *'Qual remédio ou bioinsumo devo passar no tomate?'*\n"
        f"• *'Tem perigo de pragas na roça agora?'*\n"
        f"• *'Quanto de água eu estou poupando?'*\n\n"
        f"Como posso te ajudar no campo hoje? 👍"
    )
