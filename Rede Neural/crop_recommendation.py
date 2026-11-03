"""
crop_recommendation.py
======================
Motor de Recomendação Inteligente de Culturas e Calendário de Plantio.
Combina:
1. Zoneamento Agrícola de Risco Climático (ZARC) calibrado para Rio Claro - SP e microrregião.
2. Sazonalidade histórica de preços e demanda comercial (CEAGESP / CEPEA - SP).
3. Aptidão físico-química do solo da gleba e regime de manejo (Orgânico vs Convencional).
4. Prescrição agronômica completa: "O que plantar", "Como plantar" e "Melhor período".
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


# Base de Conhecimento Edafoclimático e Comercial para o Centro-Leste Paulista (Rio Claro/Piracicaba/Limeira)
# Base de Conhecimento Edafoclimático e Comercial para o Centro-Leste Paulista (Rio Claro/Piracicaba/Limeira)
CULTURAS_REGIONAL_DATABASE = {
    "alface": {
        "nome_comum": "Alface (Crespa / Americana)",
        "aptidao_rio_claro": "Muito alta em estufas ou campo aberto irrigado",
        "janelas_plantio_meses": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Ano todo
        "meses_pico_preco_ceagesp": [1, 2, 12, 10],  # Verão e primavera
        "ciclo_dias": 45,
        "precos_mercado": {
            "convencional": "R$ 2,00 a R$ 3,20 por pé no atacado regional",
            "organico": "R$ 4,00 a R$ 5,50 por pé (+70% de ágio em feiras e empórios)",
            "rentabilidade_estimada": "R$ 60.000 a R$ 85.000 / ha por ciclo (giro em 45 dias)"
        },
        "como_plantar": {
            "espacamento": "0,25 m x 0,25 m em canteiros elevados (20 a 25 cm de altura)",
            "profundidade_muda": "Nível do torrão da muda alinhado com a superfície do solo",
            "preparo_solo_organico": "Canteiros com alta incorporação de matéria orgânica e cobertura morta (*mulching*) para reter umidade",
            "preparo_solo_convencional": "Rotocanteirador mecânico com adubação química em sulcos",
            "adubacao_base_organica": "15 a 20 t/ha de Composto Orgânico / Húmus + 100 g/m² de Farinha de Ossos e Cinzas vegetais",
            "adubacao_base_convencional": "250 a 350 kg/ha de NPK 04-14-08 com cobertura nitrogenada parcelada",
            "cuidados_solo": "Raiz rasa e muito sensível à salinidade e compactação. Exige irrigação por microaspersão ou gotejo diário."
        },
        "vantagem_organica": "Giro de caixa ultra-rápido (45 dias) e demanda orgânica contínua nas feiras e supermercados da região."
    },
    "tomate": {
        "nome_comum": "Tomate de Mesa / Rasteiro",
        "aptidao_rio_claro": "Excelente em Latossolos e solos bem drenados",
        "janelas_plantio_meses": [3, 4, 5, 6, 7, 8],  # Outono/Inverno (evita excesso de chuva de verão)
        "meses_pico_preco_ceagesp": [6, 7, 11, 12],
        "ciclo_dias": 90,
        "precos_mercado": {
            "convencional": "R$ 4,50 a R$ 6,80 / kg no CEAGESP",
            "organico": "R$ 8,50 a R$ 12,50 / kg (+65% de ágio em redes sustentáveis)",
            "rentabilidade_estimada": "R$ 90.000 a R$ 140.000 / ha"
        },
        "como_plantar": {
            "espacamento": "1,00 m entre linhas x 0,50 m entre plantas (tutorado) ou 1,20 m x 0,30 m (rasteiro)",
            "profundidade_muda": "Enterrar até o primeiro par de folhas verdadeiras",
            "preparo_solo_organico": "Sulcamento profundo, incorporação de bokashi, pó de rocha e palhada nas entrelinhas",
            "preparo_solo_convencional": "Aração, gradagem e sulcamento com calagem prévia (V% 70%)",
            "adubacao_base_organica": "20 a 30 t/ha de Composto Orgânico curtido + 250 g/cova de Fosfato Natural Reativo (Bayóvar) + Cinzas de biomassa",
            "adubacao_base_convencional": "400 a 600 kg/ha de NPK 04-14-08 com Boro e Zinco",
            "cuidados_solo": "pH ideal entre 6,0 e 6,8. Solo sensível a encharcamento e compactação."
        },
        "vantagem_organica": "Altíssimo valor agregado; o tomate orgânico certificado tem venda garantida com prêmio de até 70%."
    },
    "citros": {
        "nome_comum": "Citros (Laranja Pera / Valência / Limão Tahiti)",
        "aptidao_rio_claro": "Excelente. Polo citrícola tradicional da região de Limeira/Araras/Rio Claro",
        "janelas_plantio_meses": [9, 10, 11, 12, 1, 2],  # Estação chuvosa (primavera/verão)
        "meses_pico_preco_ceagesp": [8, 9, 10, 1],  # Entressafra e meses de alta demanda de suco
        "ciclo_dias": 730,  # Perene
        "precos_mercado": {
            "convencional": "R$ 40,00 a R$ 54,00 por caixa de 40.8 kg",
            "organico": "R$ 68,00 a R$ 90,00 por caixa (+50% para indústrias de suco orgânico)",
            "rentabilidade_estimada": "R$ 35.000 a R$ 55.000 / ha anual após o 3º ano"
        },
        "como_plantar": {
            "espacamento": "6,0 m entre linhas x 3,5 m entre plantas (476 plantas/ha)",
            "profundidade_muda": "Covas de 60x60x60 cm, colo da muda 5 cm acima do nível do solo",
            "preparo_solo_organico": "Subsolagem profunda nas linhas de plantio, calagem dolomítica e plantio de leguminosas nas entrelinhas",
            "preparo_solo_convencional": "Preparo mecânico completo com subsolagem e calagem em área total",
            "adubacao_base_organica": "15 kg de composto orgânico por cova + 1 kg de calcário dolomítico + 500 g de Fosfato Natural",
            "adubacao_base_convencional": "300 g de superfosfato simples + 100 g de FTE BR-12 por cova",
            "cuidados_solo": "Exige solo profundo (>1m) e descompactado. Monitoramento constante do Psilídeo (HLB/Greening)."
        },
        "vantagem_organica": "Alta liquidez regional para sucos orgânicos de exportação e fornecimento para merenda escolar ecológica."
    },
    "milho": {
        "nome_comum": "Milho (Grão ou Verde)",
        "aptidao_rio_claro": "Alta em Latossolos Vermelhos",
        "janelas_plantio_meses": [10, 11, 12, 1, 2],  # Safra principal (out/dez) e Safrinha (jan/fev)
        "meses_pico_preco_ceagesp": [5, 6, 7],
        "ciclo_dias": 120,
        "precos_mercado": {
            "convencional": "R$ 62,00 a R$ 74,00 por saca de 60 kg (ou R$ 1,20/espiga verde)",
            "organico": "R$ 105,00 a R$ 130,00 por saca (ou R$ 2,50/espiga verde orgânica)",
            "rentabilidade_estimada": "R$ 12.000 a R$ 19.000 / ha"
        },
        "como_plantar": {
            "espacamento": "0,80 m a 0,90 m entre linhas x 5 plantas por metro linear",
            "profundidade_muda": "3 a 5 cm de profundidade para a semente",
            "preparo_solo_organico": "Plantio direto na palhada, inoculação de sementes com Azospirillum brasilense",
            "preparo_solo_convencional": "Plantio convencional ou direto com dessecação química",
            "adubacao_base_organica": "8 t/ha de esterco de curral/aviário curtido + 300 kg/ha de pó de rocha remineralizador",
            "adubacao_base_convencional": "350 kg/ha de NPK 08-28-16 + cobertura com Ureia aos 30 dias",
            "cuidados_solo": "Alta demanda de Nitrogênio e tolerância moderada a solos argilosos."
        },
        "vantagem_organica": "Excelente para rotação de culturas, produção de palhada (*mulching*) e quebra de patógenos."
    },
    "mandioca": {
        "nome_comum": "Mandioca de Mesa (Aipim / Macaxeira)",
        "aptidao_rio_claro": "Muito alta em solos médios e arenosos",
        "janelas_plantio_meses": [8, 9, 10, 11],  # Início das chuvas de primavera
        "meses_pico_preco_ceagesp": [4, 5, 6, 7],
        "ciclo_dias": 240,
        "precos_mercado": {
            "convencional": "R$ 1,80 a R$ 2,60 / kg na roça",
            "organico": "R$ 3,80 a R$ 5,20 / kg descascada e embalada a vácuo",
            "rentabilidade_estimada": "R$ 28.000 a R$ 42.000 / ha"
        },
        "como_plantar": {
            "espacamento": "1,00 m entre linhas x 0,60 m a 0,80 m entre plantas",
            "profundidade_muda": "Manivas de 15 a 20 cm plantadas horizontalmente a 10 cm de profundidade",
            "preparo_solo_organico": "Sulcos com adubação orgânica e torta de mamona, solo bem fofo para desenvolvimento das raízes",
            "preparo_solo_convencional": "Aração e sulcamento com adubação NPK de plantio",
            "adubacao_base_organica": "10 t/ha de esterco curtido + 200 kg/ha de cinzas vegetais ricas em Potássio",
            "adubacao_base_convencional": "300 kg/ha de NPK 04-14-08",
            "cuidados_solo": "Rústica e resistente à seca moderada; evita terrenos encharcados para não apodrecer raízes."
        },
        "vantagem_organica": "Cultura rústica com quase zero perda por pragas no regime agroecológico."
    }
}


def recomendar_culturas_para_produtor(
    lat: float = -22.41,
    lon: float = -47.56,
    mes_atual: Optional[int] = None,
    is_organico: bool = True,
    tipo_solo: str = "latossolo_vermelho",
    dados_solo_atuais: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Gera a recomendação agronômica e mercadológica completa:
    - O que plantar (cultura e aptidão)
    - Como plantar (espaçamento, preparo e adubação Orgânica vs Convencional)
    - Melhor período (janela ZARC Rio Claro e colheita no pico CEAGESP)
    - Valores regionais e diferenciais de rentabilidade.
    """
    if mes_atual is None:
        mes_atual = datetime.now().month

    meses_nomes = [
        "", "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
        "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
    ]
    nome_mes = meses_nomes[mes_atual]

    recomendacoes = []

    for chave_cultura, info in CULTURAS_REGIONAL_DATABASE.items():
        score_viabilidade = 50.0
        motivos = []

        # 1. Aptidão pelo calendário ZARC regional para Rio Claro
        na_janela_ideal = mes_atual in info["janelas_plantio_meses"]
        if na_janela_ideal:
            score_viabilidade += 30.0
            motivos.append(f"Janela climática ideal para Rio Claro em {nome_mes}")
        else:
            score_viabilidade -= 20.0
            prox_mes = min([m for m in info["janelas_plantio_meses"] if m > mes_atual], default=info["janelas_plantio_meses"][0])
            motivos.append(f"Fora da janela principal de semeadura (melhor época: {meses_nomes[prox_mes]})")

        # 2. Sazonalidade de Preços no momento da colheita estimada
        meses_ciclo = round(info["ciclo_dias"] / 30)
        mes_colheita = ((mes_atual + meses_ciclo - 1) % 12) + 1
        colhe_em_pico = mes_colheita in info["meses_pico_preco_ceagesp"]
        if colhe_em_pico:
            score_viabilidade += 20.0
            motivos.append(f"Colheita prevista para {meses_nomes[mes_colheita]} coincide com pico de preços no CEAGESP/SP")
        else:
            motivos.append(f"Colheita prevista para {meses_nomes[mes_colheita]} com cotações estáveis")

        # 3. Diferencial econômico Orgânico vs Convencional
        if is_organico:
            score_viabilidade += 10.0
            motivos.append(f"Ágio Orgânico: {info['precos_mercado']['organico']}")
        else:
            motivos.append(f"Preço de Mercado Convencional: {info['precos_mercado']['convencional']}")

        # 4. Análise dos Sensores de Solo Reais
        if dados_solo_atuais:
            ph = dados_solo_atuais.get("ph") or dados_solo_atuais.get("pH", 6.3)
            comp = dados_solo_atuais.get("compactacao") or dados_solo_atuais.get("compactacao_solo_kPa", 1500.0)
            p_ppm = dados_solo_atuais.get("fosforo", 20.0)

            if ph < 5.8:
                if is_organico:
                    motivos.append(f"Solo com pH {ph:.1f} (ácido): Recomenda-se calagem com Calcário Dolomítico ou Cinzas de biomassa.")
                else:
                    motivos.append(f"Solo com pH {ph:.1f} (ácido): Exige calagem prévia para elevar saturação por bases (V% 70%).")
            if comp > 1800.0:
                motivos.append("Solo com compactação moderada/alta: Recomenda-se descompactação/subsolagem prévia.")
            if p_ppm < 15.0:
                if is_organico:
                    motivos.append("Fósforo baixo no solo: Usar Fosfato Natural Reativo (Bayóvar) ou Farinha de Ossos.")
                else:
                    motivos.append("Fósforo baixo no solo: Reforçar adubação com Superfosfato Simples/Triplo no sulco.")

        score_viabilidade = max(10.0, min(100.0, score_viabilidade))

        regime_label = "organica" if is_organico else "convencional"
        guia_plantio = info["como_plantar"]
        precos = info["precos_mercado"]

        recomendacoes.append({
            "cultura_chave": chave_cultura,
            "nome": info["nome_comum"],
            "score_viabilidade": round(score_viabilidade, 1),
            "recomendado_agora": na_janela_ideal,
            "mes_ideal_plantio": [meses_nomes[m] for m in info["janelas_plantio_meses"]],
            "previsao_colheita": meses_nomes[mes_colheita],
            "justificativas": motivos,
            "precos_regionais": {
                "cotacao": precos["organico"] if is_organico else precos["convencional"],
                "rentabilidade_estimada": precos["rentabilidade_estimada"]
            },
            "como_plantar": {
                "espacamento": guia_plantio["espacamento"],
                "profundidade": guia_plantio["profundidade_muda"],
                "preparo_solo": guia_plantio.get(f"preparo_solo_{regime_label}", guia_plantio["cuidados_solo"]),
                "adubacao_recomendada": guia_plantio[f"adubacao_base_{regime_label}"],
                "cuidados_essenciais": guia_plantio["cuidados_solo"]
            }
        })

    # Ordena por score decrescente
    recomendacoes.sort(key=lambda x: x["score_viabilidade"], reverse=True)

    return {
        "regiao": "Rio Claro - SP e Microrregião Central",
        "mes_referencia": nome_mes,
        "regime_cultivo": "Orgânico Certificado 🌱" if is_organico else "Convencional 🚜",
        "top_recomendacoes": recomendacoes[:3],
        "todas_opcoes": recomendacoes
    }
