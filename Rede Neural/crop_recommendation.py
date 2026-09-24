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
CULTURAS_REGIONAL_DATABASE = {
    "tomate": {
        "nome_comum": "Tomate de Mesa / Rasteiro",
        "aptidao_rio_claro": "Excelente em Latossolos e solos bem drenados",
        "janelas_plantio_meses": [3, 4, 5, 6, 7],  # Outono/Inverno (evita excesso de chuva que apodrece frutos)
        "meses_pico_preco_ceagesp": [6, 7, 11, 12],  # Junho/Julho (frio reduz oferta) e fim de ano
        "ciclo_dias": 90,
        "como_plantar": {
            "espacamento": "1,00 m entre linhas x 0,50 m entre plantas (tutorado) ou 1,20 m x 0,30 m (rasteiro)",
            "profundidade_muda": "Enterrar até o primeiro par de folhas verdadeiras",
            "adubacao_base_organica": "20 a 30 t/ha de Composto Orgânico curtido + 250 g/cova de Fosfato Natural Reativo + Cinzas de biomassa",
            "adubacao_base_convencional": "400 a 600 kg/ha de NPK 04-14-08 com micronutrientes (Boro e Zinco)",
            "cuidados_solo": "pH ideal entre 6,0 e 6,8. Solo sensível a encharcamento e compactação."
        },
        "vantagem_organica": "Preço de venda até 40% a 80% superior em feiras e varejo especializado de SP."
    },
    "alface": {
        "nome_comum": "Alface (Crespa / Americana)",
        "aptidao_rio_claro": "Muito alta em estufas ou campo aberto irrigado",
        "janelas_plantio_meses": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Ano todo com proteção de tela/estufa no verão
        "meses_pico_preco_ceagesp": [1, 2, 12],  # Verão (chuvas fortes destroem lavouras em campo aberto, preço sobe)
        "ciclo_dias": 45,
        "como_plantar": {
            "espacamento": "0,25 m x 0,25 m em canteiros elevados (20 a 25 cm de altura)",
            "profundidade_muda": "Nível do torrão da muda com a superfície do solo",
            "adubacao_base_organica": "15 t/ha de Húmus de Minhoca ou Esterco bovino estabilizado + 100 g/m² de Farinha de Ossos",
            "adubacao_base_convencional": "250 a 350 kg/ha de NPK 04-14-08 + adubação nitrogenada parcelada",
            "cuidados_solo": "Raiz rasa e muito sensível à salinidade e compactação. Exige umidade uniforme sem encharcar."
        },
        "vantagem_organica": "Giro rápido de caixa (40 a 50 dias) e altíssima demanda orgânica nos centros urbanos próximos."
    },
    "citros": {
        "nome_comum": "Citros (Laranja Pera / Valência / Limão Tahiti)",
        "aptidao_rio_claro": "Excelente. Polo citrícola tradicional da região de Limeira/Araras/Rio Claro",
        "janelas_plantio_meses": [9, 10, 11, 12, 1, 2],  # Período de chuvas (primavera/verão)
        "meses_pico_preco_ceagesp": [8, 9, 10, 1],  # Entressafra e meses mais quentes (demanda de suco)
        "ciclo_dias": 730,  # Perene, início de produção comercial aos 2-3 anos
        "como_plantar": {
            "espacamento": "6,0 m entre linhas x 3,0 m a 4,0 m entre plantas (densidade de 400 a 550 plantas/ha)",
            "profundidade_muda": "Berço de 60x60x60 cm, colo da muda 5 cm acima do nível do solo",
            "adubacao_base_organica": "15 kg de composto orgânico por cova + 1 kg de calcário dolomítico + 500 g de termofosfato",
            "adubacao_base_convencional": "300 g de superfosfato simples + 100 g de FTE BR-12 por cova",
            "cuidados_solo": "Exige solo profundo e descompactado (>1 metro). Tolerante à seca moderada após pegamento."
        },
        "vantagem_organica": "Valorização crescente da indústria de sucos orgânicos certificados para exportação e mercado interno."
    },
    "milho": {
        "nome_comum": "Milho (Grão ou Verde)",
        "aptidao_rio_claro": "Alta em Latossolos Vermelhos",
        "janelas_plantio_meses": [10, 11, 12, 1, 2],  # Safra principal (out/dez) e Safrinha (jan/fev)
        "meses_pico_preco_ceagesp": [5, 6, 7],  # Milho verde valorizado nas festas juninas e entressafra
        "ciclo_dias": 120,
        "como_plantar": {
            "espacamento": "0,80 m a 0,90 m entre linhas x 5 a 6 plantas por metro linear",
            "profundidade_muda": "3 a 5 cm para a semente",
            "adubacao_base_organica": "8 t/ha de cama de aviário curtida + 300 kg/ha de pó de rocha remineralizador",
            "adubacao_base_convencional": "350 kg/ha de NPK 08-28-16 + cobertura com Ureia ou Sulfato de Amônio aos 30 dias",
            "cuidados_solo": "Alta demanda por Nitrogênio e tolerância média a compactação."
        },
        "vantagem_organica": "Excelente para rotação de culturas, produção de palhada (*mulching*) e quebra de ciclo de pragas."
    },
    "cana": {
        "nome_comum": "Cana-de-açúcar",
        "aptidao_rio_claro": "Vocação histórica regional (usinas da microrregião de Piracicaba e Rio Claro)",
        "janelas_plantio_meses": [2, 3, 4, 8, 9, 10],  # Cana de ano-e-meio (fev-mar) ou cana de ano (set-out)
        "meses_pico_preco_ceagesp": [7, 8, 9],  # Pico de safra sucroalcooleira e ATR
        "ciclo_dias": 360,
        "como_plantar": {
            "espacamento": "1,40 m a 1,50 m entre sulcos de plantio",
            "profundidade_muda": "Sulcos de 25 a 30 cm de profundidade, com 12 a 15 gemas por metro",
            "adubacao_base_organica": "Aplicação de torta de filtro de usina (40 t/ha) + cinzas de bagaço + vinhaça localizada",
            "adubacao_base_convencional": "500 kg/ha de NPK 05-25-25 no fundo do sulco",
            "cuidados_solo": "Demanda subsolagem profunda para quebra de pé de arado e calagem para V% > 60%."
        },
        "vantagem_organica": "Alta liquidez para açúcar mascavo, cachaça artesanal e etanol orgânico de alto valor agregado."
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
    Gera a recomendação agronômica e mercadológica com base no calendário de Rio Claro,
    preços históricos e condições do solo.
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

        # 1. Aptidão pelo calendário ZARC regional
        na_janela_ideal = mes_atual in info["janelas_plantio_meses"]
        if na_janela_ideal:
            score_viabilidade += 30.0
            motivos.append(f"Janela climática ideal para Rio Claro em {nome_mes}")
        else:
            # Penalidade se fora de época
            score_viabilidade -= 25.0
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

        # 3. Bonificação para cultivo orgânico
        if is_organico:
            if chave_cultura in ["alface", "tomate", "citros"]:
                score_viabilidade += 10.0
                motivos.append(f"Alta rentabilidade em canal orgânico: {info['vantagem_organica']}")

        # 4. Análise de restrições do solo atual (se fornecido)
        if dados_solo_atuais:
            ph = dados_solo_atuais.get("pH", 6.5)
            comp = dados_solo_atuais.get("compactacao_solo_kPa", 1500.0)
            if ph < 5.5 and chave_cultura in ["tomate", "alface"]:
                score_viabilidade -= 10.0
                motivos.append("pH atual do solo está ácido. Exige calagem prévia de correção.")
            if comp > 2000.0 and chave_cultura in ["citros", "cana"]:
                motivos.append("Solo compactado. Recomenda-se subsolagem antes do plantio.")

        score_viabilidade = max(10.0, min(100.0, score_viabilidade))

        regime_label = "organica" if is_organico else "convencional"
        guia_plantio = info["como_plantar"]

        recomendacoes.append({
            "cultura_chave": chave_cultura,
            "nome": info["nome_comum"],
            "score_viabilidade": round(score_viabilidade, 1),
            "recomendado_agora": na_janela_ideal,
            "mes_ideal_plantio": [meses_nomes[m] for m in info["janelas_plantio_meses"]],
            "previsao_colheita": meses_nomes[mes_colheita],
            "justificativas": motivos,
            "como_plantar": {
                "espacamento": guia_plantio["espacamento"],
                "profundidade": guia_plantio["profundidade_muda"],
                "adubacao_recomendada": guia_plantio[f"adubacao_base_{regime_label}"],
                "cuidados_essenciais": guia_plantio["cuidados_solo"]
            }
        })

    # Ordena por score decrescente
    recomendacoes.sort(key=lambda x: x["score_viabilidade"], reverse=True)

    return {
        "regiao": "Rio Claro - SP e Microrregião Central",
        "mes_referencia": nome_mes,
        "regime_cultivo": "Orgânico Certificado" if is_organico else "Convencional",
        "top_recomendacoes": recomendacoes[:3],
        "todas_opcoes": recomendacoes
    }
