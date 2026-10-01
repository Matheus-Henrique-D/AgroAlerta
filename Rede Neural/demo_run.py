"""
demo_run.py
===========
Script de Demonstração e Homologação Completa do AgroAlerta.
Demonstra a execução da Rede Neural Multitarefa, a saída enriquecida
(pragas específicas, tempo operacional de rega) e a síntese prescritiva por LLM.
"""

import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Adiciona caminhos dos módulos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

sys.path.append(BASE_DIR)
sys.path.append(BACKEND_DIR)

from agro_system import predict_agro_system
from crop_recommendation import recomendar_culturas_para_produtor

try:
    from llm_adapter import gerar_texto_prescritivo_llm
    HAS_LLM = True
except Exception:
    HAS_LLM = False


def print_banner(titulo: str):
    print("\n" + "=" * 75)
    print(f"  {titulo}")
    print("=" * 75)


def run_demonstration():
    print_banner("🌾 DEMONSTRAÇÃO DO SISTEMA AGROALERTA — DEEP LEARNING & LLM 🌾")
    print(f"Diretório Base: {BASE_DIR}")
    print("Carregando modelo treinado e motor agronômico...")

    # =========================================================================
    # CENÁRIO 1: Citros Orgânico em Rio Claro - SP (Déficit Hídrico Real)
    # =========================================================================
    print_banner("CENÁRIO 1: CITROS ORGÂNICO EM RIO CLARO - SP (TALHÃO DE 4.5 HECTARES)")
    print("Condição: Pomar de citros em Latossolo Vermelho com solo seco (12.0% umidade) sob sol forte.")

    # Simula clima de dia seco e ensolarado para avaliar reposição hídrica
    from weather_client import weather_service
    original_get_features = weather_service.get_weather_features

    def mock_sol_seco(lat, lon):
        clima = original_get_features(lat, lon)
        clima["precipitation_probability_max_6h"] = 5.0
        clima["rain_sum_6h"] = 0.0
        clima["rain_sum_12h"] = 0.0
        clima["relative_humidity_2m"] = 42.0
        clima["temperature_2m"] = 29.5
        clima["et0_fao_sum_12h"] = 5.2
        return clima

    weather_service.get_weather_features = mock_sol_seco

    arduino_citros = {
        "tipo_solo": "latossolo_vermelho",
        "pH": 6.2,
        "teor_umildade_%": 12.0,
        "nitrogenio_N_ppm": 3.4,
        "fosforo_P_ppm": 3.1,
        "potassio_K_ppm": 2.6,
        "condutividade_eletrica_dS_m": 0.95,
        "compactacao_solo_kPa": 1420.0
    }

    area_talhao_1 = 4.5  # 4.5 hectares calculados pela delimitação GPS

    relatorio_1 = predict_agro_system(
        lat=-22.4114,
        lon=-47.5614,
        cultura="citros",
        is_organico=True,
        dados_arduino_dict=arduino_citros,
        area_ha=area_talhao_1
    )

    weather_service.get_weather_features = original_get_features

    # 1. Exibição da Saída Simples e Estruturada da Rede Neural (Sem texto narrativo longo)
    print("\n[1. SAÍDA SIMPLES DA REDE NEURAL — JSON ANALÍTICO ENXUTO]:")
    resumo_analitico_1 = {
        "talhao": {"area_ha": relatorio_1["meta"]["area_ha"], "cultura": relatorio_1["meta"]["cultura"], "regime": relatorio_1["meta"]["regime_cultivo"]},
        "decisao_irrigacao": {
            "classe_codigo": relatorio_1["decisao_irrigacao"]["classe_codigo"],
            "status": relatorio_1["decisao_irrigacao"]["status"],
            "lamina_mm": relatorio_1["decisao_irrigacao"]["volume_agua_recomendado_mm"]
        },
        "metricas_hidricas_area": {
            "volume_necessario_litros": relatorio_1["metricas_hidricas_talhao"]["volume_necessario_litros"],
            "tempo_gotejamento_minutos": relatorio_1["metricas_hidricas_talhao"]["tempo_gotejamento_minutos"],
            "tempo_aspersao_minutos": relatorio_1["metricas_hidricas_talhao"]["tempo_aspersao_minutos"],
            "litros_por_planta": relatorio_1["metricas_hidricas_talhao"]["litros_por_planta"]
        },
        "riscos_fitossanitarios": [
            {"alerta": a["patogeno_estresse"], "prob_pct": a["probabilidade"], "nivel": a["nivel_alerta"]}
            for a in relatorio_1["alertas_risco_patogenos"] if a["nivel_alerta"] in ["Alto", "Moderado"]
        ]
    }
    print(json.dumps(resumo_analitico_1, indent=2, ensure_ascii=False))

    # 2. Exibição do Texto Prescritivo Gerado Exclusivamente pelo LLM
    if HAS_LLM:
        print("\n[2. RESPOSTA EM TEXTO GERADA PELO LLM A PARTIR DA SAÍDA DA REDE]:")
        texto_llm_1 = gerar_texto_prescritivo_llm(relatorio_1)
        print(texto_llm_1)

    # =========================================================================
    # CENÁRIO 2: Tomate Orgânico com Alerta de Tempestade (TRAVA FÍSICA DE CHUVA)
    # =========================================================================
    print_banner("CENÁRIO 2: TOMATE ORGÂNICO COM ALERTA DE CHUVA (TALHÃO DE 2.3 HECTARES)")
    print("Condição: Solo pedindo água (19% umidade), porém Open-Meteo prevê temporal iminente.")

    def mock_chuva(lat, lon):
        clima = original_get_features(lat, lon)
        clima["precipitation_probability_max_6h"] = 85.0
        clima["rain_sum_6h"] = 22.0
        clima["rain_sum_12h"] = 38.0
        clima["relative_humidity_2m"] = 88.0
        clima["temperature_2m"] = 23.0
        return clima

    weather_service.get_weather_features = mock_chuva

    arduino_tomate = {
        "tipo_solo": "argiloso",
        "pH": 5.9,
        "teor_umildade_%": 19.0,
        "nitrogenio_N_ppm": 4.5,
        "fosforo_P_ppm": 4.0,
        "potassio_K_ppm": 3.2,
        "condutividade_eletrica_dS_m": 1.10,
        "compactacao_solo_kPa": 1380.0
    }

    area_talhao_2 = 2.3  # 2.3 hectares delimitados

    relatorio_2 = predict_agro_system(
        lat=-22.40,
        lon=-47.55,
        cultura="tomate",
        is_organico=True,
        dados_arduino_dict=arduino_tomate,
        area_ha=area_talhao_2
    )

    weather_service.get_weather_features = original_get_features

    # 1. Exibição da Saída Simples e Estruturada da Rede Neural (Sem texto narrativo longo)
    print("\n[1. SAÍDA SIMPLES DA REDE NEURAL — JSON ANALÍTICO ENXUTO]:")
    resumo_analitico_2 = {
        "talhao": {"area_ha": relatorio_2["meta"]["area_ha"], "cultura": relatorio_2["meta"]["cultura"], "regime": relatorio_2["meta"]["regime_cultivo"]},
        "decisao_irrigacao": {
            "classe_codigo": relatorio_2["decisao_irrigacao"]["classe_codigo"],
            "status": relatorio_2["decisao_irrigacao"]["status"],
            "lamina_mm": relatorio_2["decisao_irrigacao"]["volume_agua_recomendado_mm"],
            "trava_meteorologica_acionada": relatorio_2["decisao_irrigacao"]["trava_seguranca_meteorologica"]["acionada"]
        },
        "metricas_hidricas_area": {
            "volume_necessario_litros": relatorio_2["metricas_hidricas_talhao"]["volume_necessario_litros"],
            "volume_economizado_litros": relatorio_2["metricas_hidricas_talhao"]["volume_economizado_litros"]
        },
        "riscos_fitossanitarios": [
            {"alerta": a["patogeno_estresse"], "prob_pct": a["probabilidade"], "nivel": a["nivel_alerta"]}
            for a in relatorio_2["alertas_risco_patogenos"] if a["nivel_alerta"] in ["Alto", "Moderado"]
        ]
    }
    print(json.dumps(resumo_analitico_2, indent=2, ensure_ascii=False))

    # 2. Exibição do Texto Prescritivo Gerado Exclusivamente pelo LLM
    if HAS_LLM:
        print("\n[2. RESPOSTA EM TEXTO GERADA PELO LLM A PARTIR DA SAÍDA DA REDE]:")
        texto_llm_2 = gerar_texto_prescritivo_llm(relatorio_2)
        print(texto_llm_2)

    # =========================================================================
    # CENÁRIO 3: Recomendação Inteligente de Safra & Mercado (ZARC Rio Claro)
    # =========================================================================
    print_banner("CENÁRIO 3: PLANEJAMENTO DE SAFRA (ZARC RIO CLARO + CEAGESP)")
    print("Consulta: 'O que o produtor deve plantar neste mês para maximizar lucro e minimizar risco?'")

    rec_safra = recomendar_culturas_para_produtor(
        lat=-22.4114,
        lon=-47.5614,
        is_organico=True
    )

    print(f"\nRegião: {rec_safra['regiao']} | Mês Atual: {rec_safra['mes_referencia']}")
    print(f"Regime: {rec_safra['regime_cultivo']}\n")
    for i, item in enumerate(rec_safra["top_recomendacoes"], 1):
        print(f"{i}º Lugar: {item['nome']} (Pontuação: {item['score_viabilidade']}/100)")
        print(f"   👉 Motivo: {item['justificativas'][0]}")
        print(f"   📅 Colheita Prevista: {item['previsao_colheita']}")
        print(f"   📐 Espaçamento: {item['como_plantar']['espacamento']}")
        print(f"   🌱 Adubação de Base: {item['como_plantar']['adubacao_recomendada']}\n")

    print("=" * 75)
    print("  DEMONSTRAÇÃO CONCLUÍDA COM 100% DE SUCESSO!")
    print("=" * 75)


if __name__ == "__main__":
    run_demonstration()
