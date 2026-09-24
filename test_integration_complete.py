"""
test_integration_complete.py
============================
Bateria de testes automatizados de integração do AgroAlerta:
1. Geoprocessamento Point-in-Polygon (3 e 4 vértices) e tolerância de GPS.
2. Persistência de Talhões e Leituras no Banco de Dados.
3. Inferência da Rede Neural Multitarefa para Citros Orgânicos em Rio Claro - SP.
4. Motor de Recomendação de Culturas e Sazonalidade (ZARC/CEAGESP).
5. Bot do Telegram e Adaptador LLM para resposta humanizada ao produtor.
"""

import sys
import os
import unittest

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Setup de paths
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Rede Neural"))

from geo_engine import ponto_no_poligono, calcular_centroide_e_area_hectares, identificar_talhao
from db_manager import db_manager
from crop_recommendation import recomendar_culturas_para_produtor
from bot import processar_mensagem_produtor, notificar_analise_leitura
from agro_system import predict_agro_system


class TestAgroAlertaIntegracao(unittest.TestCase):

    def setUp(self):
        # Polígono de teste representando um talhão retangular em Rio Claro - SP
        # Coordenadas aproximadas em torno de (-22.4114, -47.5614)
        self.vertices_quadrado = [
            {"lat": -22.4100, "lon": -47.5630},  # Vértice 1
            {"lat": -22.4100, "lon": -47.5600},  # Vértice 2
            {"lat": -22.4130, "lon": -47.5600},  # Vértice 3
            {"lat": -22.4130, "lon": -47.5630}   # Vértice 4
        ]

        # Triângulo
        self.vertices_triangulo = [
            {"lat": -22.4200, "lon": -47.5700},
            {"lat": -22.4200, "lon": -47.5650},
            {"lat": -22.4250, "lon": -47.5675}
        ]

    def test_01_geoprocessamento_point_in_polygon(self):
        """Valida se o algoritmo Ray-Casting detecta pontos internos e externos com precisão."""
        # Ponto central dentro do quadrado
        ponto_dentro = (-22.4115, -47.5615)
        self.assertTrue(ponto_no_poligono(ponto_dentro[0], ponto_dentro[1], self.vertices_quadrado))

        # Ponto claramente fora do quadrado
        ponto_fora = (-22.4050, -47.5500)
        self.assertFalse(ponto_no_poligono(ponto_fora[0], ponto_fora[1], self.vertices_quadrado, margem_metros=0.0))

        # Ponto dentro do triângulo
        ponto_triangulo = (-22.4210, -47.5675)
        self.assertTrue(ponto_no_poligono(ponto_triangulo[0], ponto_triangulo[1], self.vertices_triangulo))

        # Cálculo de área
        centroide, area_ha = calcular_centroide_e_area_hectares(self.vertices_quadrado)
        self.assertGreater(area_ha, 5.0)  # Aproximadamente 10 ha
        print(f"-> Teste Geo: Área calculada = {area_ha:.2f} ha, Centroide = {centroide}")

    def test_02_banco_dados_talhao_e_identificacao(self):
        """Valida o cadastro de talhão com 4 vértices e busca automática por coordenada."""
        talhao_salvo = db_manager.salvar_talhao(
            chat_id="7858612258",
            nome_talhao="Talhão Citros Orgânico - Gleba A",
            poligono_tipo="quadrilatero",
            vertices=self.vertices_quadrado,
            cultura="citros",
            is_organico=True,
            tipo_solo="latossolo_vermelho"
        )
        self.assertIsNotNone(talhao_salvo["id"])

        # Identifica por coordenada interna
        talhoes = db_manager.listar_talhoes("7858612258")
        detectado = identificar_talhao(-22.4115, -47.5615, talhoes)
        self.assertIsNotNone(detectado)
        self.assertEqual(detectado["cultura"], "citros")
        self.assertTrue(detectado["is_organico"])
        print(f"-> Teste Banco: Talhão detectado com sucesso: {detectado['nome_talhao']}")

    def test_03_inferencia_rede_neural_citros_organico(self):
        """Valida a Rede Neural com a nova cultura Citros sob manejo Orgânico em Rio Claro."""
        sensor_leitura = {
            "pH": 6.3,
            "teor_umildade_%": 24.5,
            "nitrogenio_N_ppm": 3.8,
            "fosforo_P_ppm": 3.5,
            "potassio_K_ppm": 2.8,
            "condutividade_eletrica_dS_m": 1.05,
            "compactacao_solo_kPa": 1400.0,
            "tipo_solo": "latossolo_vermelho"
        }

        relatorio = predict_agro_system(
            lat=-22.4114,
            lon=-47.5614,
            cultura="citros",
            is_organico=True,
            dados_arduino_dict=sensor_leitura
        )

        self.assertIn("decisao_irrigacao", relatorio)
        self.assertIn("alertas_risco_patogenos", relatorio)
        self.assertIn("Cultivo Orgânico", relatorio["meta"]["regime_cultivo"])
        print(f"-> Teste IA: Status Irrigação = {relatorio['decisao_irrigacao']['status']} | Volume = {relatorio['decisao_irrigacao']['volume_agua_recomendado_mm']} mm")

    def test_04_recomendacao_safra_rio_claro(self):
        """Valida o motor de recomendação ZARC/Preços CEAGESP."""
        rec = recomendar_culturas_para_produtor(
            lat=-22.4114,
            lon=-47.5614,
            mes_atual=9,  # Setembro
            is_organico=True
        )
        self.assertGreater(len(rec["top_recomendacoes"]), 0)
        primeira_opcao = rec["top_recomendacoes"][0]
        self.assertIn("nome", primeira_opcao)
        self.assertIn("como_plantar", primeira_opcao)
        print(f"-> Teste Safra: 1ª Recomendação de Setembro = {primeira_opcao['nome']} (Score: {primeira_opcao['score_viabilidade']})")

    def test_05_bot_telegram_e_llm(self):
        """Valida a interação em linguagem natural com o bot e respostas orientativas."""
        resp_plantar = processar_mensagem_produtor("7858612258", "/plantar")
        self.assertIn("Planejamento de Safra", resp_plantar)

        resp_duvida = processar_mensagem_produtor("7858612258", "Preciso regar meu talhão hoje?")
        self.assertTrue(len(resp_duvida) > 20)
        print(f"-> Teste Bot Telegram: Resposta gerada com sucesso:\n{resp_duvida[:150]}...")


if __name__ == "__main__":
    unittest.main()
