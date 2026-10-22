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
        self.assertIn("orientacoes_operacionais", relatorio)
        self.assertIn("pragas_especificas_cultura", relatorio)

        operacional = relatorio["orientacoes_operacionais"]
        self.assertIn("tempo_gotejamento", operacional)
        self.assertIn("melhor_horario_irrigacao", operacional)
        self.assertIn("metricas_hidricas_talhao", relatorio)
        self.assertIn("volume_economizado_litros", operacional)

        # Verifica se as pragas específicas de Citros foram mapeadas
        pragas_esp = relatorio["pragas_especificas_cultura"]
        tem_psilideo_ou_cancro = any(
            any("Psilídeo" in p["nome"] or "Cancro" in p["nome"] for p in g.get("especies_principais", []))
            for g in pragas_esp
        )
        self.assertTrue(tem_psilideo_ou_cancro, "O catálogo específico de Citros deve conter Psilídeo ou Cancro Cítrico")

        print(f"-> Teste IA Enriquecida: Status = {relatorio['decisao_irrigacao']['status']} | Rega Gotejamento = {operacional['tempo_gotejamento']}")
        print(f"   Janela de Irrigação = {operacional['melhor_horario_irrigacao']}")

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
        resp_plantar = processar_mensagem_produtor("7858612258", "o que devo plantar agora?")
        self.assertIn("Plantio para Rio Claro", resp_plantar)

        resp_duvida = processar_mensagem_produtor("7858612258", "Preciso regar meu talhão hoje?")
        self.assertTrue(len(resp_duvida) > 20)
        print(f"-> Teste Bot Telegram: Resposta gerada com sucesso:\n{resp_duvida[:150]}...")

    def test_06_sintese_prescritiva_llm(self):
        """Valida a geração da síntese prescritiva completa pelo LLM a partir da saída enriquecida."""
        from llm_adapter import gerar_texto_prescritivo_llm

        sensor_leitura = {
            "pH": 6.1,
            "teor_umildade_%": 18.0,  # umidade mais baixa para exigir irrigação
            "nitrogenio_N_ppm": 3.0,
            "fosforo_P_ppm": 2.5,
            "potassio_K_ppm": 2.0,
            "condutividade_eletrica_dS_m": 0.9,
            "compactacao_solo_kPa": 1350.0,
            "tipo_solo": "latossolo_vermelho"
        }
        relatorio = predict_agro_system(
            lat=-22.4114,
            lon=-47.5614,
            cultura="citros",
            is_organico=True,
            dados_arduino_dict=sensor_leitura
        )

        texto_prescritivo = gerar_texto_prescritivo_llm(relatorio)
        self.assertIn("O QUE FAZER NA IRRIGAÇÃO", texto_prescritivo)
        self.assertIn("O QUE VISTORIAR", texto_prescritivo)
        self.assertIn("RECEITA E MANEJO", texto_prescritivo)
        self.assertIn("Citros", texto_prescritivo)
        self.assertIn("Psilídeo", texto_prescritivo)
        print(f"\n-> Teste Síntese LLM: Texto prescritivo gerado com sucesso (Tamanho: {len(texto_prescritivo)} chars):")
        print(texto_prescritivo[:350] + "\n...")

    def test_07_persistencia_analises_ia_e_8_sensores(self):
        """Valida a persistência completa dos 8 sensores e da análise da IA no banco de dados."""
        chat_id = "7858612258"
        t = db_manager.salvar_talhao(
            chat_id=chat_id,
            nome_talhao="Talhão Citros 4.5ha",
            poligono_tipo="quadrilatero",
            vertices=self.vertices_quadrado,
            cultura="citros",
            is_organico=True,
            tipo_solo="latossolo_vermelho",
            area_ha=4.5
        )

        sensores_8 = {
            "ph": 6.3,
            "umidade": 17.5,
            "nitrogenio": 4.2,
            "fosforo": 3.1,
            "potassio": 2.8,
            "condutividade": 1.05,
            "temperatura": 25.5,
            "compactacao": 1380.0
        }
        leitura_id = db_manager.salvar_leitura_solo(
            chat_id=chat_id,
            lat=-22.4114,
            lon=-47.5614,
            talhao_id=t["id"],
            talhao_nome=t["nome_talhao"],
            dados_sensores=sensores_8,
            origem="telegram"
        )
        self.assertIsNotNone(leitura_id)

        relatorio = predict_agro_system(
            lat=-22.4114,
            lon=-47.5614,
            cultura="citros",
            is_organico=True,
            dados_arduino_dict=sensores_8,
            area_ha=4.5
        )
        from llm_adapter import gerar_texto_prescritivo_llm
        texto_llm = gerar_texto_prescritivo_llm(relatorio)

        analise_id = db_manager.salvar_analise_ia(
            leitura_id=leitura_id,
            chat_id=chat_id,
            talhao_id=t["id"],
            talhao_nome=t["nome_talhao"],
            relatorio_ia=relatorio,
            texto_prescritivo_llm=texto_llm
        )
        self.assertIsNotNone(analise_id)

        hist = db_manager.listar_historico_completo(chat_id=chat_id, limit=1)
        self.assertGreater(len(hist), 0)
        primeiro = hist[0]
        self.assertEqual(primeiro["leitura_id"], leitura_id)
        self.assertIn("volume_necessario_litros", primeiro)
        self.assertIn("texto_prescritivo_llm", primeiro)
        print(f"-> Teste Persistência IA e 8 Sensores: Análise ID = {analise_id} salva com sucesso.")
        print(f"   Lâmina = {primeiro['lamina_recomendada_mm']} mm | Volume Total = {primeiro['volume_necessario_litros']:,.0f} L")

    def test_08_telegram_linguagem_natural_configuracao(self):
        """Valida que o bot compreende linguagem natural para setar solo, cultura e orgânico."""
        chat_id = "7858612258"
        msg_config = "Olá, estou plantando tomate orgânico em solo argiloso no meu talhão de 2.5 hectares"
        resposta = processar_mensagem_produtor(chat_id, msg_config, nome_usuario="Seu Pedro")
        self.assertIn("Tomate", resposta)
        self.assertIn("Argiloso", resposta)
        self.assertIn("Orgânico", resposta)

        # Valida que os dados foram atualizados no banco
        produtor = db_manager.obter_ou_criar_produtor(chat_id)
        self.assertEqual(produtor["cultura_ativa"], "tomate")
        self.assertEqual(produtor["tipo_solo_ativo"], "argiloso")
        self.assertTrue(produtor["is_organico_ativo"])
        print(f"-> Teste Configuração Linguagem Natural Telegram: Sucesso!")
        print(f"   Produtor atualizado para: {produtor['cultura_ativa']} | Solo: {produtor['tipo_solo_ativo']} | Orgânico: {produtor['is_organico_ativo']}")


if __name__ == "__main__":
    unittest.main()
