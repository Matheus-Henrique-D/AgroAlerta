"""
db_manager.py
=============
Gerenciador de Banco de Dados Híbrido (Supabase Cloud + Fallback Local SQLite).
Garante persistência de produtores do Telegram, talhões georreferenciados,
leituras dos 8 sensores físicos de solo e histórico analítico da Rede Neural & LLM.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests

try:
    from config import SUPABASE_URL, SUPABASE_SECRET_KEY, SUPABASE_PUBLISHABLE_KEY, DATABASE_URL
except ImportError:
    try:
        from backend.config import SUPABASE_URL, SUPABASE_SECRET_KEY, SUPABASE_PUBLISHABLE_KEY, DATABASE_URL
    except ImportError:
        SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
        SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")
        SUPABASE_PUBLISHABLE_KEY = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "")
        DATABASE_URL = os.environ.get("DATABASE_URL", "")

logger = logging.getLogger("AgroDB")
LOCAL_DB_PATH = os.path.join(os.path.dirname(__file__), "agroalerta_local.db")


class AgroDatabaseManager:
    """
    Gerencia a persistência de dados do AgroAlerta com sincronização no Supabase
    e espelho offline resiliente em SQLite local.
    """

    def __init__(
        self,
        supabase_url: str = SUPABASE_URL,
        supabase_key: str = SUPABASE_SECRET_KEY or SUPABASE_PUBLISHABLE_KEY
    ):
        self.supabase_url = (supabase_url or "").rstrip("/")
        self.supabase_key = supabase_key or ""
        self._init_sqlite()

    def _get_supabase_headers(self) -> Dict[str, str]:
        """Headers de autenticação para a API REST do Supabase (PostgREST)."""
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

    def _sync_supabase_insert(self, table: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Executa insert na tabela do Supabase via REST API."""
        if not self.supabase_url or not self.supabase_key or "seu-projeto" in self.supabase_url:
            return None
        url = f"{self.supabase_url}/rest/v1/{table}"
        try:
            resp = requests.post(url, headers=self._get_supabase_headers(), json=payload, timeout=4)
            if resp.status_code in [200, 201]:
                data = resp.json()
                return data[0] if isinstance(data, list) and len(data) > 0 else None
            else:
                logger.debug(f"Supabase REST status {resp.status_code} em {table}: {resp.text[:100]}")
        except Exception as e:
            logger.debug(f"Falha de conexão com Supabase ({table}): {e}")
        return None

    def _init_sqlite(self):
        """Cria as 5 tabelas no SQLite local garantindo funcionamento offline."""
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            
            # 1. Produtores (Telegram)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS produtores (
                    chat_id TEXT PRIMARY KEY,
                    nome_produtor TEXT,
                    cultura_ativa TEXT DEFAULT 'citros',
                    tipo_solo_ativo TEXT DEFAULT 'latossolo_vermelho',
                    is_organico_ativo INTEGER DEFAULT 1,
                    area_ha_ativa REAL DEFAULT 1.0,
                    localizacao_padrao_lat REAL DEFAULT -22.4114,
                    localizacao_padrao_lon REAL DEFAULT -47.5614,
                    notificacoes_ativas INTEGER DEFAULT 1,
                    criado_em TEXT NOT NULL,
                    atualizado_em TEXT NOT NULL
                )
            """)

            # 2. Talhões
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS talhoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    nome_talhao TEXT NOT NULL,
                    poligono_tipo TEXT NOT NULL,
                    vertices TEXT NOT NULL,
                    centroide TEXT,
                    cultura TEXT NOT NULL,
                    is_organico INTEGER NOT NULL,
                    tipo_solo TEXT NOT NULL,
                    area_ha REAL DEFAULT 1.0,
                    criado_em TEXT NOT NULL
                )
            """)

            # 3. Leituras de Solo Físicas (8 Sensores Exatos)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS leituras_solo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    talhao_id INTEGER,
                    talhao_nome TEXT,
                    origem TEXT DEFAULT 'telegram',
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    ph REAL NOT NULL,
                    umidade REAL NOT NULL,
                    nitrogenio REAL NOT NULL,
                    fosforo REAL NOT NULL,
                    potassio REAL NOT NULL,
                    condutividade REAL NOT NULL,
                    temperatura REAL NOT NULL,
                    compactacao REAL NOT NULL,
                    criado_em TEXT NOT NULL
                )
            """)

            # 4. Análises da IA (Rede Neural & LLM)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analises_ia (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    leitura_id INTEGER,
                    chat_id TEXT NOT NULL,
                    talhao_id INTEGER,
                    talhao_nome TEXT,
                    cultura TEXT NOT NULL,
                    is_organico INTEGER NOT NULL,
                    tipo_solo TEXT NOT NULL,
                    area_ha REAL NOT NULL,
                    classe_irrigacao INTEGER NOT NULL,
                    status_irrigacao TEXT NOT NULL,
                    lamina_recomendada_mm REAL DEFAULT 0.0,
                    volume_necessario_litros REAL DEFAULT 0.0,
                    volume_economizado_litros REAL DEFAULT 0.0,
                    tempo_gotejamento_minutos INTEGER DEFAULT 0,
                    tempo_aspersao_minutos INTEGER DEFAULT 0,
                    litros_por_planta REAL DEFAULT 0.0,
                    janela_irrigacao_ideal TEXT,
                    janela_pulverizacao_ideal TEXT,
                    trava_chuva_acionada INTEGER DEFAULT 0,
                    chuva_prevista_mm REAL DEFAULT 0.0,
                    probabilidade_chuva_pct REAL DEFAULT 0.0,
                    alertas_pragas TEXT,
                    pragas_especificas TEXT,
                    prescricao_insumos TEXT,
                    texto_prescritivo_llm TEXT,
                    relatorio_json TEXT,
                    notificado_telegram INTEGER DEFAULT 0,
                    criado_em TEXT NOT NULL
                )
            """)

            # 5. Mensagens Telegram (Histórico Conversacional para LLM)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mensagens_telegram (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    direcao TEXT NOT NULL,
                    tipo_mensagem TEXT DEFAULT 'texto_livre',
                    mensagem_texto TEXT NOT NULL,
                    talhao_mencionado_id INTEGER,
                    analise_ia_id INTEGER,
                    criado_em TEXT NOT NULL
                )
            """)
            conn.commit()

            # Migração automática de colunas para tabelas já existentes
            self._migrar_tabelas_sqlite(cursor)
            conn.commit()

    def _migrar_tabelas_sqlite(self, cursor):
        """Garante compatibilidade adicionando colunas faltantes em bancos locais preexistentes."""
        def colunas_da_tabela(nome_tabela):
            cursor.execute(f"PRAGMA table_info({nome_tabela})")
            return [col[1] for col in cursor.fetchall()]

        cols_talhoes = colunas_da_tabela("talhoes")
        if "centroide" not in cols_talhoes:
            cursor.execute("ALTER TABLE talhoes ADD COLUMN centroide TEXT")

        cols_leituras = colunas_da_tabela("leituras_solo")
        if "origem" not in cols_leituras:
            cursor.execute("ALTER TABLE leituras_solo ADD COLUMN origem TEXT DEFAULT 'telegram'")

        cols_analises = colunas_da_tabela("analises_ia")
        colunas_necessarias = {
            "talhao_id": "INTEGER",
            "talhao_nome": "TEXT",
            "cultura": "TEXT",
            "is_organico": "INTEGER DEFAULT 1",
            "tipo_solo": "TEXT",
            "area_ha": "REAL DEFAULT 1.0",
            "lamina_recomendada_mm": "REAL DEFAULT 0.0",
            "volume_necessario_litros": "REAL DEFAULT 0.0",
            "volume_economizado_litros": "REAL DEFAULT 0.0",
            "tempo_gotejamento_minutos": "INTEGER DEFAULT 0",
            "tempo_aspersao_minutos": "INTEGER DEFAULT 0",
            "litros_por_planta": "REAL DEFAULT 0.0",
            "janela_irrigacao_ideal": "TEXT",
            "janela_pulverizacao_ideal": "TEXT",
            "trava_chuva_acionada": "INTEGER DEFAULT 0",
            "chuva_prevista_mm": "REAL DEFAULT 0.0",
            "probabilidade_chuva_pct": "REAL DEFAULT 0.0",
            "pragas_especificas": "TEXT",
            "prescricao_insumos": "TEXT",
            "texto_prescritivo_llm": "TEXT",
            "relatorio_json": "TEXT",
            "notificado_telegram": "INTEGER DEFAULT 0"
        }
        for col_nome, col_tipo in colunas_necessarias.items():
            if col_nome not in cols_analises:
                try:
                    cursor.execute(f"ALTER TABLE analises_ia ADD COLUMN {col_nome} {col_tipo}")
                except Exception:
                    pass

    # =========================================================================
    # 1. GESTÃO DE PRODUTORES (Configuração via Telegram)
    # =========================================================================

    def obter_ou_criar_produtor(self, chat_id: str, nome: Optional[str] = None) -> Dict[str, Any]:
        """Recupera ou cadastra as preferências do produtor via Telegram."""
        cid = str(chat_id)
        agora = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM produtores WHERE chat_id = ?", (cid,))
            row = cursor.fetchone()
            if row:
                prod = dict(row)
                prod["is_organico_ativo"] = bool(prod["is_organico_ativo"])
                return prod

            # Cadastra novo produtor com valores padrão
            cursor.execute("""
                INSERT INTO produtores (
                    chat_id, nome_produtor, cultura_ativa, tipo_solo_ativo,
                    is_organico_ativo, area_ha_ativa, criado_em, atualizado_em
                ) VALUES (?, ?, 'citros', 'latossolo_vermelho', 1, 1.0, ?, ?)
            """, (cid, nome or "Produtor Rural", agora, agora))
            conn.commit()

        # Replica no Supabase
        payload_cloud = {
            "chat_id": cid,
            "nome_produtor": nome or "Produtor Rural",
            "cultura_ativa": "citros",
            "tipo_solo_ativo": "latossolo_vermelho",
            "is_organico_ativo": True,
            "area_ha_ativa": 1.0
        }
        self._sync_supabase_insert("produtores", payload_cloud)

        return {
            "chat_id": cid,
            "nome_produtor": nome or "Produtor Rural",
            "cultura_ativa": "citros",
            "tipo_solo_ativo": "latossolo_vermelho",
            "is_organico_ativo": True,
            "area_ha_ativa": 1.0
        }

    def atualizar_configuracao_produtor(
        self,
        chat_id: str,
        cultura: Optional[str] = None,
        tipo_solo: Optional[str] = None,
        is_organico: Optional[bool] = None,
        area_ha: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Atualiza o tipo de solo, cultura e regime (Orgânico vs Convencional)
        definidos pelo produtor via conversa no Telegram.
        """
        cid = str(chat_id)
        self.obter_ou_criar_produtor(cid)
        agora = datetime.now(timezone.utc).isoformat()

        campos_update = []
        valores_update = []

        if cultura is not None:
            campos_update.append("cultura_ativa = ?")
            valores_update.append(cultura.lower().strip())
        if tipo_solo is not None:
            campos_update.append("tipo_solo_ativo = ?")
            valores_update.append(tipo_solo.lower().strip())
        if is_organico is not None:
            campos_update.append("is_organico_ativo = ?")
            valores_update.append(1 if is_organico else 0)
        if area_ha is not None:
            campos_update.append("area_ha_ativa = ?")
            valores_update.append(float(area_ha))

        if campos_update:
            campos_update.append("atualizado_em = ?")
            valores_update.append(agora)
            valores_update.append(cid)

            query = f"UPDATE produtores SET {', '.join(campos_update)} WHERE chat_id = ?"
            with sqlite3.connect(LOCAL_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(query, tuple(valores_update))
                conn.commit()

        return self.obter_ou_criar_produtor(cid)

    # =========================================================================
    # 2. GESTÃO DE TALHÕES
    # =========================================================================

    def salvar_talhao(
        self,
        chat_id: str,
        nome_talhao: str,
        poligono_tipo: str,
        vertices: List[Dict[str, float]],
        cultura: str,
        is_organico: bool,
        tipo_solo: str = "latossolo_vermelho",
        area_ha: float = 1.0,
        centroide: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Salva um novo talhão com cálculo de centróide e área real."""
        cid = str(chat_id)
        self.obter_ou_criar_produtor(cid)
        agora = datetime.now(timezone.utc).isoformat()
        vertices_json = json.dumps(vertices)
        centroide_json = json.dumps(centroide) if centroide else None

        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO talhoes (
                    chat_id, nome_talhao, poligono_tipo, vertices, centroide,
                    cultura, is_organico, tipo_solo, area_ha, criado_em
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cid, nome_talhao, poligono_tipo, vertices_json, centroide_json,
                cultura.lower().strip(), 1 if is_organico else 0, tipo_solo.lower().strip(),
                float(area_ha), agora
            ))
            talhao_id = cursor.lastrowid
            conn.commit()

        # Replica no Supabase
        payload_cloud = {
            "chat_id": cid,
            "nome_talhao": nome_talhao,
            "poligono_tipo": poligono_tipo,
            "vertices": vertices,
            "centroide": centroide,
            "cultura": cultura.lower().strip(),
            "is_organico": bool(is_organico),
            "tipo_solo": tipo_solo.lower().strip(),
            "area_ha": float(area_ha)
        }
        self._sync_supabase_insert("talhoes", payload_cloud)

        return {
            "id": talhao_id,
            "chat_id": cid,
            "nome_talhao": nome_talhao,
            "poligono_tipo": poligono_tipo,
            "vertices": vertices,
            "centroide": centroide,
            "cultura": cultura.lower().strip(),
            "is_organico": bool(is_organico),
            "tipo_solo": tipo_solo.lower().strip(),
            "area_ha": float(area_ha)
        }

    def listar_talhoes(self, chat_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lista os talhões do produtor."""
        talhoes = []
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if chat_id:
                cursor.execute("SELECT * FROM talhoes WHERE chat_id = ? ORDER BY id DESC", (str(chat_id),))
            else:
                cursor.execute("SELECT * FROM talhoes ORDER BY id DESC")
            rows = cursor.fetchall()
            for r in rows:
                talhoes.append({
                    "id": r["id"],
                    "chat_id": r["chat_id"],
                    "nome_talhao": r["nome_talhao"],
                    "poligono_tipo": r["poligono_tipo"],
                    "vertices": json.loads(r["vertices"]) if r["vertices"] else [],
                    "centroide": json.loads(r["centroide"]) if r["centroide"] else None,
                    "cultura": r["cultura"],
                    "is_organico": bool(r["is_organico"]),
                    "tipo_solo": r["tipo_solo"],
                    "area_ha": r["area_ha"],
                    "criado_em": r["criado_em"]
                })
        return talhoes

    # =========================================================================
    # 3. GESTÃO DE LEITURAS DE SOLO (8 Sensores Físicos Exatos)
    # =========================================================================

    def salvar_leitura_solo(
        self,
        chat_id: str,
        lat: float,
        lon: float,
        talhao_id: Optional[int],
        talhao_nome: Optional[str],
        dados_sensores: Dict[str, Any],
        origem: str = "telegram"
    ) -> int:
        """
        Registra a leitura com os 8 sensores físicos exatos do solo:
        pH, umidade, nitrogênio, fósforo, potássio, condutividade, temperatura e compactação.
        """
        cid = str(chat_id)
        self.obter_ou_criar_produtor(cid)
        agora = datetime.now(timezone.utc).isoformat()

        # Extração normalizada dos 8 sensores
        ph = float(dados_sensores.get("ph") or dados_sensores.get("pH", 6.5))
        umidade = float(dados_sensores.get("umidade") or dados_sensores.get("teor_umildade_%", 20.0))
        nitrogenio = float(dados_sensores.get("nitrogenio") or dados_sensores.get("nitrogenio_N_ppm", 4.0))
        fosforo = float(dados_sensores.get("fosforo") or dados_sensores.get("fosforo_P_ppm", 3.0))
        potassio = float(dados_sensores.get("potassio") or dados_sensores.get("potassio_K_ppm", 2.5))
        condutividade = float(dados_sensores.get("condutividade") or dados_sensores.get("condutividade_eletrica_dS_m", 1.0))
        temperatura = float(dados_sensores.get("temperatura") or dados_sensores.get("temperatura_solo_C", 25.0))
        compactacao = float(dados_sensores.get("compactacao") or dados_sensores.get("compactacao_solo_kPa", 1400.0))

        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO leituras_solo (
                    chat_id, talhao_id, talhao_nome, origem, latitude, longitude,
                    ph, umidade, nitrogenio, fosforo, potassio, condutividade, temperatura, compactacao, criado_em
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cid, talhao_id, talhao_nome, origem, float(lat), float(lon),
                ph, umidade, nitrogenio, fosforo, potassio, condutividade, temperatura, compactacao, agora
            ))
            leitura_id = cursor.lastrowid
            conn.commit()

        # Replica no Supabase
        payload_cloud = {
            "chat_id": cid,
            "talhao_id": talhao_id,
            "talhao_nome": talhao_nome,
            "origem": origem,
            "latitude": float(lat),
            "longitude": float(lon),
            "ph": ph,
            "umidade": umidade,
            "nitrogenio": nitrogenio,
            "fosforo": fosforo,
            "potassio": potassio,
            "condutividade": condutividade,
            "temperatura": temperatura,
            "compactacao": compactacao
        }
        self._sync_supabase_insert("leituras_solo", payload_cloud)

        return leitura_id

    # =========================================================================
    # 4. GESTÃO DE ANÁLISES DA IA (Rede Neural & LLM)
    # =========================================================================

    def salvar_analise_ia(
        self,
        leitura_id: int,
        chat_id: str,
        talhao_id: Optional[int],
        talhao_nome: Optional[str],
        relatorio_ia: Dict[str, Any],
        texto_prescritivo_llm: str,
        notificado_telegram: bool = True
    ) -> int:
        """
        Persiste o resultado completo da inferência da Rede Neural e a prescrição
        humanizada da LLM, garantindo histórico contínuo para o assistente.
        """
        cid = str(chat_id)
        agora = datetime.now(timezone.utc).isoformat()

        meta = relatorio_ia.get("meta", {})
        irrig = relatorio_ia.get("decisao_irrigacao", {})
        metricas = relatorio_ia.get("metricas_hidricas_talhao", {})
        clima = relatorio_ia.get("analise_microclimatica_open_meteo", {})
        alertas_pragas = relatorio_ia.get("alertas_risco_patogenos", [])
        pragas_especificas = relatorio_ia.get("pragas_especificas_cultura", [])
        plano = relatorio_ia.get("plano_de_acao_prescritivo", {})

        cultura = meta.get("cultura", "citros")
        is_organico = "organico" in str(meta.get("regime_cultivo", "")).lower()
        tipo_solo = meta.get("tipo_solo", "latossolo_vermelho")
        area_ha = float(meta.get("area_ha") or metricas.get("area_ha", 1.0))

        classe_irrig = int(irrig.get("classe_codigo", 0))
        status_irrig = irrig.get("status", "Não avaliado")
        lamina_mm = float(irrig.get("volume_agua_recomendado_mm", 0.0))
        vol_litros = float(metricas.get("volume_necessario_litros", 0.0))
        vol_econ = float(metricas.get("volume_economizado_litros", 0.0))
        tempo_got = int(metricas.get("tempo_gotejamento_minutos", 0))
        tempo_asp = int(metricas.get("tempo_aspersao_minutos", 0))
        litros_planta = float(metricas.get("litros_por_planta", 0.0))
        janela_irrig = metricas.get("melhor_horario_irrigacao", "")
        janela_pulv = metricas.get("melhor_horario_pulverizacao", "")

        trava_chuva = 1 if irrig.get("trava_seguranca_meteorologica", {}).get("acionada") else 0
        chuva_prev = float(clima.get("chuva_prevista_6h_mm", 0.0))
        prob_chuva = float(clima.get("probabilidade_chuva_6h_pct", 0.0))

        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analises_ia (
                    leitura_id, chat_id, talhao_id, talhao_nome, cultura, is_organico,
                    tipo_solo, area_ha, classe_irrigacao, status_irrigacao, lamina_recomendada_mm,
                    volume_necessario_litros, volume_economizado_litros, tempo_gotejamento_minutos,
                    tempo_aspersao_minutos, litros_por_planta, janela_irrigacao_ideal,
                    janela_pulverizacao_ideal, trava_chuva_acionada, chuva_prevista_mm,
                    probabilidade_chuva_pct, alertas_pragas, pragas_especificas, prescricao_insumos,
                    texto_prescritivo_llm, relatorio_json, notificado_telegram, criado_em
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                leitura_id, cid, talhao_id, talhao_nome, cultura, 1 if is_organico else 0,
                tipo_solo, area_ha, classe_irrig, status_irrig, lamina_mm,
                vol_litros, vol_econ, tempo_got, tempo_asp, litros_planta,
                janela_irrig, janela_pulv, trava_chuva, chuva_prev, prob_chuva,
                json.dumps(alertas_pragas), json.dumps(pragas_especificas),
                json.dumps(plano.get("acoes_e_insumos_recomendados", [])),
                texto_prescritivo_llm, json.dumps(relatorio_ia), 1 if notificado_telegram else 0, agora
            ))
            analise_id = cursor.lastrowid
            conn.commit()

        # Replica no Supabase
        payload_cloud = {
            "leitura_id": leitura_id,
            "chat_id": cid,
            "talhao_id": talhao_id,
            "talhao_nome": talhao_nome,
            "cultura": cultura,
            "is_organico": is_organico,
            "tipo_solo": tipo_solo,
            "area_ha": area_ha,
            "classe_irrigacao": classe_irrig,
            "status_irrigacao": status_irrig,
            "lamina_recomendada_mm": lamina_mm,
            "volume_necessario_litros": vol_litros,
            "volume_economizado_litros": vol_econ,
            "tempo_gotejamento_minutos": tempo_got,
            "tempo_aspersao_minutos": tempo_asp,
            "litros_por_planta": litros_planta,
            "janela_irrigacao_ideal": janela_irrig,
            "janela_pulverizacao_ideal": janela_pulv,
            "trava_chuva_acionada": bool(trava_chuva),
            "chuva_prevista_mm": chuva_prev,
            "probabilidade_chuva_pct": prob_chuva,
            "alertas_pragas": alertas_pragas,
            "pragas_especificas": pragas_especificas,
            "prescricao_insumos": plano.get("acoes_e_insumos_recomendados", []),
            "texto_prescritivo_llm": texto_prescritivo_llm,
            "relatorio_json": relatorio_ia,
            "notificado_telegram": notificado_telegram
        }
        self._sync_supabase_insert("analises_ia", payload_cloud)

        return analise_id

    # =========================================================================
    # 5. GESTÃO DE MENSAGENS E MEMÓRIA TELEGRAM
    # =========================================================================

    def salvar_mensagem_telegram(
        self,
        chat_id: str,
        direcao: str,
        mensagem_texto: str,
        tipo_mensagem: str = "texto_livre",
        talhao_mencionado_id: Optional[int] = None,
        analise_ia_id: Optional[int] = None
    ) -> int:
        """Grava uma interação para memória conversacional."""
        cid = str(chat_id)
        agora = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO mensagens_telegram (
                    chat_id, direcao, tipo_mensagem, mensagem_texto,
                    talhao_mencionado_id, analise_ia_id, criado_em
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (cid, direcao, tipo_mensagem, mensagem_texto, talhao_mencionado_id, analise_ia_id, agora))
            msg_id = cursor.lastrowid
            conn.commit()

        # Replica no Supabase
        payload_cloud = {
            "chat_id": cid,
            "direcao": direcao,
            "tipo_mensagem": tipo_mensagem,
            "mensagem_texto": mensagem_texto,
            "talhao_mencionado_id": talhao_mencionado_id,
            "analise_ia_id": analise_ia_id
        }
        self._sync_supabase_insert("mensagens_telegram", payload_cloud)

        return msg_id

    # =========================================================================
    # 6. CONSULTAS HISTÓRICAS PARA O LLM
    # =========================================================================

    def listar_historico_completo(
        self,
        chat_id: Optional[str] = None,
        talhao_id: Optional[int] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retorna o histórico unificado de leituras físicas acopladas com as análises
        da Rede Neural e os textos da LLM, para servir de contexto factual rico.
        """
        historico = []
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            where_clauses = []
            params = []
            if chat_id:
                where_clauses.append("l.chat_id = ?")
                params.append(str(chat_id))
            if talhao_id:
                where_clauses.append("l.talhao_id = ?")
                params.append(talhao_id)

            where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            params.append(limit)

            sql = f"""
                SELECT 
                    l.id AS leitura_id, l.chat_id, l.talhao_id, l.talhao_nome, l.origem,
                    l.latitude, l.longitude, l.ph, l.umidade, l.nitrogenio, l.fosforo,
                    l.potassio, l.condutividade, l.temperatura, l.compactacao, l.criado_em AS leitura_data,
                    a.id AS analise_id, a.cultura, a.is_organico, a.tipo_solo, a.area_ha,
                    a.classe_irrigacao, a.status_irrigacao, a.lamina_recomendada_mm,
                    a.volume_necessario_litros, a.volume_economizado_litros,
                    a.tempo_gotejamento_minutos, a.tempo_aspersao_minutos,
                    a.alertas_pragas, a.pragas_especificas, a.prescricao_insumos,
                    a.texto_prescritivo_llm, a.trava_chuva_acionada
                FROM leituras_solo l
                LEFT JOIN analises_ia a ON a.leitura_id = l.id
                {where_str}
                ORDER BY l.id DESC
                LIMIT ?
            """
            cursor.execute(sql, tuple(params))
            for r in cursor.fetchall():
                d = dict(r)
                d["is_organico"] = bool(d.get("is_organico", 1))
                d["trava_chuva_acionada"] = bool(d.get("trava_chuva_acionada", 0))
                try:
                    d["alertas_pragas"] = json.loads(d["alertas_pragas"]) if d.get("alertas_pragas") else []
                except Exception:
                    pass
                try:
                    d["pragas_especificas"] = json.loads(d["pragas_especificas"]) if d.get("pragas_especificas") else []
                except Exception:
                    pass
                try:
                    d["prescricao_insumos"] = json.loads(d["prescricao_insumos"]) if d.get("prescricao_insumos") else []
                except Exception:
                    pass
                historico.append(d)
        return historico

    def obter_ultimo_diagnostico_talhao(
        self,
        chat_id: str,
        talhao_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Recupera o diagnóstico mais recente do produtor para resposta rápida."""
        hist = self.listar_historico_completo(chat_id=chat_id, talhao_id=talhao_id, limit=1)
        return hist[0] if hist else None

    def listar_ultimas_leituras(self, chat_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Método de retrocompatibilidade para listar dados de sensores."""
        leituras = []
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if chat_id:
                cursor.execute("SELECT * FROM leituras_solo WHERE chat_id = ? ORDER BY id DESC LIMIT ?", (str(chat_id), limit))
            else:
                cursor.execute("SELECT * FROM leituras_solo ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            for r in rows:
                leituras.append(dict(r))
        return leituras


# Instância Singleton do Banco
db_manager = AgroDatabaseManager()
