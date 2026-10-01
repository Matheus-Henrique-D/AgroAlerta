"""
db_manager.py
=============
Gerenciador de Banco de Dados Híbrido (Supabase PostgreSQL com Fallback Local SQLite).
Garante persistência de talhões georreferenciados, leituras de campo e predições da IA.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from sqlalchemy import create_engine, text
    import pandas as pd
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres.acloppaxgohqupmdosoi:hanumanhitogaM4@aws-0-us-west-2.pooler.supabase.com:5432/postgres"
)
LOCAL_DB_PATH = os.path.join(os.path.dirname(__file__), "agroalerta_local.db")


class AgroDatabaseManager:
    """Gerenciador de dados que opera tanto na nuvem (Supabase) quanto localmente (SQLite)."""

    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url
        self.engine = None
        self._init_sqlite()
        if HAS_SQLALCHEMY:
            try:
                self.engine = create_engine(self.db_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
                self._init_cloud_tables()
            except Exception:
                self.engine = None

    def _init_sqlite(self):
        """Cria tabelas no SQLite local para garantir funcionamento mesmo offline."""
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS talhoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    nome_talhao TEXT NOT NULL,
                    poligono_tipo TEXT NOT NULL,
                    vertices TEXT NOT NULL,
                    cultura TEXT NOT NULL,
                    is_organico INTEGER NOT NULL,
                    tipo_solo TEXT NOT NULL,
                    area_ha REAL DEFAULT 0.0,
                    criado_em TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS leituras_solo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    talhao_id INTEGER,
                    talhao_nome TEXT,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    ph REAL,
                    umidade REAL,
                    nitrogenio REAL,
                    fosforo REAL,
                    potassio REAL,
                    condutividade REAL,
                    temperatura REAL,
                    compactacao REAL,
                    criado_em TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analises_ia (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    leitura_id INTEGER,
                    chat_id TEXT,
                    classe_irrigacao INTEGER,
                    status_irrigacao TEXT,
                    volume_mm REAL,
                    alertas_pragas TEXT,
                    relatorio_completo TEXT,
                    criado_em TEXT NOT NULL
                )
            """)
            conn.commit()

    def _init_cloud_tables(self):
        """Garante a existência das tabelas estruturadas no PostgreSQL do Supabase."""
        if not self.engine:
            return
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS talhoes (
                        id SERIAL PRIMARY KEY,
                        chat_id VARCHAR(64) NOT NULL,
                        nome_talhao VARCHAR(128) NOT NULL,
                        poligono_tipo VARCHAR(32) NOT NULL,
                        vertices JSONB NOT NULL,
                        cultura VARCHAR(64) NOT NULL,
                        is_organico BOOLEAN NOT NULL,
                        tipo_solo VARCHAR(64) NOT NULL,
                        area_ha FLOAT DEFAULT 0.0,
                        criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                conn.commit()
        except Exception:
            pass

    def salvar_talhao(
        self,
        chat_id: str,
        nome_talhao: str,
        poligono_tipo: str,
        vertices: List[Dict[str, float]],
        cultura: str,
        is_organico: bool,
        tipo_solo: str = "latossolo_vermelho",
        area_ha: float = 0.0
    ) -> Dict[str, Any]:
        """Salva um novo talhão com seus 3 ou 4 vértices."""
        agora = datetime.utcnow().isoformat()
        vertices_json = json.dumps(vertices)

        # Salva no SQLite
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO talhoes (chat_id, nome_talhao, poligono_tipo, vertices, cultura, is_organico, tipo_solo, area_ha, criado_em)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (str(chat_id), nome_talhao, poligono_tipo, vertices_json, cultura.lower(), int(is_organico), tipo_solo.lower(), area_ha, agora))
            talhao_id = cursor.lastrowid
            conn.commit()

        # Tenta replicar na nuvem se houver conexão
        if self.engine:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO talhoes (chat_id, nome_talhao, poligono_tipo, vertices, cultura, is_organico, tipo_solo, area_ha)
                        VALUES (:cid, :nome, :ptipo, :verts, :cult, :org, :solo, :area)
                    """), {
                        "cid": str(chat_id),
                        "nome": nome_talhao,
                        "ptipo": poligono_tipo,
                        "verts": vertices_json,
                        "cult": cultura.lower(),
                        "org": is_organico,
                        "solo": tipo_solo.lower(),
                        "area": area_ha
                    })
                    conn.commit()
            except Exception:
                pass

        return {
            "id": talhao_id,
            "chat_id": chat_id,
            "nome_talhao": nome_talhao,
            "poligono_tipo": poligono_tipo,
            "vertices": vertices,
            "cultura": cultura.lower(),
            "is_organico": is_organico,
            "tipo_solo": tipo_solo.lower(),
            "area_ha": area_ha
        }

    def listar_talhoes(self, chat_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lista todos os talhões cadastrados para um produtor."""
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
                    "vertices": json.loads(r["vertices"]),
                    "cultura": r["cultura"],
                    "is_organico": bool(r["is_organico"]),
                    "tipo_solo": r["tipo_solo"],
                    "area_ha": r["area_ha"],
                    "criado_em": r["criado_em"]
                })
        return talhoes

    def salvar_leitura_solo(
        self,
        chat_id: str,
        lat: float,
        lon: float,
        talhao_id: Optional[int],
        talhao_nome: Optional[str],
        dados_sensores: Dict[str, float]
    ) -> int:
        """Registra a leitura física do sensor com coordenada geográfica."""
        agora = datetime.utcnow().isoformat()
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO leituras_solo (
                    chat_id, talhao_id, talhao_nome, latitude, longitude,
                    ph, umidade, nitrogenio, fosforo, potassio, condutividade, temperatura, compactacao, criado_em
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(chat_id), talhao_id, talhao_nome, lat, lon,
                float(dados_sensores.get("ph", 6.5)),
                float(dados_sensores.get("umidade", 0.0)),
                float(dados_sensores.get("nitrogenio", 0.0)),
                float(dados_sensores.get("fosforo", 0.0)),
                float(dados_sensores.get("potassio", 0.0)),
                float(dados_sensores.get("condutividade", 1.0)),
                float(dados_sensores.get("temperatura", 25.0)),
                float(dados_sensores.get("compactacao", 1500.0)),
                agora
            ))
            leitura_id = cursor.lastrowid
            conn.commit()
            return leitura_id

    def listar_ultimas_leituras(self, chat_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Retorna as medições mais recentes."""
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


# Instância Singleton
db_manager = AgroDatabaseManager()
