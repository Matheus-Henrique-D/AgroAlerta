"""
geo_engine.py
=============
Motor de Geoprocessamento e Delimitação Espacial para o AgroAlerta.
Permite:
1. Teste de Ponto no Polígono (Point-in-Polygon) via Ray-Casting com buffer de tolerância GPS.
2. Cálculo de centróide e área aproximada em hectares.
3. Validação de polígonos de 3 (triângulo) ou 4 (quadrilátero) vértices.
"""

from typing import List, Dict, Any, Optional, Tuple
import math


def ponto_no_poligono(lat: float, lon: float, vertices: List[Dict[str, float]], margem_metros: float = 8.0) -> bool:
    """
    Verifica se a coordenada (lat, lon) está dentro do polígono de vértices [{'lat': ..., 'lon': ...}].
    Implementa Ray-Casting com tolerância de margem em metros para compensar oscilação de GPS do celular.
    """
    n = len(vertices)
    if n < 3:
        return False

    # 1. Teste padrão Ray-Casting
    dentro = False
    p1 = vertices[0]
    for i in range(1, n + 1):
        p2 = vertices[i % n]
        lat1, lon1 = float(p1["lat"]), float(p1["lon"])
        lat2, lon2 = float(p2["lat"]), float(p2["lon"])

        if min(lat1, lat2) <= lat <= max(lat1, lat2):
            if lon <= max(lon1, lon2):
                if lat1 != lat2:
                    lon_inters = (lat - lat1) * (lon2 - lon1) / (lat2 - lat1) + lon1
                    if lon1 == lon2 or lon <= lon_inters:
                        dentro = not dentro
        p1 = p2

    if dentro:
        return True

    # 2. Se não estiver estritamente dentro, testa distância aos segmentos (buffer de borda/cerca)
    if margem_metros > 0.0:
        dist_minima = distancia_minima_poligono_metros(lat, lon, vertices)
        if dist_minima <= margem_metros:
            return True

    return False


def calcular_distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distância geodésica em metros entre dois pontos (Haversine)."""
    R = 6371000.0  # Raio da Terra em metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def distancia_ponto_segmento_metros(lat: float, lon: float, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula a menor distância em metros entre um ponto e um segmento de reta de dois vértices."""
    # Conversão local para metros planos em torno de lat, lon
    lat_rad = math.radians(lat)
    fator_lat = 111320.0
    fator_lon = 111320.0 * math.cos(lat_rad)

    x0 = lon * fator_lon
    y0 = lat * fator_lat
    x1 = lon1 * fator_lon
    y1 = lat1 * fator_lat
    x2 = lon2 * fator_lon
    y2 = lat2 * fator_lat

    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(x0 - x1, y0 - y1)

    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return math.hypot(x0 - proj_x, y0 - proj_y)


def distancia_minima_poligono_metros(lat: float, lon: float, vertices: List[Dict[str, float]]) -> float:
    """Calcula a distância mínima de um ponto às arestas do polígono."""
    n = len(vertices)
    dist_min = float("inf")
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        d = distancia_ponto_segmento_metros(lat, lon, p1["lat"], p1["lon"], p2["lat"], p2["lon"])
        if d < dist_min:
            dist_min = d
    return dist_min


def calcular_centroide_e_area_hectares(vertices: List[Dict[str, float]]) -> Tuple[Dict[str, float], float]:
    """
    Calcula o centroide (lat, lon) médio e a área aproximada em hectares via fórmula de Shoelace (Gauss).
    """
    n = len(vertices)
    if n < 3:
        return {"lat": 0.0, "lon": 0.0}, 0.0

    soma_lat = sum(p["lat"] for p in vertices)
    soma_lon = sum(p["lon"] for p in vertices)
    centroide = {"lat": round(soma_lat / n, 6), "lon": round(soma_lon / n, 6)}

    # Conversão para metros locais
    lat_med = math.radians(centroide["lat"])
    fator_y = 111320.0
    fator_x = 111320.0 * math.cos(lat_med)

    area_m2 = 0.0
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        x1 = p1["lon"] * fator_x
        y1 = p1["lat"] * fator_y
        x2 = p2["lon"] * fator_x
        y2 = p2["lat"] * fator_y
        area_m2 += (x1 * y2 - x2 * y1)

    area_m2 = abs(area_m2) / 2.0
    area_hectares = round(area_m2 / 10000.0, 4)

    return centroide, area_hectares


def identificar_talhao(lat: float, lon: float, lista_talhoes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Dada uma lista de talhões cadastrados no banco de dados,
    retorna o talhão que engloba a coordenada fornecida.
    """
    for talhao in lista_talhoes:
        vertices = talhao.get("vertices", [])
        if ponto_no_poligono(lat, lon, vertices):
            return talhao
    return None
