import os
import requests
import json
import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Union, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import hashlib
import traceback
import random
import streamlit as st
from tavily import TavilyClient

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Clase para control de throttling
class ThrottlingController:
    """Controlador avanzado de tasas de solicitud para proveedores de datos de mercado"""

    def __init__(self):
        # Tasas máximas de solicitud por minuto por proveedor
        self.rate_limits = {
            "yfinance": 10,  # Límite conservador para Yahoo Finance
            "alphavantage": 5,  # Límite de API gratuita
            "finnhub": 30,  # Límite de plan básico
            "tavily": 5,  # Límite conservador
            "duckduckgo": 3,  # Muy restrictivo con web scraping
            "you": 5,  # Límite conservador para YOU API
        }

        # Ventanas de tiempo para solicitudes (últimas marcas de tiempo)
        self.request_windows = {provider: [] for provider in self.rate_limits}

        # Jitter aleatorio para evitar sincronización de solicitudes
        self.jitter_range = (0.1, 1.2)  # segundos

        # Contadores para estadísticas
        self.throttled_requests = {provider: 0 for provider in self.rate_limits}
        self.total_requests = {provider: 0 for provider in self.rate_limits}

        # Estado de disponibilidad de proveedores
        self.provider_status = {provider: True for provider in self.rate_limits}

        # Tiempo de cooldown para providers con error (en segundos)
        self.error_cooldown = 300  # 5 minutos
        self.provider_errors = {provider: [] for provider in self.rate_limits}

    def can_request(self, provider: str) -> bool:
        """Determina si se puede realizar una solicitud al proveedor especificado"""
        if provider not in self.rate_limits:
            # Si no tenemos límite registrado, permitimos con advertencia
            logger.warning(f"Proveedor desconocido: {provider}. Permitiendo solicitud.")
            return True

        self.total_requests[provider] += 1

        # Verificar si el proveedor está en cooldown por errores previos
        if not self.provider_status[provider]:
            last_error = (
                self.provider_errors[provider][-1]
                if self.provider_errors[provider]
                else datetime.min
            )
            if (datetime.now() - last_error).total_seconds() < self.error_cooldown:
                logger.info(f"Proveedor {provider} en cooldown por errores previos.")
                self.throttled_requests[provider] += 1
                return False
            else:
                # Restaurar proveedor después del período de cooldown
                self.provider_status[provider] = True

        # Limpiar ventana de tiempo (eliminar marcas de tiempo antiguas)
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        self.request_windows[provider] = [
            ts for ts in self.request_windows[provider] if ts > cutoff
        ]

        # Verificar si estamos dentro del límite
        if len(self.request_windows[provider]) < self.rate_limits[provider]:
            # Añadir jitter aleatorio para evitar patrones de solicitud
            jitter = random.uniform(*self.jitter_range)
            time.sleep(jitter)

            # Registrar solicitud en la ventana
            self.request_windows[provider].append(now)
            return True
        else:
            # Solicitud throttled
            self.throttled_requests[provider] += 1
            logger.info(f"Solicitud a {provider} limitada por rate limit.")
            return False

    def get_backoff_time(self, provider: str) -> float:
        """Calcula tiempo de espera adaptativo basado en la congestión del proveedor"""
        if provider not in self.rate_limits:
            return 1.0

        # Número de solicitudes en ventana actual
        request_count = len(self.request_windows[provider])

        # Si estamos cerca del límite, retornar tiempo proporcional
        limit = self.rate_limits[provider]
        if request_count > 0:
            utilization = request_count / limit
            # Fórmula exponencial para backoff
            return max(1.0, 5.0 * (2 ** (utilization * 2) - 1))
        return 1.0

    def report_error(self, provider: str):
        """Reporta un error con un proveedor para activar cooldown"""
        if provider not in self.provider_status:
            return

        self.provider_errors[provider].append(datetime.now())
        # Si acumulamos más de 3 errores en 5 minutos, deshabilitar temporalmente
        recent_errors = [
            ts
            for ts in self.provider_errors[provider]
            if (datetime.now() - ts).total_seconds() < 300
        ]

        if len(recent_errors) >= 3:
            self.provider_status[provider] = False
            logger.warning(
                f"Proveedor {provider} deshabilitado temporalmente por errores frecuentes."
            )

    def get_stats(self):
        """Obtiene estadísticas de throttling"""
        stats = {}
        for provider in self.rate_limits:
            success_rate = 0
            if self.total_requests[provider] > 0:
                success_rate = (
                    (self.total_requests[provider] - self.throttled_requests[provider])
                    / self.total_requests[provider]
                ) * 100

            stats[provider] = {
                "success_rate": f"{success_rate:.1f}%",
                "throttled": self.throttled_requests[provider],
                "total": self.total_requests[provider],
                "available": self.provider_status[provider],
                "current_utilization": f"{(len(self.request_windows[provider]) / self.rate_limits[provider]) * 100:.1f}%",
            }
        return stats

    def suggest_alternative_provider(self, provider: str) -> str:
        """Sugiere un proveedor alternativo cuando uno está limitado"""
        alternatives = {
            "yfinance": ["alphavantage", "finnhub"],
            "alphavantage": ["yfinance", "finnhub"],
            "finnhub": ["alphavantage", "yfinance"],
            "tavily": ["duckduckgo", "you"],
            "duckduckgo": ["tavily", "you"],
            "you": ["tavily", "duckduckgo"],
        }

        if provider not in alternatives:
            return None

        # Buscar alternativa disponible con menor utilización
        candidates = []
        for alt in alternatives[provider]:
            if self.provider_status[alt]:
                # Calcular utilización actual (0-1)
                utilization = (
                    len(self.request_windows[alt]) / self.rate_limits[alt]
                    if alt in self.rate_limits
                    else 1.0
                )
                candidates.append((alt, utilization))

        if not candidates:
            return None

        # Ordenar por utilización (menor primero)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]


# Inicializar throttling controller
throttling_controller = ThrottlingController()


class MarketDataCache:
    """Caché para datos de mercado y resultados de búsqueda web"""

    def __init__(self, ttl_minutes: int = 30):
        self.cache = {}
        self.ttl_minutes = ttl_minutes
        self.request_timestamps = {}
        self.hit_counter = 0
        self.miss_counter = 0

    def get(self, key: str) -> Optional[Dict]:
        """Obtener datos de caché si son válidos"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        if key_hash in self.cache:
            timestamp, data = self.cache[key_hash]
            if (datetime.now() - timestamp).total_seconds() < (self.ttl_minutes * 60):
                self.hit_counter += 1
                return data
        self.miss_counter += 1
        return None

    def set(self, key: str, data: Dict) -> None:
        """Almacenar datos en caché con timestamp"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        self.cache[key_hash] = (datetime.now(), data)

    def clear(self) -> int:
        """Limpiar caché completo y retornar número de entradas eliminadas"""
        old_count = len(self.cache)
        self.cache = {}
        return old_count

    def get_stats(self) -> Dict:
        """Retornar estadísticas del caché"""
        total_requests = self.hit_counter + self.miss_counter
        hit_rate = (
            (self.hit_counter / total_requests * 100) if total_requests > 0 else 0
        )

        return {
            "entradas": len(self.cache),
            "hit_rate": f"{hit_rate:.1f}%",
            "hits": self.hit_counter,
            "misses": self.miss_counter,
        }


# Inicializar caché global
_data_cache = MarketDataCache()


def get_api_keys_from_secrets():
    """Obtener claves API desde el archivo secrets.toml de Streamlit"""
    api_keys = {}

    try:
        # Intentar obtener las claves de API de Streamlit secrets
        if "api_keys" in st.secrets:
            if "YOU_API_KEY" in st.secrets["api_keys"]:
                api_keys["you"] = st.secrets["api_keys"]["YOU_API_KEY"]

            if "TAVILY_API_KEY" in st.secrets["api_keys"]:
                api_keys["tavily"] = st.secrets["api_keys"]["TAVILY_API_KEY"]

            # APIs adicionales que podrían ser útiles
            if "ALPHA_VANTAGE_API_KEY" in st.secrets["api_keys"]:
                api_keys["alpha_vantage"] = st.secrets["api_keys"][
                    "ALPHA_VANTAGE_API_KEY"
                ]

            if "FINNHUB_API_KEY" in st.secrets["api_keys"]:
                api_keys["finnhub"] = st.secrets["api_keys"]["FINNHUB_API_KEY"]

            if "MARKETSTACK_API_KEY" in st.secrets["api_keys"]:
                api_keys["marketstack"] = st.secrets["api_keys"]["MARKETSTACK_API_KEY"]

        return api_keys
    except Exception as e:
        logger.error(f"Error cargando claves API desde secrets: {str(e)}")
        return {}


class WebSearchEngine:
    """Motor de búsqueda web con múltiples proveedores"""

    def __init__(self, api_keys: Dict[str, str] = None):
        # Si no se proporcionaron claves, intentar obtenerlas de secrets.toml
        if not api_keys:
            api_keys = get_api_keys_from_secrets()

        self.api_keys = api_keys or {}
        self.cache = _data_cache

    def _get_api_key(self, service: str) -> str:
        """Obtener clave API con estrategia de fallback robusta"""
        try:
            service_lower = service.lower()

            # Buscar en diccionario de claves normalizadas
            if service_lower in self.api_keys and self.api_keys[service_lower]:
                return self.api_keys[service_lower]

            # Versión mayúscula
            if service.upper() in self.api_keys and self.api_keys[service.upper()]:
                return self.api_keys[service.upper()]

            # Búsqueda insensible a mayúsculas/minúsculas
            for key, value in self.api_keys.items():
                if key.lower() == service_lower and value:
                    return value

            # Buscar en variables de entorno como última opción
            env_key = f"{service.upper()}_API_KEY"
            return os.environ.get(env_key, "")
        except Exception as e:
            logger.error(f"Error obteniendo API key para {service}: {str(e)}")
            return ""

    def search_you_api(self, query: str, max_results: int = 3) -> List[Dict]:
        """Buscar usando API de YOU"""
        if not throttling_controller.can_request("you"):
            logger.info(f"Solicitud a YOU API limitada por throttling")
            return []

        try:
            api_key = self._get_api_key("you")
            if not api_key:
                logger.warning("YOU API key no disponible")
                return []

            headers = {"X-API-Key": api_key}
            params = {"query": query}
            response = requests.get(
                "https://api.ydc-index.io/search",
                params=params,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            if "snippets" in data:
                for snippet in data["snippets"][:max_results]:
                    results.append(
                        {
                            "title": snippet.get("title", "Sin título"),
                            "url": snippet.get("url", ""),
                            "content": snippet.get("content", ""),
                            "source": "YOU",
                        }
                    )

            return results
        except Exception as e:
            throttling_controller.report_error("you")
            logger.error(f"Error en búsqueda YOU: {str(e)}")
            return []

    def search_tavily(self, query: str, max_results: int = 3) -> List[Dict]:
        """Buscar usando API de Tavily"""
        if not throttling_controller.can_request("tavily"):
            logger.info(f"Solicitud a Tavily API limitada por throttling")
            return []

        try:
            api_key = self._get_api_key("tavily")
            if not api_key:
                logger.warning("Tavily API key no disponible")
                return []

            # Instalar tavily-python si aún no está instalado
            try:
                from tavily import TavilyClient
            except ImportError:
                logger.warning("Tavily SDK no encontrado, intentando instalarlo...")
                try:
                    import subprocess

                    subprocess.check_call(["pip", "install", "tavily-python"])
                    from tavily import TavilyClient

                    logger.info("SDK de Tavily instalado correctamente")
                except Exception as e:
                    logger.error(f"Error instalando Tavily SDK: {str(e)}")
                    return []

            # Usar el SDK oficial de Tavily
            tavily_client = TavilyClient(api_key=api_key)
            response = tavily_client.search(
                query, search_depth="advanced", max_results=max_results
            )

            results = []
            if "results" in response:
                for result in response["results"][:max_results]:
                    results.append(
                        {
                            "title": result.get("title", "Sin título"),
                            "url": result.get("url", ""),
                            "content": result.get("content", ""),
                            "source": "Tavily",
                        }
                    )

            return results
        except Exception as e:
            throttling_controller.report_error("tavily")
            logger.error(f"Error en búsqueda Tavily: {str(e)}")
            return []

    def search_duckduckgo(self, query: str, max_results: int = 3) -> List[Dict]:
        """Buscar usando DuckDuckGo"""
        if not throttling_controller.can_request("duckduckgo"):
            logger.info(f"Solicitud a DuckDuckGo limitada por throttling")
            return []

        try:
            from duckduckgo_search import DDGS

            results = []
            for attempt in range(3):  # Intentar hasta 3 veces
                try:
                    with DDGS() as ddgs:
                        ddg_results = list(ddgs.text(query, max_results=max_results))

                        for result in ddg_results:
                            results.append(
                                {
                                    "title": result.get("title", "Sin título"),
                                    "url": result.get("href", ""),
                                    "content": result.get("body", ""),
                                    "source": "DuckDuckGo",
                                }
                            )
                        break  # Éxito, salir del bucle
                except Exception as e:
                    if "Rate limit" in str(e) and attempt < 2:
                        throttling_controller.report_error("duckduckgo")
                        time.sleep(2**attempt)  # Backoff exponencial
                    else:
                        raise

            return results
        except Exception as e:
            throttling_controller.report_error("duckduckgo")
            logger.error(f"Error en búsqueda DuckDuckGo: {str(e)}")
            return []

    def search_alpha_vantage_news(
        self, symbol: str, max_results: int = 3
    ) -> List[Dict]:
        """Buscar noticias usando Alpha Vantage News API"""
        if not throttling_controller.can_request("alphavantage"):
            logger.info(f"Solicitud a Alpha Vantage limitada por throttling")
            return []

        try:
            api_key = self._get_api_key("alpha_vantage")
            if not api_key:
                logger.warning("Alpha Vantage API key no disponible")
                return []

            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}&limit={max_results}"

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            if "feed" in data:
                for item in data["feed"][:max_results]:
                    results.append(
                        {
                            "title": item.get("title", "Sin título"),
                            "url": item.get("url", ""),
                            "content": item.get("summary", ""),
                            "source": "Alpha Vantage News",
                            "sentiment": item.get("overall_sentiment_score", 0),
                        }
                    )

            return results
        except Exception as e:
            throttling_controller.report_error("alphavantage")
            logger.error(f"Error en búsqueda Alpha Vantage News: {str(e)}")
            return []

    def perform_web_search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Realizar búsqueda web utilizando múltiples proveedores con throttling inteligente"""
        cache_key = f"web_search_{query}_{max_results}"
        cached_results = self.cache.get(cache_key)

        if cached_results:
            return cached_results

        all_results = []

        # Lista de proveedores a intentar
        providers = [
            {"name": "tavily", "func": lambda: self.search_tavily(query, max_results)},
            {"name": "you", "func": lambda: self.search_you_api(query, max_results)},
            {
                "name": "duckduckgo",
                "func": lambda: self.search_duckduckgo(query, max_results),
            },
        ]

        # Intentar cada proveedor hasta obtener suficientes resultados
        for provider in providers:
            provider_name = provider["name"]

            # Verificar throttling
            if not throttling_controller.can_request(provider_name):
                logger.info(
                    f"Proveedor {provider_name} limitado por throttling, intentando alternativa"
                )

                # Buscar alternativa
                alt_provider = throttling_controller.suggest_alternative_provider(
                    provider_name
                )
                if alt_provider:
                    logger.info(f"Usando proveedor alternativo: {alt_provider}")
                    # Ya lo intentaremos en otra iteración
                    continue
                else:
                    # Sin alternativa, aplicar backoff y continuar
                    backoff_time = throttling_controller.get_backoff_time(provider_name)
                    logger.info(
                        f"Aplicando backoff de {backoff_time:.2f}s para {provider_name}"
                    )
                    time.sleep(backoff_time)

            try:
                # Intentar búsqueda con este proveedor
                provider_results = provider["func"]()

                if provider_results:
                    all_results.extend(provider_results)

                # Verificar si ya tenemos suficientes resultados
                if len(all_results) >= max_results:
                    break

            except Exception as e:
                # Reportar error al sistema de throttling
                throttling_controller.report_error(provider_name)
                logger.error(f"Error en búsqueda con {provider_name}: {str(e)}")
                continue

        # Si es una búsqueda relacionada con un símbolo, intentar con Alpha Vantage News
        if query.split()[0].isupper() and len(query.split()[0]) <= 5:
            symbol = query.split()[0]
            av_results = self.search_alpha_vantage_news(symbol, max_results=2)
            if av_results and len(all_results) < max_results:
                all_results.extend(av_results[: max_results - len(all_results)])

        # Almacenar en caché
        if all_results:
            self.cache.set(cache_key, all_results)

        return all_results


class WebScraper:
    """Scraper para datos fundamentales y técnicos de acciones"""

    def __init__(self):
        self.cache = _data_cache
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.api_keys = get_api_keys_from_secrets()

    def _clean_html(self, text: str) -> str:
        """Limpia texto HTML"""
        return re.sub(r"<.*?>", "", text).strip()

    def scrape_yahoo_finance(self, symbol: str) -> Dict:
        """Scrape datos de Yahoo Finance"""
        cache_key = f"yahoo_finance_{symbol}"
        cached_data = self.cache.get(cache_key)

        if cached_data:
            return cached_data

        if not throttling_controller.can_request("yfinance"):
            logger.info(
                f"Solicitud a Yahoo Finance limitada por throttling, buscando alternativa"
            )
            # Buscar datos alternativos o sintéticos
            return {
                "symbol": symbol,
                "source": "Synthetic Data",
                "timestamp": datetime.now().isoformat(),
                "price": {"current": 100.0, "change": 0.0, "change_percent": 0.0},
                "fundamentals": {},
                "indicators": {},
            }

        try:
            url = f"https://finance.yahoo.com/quote/{symbol}"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            # Datos básicos
            data = {
                "symbol": symbol,
                "source": "Yahoo Finance",
                "timestamp": datetime.now().isoformat(),
                "price": {},
                "fundamentals": {},
                "indicators": {},
            }

            # Precio actual
            try:
                price_elem = soup.select_one('[data-test="qsp-price"]')
                if price_elem:
                    data["price"]["current"] = float(price_elem.text.replace(",", ""))
            except Exception as e:
                logger.debug(f"Error obteniendo precio actual: {str(e)}")

            # Cambio diario
            try:
                change_elems = soup.select('[data-test="qsp-price-change"]')
                if change_elems and len(change_elems) >= 2:
                    data["price"]["change"] = float(
                        change_elems[0].text.replace(",", "")
                    )
                    data["price"]["change_percent"] = float(
                        change_elems[1].text.strip("()%")
                    )
            except Exception as e:
                logger.debug(f"Error obteniendo cambio diario: {str(e)}")

            # Fundamentales
            fundamentals_mapping = {
                "Market Cap": "market_cap",
                "PE Ratio": "pe_ratio",
                "EPS": "eps",
                "Dividend": "dividend",
                "Yield": "yield",
                "Volume": "volume",
                "Avg Vol": "avg_volume",
                "52 Week Range": "year_range",
            }

            # Buscar tabla de stats
            tables = soup.select("table")
            for table in tables:
                rows = table.select("tr")
                for row in rows:
                    cells = row.select("td")
                    if len(cells) >= 2:
                        label = cells[0].text.strip()
                        value = cells[1].text.strip()

                        for key, mapped_key in fundamentals_mapping.items():
                            if key in label:
                                data["fundamentals"][mapped_key] = value

            # Intentar obtener datos adicionales de Alpha Vantage si está disponible
            if "alpha_vantage" in self.api_keys:
                try:
                    av_data = self._get_alpha_vantage_overview(symbol)
                    if av_data and "fundamentals" in data:
                        # Añadir datos que no se encuentran fácilmente en Yahoo Finance
                        metrics = {
                            "PEGRatio": "peg_ratio",
                            "PriceToBookRatio": "price_to_book",
                            "ReturnOnEquityTTM": "roe",
                            "ReturnOnAssetsTTM": "roa",
                            "ProfitMargin": "profit_margin",
                            "OperatingMarginTTM": "operating_margin",
                        }

                        for av_key, mapped_key in metrics.items():
                            if av_key in av_data:
                                data["fundamentals"][mapped_key] = av_data[av_key]
                except Exception as e:
                    logger.warning(
                        f"Error obteniendo datos adicionales de Alpha Vantage: {str(e)}"
                    )

            # Almacenar en caché
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            throttling_controller.report_error("yfinance")
            logger.error(f"Error scraping Yahoo Finance para {symbol}: {str(e)}")
            return {"symbol": symbol, "error": str(e)}

    def _get_alpha_vantage_overview(self, symbol: str) -> Dict:
        """Obtener resumen de company de Alpha Vantage"""
        if not throttling_controller.can_request("alphavantage"):
            logger.info(f"Solicitud a Alpha Vantage limitada por throttling")
            return {}

        try:
            api_key = self.api_keys.get("alpha_vantage", "")
            if not api_key:
                return {}

            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return {}

            return response.json()
        except Exception as e:
            throttling_controller.report_error("alphavantage")
            logger.error(f"Error obteniendo overview de Alpha Vantage: {str(e)}")
            return {}

    def scrape_financial_news(self, symbol: str, max_news: int = 5) -> List[Dict]:
        """Scrape noticias financieras relevantes"""
        cache_key = f"financial_news_{symbol}_{max_news}"
        cached_data = self.cache.get(cache_key)

        if cached_data:
            return cached_data

        try:
            # Primero intentar con Alpha Vantage si está disponible
            if "alpha_vantage" in self.api_keys and throttling_controller.can_request(
                "alphavantage"
            ):
                av_news = self._get_alpha_vantage_news(symbol, max_news)
                if av_news:
                    # Almacenar en caché
                    self.cache.set(cache_key, av_news)
                    return av_news

            # Si Alpha Vantage falla o no está disponible, usar FinViz
            if throttling_controller.can_request("finviz"):
                url = f"https://finviz.com/quote.ashx?t={symbol}"
                response = requests.get(url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")

                news = []
                news_table = soup.find(id="news-table")
                if news_table:
                    rows = news_table.findAll("tr")

                    for row in rows[:max_news]:
                        cells = row.findAll("td")
                        if len(cells) >= 2:
                            date_cell = cells[0].text.strip()
                            title_cell = cells[1]

                            # Extraer fecha/hora
                            if " " in date_cell:
                                date = date_cell
                            else:
                                time = date_cell
                                date = rows[0].td.text.strip().split(" ")[0]

                            # Extraer título y enlace
                            a_tag = title_cell.a
                            if a_tag:
                                title = a_tag.text
                                href = a_tag["href"]

                                news.append({"date": date, "title": title, "url": href})

                # Almacenar en caché
                self.cache.set(cache_key, news)
                return news
            else:
                # Si todo falla, devolver lista vacía
                logger.warning(
                    f"No se pudieron obtener noticias para {symbol} debido a limitaciones de rate limit"
                )
                return []

        except Exception as e:
            logger.error(f"Error scraping noticias para {symbol}: {str(e)}")
            return []

    def _get_alpha_vantage_news(self, symbol: str, max_news: int = 5) -> List[Dict]:
        """Obtener noticias desde Alpha Vantage"""
        try:
            api_key = self.api_keys.get("alpha_vantage", "")
            if not api_key:
                return []

            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}&limit={max_news}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return []

            data = response.json()
            news = []

            if "feed" in data:
                for item in data["feed"][:max_news]:
                    timestamp = item.get("time_published", "")
                    if timestamp:
                        try:
                            date = datetime.strptime(
                                timestamp, "%Y%m%dT%H%M%S"
                            ).strftime("%Y-%m-%d %H:%M")
                        except:
                            date = timestamp
                    else:
                        date = "N/A"

                    news.append(
                        {
                            "date": date,
                            "title": item.get("title", "Sin título"),
                            "url": item.get("url", "#"),
                            "source": item.get("source", "Alpha Vantage"),
                        }
                    )

            return news
        except Exception as e:
            logger.error(f"Error obteniendo noticias de Alpha Vantage: {str(e)}")
            return []

    def get_finnhub_sentiment(self, symbol: str) -> Dict:
        """Obtener análisis de sentimiento de noticias de Finnhub"""
        if not throttling_controller.can_request("finnhub"):
            logger.info(f"Solicitud a Finnhub limitada por throttling")
            return {"symbol": symbol, "sentiment": "neutral", "score": 0.5}

        try:
            api_key = self.api_keys.get("finnhub", "")
            if not api_key:
                return {"symbol": symbol, "sentiment": "neutral", "score": 0.5}

            url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return {"symbol": symbol, "sentiment": "neutral", "score": 0.5}

            data = response.json()

            if "sentiment" in data and "bullishPercent" in data["sentiment"]:
                score = data["sentiment"]["bullishPercent"]

                # Determinar sentimiento
                if score > 0.6:
                    sentiment = "bullish"
                elif score < 0.4:
                    sentiment = "bearish"
                else:
                    sentiment = "neutral"

                result = {
                    "symbol": symbol,
                    "sentiment": sentiment,
                    "score": score,
                    "company_news_score": data.get("companyNewsScore", 0),
                    "sector_avg_bullish": data.get("sectorAverageBullishPercent", 0),
                    "sector_avg_bearish": data.get("sectorAverageBearishPercent", 0),
                }

                return result
            else:
                return {"symbol": symbol, "sentiment": "neutral", "score": 0.5}

        except Exception as e:
            throttling_controller.report_error("finnhub")
            logger.error(f"Error obteniendo sentimiento de Finnhub: {str(e)}")
            return {"symbol": symbol, "sentiment": "neutral", "score": 0.5}


def validate_market_data(data: pd.DataFrame) -> bool:
    """Valida la integridad de los datos de mercado"""
    try:
        if data is None or len(data) == 0:
            return False

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required_columns):
            return False

        # Verificar valores nulos
        if data[required_columns].isnull().any().any():
            # Intentar reparar datos nulos
            data.fillna(method="ffill", inplace=True)
            data.fillna(method="bfill", inplace=True)
            # Verificar si aún hay nulos después de reparar
            if data[required_columns].isnull().any().any():
                return False

        # Verificar valores negativos en precio
        price_cols = ["Open", "High", "Low", "Close"]
        if (data[price_cols] <= 0).any().any():
            return False

        # Verificar coherencia OHLC (con pequeña tolerancia)
        if (
            not all(data["High"] >= data["Low"])
            or not all(data["High"] >= data["Open"] * 0.99)
            or not all(data["High"] >= data["Close"] * 0.99)
            or not all(data["Low"] <= data["Open"] * 1.01)
            or not all(data["Low"] <= data["Close"] * 1.01)
        ):
            return False

        return True

    except Exception as e:
        logger.error(f"Error en validate_market_data: {str(e)}")
        return False


def validate_and_fix_data(data: pd.DataFrame) -> pd.DataFrame:
    """Valida y corrige problemas en datos de mercado"""
    if data is None or data.empty:
        return pd.DataFrame()

    # Asegurar índice de tiempo
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            logger.warning(f"Error al convertir índice: {str(e)}")

    # Asegurar columnas OHLCV
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in data.columns:
            if col == "Volume":
                data[col] = 0  # Valor por defecto para volumen
            elif "Close" in data.columns:
                # Si falta una columna crítica pero hay Close, usar Close como base
                if col == "Open":
                    data[col] = data["Close"].shift(1).fillna(data["Close"])
                elif col == "High":
                    data[col] = data["Close"] * 1.001  # Leve ajuste para High
                elif col == "Low":
                    data[col] = data["Close"] * 0.999  # Leve ajuste para Low
            else:
                # Último recurso, crear datos sintéticos
                data[col] = np.random.normal(100, 1, len(data))

    # Convertir a tipos numéricos
    for col in required_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Asegurarnos que High siempre es el máximo y Low siempre es el mínimo
    if all(col in data.columns for col in ["Open", "Close", "High", "Low"]):
        data["High"] = data[["Open", "Close", "High"]].max(axis=1)
        data["Low"] = data[["Open", "Close", "Low"]].min(axis=1)

    # Asegurar que el volumen es no-negativo
    if "Volume" in data.columns:
        data["Volume"] = data["Volume"].abs()

    # Rellenar valores NaN - Corregido para evitar método deprecado
    data = data.ffill().bfill().fillna(0)

    return data


def _generate_synthetic_data(symbol: str, periods: int = 180) -> pd.DataFrame:
    """Genera datos sintéticos para fallback de interfaz"""
    try:
        # Crear datos determinísticos pero realistas basados en el símbolo
        seed_value = sum(ord(c) for c in symbol)
        np.random.seed(seed_value)

        # Fechas para los días solicitados hasta hoy
        end_date = datetime.now()
        start_date = end_date - timedelta(days=periods)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Precio base variable según símbolo
        base_price = 100 + (seed_value % 900)

        # Generar precios con tendencia y volatilidad realista
        prices = []
        price = base_price

        # Volatilidad dependiente del símbolo
        volatility = 0.01 + (seed_value % 10) / 100
        trend = 0.0005 * ((seed_value % 10) - 5)  # Entre -0.0025 y +0.0025

        # Generar serie de precios
        for _ in range(len(dates)):
            noise = np.random.normal(trend, volatility)
            price *= 1 + noise
            prices.append(max(price, 0.01))  # Evitar precios negativos

        # Crear DataFrame OHLCV sintético
        df = pd.DataFrame(index=dates)
        df["Close"] = prices
        df["Open"] = [p * (1 - np.random.uniform(0, volatility)) for p in prices]
        df["High"] = [
            max(o, c) * (1 + np.random.uniform(0, volatility))
            for o, c in zip(df["Open"], df["Close"])
        ]
        df["Low"] = [
            min(o, c) * (1 - np.random.uniform(0, volatility))
            for o, c in zip(df["Open"], df["Close"])
        ]
        df["Volume"] = [int(1e6 * (1 + np.random.normal(0, 0.3))) for _ in prices]
        df["Adj Close"] = df["Close"]

        # Flag para identificar como sintético
        df.attrs["synthetic"] = True
        logger.info(f"Datos sintéticos generados para {symbol}")

        return df

    except Exception as e:
        logger.error(f"Error generando datos sintéticos: {str(e)}")

        # Crear un DataFrame mínimo para evitar errores
        df = pd.DataFrame(index=pd.date_range(end=datetime.now(), periods=30))
        df["Close"] = np.linspace(100, 110, 30)
        df["Open"] = df["Close"] * 0.99
        df["High"] = df["Close"] * 1.01
        df["Low"] = df["Open"] * 0.99
        df["Volume"] = 1000000
        df["Adj Close"] = df["Close"]
        df.attrs["synthetic"] = True
        return df


def _get_alpha_vantage_data(symbol: str, interval: str = "1d") -> pd.DataFrame:
    """Obtiene datos desde Alpha Vantage como respaldo"""
    if not throttling_controller.can_request("alphavantage"):
        logger.info(f"Solicitud a Alpha Vantage limitada por throttling")
        return None

    api_keys = get_api_keys_from_secrets()
    alpha_vantage_key = api_keys.get("alpha_vantage", "")
    if not alpha_vantage_key:
        logger.warning("Alpha Vantage API key no disponible")
        return None

    try:
        # Mapear intervalo
        av_function = "TIME_SERIES_DAILY"
        av_interval = None

        if interval in ["1m", "5m", "15m", "30m", "60m", "1h"]:
            av_function = "TIME_SERIES_INTRADAY"
            av_interval = (
                interval.replace("m", "min")
                .replace("h", "min")
                .replace("60min", "60min")
            )

        # Construir URL
        url_params = f"&interval={av_interval}" if av_interval else ""
        url = f"https://www.alphavantage.co/query?function={av_function}&symbol={symbol}&outputsize=full{url_params}&apikey={alpha_vantage_key}"

        # Realizar solicitud con timeout
        response = requests.get(url, timeout=10)
        data = response.json()

        # Parsear respuesta
        time_series_key = next((k for k in data.keys() if "Time Series" in k), None)

        if not time_series_key or time_series_key not in data:
            raise ValueError(f"Datos no encontrados en Alpha Vantage para {symbol}")

        # Convertir a DataFrame
        time_series = data[time_series_key]
        df = pd.DataFrame.from_dict(time_series, orient="index")

        # Renombrar columnas
        column_map = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume",
        }

        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Convertir a tipos numéricos
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Establecer índice de tiempo
        df.index = pd.to_datetime(df.index)

        # Añadir columna Adj Close si no existe
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        return df

    except Exception as e:
        throttling_controller.report_error("alphavantage")
        logger.error(f"Error en Alpha Vantage para {symbol}: {str(e)}")
        return None


def _get_finnhub_data(symbol: str, resolution: str = "D") -> pd.DataFrame:
    """Obtiene datos desde Finnhub como respaldo adicional"""
    if not throttling_controller.can_request("finnhub"):
        logger.info(f"Solicitud a Finnhub limitada por throttling")
        return None

    api_keys = get_api_keys_from_secrets()
    finnhub_key = api_keys.get("finnhub", "")
    if not finnhub_key:
        logger.warning("Finnhub API key no disponible")
        return None

    try:
        # Convertir período a resolución Finnhub
        resolution_map = {
            "1d": "D",
            "1h": "60",
            "30m": "30",
            "15m": "15",
            "5m": "5",
            "1m": "1",
        }
        finnhub_resolution = resolution_map.get(resolution, "D")

        # Calcular fechas (unix timestamp)
        end_time = int(time.time())
        # 6 meses de datos
        start_time = end_time - (180 * 24 * 60 * 60)

        # Construir URL
        url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution={finnhub_resolution}&from={start_time}&to={end_time}&token={finnhub_key}"

        # Realizar solicitud
        response = requests.get(url, timeout=10)
        data = response.json()

        # Verificar si hay datos válidos
        if data.get("s") != "ok":
            logger.warning(f"Finnhub no retornó datos válidos para {symbol}")
            return None

        # Construir DataFrame
        df = pd.DataFrame(
            {
                "Open": data["o"],
                "High": data["h"],
                "Low": data["l"],
                "Close": data["c"],
                "Volume": data["v"],
            },
            index=pd.to_datetime([datetime.fromtimestamp(ts) for ts in data["t"]]),
        )

        # Añadir Adj Close
        df["Adj Close"] = df["Close"]

        return df

    except Exception as e:
        throttling_controller.report_error("finnhub")
        logger.error(f"Error en Finnhub para {symbol}: {str(e)}")
        return None


def _get_marketstack_data(symbol: str) -> pd.DataFrame:
    """Obtiene datos desde MarketStack como otra fuente alternativa"""
    api_keys = get_api_keys_from_secrets()
    marketstack_key = api_keys.get("marketstack", "")
    if not marketstack_key:
        logger.warning("MarketStack API key no disponible")
        return None

    try:
        # Calcular fechas
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

        # Construir URL
        url = f"http://api.marketstack.com/v1/eod?access_key={marketstack_key}&symbols={symbol}&date_from={start_date}&date_to={end_date}&limit=1000"

        # Realizar solicitud
        response = requests.get(url, timeout=10)
        data = response.json()

        # Verificar datos válidos
        if "data" not in data or not data["data"]:
            logger.warning(f"MarketStack no retornó datos válidos para {symbol}")
            return None

        # Construir DataFrame
        df = pd.DataFrame(data["data"])

        # Convertir columnas y formato
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "adj_close": "Adj Close",
                "date": "Date",
            }
        )

        # Establecer fecha como índice
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

        # Invertir orden para tener el más reciente al final
        df = df.sort_index()

        return df

    except Exception as e:
        logger.error(f"Error en MarketStack para {symbol}: {str(e)}")
        return None


def fetch_market_data(
    symbol: str, period: str = "6mo", interval: str = "1d"
) -> pd.DataFrame:
    """
    Obtiene datos de mercado con throttling inteligente, multiple fallbacks y validación.

    Args:
        symbol (str): Símbolo de la acción o ETF
        period (str): Período de tiempo ('1d', '1mo', '3mo', '6mo', '1y', '2y', '5y')
        interval (str): Intervalo de velas ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')

    Returns:
        pd.DataFrame: DataFrame con datos OHLCV
    """
    # Clave de caché
    cache_key = f"market_data_{symbol}_{period}_{interval}"

    # Verificar caché
    cached_data = _data_cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    # Verificar throttling para yfinance
    if not throttling_controller.can_request("yfinance"):
        # Si no podemos solicitar a yfinance, buscar proveedor alternativo o datos históricos
        logger.info(
            f"Limitando solicitudes para {symbol}, buscando alternativa o caché histórico"
        )

        # Buscar cualquier dato previo para este símbolo
        for k, v in _data_cache.cache.items():
            if symbol in k and k.startswith("market_data_"):
                logger.info(f"Usando datos históricos para {symbol}")
                return v[1]  # v[1] contiene los datos, v[0] el timestamp

        # Calcular tiempo de backoff y esperar
        backoff_time = throttling_controller.get_backoff_time("yfinance")
        logger.info(f"Aplicando backoff de {backoff_time:.2f}s para {symbol}")
        time.sleep(backoff_time)

        # Intentar con proveedores alternativos
        alt_provider = throttling_controller.suggest_alternative_provider("yfinance")
        if alt_provider == "alphavantage":
            data = _get_alpha_vantage_data(symbol, interval)
            if data is not None and not data.empty:
                # Procesar datos y agregar a caché
                data = validate_and_fix_data(data)
                _data_cache.set(cache_key, data)
                return data
        elif alt_provider == "finnhub":
            data = _get_finnhub_data(symbol, interval)
            if data is not None and not data.empty:
                # Procesar datos y agregar a caché
                data = validate_and_fix_data(data)
                _data_cache.set(cache_key, data)
                return data

        # Si todo falla, generar datos sintéticos
        synthetic_data = _generate_synthetic_data(symbol)
        logger.warning(
            f"Generando datos sintéticos para {symbol} debido a rate limiting"
        )
        _data_cache.set(cache_key, synthetic_data)
        return synthetic_data

    try:
        # Intentar obtener datos con yfinance con manejo de errores mejorado
        for attempt in range(3):  # Intentar hasta 3 veces con backoff exponencial
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)

                if data.empty and attempt < 2:
                    wait_time = (2**attempt) * (1 + random.random())
                    logger.warning(
                        f"Datos vacíos para {symbol}, reintentando en {wait_time:.2f}s (intento {attempt+1})"
                    )
                    time.sleep(wait_time)
                    continue

                break  # Éxito, salir del bucle
            except Exception as e:
                if "Too Many Requests" in str(e):
                    # Reportar error de rate limit
                    throttling_controller.report_error("yfinance")
                    if attempt < 2:
                        wait_time = (2**attempt) * 2 * (1 + random.random())
                        logger.warning(
                            f"Rate limit para {symbol}, reintentando en {wait_time:.2f}s (intento {attempt+1})"
                        )
                        time.sleep(wait_time)
                    else:
                        # Último intento fallido, intentar alternativas
                        raise
                else:
                    # Otro tipo de error, intentar de nuevo con backoff
                    if attempt < 2:
                        wait_time = (2**attempt) * (1 + random.random())
                        logger.warning(
                            f"Error para {symbol}: {str(e)}, reintentando en {wait_time:.2f}s (intento {attempt+1})"
                        )
                        time.sleep(wait_time)
                    else:
                        # Último intento fallido, re-lanzar excepción
                        raise

        # Validar datos obtenidos
        if data.empty or not validate_market_data(data):
            logger.warning(
                f"Datos inválidos para {symbol} en yfinance, intentando fuentes alternativas"
            )

            # Intentar con Alpha Vantage
            data = _get_alpha_vantage_data(symbol, interval)

            # Si Alpha Vantage falla, intentar con Finnhub
            if data is None or not validate_market_data(data):
                data = _get_finnhub_data(symbol, interval)

                # Si Finnhub falla, intentar con MarketStack
                if data is None or not validate_market_data(data):
                    data = _get_marketstack_data(symbol)

                    # Si todo falla, generar datos sintéticos
                    if data is None or not validate_market_data(data):
                        logger.warning(
                            f"Todas las fuentes fallaron para {symbol}, generando datos sintéticos"
                        )
                        data = _generate_synthetic_data(symbol)

        # Corrección final de datos
        data = validate_and_fix_data(data)

        # Guardar en caché
        _data_cache.set(cache_key, data)
        return data

    except Exception as e:
        logger.error(f"Error obteniendo datos para {symbol}: {str(e)}")
        traceback.print_exc()

        # Manejo de errores mejorado con reintento progresivo
        data_sources = [
            {
                "name": "alpha_vantage",
                "func": lambda: _get_alpha_vantage_data(symbol, interval),
            },
            {"name": "finnhub", "func": lambda: _get_finnhub_data(symbol, interval)},
            {"name": "marketstack", "func": lambda: _get_marketstack_data(symbol)},
            {"name": "synthetic", "func": lambda: _generate_synthetic_data(symbol)},
        ]

        for source in data_sources:
            source_name = source["name"]

            # Verificar throttling excepto para datos sintéticos
            if source_name != "synthetic" and not throttling_controller.can_request(
                source_name
            ):
                logger.info(f"Fuente {source_name} limitada por throttling, saltando")
                continue

            try:
                logger.info(f"Intentando con fuente alternativa: {source_name}")
                data = source["func"]()

                if data is not None and not data.empty:
                    data = validate_and_fix_data(data)
                    # Guardar en caché
                    _data_cache.set(cache_key, data)
                    return data
            except Exception as source_error:
                # Reportar error al sistema de throttling
                if source_name != "synthetic":
                    throttling_controller.report_error(source_name)
                logger.error(f"Error en fuente {source_name}: {str(source_error)}")
                continue

        # Si absolutamente todo falla, retornar DataFrame vacío
        logger.error(f"Todas las fuentes de datos fallaron para {symbol}")
        return pd.DataFrame()


class StockDataAnalyzer:
    """Analizador de datos de acciones para trading de opciones"""

    def __init__(self, api_keys: Dict[str, str] = None):
        # Si no se proporcionaron claves, intentar obtenerlas de secrets.toml
        if not api_keys:
            api_keys = get_api_keys_from_secrets()

        self.web_search = WebSearchEngine(api_keys)
        self.scraper = WebScraper()
        self.cache = _data_cache
        self.api_keys = api_keys

    def get_stock_technical_data(
        self, symbol: str, period: str = "6mo", interval: str = "1d"
    ) -> pd.DataFrame:
        """Obtiene datos técnicos históricos de una acción"""
        cache_key = f"tech_data_{symbol}_{period}_{interval}"
        cached_data = self.cache.get(cache_key)

        if cached_data is not None:
            return pd.DataFrame(cached_data)

        try:
            # Obtener datos históricos usando el sistema optimizado
            data = fetch_market_data(symbol, period=period, interval=interval)

            if data.empty:
                return pd.DataFrame()

            # Calcular indicadores técnicos
            # Medias móviles
            data["SMA_20"] = data["Close"].rolling(window=20).mean()
            data["SMA_50"] = data["Close"].rolling(window=50).mean()
            data["SMA_200"] = data["Close"].rolling(window=200).mean()

            # RSI
            delta = data["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            data["RSI"] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = data["Close"].ewm(span=12, adjust=False).mean()
            exp2 = data["Close"].ewm(span=26, adjust=False).mean()
            data["MACD"] = exp1 - exp2
            data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

            # Bollinger Bands
            data["BB_MA20"] = data["Close"].rolling(window=20).mean()
            std_dev = data["Close"].rolling(window=20).std()
            data["BB_Upper"] = data["BB_MA20"] + (std_dev * 2)
            data["BB_Lower"] = data["BB_MA20"] - (std_dev * 2)

            # Cálculo de ATR (Average True Range) para medir volatilidad
            data["TR"] = np.maximum(
                np.maximum(
                    data["High"] - data["Low"],
                    np.abs(data["High"] - data["Close"].shift()),
                ),
                np.abs(data["Low"] - data["Close"].shift()),
            )
            data["ATR"] = data["TR"].rolling(window=14).mean()

            # Volumen relativo
            data["Volume_SMA"] = data["Volume"].rolling(window=20).mean()
            data["Volume_Ratio"] = data["Volume"] / data["Volume_SMA"]

            # Almacenar en caché
            self.cache.set(cache_key, data.reset_index().to_dict("records"))

            return data

        except Exception as e:
            logger.error(f"Error obteniendo datos técnicos para {symbol}: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()

    def get_news_sentiment(self, symbol: str) -> Dict:
        """Analiza sentimiento de noticias para una acción"""
        cache_key = f"news_sentiment_{symbol}"
        cached_data = self.cache.get(cache_key)

        if cached_data:
            return cached_data

        try:
            # Primero intentar con Finnhub si está disponible
            if "finnhub" in self.api_keys and throttling_controller.can_request(
                "finnhub"
            ):
                finnhub_sentiment = self.scraper.get_finnhub_sentiment(symbol)
                if finnhub_sentiment and "sentiment" in finnhub_sentiment:
                    self.cache.set(cache_key, finnhub_sentiment)
                    return finnhub_sentiment

            # Obtener noticias recientes
            news = self.scraper.scrape_financial_news(symbol, max_news=10)

            if not news:
                return {"symbol": symbol, "sentiment": "neutral", "score": 0.5}

            # Extraer títulos para análisis de sentimiento
            titles = [item["title"] for item in news]

            # Análisis de sentimiento simple basado en palabras clave
            positive_words = [
                "bull",
                "bullish",
                "buy",
                "outperform",
                "strong",
                "positive",
                "upgrade",
                "growth",
                "profit",
                "gain",
                "rise",
                "up",
                "higher",
                "beat",
                "exceed",
            ]

            negative_words = [
                "bear",
                "bearish",
                "sell",
                "underperform",
                "weak",
                "negative",
                "downgrade",
                "decline",
                "loss",
                "fall",
                "down",
                "lower",
                "miss",
                "disappoint",
            ]

            # Contar palabras positivas y negativas
            positive_count = 0
            negative_count = 0

            for title in titles:
                title_lower = title.lower()
                for word in positive_words:
                    if word in title_lower:
                        positive_count += 1

                for word in negative_words:
                    if word in title_lower:
                        negative_count += 1

            total_count = positive_count + negative_count
            if total_count > 0:
                sentiment_score = positive_count / total_count
            else:
                sentiment_score = 0.5  # Neutral

            # Determinar sentimiento
            if sentiment_score > 0.6:
                sentiment = "bullish"
            elif sentiment_score < 0.4:
                sentiment = "bearish"
            else:
                sentiment = "neutral"

            result = {
                "symbol": symbol,
                "sentiment": sentiment,
                "score": sentiment_score,
                "positive_mentions": positive_count,
                "negative_mentions": negative_count,
                "total_analyzed": len(titles),
            }

            # Almacenar en caché
            self.cache.set(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Error analizando sentimiento para {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "sentiment": "neutral",
                "score": 0.5,
                "error": str(e),
            }

    def get_options_recommendation(self, symbol: str) -> Dict:
        """Genera recomendación para operaciones de opciones"""
        try:
            # Recopilar todos los datos relevantes
            technical_data = self.get_stock_technical_data(symbol)
            fundamental_data = self.scraper.scrape_yahoo_finance(symbol)
            news_sentiment = self.get_news_sentiment(symbol)

            # Buscar información relevante en la web
            search_query = f"{symbol} stock options analysis forecast"
            web_results = self.web_search.perform_web_search(
                search_query, max_results=5
            )

            # Puntuación para CALL
            call_score = 0

            # Factores técnicos (40% del peso total)
            if not technical_data.empty:
                latest = technical_data.iloc[-1]

                # Tendencia de precios (precio sobre medias móviles)
                if latest["Close"] > latest["SMA_20"]:
                    call_score += 5  # +5%
                if latest["Close"] > latest["SMA_50"]:
                    call_score += 5  # +5%
                if latest["Close"] > latest["SMA_200"]:
                    call_score += 5  # +5%

                # RSI (sobrecompra/sobreventa)
                if latest["RSI"] < 30:  # Sobreventa = señal alcista
                    call_score += 10  # +10%
                elif latest["RSI"] > 70:  # Sobrecompra = señal bajista
                    call_score -= 10  # -10%

                # MACD
                if latest["MACD"] > latest["MACD_Signal"]:
                    call_score += 7.5  # +7.5%
                else:
                    call_score -= 7.5  # -7.5%

                # Momentum (últimos 5 días)
                price_5d_ago = (
                    technical_data.iloc[-6]["Close"]
                    if len(technical_data) > 5
                    else technical_data.iloc[0]["Close"]
                )
                price_change_5d = (latest["Close"] / price_5d_ago - 1) * 100

                if price_change_5d > 5:  # Fuerte momentum positivo
                    call_score += 5  # +5%
                elif price_change_5d < -5:  # Fuerte momentum negativo
                    call_score -= 5  # -5%

                # ATR - Volatilidad
                if "ATR" in latest and "Close" in latest:
                    atr_pct = (latest["ATR"] / latest["Close"]) * 100
                    if atr_pct > 3:  # Alta volatilidad reduce confianza
                        call_score *= 0.9  # Reducir score en 10%

            # Factores fundamentales (30% del peso total)
            if fundamental_data and "error" not in fundamental_data:
                # Analizar PE ratio si está disponible
                if "pe_ratio" in fundamental_data["fundamentals"]:
                    try:
                        pe_ratio = float(fundamental_data["fundamentals"]["pe_ratio"])
                        if pe_ratio < 15:  # PE bajo = potencialmente infravalorado
                            call_score += 10  # +10%
                        elif pe_ratio > 30:  # PE alto = potencialmente sobrevalorado
                            call_score -= 10  # -10%
                    except (ValueError, TypeError):
                        pass

                # Volumen relativo
                if (
                    "volume" in fundamental_data["fundamentals"]
                    and "avg_volume" in fundamental_data["fundamentals"]
                ):
                    try:
                        vol = float(
                            fundamental_data["fundamentals"]["volume"].replace(",", "")
                        )
                        avg_vol = float(
                            fundamental_data["fundamentals"]["avg_volume"].replace(
                                ",", ""
                            )
                        )
                        if vol > avg_vol * 1.5:  # Alto volumen = señal fuerte
                            # La dirección depende del precio
                            if fundamental_data["price"].get("change_percent", 0) > 0:
                                call_score += 10  # +10% para tendencia alcista
                            else:
                                call_score -= 10  # -10% para tendencia bajista
                    except (ValueError, TypeError, KeyError):
                        pass

            # Sentimiento de noticias (20% del peso total)
            if news_sentiment["sentiment"] == "bullish":
                call_score += 20  # +20%
            elif news_sentiment["sentiment"] == "bearish":
                call_score -= 20  # -20%

            # Información web (10% del peso total)
            web_sentiment_score = 0
            bullish_count = 0
            bearish_count = 0

            if web_results:
                # Analizar contenido en busca de señales
                bullish_keywords = [
                    "buy",
                    "call",
                    "bullish",
                    "upside",
                    "growth",
                    "positive",
                    "outperform",
                ]
                bearish_keywords = [
                    "sell",
                    "put",
                    "bearish",
                    "downside",
                    "decline",
                    "negative",
                    "underperform",
                ]

                for result in web_results:
                    content = result["content"].lower()
                    for word in bullish_keywords:
                        bullish_count += content.count(word)
                    for word in bearish_keywords:
                        bearish_count += content.count(word)

                total_count = bullish_count + bearish_count
                if total_count > 0:
                    web_ratio = bullish_count / total_count
                    web_sentiment_score = (web_ratio - 0.5) * 20  # -10% a +10%

            call_score += web_sentiment_score

            # Normalizar el score entre 0-100
            call_score = max(
                min(call_score + 50, 100), 0
            )  # Convertir de -50/+50 a 0-100

            # Determinar recomendación final
            if call_score >= 65:
                recommendation = "CALL"
                confidence = "alta" if call_score >= 80 else "media"
                timeframe = "corto a medio plazo"
            elif call_score <= 35:
                recommendation = "PUT"
                confidence = "alta" if call_score <= 20 else "media"
                timeframe = "corto a medio plazo"
            else:
                recommendation = "NEUTRAL"
                confidence = "baja"
                timeframe = "esperar mejores condiciones"

            # Formatear resultado
            return {
                "symbol": symbol,
                "recommendation": recommendation,
                "confidence": confidence,
                "score": call_score,
                "timeframe": timeframe,
                "analysis_date": datetime.now().isoformat(),
                "technical_factors": {
                    "price_vs_sma20": (
                        technical_data.iloc[-1]["Close"]
                        > technical_data.iloc[-1]["SMA_20"]
                        if not technical_data.empty
                        else None
                    ),
                    "price_vs_sma50": (
                        technical_data.iloc[-1]["Close"]
                        > technical_data.iloc[-1]["SMA_50"]
                        if not technical_data.empty
                        else None
                    ),
                    "price_vs_sma200": (
                        technical_data.iloc[-1]["Close"]
                        > technical_data.iloc[-1]["SMA_200"]
                        if not technical_data.empty
                        else None
                    ),
                    "rsi": (
                        technical_data.iloc[-1]["RSI"]
                        if not technical_data.empty
                        else None
                    ),
                    "macd_signal": (
                        "bullish"
                        if not technical_data.empty
                        and technical_data.iloc[-1]["MACD"]
                        > technical_data.iloc[-1]["MACD_Signal"]
                        else "bearish"
                    ),
                },
                "fundamental_factors": fundamental_data.get("fundamentals", {}),
                "news_sentiment": news_sentiment,
                "web_analysis": {
                    "bullish_mentions": bullish_count,
                    "bearish_mentions": bearish_count,
                },
            }

        except Exception as e:
            logger.error(
                f"Error generando recomendación de opciones para {symbol}: {str(e)}"
            )
            traceback.print_exc()
            return {
                "symbol": symbol,
                "recommendation": "ERROR",
                "confidence": "baja",
                "score": 50,
                "error": str(e),
            }


# Función principal para análisis completo
def analyze_stock_options(symbol: str, api_keys: Dict[str, str] = None) -> Dict:
    """
    Función principal para análisis completo de opciones de un símbolo.

    Args:
        symbol: Símbolo de la acción a analizar
        api_keys: Diccionario opcional con claves API

    Returns:
        Diccionario con análisis completo y recomendación
    """
    try:
        # Si no se proporcionaron API keys, obtenerlas de secrets.toml
        if not api_keys:
            api_keys = get_api_keys_from_secrets()

        analyzer = StockDataAnalyzer(api_keys)

        # Generar recomendación
        recommendation = analyzer.get_options_recommendation(symbol)

        # Obtener datos técnicos para visualización
        technical_data = analyzer.get_stock_technical_data(symbol)
        if not technical_data.empty:
            # Convertir a diccionario para transporte
            recommendation["chart_data"] = technical_data.reset_index().to_dict(
                "records"
            )

        # Obtener noticias relacionadas
        news = analyzer.scraper.scrape_financial_news(symbol)
        recommendation["news"] = news

        # Obtener resultados de búsqueda web
        search_query = f"{symbol} stock options analysis"
        web_results = analyzer.web_search.perform_web_search(search_query)
        recommendation["web_results"] = web_results

        return recommendation

    except Exception as e:
        logger.error(f"Error en analyze_stock_options para {symbol}: {str(e)}")
        traceback.print_exc()
        return {
            "symbol": symbol,
            "recommendation": "ERROR",
            "confidence": "baja",
            "score": 50,
            "error": str(e),
        }
