"""
Market Data Throttling System para MarketIntel Options Analyzer
Controla tasas de solicitud y maneja fallbacks para garantizar disponibilidad continua
"""

import time
import random
from datetime import datetime, timedelta
import logging

# Configurar logging específico para throttling
logger = logging.getLogger("market_data_throttling")
logger.setLevel(logging.INFO)


class ThrottlingController:
    """Controlador avanzado de tasas de solicitud para proveedores de datos de mercado"""

    def __init__(self):
        # Tasas máximas de solicitud por minuto por proveedor
        self.rate_limits = {
            "yfinance": 10,  # Límite conservador para Yahoo Finance
            "alphavantage": 5,  # Límite de API gratuita
            "alpha_vantage": 5,  # Alias para mayor compatibilidad
            "finnhub": 30,  # Límite de plan básico
            "tavily": 5,  # Límite conservador
            "duckduckgo": 3,  # Muy restrictivo con web scraping
            "you": 5,  # Límite conservador para YOU API
            "finviz": 8,  # Finviz scraping
            "marketstack": 5,  # MarketStack API
            "synthetic": 9999,  # Datos sintéticos (sin límite real)
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

        # Mapeo de alias para normalización de nombres de proveedores
        self.provider_aliases = {
            "alpha_vantage": "alphavantage",
            "av": "alphavantage",
            "yahoo": "yfinance",
            "yf": "yfinance",
            "ddg": "duckduckgo",
            "duck": "duckduckgo",
            "financial_news": "finviz",
            "news": "finviz",
            "ms": "marketstack",
        }

        logger.info(
            f"ThrottlingController inicializado con {len(self.rate_limits)} proveedores"
        )

    def normalize_provider_name(self, provider: str) -> str:
        """Normaliza el nombre del proveedor para manejar diferentes variantes"""
        provider_lower = provider.lower()

        # Verificar si es un alias conocido
        if provider_lower in self.provider_aliases:
            return self.provider_aliases[provider_lower]

        # Intentar encontrar coincidencia parcial
        for known_provider in self.rate_limits.keys():
            if (
                known_provider.lower() in provider_lower
                or provider_lower in known_provider.lower()
            ):
                return known_provider

        return provider

    def can_request(self, provider: str) -> bool:
        """Determina si se puede realizar una solicitud al proveedor especificado"""
        normalized_provider = self.normalize_provider_name(provider)

        if normalized_provider not in self.rate_limits:
            # Si no tenemos límite registrado, permitimos con advertencia
            logger.warning(
                f"Proveedor desconocido: {provider} (normalizado: {normalized_provider}). Permitiendo solicitud."
            )
            return True

        self.total_requests[normalized_provider] += 1

        # Verificar si el proveedor está en cooldown por errores previos
        if not self.provider_status[normalized_provider]:
            last_error = (
                self.provider_errors[normalized_provider][-1]
                if self.provider_errors[normalized_provider]
                else datetime.min
            )
            if (datetime.now() - last_error).total_seconds() < self.error_cooldown:
                logger.info(
                    f"Proveedor {normalized_provider} en cooldown por errores previos."
                )
                self.throttled_requests[normalized_provider] += 1
                return False
            else:
                # Restaurar proveedor después del período de cooldown
                self.provider_status[normalized_provider] = True

        # Limpiar ventana de tiempo (eliminar marcas de tiempo antiguas)
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        self.request_windows[normalized_provider] = [
            ts for ts in self.request_windows[normalized_provider] if ts > cutoff
        ]

        # Verificar si estamos dentro del límite
        if (
            len(self.request_windows[normalized_provider])
            < self.rate_limits[normalized_provider]
        ):
            # Añadir jitter aleatorio para evitar patrones de solicitud
            jitter = random.uniform(*self.jitter_range)
            time.sleep(jitter)

            # Registrar solicitud en la ventana
            self.request_windows[normalized_provider].append(now)
            return True
        else:
            # Solicitud throttled
            self.throttled_requests[normalized_provider] += 1
            logger.info(f"Solicitud a {normalized_provider} limitada por rate limit.")
            return False

    def get_backoff_time(self, provider: str) -> float:
        """Calcula tiempo de espera adaptativo basado en la congestión del proveedor"""
        normalized_provider = self.normalize_provider_name(provider)

        if normalized_provider not in self.rate_limits:
            return 1.0

        # Número de solicitudes en ventana actual
        request_count = len(self.request_windows[normalized_provider])

        # Si estamos cerca del límite, retornar tiempo proporcional
        limit = self.rate_limits[normalized_provider]
        if request_count > 0:
            utilization = request_count / limit
            # Fórmula exponencial para backoff
            return max(1.0, 5.0 * (2 ** (utilization * 2) - 1))
        return 1.0

    def report_error(self, provider: str):
        """Reporta un error con un proveedor para activar cooldown"""
        normalized_provider = self.normalize_provider_name(provider)

        if normalized_provider not in self.rate_limits:
            logger.warning(f"Reporte de error para proveedor desconocido: {provider}")
            return

        self.provider_errors[normalized_provider].append(datetime.now())
        # Si acumulamos más de 3 errores en 5 minutos, deshabilitar temporalmente
        recent_errors = [
            ts
            for ts in self.provider_errors[normalized_provider]
            if (datetime.now() - ts).total_seconds() < 300
        ]

        if len(recent_errors) >= 3:
            self.provider_status[normalized_provider] = False
            logger.warning(
                f"Proveedor {normalized_provider} deshabilitado temporalmente por errores frecuentes."
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
        normalized_provider = self.normalize_provider_name(provider)

        alternatives = {
            "yfinance": ["alphavantage", "finnhub"],
            "alphavantage": ["yfinance", "finnhub"],
            "finnhub": ["alphavantage", "yfinance"],
            "tavily": ["duckduckgo", "you"],
            "duckduckgo": ["tavily", "you"],
            "you": ["tavily", "duckduckgo"],
            "finviz": ["alphavantage"],
            "marketstack": ["alphavantage", "yfinance"],
        }

        if normalized_provider not in alternatives:
            return None

        # Buscar alternativa disponible con menor utilización
        candidates = []
        for alt in alternatives[normalized_provider]:
            if alt in self.provider_status and self.provider_status[alt]:
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


# Instancia global del controlador de throttling
throttling_controller = ThrottlingController()
