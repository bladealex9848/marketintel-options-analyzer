"""
Generador de Datos Sintéticos para Market Intel Options Analyzer
Sistema avanzado para crear datos de mercado con características realistas cuando las APIs fallan
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import hashlib
import math

logger = logging.getLogger(__name__)


class SyntheticMarketDataGenerator:
    """
    Generador de datos de mercado sintéticos con características realistas
    para trading de opciones y análisis técnico.
    """

    def __init__(self):
        # Características base por sector/categoría
        self.sector_characteristics = {
            "tech": {
                "volatility": (0.015, 0.035),  # (min, max) daily volatility
                "trend_bias": (0.0002, 0.0010),  # (min, max) daily drift
                "price_range": (80, 800),  # (min, max) price range
                "volume_profile": "high",  # volume characteristic
            },
            "finance": {
                "volatility": (0.008, 0.025),
                "trend_bias": (0.0001, 0.0006),
                "price_range": (30, 300),
                "volume_profile": "medium",
            },
            "energy": {
                "volatility": (0.012, 0.030),
                "trend_bias": (0.0000, 0.0008),
                "price_range": (40, 250),
                "volume_profile": "medium-high",
            },
            "healthcare": {
                "volatility": (0.010, 0.028),
                "trend_bias": (0.0002, 0.0007),
                "price_range": (50, 400),
                "volume_profile": "medium",
            },
            "consumer": {
                "volatility": (0.010, 0.020),
                "trend_bias": (0.0001, 0.0005),
                "price_range": (40, 200),
                "volume_profile": "medium-low",
            },
            "index": {
                "volatility": (0.005, 0.018),
                "trend_bias": (0.0001, 0.0004),
                "price_range": (250, 5000),
                "volume_profile": "very-high",
            },
            "crypto": {
                "volatility": (0.025, 0.060),
                "trend_bias": (0.0002, 0.0015),
                "price_range": (20, 150),
                "volume_profile": "extreme",
            },
            "commodity": {
                "volatility": (0.010, 0.025),
                "trend_bias": (0.0000, 0.0006),
                "price_range": (50, 300),
                "volume_profile": "low-medium",
            },
            "default": {
                "volatility": (0.010, 0.025),
                "trend_bias": (0.0001, 0.0006),
                "price_range": (50, 250),
                "volume_profile": "medium",
            },
        }

        # Mapeo de símbolos a sectores
        self.symbol_sectors = {
            # Índices
            "SPY": "index",
            "QQQ": "index",
            "DIA": "index",
            "IWM": "index",
            "EFA": "index",
            "VWO": "index",
            "IYR": "index",
            "XLE": "index",
            "XLF": "index",
            "XLV": "index",
            # Tecnología
            "AAPL": "tech",
            "MSFT": "tech",
            "GOOGL": "tech",
            "AMZN": "tech",
            "TSLA": "tech",
            "NVDA": "tech",
            "META": "tech",
            "NFLX": "tech",
            "PYPL": "tech",
            "CRM": "tech",
            # Finanzas
            "JPM": "finance",
            "BAC": "finance",
            "WFC": "finance",
            "C": "finance",
            "GS": "finance",
            "MS": "finance",
            "V": "finance",
            "MA": "finance",
            "AXP": "finance",
            "BLK": "finance",
            # Energía
            "XOM": "energy",
            "CVX": "energy",
            "SHEL": "energy",
            "TTE": "energy",
            "COP": "energy",
            "EOG": "energy",
            "PXD": "energy",
            "DVN": "energy",
            "MPC": "energy",
            "PSX": "energy",
            # Salud
            "JNJ": "healthcare",
            "UNH": "healthcare",
            "PFE": "healthcare",
            "MRK": "healthcare",
            "ABBV": "healthcare",
            "LLY": "healthcare",
            "AMGN": "healthcare",
            "BMY": "healthcare",
            "GILD": "healthcare",
            "TMO": "healthcare",
            # Consumo
            "MCD": "consumer",
            "SBUX": "consumer",
            "NKE": "consumer",
            "TGT": "consumer",
            "HD": "consumer",
            "LOW": "consumer",
            "TJX": "consumer",
            "ROST": "consumer",
            "CMG": "consumer",
            "DHI": "consumer",
            # Cripto ETFs
            "BITO": "crypto",
            "GBTC": "crypto",
            "ETHE": "crypto",
            "ARKW": "crypto",
            "BLOK": "crypto",
            # Materias Primas
            "GLD": "commodity",
            "SLV": "commodity",
            "USO": "commodity",
            "UNG": "commodity",
            "CORN": "commodity",
            "SOYB": "commodity",
            "WEAT": "commodity",
        }

        # Parámetros para ciclos de mercado
        self.market_cycles = {
            "bull": {
                "trend_multiplier": 2.0,  # Amplifica tendencia alcista
                "volatility_multiplier": 0.8,  # Reduce volatilidad
                "volume_profile": "increasing",  # Volumen creciente
                "gap_probability": 0.05,  # Probabilidad de gaps alcistas
                "gap_size": (0.01, 0.03),  # Tamaño típico de gaps (%)
            },
            "bear": {
                "trend_multiplier": -1.5,  # Crea tendencia bajista
                "volatility_multiplier": 1.5,  # Aumenta volatilidad
                "volume_profile": "spike-decline",  # Picos de volumen seguidos de declive
                "gap_probability": 0.08,  # Probabilidad de gaps bajistas
                "gap_size": (0.02, 0.05),  # Tamaño típico de gaps (%)
            },
            "sideways": {
                "trend_multiplier": 0.2,  # Tendencia muy débil
                "volatility_multiplier": 0.7,  # Volatilidad reducida
                "volume_profile": "declining",  # Volumen decreciente
                "gap_probability": 0.02,  # Baja probabilidad de gaps
                "gap_size": (0.005, 0.015),  # Gaps pequeños
            },
            "volatile": {
                "trend_multiplier": 0.5,  # Tendencia moderada
                "volatility_multiplier": 2.0,  # Alta volatilidad
                "volume_profile": "erratic",  # Volumen errático
                "gap_probability": 0.10,  # Alta probabilidad de gaps
                "gap_size": (0.01, 0.04),  # Gaps potencialmente grandes
            },
        }

    def get_symbol_characteristics(self, symbol: str) -> dict:
        """
        Determina las características del símbolo basado en su sector
        y características específicas.
        """
        # Determinar sector
        sector = self.symbol_sectors.get(symbol, "default")
        base_characteristics = self.sector_characteristics.get(
            sector, self.sector_characteristics["default"]
        )

        # Factores específicos del símbolo
        # (determinismo basado en hash del símbolo para consistencia entre generaciones)
        symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest(), 16)

        # Seed para reproducibilidad
        np.random.seed(symbol_hash)

        # Determinar ciclo de mercado basado en hash para consistencia
        market_cycle_types = list(self.market_cycles.keys())
        cycle_index = symbol_hash % len(market_cycle_types)
        market_cycle = market_cycle_types[cycle_index]
        cycle_characteristics = self.market_cycles[market_cycle]

        # Combinar características base y ciclo
        volatility_range = base_characteristics["volatility"]
        base_volatility = np.random.uniform(volatility_range[0], volatility_range[1])
        adjusted_volatility = (
            base_volatility * cycle_characteristics["volatility_multiplier"]
        )

        trend_range = base_characteristics["trend_bias"]
        base_trend = np.random.uniform(trend_range[0], trend_range[1])
        adjusted_trend = base_trend * cycle_characteristics["trend_multiplier"]

        price_range = base_characteristics["price_range"]
        base_price = np.random.uniform(price_range[0], price_range[1])

        # Construir características finales
        characteristics = {
            "symbol": symbol,
            "sector": sector,
            "market_cycle": market_cycle,
            "volatility": adjusted_volatility,
            "trend": adjusted_trend,
            "base_price": base_price,
            "volume_profile": cycle_characteristics["volume_profile"],
            "gap_probability": cycle_characteristics["gap_probability"],
            "gap_size": cycle_characteristics["gap_size"],
        }

        return characteristics

    def generate_synthetic_data(
        self, symbol: str, periods: int = 180, frequency: str = "D"
    ) -> pd.DataFrame:
        """
        Genera datos sintéticos realistas para el símbolo especificado.

        Args:
            symbol: Símbolo para el que generar datos
            periods: Número de períodos a generar
            frequency: Frecuencia de datos ('D' para diario, 'H' para horario, etc.)

        Returns:
            DataFrame con datos OHLCV sintéticos
        """
        characteristics = self.get_symbol_characteristics(symbol)

        # Definir fechas
        end_date = datetime.now()
        if frequency == "D":
            start_date = end_date - timedelta(days=periods)
            dates = pd.date_range(
                start=start_date, end=end_date, freq="B"
            )  # Días hábiles
        elif frequency == "H":
            start_date = end_date - timedelta(hours=periods)
            # Generar solo horas de mercado (9:30 AM - 4:00 PM EST)
            market_hours = []
            current = start_date
            while current <= end_date:
                hour = current.hour
                if (
                    9 <= hour <= 16 and current.weekday() < 5
                ):  # Horas de mercado en días hábiles
                    market_hours.append(current)
                current += timedelta(hours=1)
            dates = pd.DatetimeIndex(market_hours)
        else:
            # Por defecto, usar días
            start_date = end_date - timedelta(days=periods)
            dates = pd.date_range(start=start_date, end=end_date, freq="B")

        # Ajustar para tener exactamente el número de períodos solicitados
        if len(dates) > periods:
            dates = dates[-periods:]

        # Generar precios con proceso de Ornstein-Uhlenbeck modificado para mayor realismo
        volatility = characteristics["volatility"]
        trend = characteristics["trend"]
        mean_reversion = (
            0.05  # Parámetro de reversión a la media (bajo para tendencias más claras)
        )
        price = characteristics["base_price"]

        # Listas para almacenar resultados
        prices = []
        price_open = []
        price_high = []
        price_low = []
        price_close = []
        volumes = []

        # Generar serie de precios estocástica
        previous_price = price
        for i in range(len(dates)):
            # Precio de apertura (con gap ocasional)
            if i == 0:
                open_price = price
            else:
                # Posible gap overnight
                if np.random.random() < characteristics["gap_probability"]:
                    gap_direction = 1 if np.random.random() > 0.5 else -1
                    gap_min, gap_max = characteristics["gap_size"]
                    gap_pct = np.random.uniform(gap_min, gap_max) * gap_direction
                    open_price = price_close[-1] * (1 + gap_pct)
                else:
                    # Ligera variación desde el cierre anterior
                    open_price = price_close[-1] * (
                        1 + np.random.normal(0, volatility * 0.3)
                    )

            # Proceso de Ornstein-Uhlenbeck para el precio diario
            ou_drift = trend + mean_reversion * (
                characteristics["base_price"] - open_price
            )
            price = open_price * (1 + np.random.normal(ou_drift, volatility))

            # Añadir comportamiento intradiario realista
            daily_volatility = volatility * np.random.uniform(
                0.8, 1.2
            )  # Volatilidad variable
            daily_range = open_price * daily_volatility * 2  # Rango aproximado diario

            # Establecer high, low, close con patrón realista
            direction = np.sign(price - open_price)  # Dirección del movimiento

            # Precio más alto
            high_range = abs(price - open_price) + daily_range * np.random.uniform(
                0.2, 0.5
            )
            high_price = max(open_price, price) + high_range * np.random.uniform(
                0.2, 0.6
            )

            # Precio más bajo
            low_range = abs(price - open_price) + daily_range * np.random.uniform(
                0.2, 0.5
            )
            low_price = min(open_price, price) - low_range * np.random.uniform(0.2, 0.6)

            # Asegurar que el precio de cierre está dentro del rango
            close_price = max(low_price, min(high_price, price))

            # Generar volumen basado en perfil
            volume_profile = characteristics["volume_profile"]
            base_volume = 1000000  # Volumen base

            if volume_profile == "high":
                volume = base_volume * np.random.uniform(5, 15)
            elif volume_profile == "medium-high":
                volume = base_volume * np.random.uniform(3, 8)
            elif volume_profile == "medium":
                volume = base_volume * np.random.uniform(1.5, 5)
            elif volume_profile == "medium-low":
                volume = base_volume * np.random.uniform(0.8, 3)
            elif volume_profile == "low":
                volume = base_volume * np.random.uniform(0.3, 1.5)
            elif volume_profile == "very-high":
                volume = base_volume * np.random.uniform(10, 30)
            elif volume_profile == "extreme":
                volume = base_volume * np.random.uniform(15, 50)
            elif volume_profile == "increasing":
                volume = base_volume * np.random.uniform(1, 8) * (1 + i / len(dates))
            elif volume_profile == "declining":
                volume = base_volume * np.random.uniform(1, 8) * (2 - i / len(dates))
            elif volume_profile == "erratic":
                spike = np.random.random() < 0.2  # 20% de probabilidad de pico
                volume = base_volume * (
                    np.random.uniform(5, 20) if spike else np.random.uniform(1, 5)
                )
            elif volume_profile == "spike-decline":
                spike = abs(close_price - open_price) > (volatility * open_price * 1.5)
                volume = base_volume * (
                    np.random.uniform(8, 25) if spike else np.random.uniform(2, 6)
                )
            else:
                volume = base_volume * np.random.uniform(1, 5)  # Default

            # Volumen entero
            volume = int(volume)

            # Añadir a listas
            prices.append(price)
            price_open.append(open_price)
            price_high.append(high_price)
            price_low.append(low_price)
            price_close.append(close_price)
            volumes.append(volume)

            previous_price = price

        # Crear DataFrame
        df = pd.DataFrame(
            {
                "Open": price_open,
                "High": price_high,
                "Low": price_low,
                "Close": price_close,
                "Volume": volumes,
            },
            index=dates,
        )

        # Añadir columna Adj Close
        df["Adj Close"] = df["Close"]

        # Agregar metadata para identificar como sintético
        df.attrs["synthetic"] = True
        df.attrs["generated_at"] = datetime.now().isoformat()
        df.attrs["symbol"] = symbol
        df.attrs["market_cycle"] = characteristics["market_cycle"]

        logger.info(
            f"Datos sintéticos generados para {symbol} - Ciclo: {characteristics['market_cycle']}"
        )

        return df


# Instancia global
synthetic_data_generator = SyntheticMarketDataGenerator()


# Función simplificada para reutilización
def generate_synthetic_market_data(
    symbol: str, periods: int = 180, frequency: str = "D"
) -> pd.DataFrame:
    """Genera datos de mercado sintéticos para el símbolo especificado"""
    return synthetic_data_generator.generate_synthetic_data(symbol, periods, frequency)


# Función para crear gráficas de mercado realistas
def create_market_pattern(pattern_type: str, periods: int = 180) -> pd.DataFrame:
    """
    Crea un patrón de mercado específico para backtest de estrategias.

    Args:
        pattern_type: Tipo de patrón ('double_top', 'head_shoulders', 'triangle', etc.)
        periods: Número de períodos

    Returns:
        DataFrame con el patrón solicitado
    """
    # Base para todos los patrones
    dates = pd.date_range(end=datetime.now(), periods=periods, freq="B")
    df = pd.DataFrame(index=dates)

    # Generación del patrón
    if pattern_type == "double_top":
        # Generación de double top
        x = np.linspace(0, 1, periods)
        base_trend = 100 + 20 * np.sin(x * 4 * np.pi + 1.5) + 10 * x
        noise = np.random.normal(0, 1, periods)

        close = base_trend + noise

    elif pattern_type == "head_shoulders":
        # Generación de head and shoulders
        x = np.linspace(0, 1, periods)
        base = 100 + 10 * np.sin(x * 6 * np.pi) + 8 * (1 - x)
        # Añadir "cabeza" más alta en el medio
        head_pos = periods // 2
        head_width = periods // 10
        base[head_pos - head_width : head_pos + head_width] += 5

        noise = np.random.normal(0, 0.8, periods)
        close = base + noise

    elif pattern_type == "triangle":
        # Generación de triángulo convergente
        x = np.linspace(0, 1, periods)
        amplitude = 20 * (1 - x)  # amplitud decreciente
        base = 100 + amplitude * np.sin(x * 8 * np.pi)

        noise = np.random.normal(0, 0.5, periods)
        close = base + noise

    elif pattern_type == "breakout":
        # Generación de breakout
        x = np.linspace(0, 1, periods)
        consolidation_point = int(periods * 0.7)
        base = np.zeros(periods)

        # Fase de consolidación
        base[:consolidation_point] = 100 + np.random.normal(0, 1, consolidation_point)

        # Fase de breakout
        breakout_size = 15
        base[consolidation_point:] = base[consolidation_point - 1] + np.linspace(
            0, breakout_size, periods - consolidation_point
        )

        noise = np.random.normal(0, 0.8, periods)
        close = base + noise

    else:  # patrón por defecto - tendencia con ruido
        x = np.linspace(0, 1, periods)
        trend_factor = 20  # pendiente de la tendencia
        base = 100 + trend_factor * x
        noise = np.random.normal(0, 2, periods)
        close = base + noise

    # Generar resto de columnas OHLC basadas en Close
    df["Close"] = close
    df["Open"] = df["Close"].shift(1).fillna(df["Close"].iloc[0])

    volatility = np.std(np.diff(close)) * 2

    # Agregar algo de variabilidad al trading intradiario
    df["High"] = df["Close"] + abs(np.random.normal(0, volatility, periods))
    df["Low"] = df["Open"] - abs(np.random.normal(0, volatility, periods))

    # Asegurarse que High es siempre el máximo y Low es siempre el mínimo
    for i in range(len(df)):
        high = max(df["Open"].iloc[i], df["Close"].iloc[i]) + abs(
            np.random.normal(0, volatility)
        )
        low = min(df["Open"].iloc[i], df["Close"].iloc[i]) - abs(
            np.random.normal(0, volatility)
        )
        df.loc[df.index[i], "High"] = high
        df.loc[df.index[i], "Low"] = low

    # Volumen - mayor en puntos de inflexión o breakouts
    df["Volume"] = 1000000 + 500000 * np.abs(np.diff(close, prepend=close[0]))

    # Añadir Adj Close
    df["Adj Close"] = df["Close"]

    # Metadata
    df.attrs["synthetic"] = True
    df.attrs["pattern"] = pattern_type

    return df
