import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional


def detect_support_resistance(
    df: pd.DataFrame, window: int = 20, threshold: float = 0.03
) -> Tuple[List[float], List[float]]:
    """
    Detecta niveles de soporte y resistencia en un DataFrame de precios

    Args:
        df: DataFrame con datos de precios
        window: Ventana para detectar máximos y mínimos locales
        threshold: Umbral para agrupar niveles similares (% del precio)

    Returns:
        supports: Lista de niveles de soporte
        resistances: Lista de niveles de resistencia
    """
    supports = []
    resistances = []

    # Asegurarse de que el DataFrame tiene suficientes datos
    if len(df) < window * 2:
        return supports, resistances

    # Obtener precio actual para comparaciones
    close_price = df["Close"].iloc[-1]

    # Encontrar mínimos locales (soportes potenciales)
    for i in range(window, len(df) - window):
        is_min = True
        for j in range(i - window, i + window + 1):
            if j == i:
                continue
            if j < 0 or j >= len(df):
                continue
            if df["Low"].iloc[j] <= df["Low"].iloc[i]:
                is_min = False
                break

        if is_min:
            support_level = df["Low"].iloc[i]

            # Verificar si este nivel está cerca de uno ya detectado
            add_level = True
            for level in supports:
                if abs(level - support_level) / support_level < threshold:
                    add_level = False
                    break

            if add_level and support_level < close_price:
                supports.append(support_level)

    # Encontrar máximos locales (resistencias potenciales)
    for i in range(window, len(df) - window):
        is_max = True
        for j in range(i - window, i + window + 1):
            if j == i:
                continue
            if j < 0 or j >= len(df):
                continue
            if df["High"].iloc[j] >= df["High"].iloc[i]:
                is_max = False
                break

        if is_max:
            resistance_level = df["High"].iloc[i]

            # Verificar si este nivel está cerca de uno ya detectado
            add_level = True
            for level in resistances:
                if abs(level - resistance_level) / resistance_level < threshold:
                    add_level = False
                    break

            if add_level and resistance_level > close_price:
                resistances.append(resistance_level)

    # Ordenar niveles y limitar a los 5 más cercanos al precio actual
    supports = sorted(supports, reverse=True)
    resistances = sorted(resistances)

    # Filtrar por cercanía al precio actual
    if supports:
        supports = sorted(supports, key=lambda x: abs(close_price - x))[:5]
    if resistances:
        resistances = sorted(resistances, key=lambda x: abs(close_price - x))[:5]

    # Reordenar para presentación (del más cercano al más lejano del precio actual)
    supports = sorted(supports, reverse=True)
    resistances = sorted(resistances)

    return supports, resistances


def detect_trend_lines(
    df: pd.DataFrame, min_points: int = 5
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Detecta líneas de tendencia alcistas y bajistas en un DataFrame de precios

    Args:
        df: DataFrame con datos de precios
        min_points: Número mínimo de puntos para formar una línea de tendencia

    Returns:
        bullish_lines: Lista de líneas de tendencia alcistas [(x1,y1,x2,y2),...]
        bearish_lines: Lista de líneas de tendencia bajistas [(x1,y1,x2,y2),...]
    """
    bullish_lines = []
    bearish_lines = []

    # Asegurarse de que el DataFrame tiene suficientes datos
    if len(df) < min_points * 2:
        return bullish_lines, bearish_lines

    # Ventana para buscar pivotes (ajustar según timeframe)
    window = min(int(len(df) * 0.05) + 1, 20)  # Máximo 20, o 5% de los datos

    # Para líneas de tendencia alcistas (conectar mínimos)
    # Detectar potenciales puntos de pivote para los mínimos
    pivot_lows = []
    for i in range(window, len(df) - window):
        if all(
            df["Low"].iloc[i] <= df["Low"].iloc[i - j] for j in range(1, window + 1)
        ) and all(
            df["Low"].iloc[i] <= df["Low"].iloc[i + j] for j in range(1, window + 1)
        ):
            pivot_lows.append((i, df["Low"].iloc[i]))

    # Encontrar líneas de tendencia alcistas
    for i in range(len(pivot_lows) - 1):
        for j in range(i + 1, len(pivot_lows)):
            # Calcular pendiente
            x1, y1 = pivot_lows[i]
            x2, y2 = pivot_lows[j]

            if x2 <= x1:
                continue

            # Una línea de tendencia alcista debe tener pendiente positiva
            slope = (y2 - y1) / (x2 - x1)
            if slope >= 0:
                # Contar puntos que respetan la línea de tendencia
                points_above = 0
                all_points = 0

                for k in range(x1 + 1, x2):
                    expected_y = y1 + slope * (k - x1)
                    actual_y = df["Low"].iloc[k]
                    all_points += 1

                    # Un punto respeta la línea si está por encima
                    if actual_y >= expected_y * 0.995:  # 0.5% de tolerancia
                        points_above += 1

                # Si al menos el 80% de los puntos respetan la línea, es válida
                if all_points > 0 and points_above / all_points >= 0.8:
                    bullish_lines.append(((x1, y1), (x2, y2)))

    # Para líneas de tendencia bajistas (conectar máximos)
    # Detectar potenciales puntos de pivote para los máximos
    pivot_highs = []
    for i in range(window, len(df) - window):
        if all(
            df["High"].iloc[i] >= df["High"].iloc[i - j] for j in range(1, window + 1)
        ) and all(
            df["High"].iloc[i] >= df["High"].iloc[i + j] for j in range(1, window + 1)
        ):
            pivot_highs.append((i, df["High"].iloc[i]))

    # Encontrar líneas de tendencia bajistas
    for i in range(len(pivot_highs) - 1):
        for j in range(i + 1, len(pivot_highs)):
            # Calcular pendiente
            x1, y1 = pivot_highs[i]
            x2, y2 = pivot_highs[j]

            if x2 <= x1:
                continue

            # Una línea de tendencia bajista debe tener pendiente negativa
            slope = (y2 - y1) / (x2 - x1)
            if slope <= 0:
                # Contar puntos que respetan la línea de tendencia
                points_below = 0
                all_points = 0

                for k in range(x1 + 1, x2):
                    expected_y = y1 + slope * (k - x1)
                    actual_y = df["High"].iloc[k]
                    all_points += 1

                    # Un punto respeta la línea si está por debajo
                    if actual_y <= expected_y * 1.005:  # 0.5% de tolerancia
                        points_below += 1

                # Si al menos el 80% de los puntos respetan la línea, es válida
                if all_points > 0 and points_below / all_points >= 0.8:
                    bearish_lines.append(((x1, y1), (x2, y2)))

    # Limitar a las 3 líneas más recientes
    bullish_lines = (
        sorted(bullish_lines, key=lambda x: x[1][0])[-3:] if bullish_lines else []
    )
    bearish_lines = (
        sorted(bearish_lines, key=lambda x: x[1][0])[-3:] if bearish_lines else []
    )

    return bullish_lines, bearish_lines


def detect_channels(
    df: pd.DataFrame, bullish_lines: List[Tuple], bearish_lines: List[Tuple]
) -> List[Tuple]:
    """
    Detecta canales de precio basados en líneas de tendencia

    Args:
        df: DataFrame con datos de precios
        bullish_lines: Líneas de tendencia alcistas
        bearish_lines: Líneas de tendencia bajistas

    Returns:
        channels: Lista de canales [(soporte, resistencia, tipo),...]
    """
    channels = []

    # Detectar canales alcistas
    for line in bullish_lines:
        (x1, y1), (x2, y2) = line
        slope = (y2 - y1) / (x2 - x1)

        # Calcular los puntos máximos por encima de la línea
        max_distance = 0
        max_point = None

        for i in range(x1, x2 + 1):
            if i >= len(df):
                continue

            base_y = y1 + slope * (i - x1)
            distance = df["High"].iloc[i] - base_y

            if distance > max_distance:
                max_distance = distance
                max_point = (i, df["High"].iloc[i])

        if max_point and max_distance > 0:
            # Crear línea paralela en la parte superior
            mx, my = max_point

            # Puntos de la línea paralela superior
            px1 = x1
            py1 = y1 + max_distance
            px2 = x2
            py2 = y2 + max_distance

            channels.append((line, ((px1, py1), (px2, py2)), "bullish"))

    # Detectar canales bajistas
    for line in bearish_lines:
        (x1, y1), (x2, y2) = line
        slope = (y2 - y1) / (x2 - x1)

        # Calcular los puntos mínimos por debajo de la línea
        max_distance = 0
        max_point = None

        for i in range(x1, x2 + 1):
            if i >= len(df):
                continue

            base_y = y1 + slope * (i - x1)
            distance = base_y - df["Low"].iloc[i]

            if distance > max_distance:
                max_distance = distance
                max_point = (i, df["Low"].iloc[i])

        if max_point and max_distance > 0:
            # Crear línea paralela en la parte inferior
            mx, my = max_point

            # Puntos de la línea paralela inferior
            px1 = x1
            py1 = y1 - max_distance
            px2 = x2
            py2 = y2 - max_distance

            channels.append((line, ((px1, py1), (px2, py2)), "bearish"))

    # Limitar a los 2 canales más recientes
    channels = sorted(channels, key=lambda x: x[0][1][0])[-2:] if channels else []

    return channels


def improve_technical_analysis(
    df: pd.DataFrame, recommendation: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Mejora los indicadores técnicos y corrige valores N/A

    Args:
        df: DataFrame con datos de precios
        recommendation: Diccionario con el análisis y recomendaciones

    Returns:
        recommendation: Diccionario mejorado con análisis técnico
    """
    # Asegurarse de que todos los indicadores técnicos estén presentes
    if "technical_factors" not in recommendation:
        recommendation["technical_factors"] = {}

    tech_factors = recommendation["technical_factors"]

    # Verificar y proporcionar valores predeterminados para indicadores clave
    if "price_vs_sma20" not in tech_factors or tech_factors["price_vs_sma20"] is None:
        # Calcular manualmente si hay datos suficientes
        if len(df) >= 20:
            sma20 = df["Close"].rolling(window=20).mean().iloc[-1]
            tech_factors["price_vs_sma20"] = df["Close"].iloc[-1] > sma20
        else:
            tech_factors["price_vs_sma20"] = None

    if "price_vs_sma50" not in tech_factors or tech_factors["price_vs_sma50"] is None:
        if len(df) >= 50:
            sma50 = df["Close"].rolling(window=50).mean().iloc[-1]
            tech_factors["price_vs_sma50"] = df["Close"].iloc[-1] > sma50
        else:
            tech_factors["price_vs_sma50"] = None

    if "price_vs_sma200" not in tech_factors or tech_factors["price_vs_sma200"] is None:
        if len(df) >= 200:
            sma200 = df["Close"].rolling(window=200).mean().iloc[-1]
            tech_factors["price_vs_sma200"] = df["Close"].iloc[-1] > sma200
        else:
            tech_factors["price_vs_sma200"] = None

    if "rsi" not in tech_factors or tech_factors["rsi"] is None:
        # Calcular RSI manualmente
        if len(df) >= 14:
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            tech_factors["rsi"] = round(rsi, 2)
        else:
            tech_factors["rsi"] = 50  # Valor neutral predeterminado

    if "macd_signal" not in tech_factors or tech_factors["macd_signal"] is None:
        # Proporcionar un valor predeterminado basado en la tendencia reciente
        if len(df) >= 26:
            exp1 = df["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df["Close"].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            tech_factors["macd_signal"] = (
                "bullish" if macd.iloc[-1] > signal.iloc[-1] else "bearish"
            )
        else:
            # Sin datos suficientes, usar tendencia reciente
            if len(df) >= 3:
                trend = (
                    "bullish"
                    if df["Close"].iloc[-1] > df["Close"].iloc[-3]
                    else "bearish"
                )
                tech_factors["macd_signal"] = trend
            else:
                tech_factors["macd_signal"] = "neutral"

    return recommendation


def improve_sentiment_analysis(
    recommendation: Dict[str, Any], symbol: str
) -> Dict[str, Any]:
    """
    Mejora el análisis de sentimiento y muestra fuentes

    Args:
        recommendation: Diccionario con el análisis y recomendaciones
        symbol: Símbolo del activo analizado

    Returns:
        recommendation: Diccionario mejorado con análisis de sentimiento
    """
    if "news_sentiment" not in recommendation:
        recommendation["news_sentiment"] = {}

    sentiment = recommendation["news_sentiment"]

    # Asegurarse de que existan campos clave
    if "sentiment" not in sentiment:
        sentiment["sentiment"] = "neutral"

    if "score" not in sentiment:
        sentiment["score"] = 0.5

    # Añadir análisis de fuentes si no existe
    if "sources" not in sentiment:
        # Identificar fuentes basadas en web_results
        sources = []
        if "web_results" in recommendation and recommendation["web_results"]:
            for result in recommendation["web_results"]:
                source_name = result.get("source", "Desconocido")
                url = result.get("url", "#")
                sentiment_type = (
                    "positivo"
                    if "bullish" in result.get("content", "").lower()
                    else (
                        "negativo"
                        if "bearish" in result.get("content", "").lower()
                        else "neutral"
                    )
                )

                sources.append(
                    {
                        "name": source_name,
                        "url": url,
                        "sentiment": sentiment_type,
                        "relevance": (
                            "alta"
                            if symbol.upper() in result.get("title", "").upper()
                            else "media"
                        ),
                    }
                )

        # Si no hay fuentes de web_results, añadir fuentes predefinidas basadas en el símbolo
        if not sources:
            default_sources = [
                {
                    "name": "Yahoo Finance",
                    "url": f"https://finance.yahoo.com/quote/{symbol}",
                    "sentiment": "neutral",
                    "relevance": "alta",
                },
                {
                    "name": "MarketWatch",
                    "url": f"https://www.marketwatch.com/investing/stock/{symbol}",
                    "sentiment": "neutral",
                    "relevance": "media",
                },
                {
                    "name": "Seeking Alpha",
                    "url": f"https://seekingalpha.com/symbol/{symbol}",
                    "sentiment": "neutral",
                    "relevance": "media",
                },
            ]
            sources = default_sources

        sentiment["sources"] = sources

        # Actualizar el conteo de menciones
        if "positive_mentions" not in sentiment:
            sentiment["positive_mentions"] = sum(
                1 for s in sources if s["sentiment"] == "positivo"
            )

        if "negative_mentions" not in sentiment:
            sentiment["negative_mentions"] = sum(
                1 for s in sources if s["sentiment"] == "negativo"
            )

        if "total_analyzed" not in sentiment:
            sentiment["total_analyzed"] = len(sources)

    return recommendation


def detect_improved_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detección mejorada de patrones técnicos (tendencias, canales, soportes, resistencias)

    Args:
        df: DataFrame con datos de precios

    Returns:
        patterns: Diccionario con patrones detectados
    """
    patterns = {
        "supports": [],
        "resistances": [],
        "trend_lines": {"bullish": [], "bearish": []},
        "channels": [],
    }

    # No hay suficientes datos para análisis
    if len(df) < 20:
        return patterns

    # 1. Detectar soportes y resistencias con método mejorado
    close_price = df["Close"].iloc[-1]
    highs = df["High"].values
    lows = df["Low"].values

    # Método: Histograma de precios
    price_range = max(highs) - min(lows)
    bin_size = price_range / 100  # Dividir en 100 bins

    high_bins = np.histogram(highs, bins=100)[0]
    low_bins = np.histogram(lows, bins=100)[0]

    bin_edges = np.histogram(highs, bins=100)[1]

    # Encontrar picos en el histograma (concentraciones de precios)
    support_levels = []
    resistance_levels = []

    for i in range(1, len(high_bins) - 1):
        # Resistencias: concentraciones de máximos
        if (
            high_bins[i] > high_bins[i - 1]
            and high_bins[i] > high_bins[i + 1]
            and high_bins[i] > np.mean(high_bins)
        ):
            level = (bin_edges[i] + bin_edges[i + 1]) / 2
            if level > close_price:  # Solo considerar niveles encima del precio actual
                resistance_levels.append(level)

        # Soportes: concentraciones de mínimos
        if (
            low_bins[i] > low_bins[i - 1]
            and low_bins[i] > low_bins[i + 1]
            and low_bins[i] > np.mean(low_bins)
        ):
            level = (bin_edges[i] + bin_edges[i + 1]) / 2
            if level < close_price:  # Solo considerar niveles debajo del precio actual
                support_levels.append(level)

    # Ordenar y limitar niveles
    support_levels = sorted(support_levels, reverse=True)[:3]  # Top 3 más cercanos
    resistance_levels = sorted(resistance_levels)[:3]  # Top 3 más cercanos

    patterns["supports"] = support_levels
    patterns["resistances"] = resistance_levels

    # 2. Detectar líneas de tendencia y canales
    # Utilizar las funciones existentes pero con parámetros optimizados
    bullish_lines, bearish_lines = detect_trend_lines(df, min_points=3)
    channels = detect_channels(df, bullish_lines, bearish_lines)

    patterns["trend_lines"]["bullish"] = bullish_lines
    patterns["trend_lines"]["bearish"] = bearish_lines

    # Formatear canales para compatibilidad con el frontend
    formatted_channels = []
    for channel in channels:
        support_line, resistance_line, channel_type = channel
        formatted_channels.append(
            {
                "type": channel_type,
                "support": support_line,
                "resistance": resistance_line,
            }
        )

    patterns["channels"] = formatted_channels

    return patterns


def detect_candle_patterns(
    df: pd.DataFrame, lookback: int = 10
) -> List[Dict[str, Any]]:
    """
    Detecta patrones de velas japonesas en los datos de precios

    Args:
        df: DataFrame con datos de precios
        lookback: Número de velas a analizar hacia atrás

    Returns:
        patterns: Lista de diccionarios con patrones detectados
    """
    patterns = []

    if len(df) < lookback + 3:
        return patterns

    # Analizar solo las velas más recientes
    df_subset = df.iloc[-lookback:].copy()

    # Calcular características de las velas
    df_subset["body_size"] = abs(df_subset["Close"] - df_subset["Open"])
    df_subset["upper_shadow"] = df_subset["High"] - df_subset[["Open", "Close"]].max(
        axis=1
    )
    df_subset["lower_shadow"] = (
        df_subset[["Open", "Close"]].min(axis=1) - df_subset["Low"]
    )
    df_subset["total_range"] = df_subset["High"] - df_subset["Low"]
    df_subset["body_ratio"] = df_subset["body_size"] / df_subset["total_range"]
    df_subset["is_bullish"] = df_subset["Close"] > df_subset["Open"]

    # Detectar patrón "Doji"
    for i in range(len(df_subset) - 1, -1, -1):
        idx = df_subset.index[i]
        pos = len(df) - lookback + i

        # Doji (cuerpo muy pequeño)
        if df_subset.loc[idx, "body_ratio"] < 0.1:
            patterns.append(
                {
                    "position": pos,
                    "pattern": "Doji",
                    "type": "neutral",
                    "strength": "media",
                }
            )
            continue

        # Martillo / Hombre Colgado
        if (
            df_subset.loc[idx, "lower_shadow"] > 2 * df_subset.loc[idx, "body_size"]
            and df_subset.loc[idx, "upper_shadow"]
            < 0.2 * df_subset.loc[idx, "lower_shadow"]
        ):
            pattern_type = (
                "bullish" if not df_subset.loc[idx, "is_bullish"] else "bearish"
            )
            pattern_name = "Martillo" if pattern_type == "bullish" else "Hombre Colgado"
            patterns.append(
                {
                    "position": pos,
                    "pattern": pattern_name,
                    "type": pattern_type,
                    "strength": "alta",
                }
            )
            continue

        # Estrella Fugaz / Martillo Invertido
        if (
            df_subset.loc[idx, "upper_shadow"] > 2 * df_subset.loc[idx, "body_size"]
            and df_subset.loc[idx, "lower_shadow"]
            < 0.2 * df_subset.loc[idx, "upper_shadow"]
        ):
            pattern_type = "bearish" if df_subset.loc[idx, "is_bullish"] else "bullish"
            pattern_name = (
                "Estrella Fugaz" if pattern_type == "bearish" else "Martillo Invertido"
            )
            patterns.append(
                {
                    "position": pos,
                    "pattern": pattern_name,
                    "type": pattern_type,
                    "strength": "alta",
                }
            )
            continue

        # Vela alcista/bajista grande (tendencias fuertes)
        if df_subset.loc[idx, "body_ratio"] > 0.7:
            pattern_type = "bullish" if df_subset.loc[idx, "is_bullish"] else "bearish"
            patterns.append(
                {
                    "position": pos,
                    "pattern": f"Vela {pattern_type.capitalize()} Fuerte",
                    "type": pattern_type,
                    "strength": "alta",
                }
            )

    # Patrones de varias velas
    if len(df_subset) >= 3:
        for i in range(len(df_subset) - 3, -1, -1):
            idx = df_subset.index[i]
            pos = len(df) - lookback + i

            # Patrón de Engulfing
            if (
                i < len(df_subset) - 1
                and df_subset.loc[idx, "is_bullish"]
                != df_subset.loc[df_subset.index[i + 1], "is_bullish"]
                and df_subset.loc[idx, "body_size"]
                > df_subset.loc[df_subset.index[i + 1], "body_size"]
                and (
                    (
                        df_subset.loc[idx, "is_bullish"]
                        and df_subset.loc[idx, "Open"]
                        < df_subset.loc[df_subset.index[i + 1], "Close"]
                    )
                    or (
                        not df_subset.loc[idx, "is_bullish"]
                        and df_subset.loc[idx, "Open"]
                        > df_subset.loc[df_subset.index[i + 1], "Close"]
                    )
                )
            ):

                pattern_type = (
                    "bullish" if df_subset.loc[idx, "is_bullish"] else "bearish"
                )
                patterns.append(
                    {
                        "position": pos,
                        "pattern": f"Engulfing {pattern_type.capitalize()}",
                        "type": pattern_type,
                        "strength": "alta",
                    }
                )

    return patterns


def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 20) -> Dict[str, Any]:
    """
    Calcula el perfil de volumen para identificar zonas de soporte/resistencia basadas en volumen

    Args:
        df: DataFrame con datos de precios y volumen
        num_bins: Número de divisiones de precio para el análisis

    Returns:
        profile: Diccionario con el perfil de volumen
    """
    if "Volume" not in df.columns or len(df) < 10:
        return {"value_areas": [], "poc": None}

    # Definir el rango de precios para el análisis
    price_range = (df["Low"].min(), df["High"].max())
    bin_size = (price_range[1] - price_range[0]) / num_bins

    # Crear bins de precio
    price_bins = np.linspace(price_range[0], price_range[1], num_bins + 1)

    # Inicializar array para acumular volumen por nivel de precio
    volume_profile = np.zeros(num_bins)

    # Calcular contribución de volumen a cada nivel de precio
    for i, row in df.iterrows():
        # Determinar qué bins caen dentro del rango de la vela
        bin_min = max(0, int((row["Low"] - price_range[0]) / bin_size))
        bin_max = min(num_bins - 1, int((row["High"] - price_range[0]) / bin_size))

        # Distribuir el volumen proporcionalmente
        if bin_max >= bin_min:  # Asegurarse de que hay al menos un bin
            vol_per_bin = row["Volume"] / (bin_max - bin_min + 1)
            volume_profile[bin_min : bin_max + 1] += vol_per_bin

    # Determinar el Point of Control (POC) - nivel de precio con mayor volumen
    poc_idx = np.argmax(volume_profile)
    poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2

    # Calcular Value Areas (70% del volumen)
    total_volume = np.sum(volume_profile)
    target_volume = total_volume * 0.7

    # Ordenar bins por volumen de mayor a menor
    volume_sorted_idx = np.argsort(volume_profile)[::-1]

    # Acumular volumen hasta alcanzar el 70%
    cum_volume = 0
    value_areas = []

    for idx in volume_sorted_idx:
        if cum_volume >= target_volume:
            break

        bin_low = price_bins[idx]
        bin_high = price_bins[idx + 1]

        value_areas.append(
            {
                "price_low": bin_low,
                "price_high": bin_high,
                "volume": float(volume_profile[idx]),
            }
        )

        cum_volume += volume_profile[idx]

    return {
        "poc": {"price": float(poc_price), "volume": float(volume_profile[poc_idx])},
        "value_areas": value_areas,
    }
