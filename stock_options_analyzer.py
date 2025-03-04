import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import json
import time
import openai
import logging
from typing import Dict, List, Tuple, Any, Optional

# Importaciones de módulos personalizados
from market_data_engine import (
    analyze_stock_options,
    _data_cache,
    get_api_keys_from_secrets,
)
from authenticator import (
    check_password,
    validate_session,
    clear_session,
    get_session_info,
)
from technical_analysis import (
    detect_support_resistance,
    detect_trend_lines,
    detect_channels,
    improve_technical_analysis,
    improve_sentiment_analysis,
    detect_improved_patterns,
    detect_candle_patterns,
    calculate_volume_profile,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Encoder personalizado mejorado para manejar diversos tipos de datos NumPy y Pandas"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)


# Configuración de página
st.set_page_config(
    page_title="MarketIntel Options Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Universo de Trading
SYMBOLS = {
    "Índices": ["SPY", "QQQ", "DIA", "IWM", "EFA", "VWO", "IYR", "XLE", "XLF", "XLV"],
    "Tecnología": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "NFLX",
        "PYPL",
        "CRM",
    ],
    "Finanzas": ["JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP", "BLK"],
    "Energía": ["XOM", "CVX", "SHEL", "TTE", "COP", "EOG", "PXD", "DVN", "MPC", "PSX"],
    "Salud": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "AMGN", "BMY", "GILD", "TMO"],
    "Consumo Discrecional": [
        "MCD",
        "SBUX",
        "NKE",
        "TGT",
        "HD",
        "LOW",
        "TJX",
        "ROST",
        "CMG",
        "DHI",
    ],
    "Cripto ETFs": ["BITO", "GBTC", "ETHE", "ARKW", "BLOK"],
    "Materias Primas": ["GLD", "SLV", "USO", "UNG", "CORN", "SOYB", "WEAT"],
    "Bonos": ["AGG", "BND", "IEF", "TLT", "LQD", "HYG", "JNK", "TIP", "MUB", "SHY"],
    "Inmobiliario": [
        "VNQ",
        "XLRE",
        "IYR",
        "REIT",
        "HST",
        "EQR",
        "AVB",
        "PLD",
        "SPG",
        "AMT",
    ],
}

# Estilos personalizados (CSS omitido por brevedad)
st.markdown(
    """<style>/* Estilos CSS omitidos por brevedad */</style>""", unsafe_allow_html=True
)


# Inicialización del cliente de OpenAI y variables de estado
def init_openai_client():
    """Inicializa el cliente de OpenAI con manejo mejorado de credenciales"""
    OPENAI_API_KEY = None
    ASSISTANT_ID = None

    try:
        # Estrategia de búsqueda de credenciales en múltiples ubicaciones
        credential_sources = [
            # Nivel principal
            {
                "container": st.secrets,
                "key": "OPENAI_API_KEY",
                "target": "OPENAI_API_KEY",
            },
            {"container": st.secrets, "key": "ASSISTANT_ID", "target": "ASSISTANT_ID"},
            # Sección api_keys
            {
                "container": st.secrets.get("api_keys", {}),
                "key": "OPENAI_API_KEY",
                "target": "OPENAI_API_KEY",
            },
            {
                "container": st.secrets.get("api_keys", {}),
                "key": "ASSISTANT_ID",
                "target": "ASSISTANT_ID",
            },
        ]

        # Nombres alternativos
        api_key_alternatives = ["openai_api_key", "OpenAIAPIKey", "OPENAI_KEY"]
        assistant_id_alternatives = ["assistant_id", "AssistantID", "ASSISTANT"]

        # Buscar en todas las posibles ubicaciones
        for source in credential_sources:
            container = source["container"]
            key = source["key"]
            target = source["target"]

            if key in container:
                if target == "OPENAI_API_KEY":
                    OPENAI_API_KEY = container[key]
                    logger.info(f"✅ OPENAI_API_KEY encontrada en {key}")
                elif target == "ASSISTANT_ID":
                    ASSISTANT_ID = container[key]
                    logger.info(f"✅ ASSISTANT_ID encontrado en {key}")

        # Buscar nombres alternativos
        if not OPENAI_API_KEY:
            for alt_key in api_key_alternatives:
                if alt_key in st.secrets:
                    OPENAI_API_KEY = st.secrets[alt_key]
                    logger.info(f"✅ API Key encontrada como {alt_key}")
                    break
                elif "api_keys" in st.secrets and alt_key in st.secrets["api_keys"]:
                    OPENAI_API_KEY = st.secrets["api_keys"][alt_key]
                    logger.info(f"✅ API Key encontrada en api_keys.{alt_key}")
                    break

        if not ASSISTANT_ID:
            for alt_id in assistant_id_alternatives:
                if alt_id in st.secrets:
                    ASSISTANT_ID = st.secrets[alt_id]
                    logger.info(f"✅ Assistant ID encontrado como {alt_id}")
                    break
                elif "api_keys" in st.secrets and alt_id in st.secrets["api_keys"]:
                    ASSISTANT_ID = st.secrets["api_keys"][alt_id]
                    logger.info(f"✅ Assistant ID encontrado en api_keys.{alt_id}")
                    break

    except Exception as e:
        logger.error(f"Error accediendo a secrets: {str(e)}")
        st.sidebar.error(f"Error accediendo a secrets: {str(e)}")

    # Si no se encontraron credenciales, solicitar manualmente
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
        if OPENAI_API_KEY:
            logger.info("API Key ingresada manualmente")

    if not ASSISTANT_ID:
        ASSISTANT_ID = st.sidebar.text_input("Assistant ID", type="password")
        if ASSISTANT_ID:
            logger.info("Assistant ID ingresado manualmente")

    # Si aún no hay credenciales, retornar None
    if not OPENAI_API_KEY or not ASSISTANT_ID:
        return None, None

    # Configurar cliente OpenAI
    client = openai
    client.api_key = OPENAI_API_KEY

    # Inicializar variables de estado para el thread
    if "thread_id" not in st.session_state:
        try:
            thread = client.beta.threads.create()
            st.session_state.thread_id = thread.id
            logger.info(f"Thread creado: {thread.id}")
        except Exception as thread_error:
            logger.error(f"Error creando thread: {str(thread_error)}")
            st.sidebar.error(f"Error creando thread: {str(thread_error)}")
            return None, None

    return client, ASSISTANT_ID


def format_patterns_for_prompt(patterns):
    """Formatea los patrones técnicos para incluirlos en el prompt del asistente IA"""
    if not patterns:
        return "No se detectaron patrones técnicos significativos."

    formatted_text = ""

    # Soportes y resistencias
    if patterns.get("supports"):
        formatted_text += "SOPORTES:\n"
        for level in patterns.get("supports", []):
            formatted_text += f"- {level:.2f}\n"

    if patterns.get("resistances"):
        formatted_text += "\nRESISTENCIAS:\n"
        for level in patterns.get("resistances", []):
            formatted_text += f"- {level:.2f}\n"

    # Tendencias
    if patterns.get("trend_lines", {}).get("bullish"):
        formatted_text += "\nLÍNEAS DE TENDENCIA ALCISTA: Identificadas\n"

    if patterns.get("trend_lines", {}).get("bearish"):
        formatted_text += "\nLÍNEAS DE TENDENCIA BAJISTA: Identificadas\n"

    # Canales
    if patterns.get("channels"):
        formatted_text += "\nCANALES DE PRECIO:\n"
        for i, channel in enumerate(patterns.get("channels", [])):
            formatted_text += (
                f"- Canal {i+1}: Tipo {channel.get('type', 'desconocido')}\n"
            )

    # Patrones de velas
    if "candle_patterns" in patterns and patterns["candle_patterns"]:
        formatted_text += "\nPATRONES DE VELAS JAPONESAS:\n"
        for pattern in patterns["candle_patterns"]:
            formatted_text += f"- {pattern['pattern']} ({pattern['type'].capitalize()}, fuerza {pattern['strength']})\n"

    return formatted_text


def process_message_with_citations(message):
    """Extrae y devuelve el texto del mensaje del asistente con manejo mejorado de errores"""
    try:
        if hasattr(message, "content") and len(message.content) > 0:
            message_content = message.content[0]
            if hasattr(message_content, "text"):
                nested_text = message_content.text
                if hasattr(nested_text, "value"):
                    return nested_text.value
                elif isinstance(nested_text, str):
                    return nested_text
            elif isinstance(message_content, dict) and "text" in message_content:
                return message_content["text"].get("value", message_content["text"])
    except Exception as e:
        logger.error(f"Error procesando mensaje: {str(e)}")

    return "No se pudo procesar el mensaje del asistente"


def consult_expert_ia(client, assistant_id, recommendation, symbol):
    """Consulta al experto de IA con análisis detallado y solicitud de contraste entre noticias y fundamentos"""
    if not client or not assistant_id:
        return "Error: No se ha configurado correctamente el Experto IA. Por favor, verifica las credenciales de OpenAI."

    # Extraer factores clave para el prompt
    tech_factors = recommendation.get("technical_factors", {})
    fundamentals = recommendation.get("fundamental_factors", {})
    sentiment = recommendation.get("news_sentiment", {})
    news = recommendation.get("news", [])
    web_results = recommendation.get("web_results", [])

    # Formatear noticias recientes para el prompt
    news_text = ""
    if news:
        news_text = "NOTICIAS RECIENTES:\n"
        for item in news[:5]:  # Limitar a 5 noticias
            news_text += f"- {item.get('date', 'Fecha N/A')}: {item.get('title', 'Sin título')}\n"

    # Formatear factores fundamentales
    fundamentals_text = "ANÁLISIS FUNDAMENTAL:\n"
    for key, value in fundamentals.items():
        fundamentals_text += f"- {key}: {value}\n"

    # Prompt mejorado y estructurado
    prompt = f"""
    Como Especialista en Trading y Análisis Técnico Avanzado, realiza un análisis profesional completo del siguiente activo:
    
    SÍMBOLO: {symbol}
    
    DATOS TÉCNICOS:
    - Recomendación actual: {recommendation.get('recommendation')}
    - Confianza calculada: {recommendation.get('confidence')}
    - Score algorítmico: {recommendation.get('score')}%
    - Horizonte temporal sugerido: {recommendation.get('timeframe')}
    
    INDICADORES TÉCNICOS:
    - SMA 20: {"Por encima" if tech_factors.get("price_vs_sma20") is True else "Por debajo" if tech_factors.get("price_vs_sma20") is False else "N/A"}
    - SMA 50: {"Por encima" if tech_factors.get("price_vs_sma50") is True else "Por debajo" if tech_factors.get("price_vs_sma50") is False else "N/A"} 
    - SMA 200: {"Por encima" if tech_factors.get("price_vs_sma200") is True else "Por debajo" if tech_factors.get("price_vs_sma200") is False else "N/A"}
    - RSI: {tech_factors.get("rsi", "N/A")}
    - MACD: {tech_factors.get("macd_signal", "N/A")}
    
    {fundamentals_text}
    
    ANÁLISIS DE SENTIMIENTO:
    - Sentimiento general: {sentiment.get("sentiment", "neutral")}
    - Score de sentimiento: {sentiment.get("score", 0.5)*100:.1f}%
    - Menciones positivas: {sentiment.get("positive_mentions", 0)}
    - Menciones negativas: {sentiment.get("negative_mentions", 0)}
    
    {news_text}
    
    INSTRUCCIONES ESPECÍFICAS:
    1. Evalúa la coherencia entre indicadores técnicos, datos fundamentales y el sentimiento de noticias.
    2. Contrasta las noticias recientes con los datos fundamentales. ¿Las noticias justifican la valoración actual?
    3. Proporciona una recomendación de trading estructurada con horizonte temporal preciso.
    4. Sugiere estrategias específicas de opciones, incluyendo estructuras concretas (strikes y vencimientos).
    5. Identifica riesgos clave que podrían invalidar tu análisis, incluyendo niveles específicos de stop loss.
    6. Proyecta el movimiento esperado con rango de precios objetivo.
    
    FORMATO DE RESPUESTA:
    Utiliza los siguientes encabezados obligatorios en tu respuesta:
    - EVALUACIÓN INTEGRAL:
    - CONTRASTE NOTICIAS-FUNDAMENTOS:
    - RECOMENDACIÓN DE TRADING:
    - ESTRATEGIAS DE OPCIONES:
    - RIESGOS Y STOP LOSS:
    - PROYECCIÓN DE MOVIMIENTO:
    
    Tu análisis debe ser conciso, directo y con información accionable específica para un trader profesional.
    """

    try:
        # Enviar mensaje al thread
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id, role="user", content=prompt
        )

        # Crear una ejecución para el thread
        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id, assistant_id=assistant_id
        )

        # Mostrar progreso
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Esperar a que se complete la ejecución con timeout
        start_time = time.time()
        timeout = 60  # 60 segundos máximo

        while run.status not in ["completed", "failed", "cancelled", "expired"]:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                status_text.error(
                    "El análisis del experto está tardando demasiado. Por favor, inténtalo de nuevo."
                )
                return "Error: Timeout en la consulta al experto"

            # Actualizar progreso
            progress = min(0.9, elapsed / timeout)
            progress_bar.progress(progress)
            status_text.text(f"El experto está analizando {symbol}... ({run.status})")

            # Esperar antes de verificar de nuevo
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id, run_id=run.id
            )

        # Completar barra de progreso
        progress_bar.progress(1.0)
        status_text.empty()

        if run.status != "completed":
            return f"Error: La consulta al experto falló con estado {run.status}"

        # Recuperar mensajes
        messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread_id
        )

        # Obtener respuesta
        for message in messages:
            if message.run_id == run.id and message.role == "assistant":
                return process_message_with_citations(message)

        return "No se recibió respuesta del experto."

    except Exception as e:
        return f"Error al consultar al experto: {str(e)}"


def create_technical_chart(data):
    """Crea gráfico técnico con indicadores y patrones técnicos"""
    # Verificación adecuada de DataFrame vacío
    if (
        data is None
        or (isinstance(data, pd.DataFrame) and data.empty)
        or (isinstance(data, list) and (len(data) < 20))
    ):
        return None

    # Convertir a DataFrame si es necesario
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # Crear figura con subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("OHLC con Medias Móviles y Bandas Bollinger", "MACD", "RSI"),
    )

    # Determinar los datos del eje X
    if "Date" in df.columns:
        x_data = df["Date"]
        x_first = x_data.iloc[0] if len(x_data) > 0 else None
        x_last = x_data.iloc[-1] if len(x_data) > 0 else None
    else:
        x_data = df.index
        x_first = x_data[0] if len(x_data) > 0 else None
        x_last = x_data[len(x_data) - 1] if len(x_data) > 0 else None

    # Añadir Candlestick
    fig.add_trace(
        go.Candlestick(
            x=x_data,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # Añadir Medias Móviles
    for ma, color in [
        ("SMA_20", "rgba(13, 71, 161, 0.7)"),
        ("SMA_50", "rgba(141, 110, 99, 0.7)"),
        ("SMA_200", "rgba(183, 28, 28, 0.7)"),
    ]:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=df[ma],
                    name=ma,
                    line=dict(color=color, width=1),
                ),
                row=1,
                col=1,
            )

    # Añadir Bandas Bollinger
    for bb, color, fill in [
        ("BB_Upper", "rgba(0, 150, 136, 0.3)", None),
        ("BB_MA20", "rgba(0, 150, 136, 0.7)", None),
        ("BB_Lower", "rgba(0, 150, 136, 0.3)", "tonexty"),
    ]:
        if bb in df.columns or (bb == "BB_MA20" and "SMA_20" in df.columns):
            y_data = df[bb] if bb in df.columns else df["SMA_20"]
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    name=bb,
                    line=dict(color=color, width=1),
                    fill=fill,
                ),
                row=1,
                col=1,
            )

    # Añadir MACD
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=df["MACD"],
                name="MACD",
                line=dict(color="rgba(33, 150, 243, 0.7)", width=1),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=df["MACD_Signal"],
                name="Señal MACD",
                line=dict(color="rgba(255, 87, 34, 0.7)", width=1),
            ),
            row=2,
            col=1,
        )

        # Añadir histograma MACD
        colors = [
            "rgba(33, 150, 243, 0.7)" if val >= 0 else "rgba(255, 87, 34, 0.7)"
            for val in (df["MACD"] - df["MACD_Signal"])
        ]

        fig.add_trace(
            go.Bar(
                x=x_data,
                y=df["MACD"] - df["MACD_Signal"],
                name="Histograma MACD",
                marker_color=colors,
            ),
            row=2,
            col=1,
        )

    # Añadir RSI
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=df["RSI"],
                name="RSI",
                line=dict(color="rgba(156, 39, 176, 0.7)", width=1),
            ),
            row=3,
            col=1,
        )

        # Líneas de referencia RSI
        for level, color in [
            (30, "rgba(76, 175, 80, 0.5)"),
            (70, "rgba(255, 87, 34, 0.5)"),
        ]:
            fig.add_shape(
                type="line",
                x0=x_first,
                x1=x_last,
                y0=level,
                y1=level,
                line=dict(color=color, width=1, dash="dash"),
                row=3,
                col=1,
            )

    # Detectar soportes y resistencias
    try:
        supports, resistances = detect_support_resistance(df)

        # Añadir líneas de soporte
        for level in supports:
            fig.add_shape(
                type="line",
                x0=x_first,
                x1=x_last,
                y0=level,
                y1=level,
                line=dict(color="rgba(0, 128, 0, 0.7)", width=1, dash="dot"),
                row=1,
                col=1,
            )

            # Añadir etiqueta
            fig.add_annotation(
                x=x_last,
                y=level,
                text=f"Soporte: {level:.2f}",
                showarrow=False,
                xshift=10,
                font=dict(color="rgba(0, 128, 0, 1)"),
                row=1,
                col=1,
            )

        # Añadir líneas de resistencia
        for level in resistances:
            fig.add_shape(
                type="line",
                x0=x_first,
                x1=x_last,
                y0=level,
                y1=level,
                line=dict(color="rgba(255, 0, 0, 0.7)", width=1, dash="dot"),
                row=1,
                col=1,
            )

            # Añadir etiqueta
            fig.add_annotation(
                x=x_last,
                y=level,
                text=f"Resistencia: {level:.2f}",
                showarrow=False,
                xshift=10,
                font=dict(color="rgba(255, 0, 0, 1)"),
                row=1,
                col=1,
            )
    except Exception as e:
        st.warning(f"No se pudieron detectar niveles de soporte/resistencia: {str(e)}")

    # Detectar líneas de tendencia
    try:
        if "Date" in df.columns:
            # Si hay fechas, convertir a índices numéricos para cálculos de tendencia
            df_idx = df.copy()
            df_idx["idx"] = range(len(df))
            bullish_lines, bearish_lines = detect_trend_lines(df_idx)

            # Convertir índices de vuelta a fechas
            bullish_lines_dates = [
                (df["Date"].iloc[x1], y1, df["Date"].iloc[x2], y2)
                for x1, y1, x2, y2 in bullish_lines
                if x1 < len(df) and x2 < len(df)
            ]

            bearish_lines_dates = [
                (df["Date"].iloc[x1], y1, df["Date"].iloc[x2], y2)
                for x1, y1, x2, y2 in bearish_lines
                if x1 < len(df) and x2 < len(df)
            ]
        else:
            # Usar índices directamente
            bullish_lines, bearish_lines = detect_trend_lines(df)
            bullish_lines_dates = [
                (df.index[x1], y1, df.index[x2], y2)
                for x1, y1, x2, y2 in bullish_lines
                if x1 < len(df) and x2 < len(df)
            ]

            bearish_lines_dates = [
                (df.index[x1], y1, df.index[x2], y2)
                for x1, y1, x2, y2 in bearish_lines
                if x1 < len(df) and x2 < len(df)
            ]

        # Añadir líneas de tendencia alcistas
        for i, (x1, y1, x2, y2) in enumerate(bullish_lines_dates):
            fig.add_shape(
                type="line",
                x0=x1,
                y0=y1,
                x1=x2,
                y1=y2,
                line=dict(color="rgba(0, 128, 0, 0.7)", width=2),
                row=1,
                col=1,
            )

            # Añadir etiqueta
            fig.add_annotation(
                x=x2,
                y=y2,
                text=f"Tendencia Alcista",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                font=dict(color="rgba(0, 128, 0, 1)"),
                row=1,
                col=1,
            )

        # Añadir líneas de tendencia bajistas
        for i, (x1, y1, x2, y2) in enumerate(bearish_lines_dates):
            fig.add_shape(
                type="line",
                x0=x1,
                y0=y1,
                x1=x2,
                y1=y2,
                line=dict(color="rgba(255, 0, 0, 0.7)", width=2),
                row=1,
                col=1,
            )

            # Añadir etiqueta
            fig.add_annotation(
                x=x2,
                y=y2,
                text=f"Tendencia Bajista",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=30,
                font=dict(color="rgba(255, 0, 0, 1)"),
                row=1,
                col=1,
            )
    except Exception as e:
        st.warning(f"No se pudieron detectar líneas de tendencia: {str(e)}")

    # Ajustar layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        title=f"Análisis Técnico con Patrones",
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Configuración de ejes y rangos
    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

    return fig


def display_recommendation_summary(recommendation):
    """Muestra resumen de recomendación con manejo mejorado de datos"""
    rec_type = recommendation.get("recommendation", "NEUTRAL")
    confidence = recommendation.get("confidence", "baja")
    score = recommendation.get("score", 50)

    # Determinar clases CSS
    badge_class = (
        "call-badge"
        if rec_type == "CALL"
        else "put-badge" if rec_type == "PUT" else "neutral-badge"
    )
    confidence_class = f"confidence-{'high' if confidence == 'alta' else 'medium' if confidence == 'media' else 'low'}"

    # Crear columnas para mostrar métricas
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value"><span class="{badge_class}">{rec_type}</span></div>
            <div class="metric-label">Recomendación</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value"><span class="{confidence_class}">{confidence.upper()}</span></div>
            <div class="metric-label">Confianza</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        # Crear color gradiente basado en score
        color = "green" if score >= 65 else "red" if score <= 35 else "#FFA500"
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {color};">{score}%</div>
            <div class="metric-label">Score de Operación</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Timeframe
    timeframe = recommendation.get("timeframe", "No especificado")
    st.markdown(
        f"""
    <div style="margin-top: 1rem; text-align: center;">
        <span style="font-weight: 500;">Horizonte recomendado:</span> {timeframe}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Fecha de análisis
    try:
        analysis_date = recommendation.get("analysis_date", datetime.now().isoformat())
        if isinstance(analysis_date, str):
            analysis_date = datetime.fromisoformat(analysis_date)

        st.markdown(
            f"""
        <div style="margin-top: 0.5rem; text-align: center; font-size: 0.8rem; color: #6c757d;">
            Análisis realizado el {analysis_date.strftime('%d/%m/%Y a las %H:%M:%S')}
        </div>
        """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        logger.error(f"Error formateando fecha de análisis: {str(e)}")

    # Análisis institucional
    if rec_type == "CALL":
        risk_level = "media" if score < 75 else "baja"
        strategy_type = "direccional alcista"
    elif rec_type == "PUT":
        risk_level = "media" if score > 25 else "baja"
        strategy_type = "direccional bajista"
    else:
        risk_level = "alta"
        strategy_type = "neutral (Iron Condor o Calendar)"

    risk_class = f"risk-{'low' if risk_level == 'baja' else 'medium' if risk_level == 'media' else 'high'}"

    st.markdown(
        f"""
        <div class="institutional-insight">
            <h4>Análisis Institucional</h4>
            <p>Probabilidad de éxito: <strong>{int((100-abs(score-50))*0.9)}%</strong></p>
            <p>Perfil de riesgo: <span class="{risk_class}">{risk_level.upper()}</span></p>
            <p>Estrategia recomendada: <strong>{strategy_type}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_expert_opinion(expert_opinion):
    """Muestra la opinión del experto IA con manejo mejorado de salida"""
    if not expert_opinion:
        return

    st.markdown("## 🧠 Análisis del Experto")

    # Procesamiento mejorado del texto: buscar secciones clave
    sections = {
        "evaluación": "",
        "recomendación": "",
        "estrategias": "",
        "riesgos": "",
        "proyección": "",
    }

    current_section = None

    try:
        # Intentar identificar secciones numeradas en el texto
        lines = expert_opinion.split("\n")
        for line in lines:
            line = line.strip()

            # Detectar secciones numeradas
            if line.startswith("1.") and "evaluación" in line.lower():
                current_section = "evaluación"
                sections[current_section] += line[2:].strip() + "\n"
            elif line.startswith("2.") and "recomendación" in line.lower():
                current_section = "recomendación"
                sections[current_section] += line[2:].strip() + "\n"
            elif line.startswith("3.") and "estrategia" in line.lower():
                current_section = "estrategias"
                sections[current_section] += line[2:].strip() + "\n"
            elif line.startswith("4.") and "riesgo" in line.lower():
                current_section = "riesgos"
                sections[current_section] += line[2:].strip() + "\n"
            elif line.startswith("5.") and "proyección" in line.lower():
                current_section = "proyección"
                sections[current_section] += line[2:].strip() + "\n"
            elif current_section:
                sections[current_section] += line + "\n"
    except Exception as e:
        logger.error(f"Error al procesar la respuesta del experto: {str(e)}")

    # Si no se identificaron secciones, mostrar el texto completo
    if all(not v for v in sections.values()):
        st.markdown(
            f"""
            <div class="expert-container">
                <div class="expert-header">
                    <div class="expert-avatar">E</div>
                    <div class="expert-title">Analista de Mercados</div>
                </div>
                <div class="expert-content">
                    {expert_opinion}
                </div>
                <div class="expert-footer">
                    Análisis generado por IA - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Mostrar secciones identificadas
        st.markdown(
            f"""
            <div class="expert-container">
                <div class="expert-header">
                    <div class="expert-avatar">E</div>
                    <div class="expert-title">Analista de Mercados</div>
                </div>
                <div class="expert-content">
            """,
            unsafe_allow_html=True,
        )

        # Mostrar cada sección identificada en un formato más estructurado
        if sections["evaluación"]:
            st.markdown("### 📊 Evaluación del Activo")
            st.markdown(sections["evaluación"])

        if sections["recomendación"]:
            st.markdown("### 🎯 Recomendación de Trading")
            st.markdown(sections["recomendación"])

        if sections["estrategias"]:
            st.markdown("### 📈 Estrategias con Opciones")
            st.markdown(sections["estrategias"])

        if sections["riesgos"]:
            st.markdown("### ⚠️ Riesgos y Niveles Clave")
            st.markdown(sections["riesgos"])

        if sections["proyección"]:
            st.markdown("### 🔮 Proyección de Movimiento")
            st.markdown(sections["proyección"])

        st.markdown(
            f"""
                </div>
                <div class="expert-footer">
                    Análisis generado por IA - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def synchronize_market_data(recommendation, symbol):
    """Intenta sincronizar los datos con valores de mercado recientes y añade información diagnóstica"""
    try:
        # Log de diagnóstico
        print(f"DEBUG: Sincronizando datos de mercado para {symbol}")
        chart_data = recommendation.get("chart_data", [])

        if not chart_data or len(chart_data) == 0:
            print("DEBUG: No hay chart_data disponible para sincronización")
            return recommendation

        # Verificar datos actuales
        latest_price = (
            chart_data[-1]["Close"] if isinstance(chart_data[-1], dict) else None
        )
        oldest_price = (
            chart_data[0]["Close"] if isinstance(chart_data[0], dict) else None
        )

        print(
            f"DEBUG: Precio más reciente en datos: {latest_price}, Precio más antiguo: {oldest_price}"
        )
        print(
            f"DEBUG: Período analizado: {chart_data[0].get('Date', 'N/A')} a {chart_data[-1].get('Date', 'N/A')}"
        )

        # Añadir metadatos de sincronización
        if "metadata" not in recommendation:
            recommendation["metadata"] = {}

        recommendation["metadata"]["last_price"] = latest_price
        recommendation["metadata"]["data_source"] = "API con posible retraso"
        recommendation["metadata"]["data_timestamp"] = datetime.now().isoformat()
        recommendation["metadata"][
            "disclaimer"
        ] = "Los precios pueden diferir de los datos de mercado en tiempo real. Use fuentes primarias para trading activo."

        # Si hay datos técnicos, verificar concordancia
        if "technical_factors" in recommendation:
            sma20 = chart_data[-20:] if len(chart_data) >= 20 else []
            if sma20:
                sma20_value = sum(item["Close"] for item in sma20) / len(sma20)
                print(
                    f"DEBUG: SMA20 calculado manualmente: {sma20_value}, diferencia con último precio: {latest_price - sma20_value}"
                )

        return recommendation
    except Exception as e:
        print(f"ERROR en sincronización de datos: {e}")
        return recommendation


def display_fundamental_factors(recommendation):
    """Muestra factores fundamentales con diagnóstico avanzado"""
    fundamentals = recommendation.get("fundamental_factors", {})

    # Añadir debug de datos fundamentales
    print(f"DEBUG: Datos fundamentales originales: {fundamentals}")

    # Intentar obtener datos fundamentales del análisis si están vacíos
    if not fundamentals:
        # Buscar en otros lugares del objeto recomendación
        print("DEBUG: Intentando buscar datos fundamentales en otras secciones...")

        for key in recommendation:
            if (
                isinstance(recommendation[key], dict)
                and "pe_ratio" in recommendation[key]
            ):
                print(f"DEBUG: Encontrados datos fundamentales en sección '{key}'")
                fundamentals = recommendation[key]
                break

        # Si aún no hay datos, intentar crear algunos datos ficticios
        # basados en webResults para no dejar esta sección vacía
        if (
            not fundamentals
            and "web_results" in recommendation
            and recommendation["web_results"]
        ):
            fundamentals = {
                "market_cap": "Datos no disponibles",
                "pe_ratio": "N/A",
                "eps": "N/A",
                "yield": "N/A",
                "note": "Los datos fundamentales no pudieron ser recuperados.",
            }
            print("DEBUG: Creados placeholder para datos fundamentales")

    if not fundamentals:
        st.warning(
            "⚠️ No se encontraron datos fundamentales disponibles. Es posible que los API de datos estén limitadas o en mantenimiento."
        )
        st.info(
            "💡 Consideración: Para decisiones de trading con opciones, los factores técnicos y el análisis de volatilidad tienen mayor peso que los factores fundamentales a corto plazo."
        )
        return

    st.markdown(
        '<div class="sub-header">Factores Fundamentales</div>', unsafe_allow_html=True
    )

    # Crear columnas para mostrar métricas fundamentales
    cols = st.columns(4)

    metrics = [
        ("Market Cap", "market_cap", "Capitalización"),
        ("PE Ratio", "pe_ratio", "P/E Ratio"),
        ("EPS", "eps", "Beneficio por Acción"),
        ("Dividend Yield", "yield", "Rendimiento Div."),
    ]

    # Mostrar métricas disponibles, con control de errores
    for i, (key, field, label) in enumerate(metrics):
        try:
            with cols[i % 4]:
                value = fundamentals.get(field, "N/A")
                # Verificar si el valor es numérico para formatearlo correctamente
                if isinstance(value, (int, float)):
                    if field == "market_cap":
                        # Formatear market cap en billones/millones
                        if value >= 1e12:
                            value = f"${value/1e12:.2f}T"
                        elif value >= 1e9:
                            value = f"${value/1e9:.2f}B"
                        elif value >= 1e6:
                            value = f"${value/1e6:.2f}M"
                    elif field == "yield":
                        # Formatear yield como porcentaje
                        value = f"{value:.2f}%"
                st.metric(label=label, value=value)
        except Exception as e:
            print(f"ERROR mostrando métrica {field}: {e}")
            with cols[i % 4]:
                st.metric(label=label, value="Error")

    # Mostrar métricas adicionales avanzadas si existen
    advanced_metrics = [
        ("PEG Ratio", "peg_ratio", "PEG Ratio"),
        ("Price to Book", "price_to_book", "P/B"),
        ("Return on Equity", "roe", "ROE"),
        ("Return on Assets", "roa", "ROA"),
        ("Profit Margin", "profit_margin", "Margen"),
        ("Operating Margin", "operating_margin", "Margen Op."),
    ]

    # Comprobar si hay al menos una métrica avanzada
    has_advanced = any(field in fundamentals for _, field, _ in advanced_metrics)

    if has_advanced:
        st.markdown("##### Métricas Avanzadas")
        adv_cols = st.columns(6)

        for i, (key, field, label) in enumerate(advanced_metrics):
            if field in fundamentals:
                try:
                    with adv_cols[i % 6]:
                        value = fundamentals.get(field, "N/A")
                        # Convertir a porcentaje si es necesario
                        if field in ["roe", "roa", "profit_margin", "operating_margin"]:
                            try:
                                if isinstance(value, (int, float)):
                                    value = (
                                        f"{value*100:.2f}%"
                                        if value < 1
                                        else f"{value:.2f}%"
                                    )
                                elif isinstance(value, str) and value != "N/A":
                                    value_float = float(
                                        value.replace("%", "").replace(",", "")
                                    )
                                    value = f"{value_float:.2f}%"
                            except Exception as e:
                                print(f"ERROR convirtiendo valor porcentual: {e}")
                        st.metric(label=label, value=value)
                except Exception as e:
                    print(f"ERROR mostrando métrica avanzada {field}: {e}")
                    with adv_cols[i % 6]:
                        st.metric(label=label, value="Error")

    # Añadir nota sobre precios
    st.info(
        "📊 Los datos fundamentales pueden tener un retraso de hasta 15 minutos respecto al mercado en tiempo real y variar según la fuente. Para trading activo, se recomienda confirmar con fuentes primarias."
    )


def display_technical_factors(recommendation):
    """Muestra factores técnicos con cálculos in-situ garantizados"""
    tech_factors = recommendation.get("technical_factors", {})
    chart_data = recommendation.get("chart_data", [])

    # Log para depuración
    print(f"DEBUG: Técnicos originales: {tech_factors}")
    print(f"DEBUG: Tamaño de chart_data: {len(chart_data)}")

    # Siempre calcular los indicadores con los datos del gráfico si están disponibles
    if chart_data and len(chart_data) > 0:
        try:
            # Convertir a DataFrame y asegurar tipos numéricos
            df = pd.DataFrame(chart_data)

            # Verificar si Date está presente y convertirla a datetime
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # Asegurar columnas numéricas
            for col in ["Open", "High", "Low", "Close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Imprimir para depuración
            print(f"DEBUG: Últimos precios: {df['Close'].tail(5).tolist()}")

            # Calcular SMAs directamente - siempre
            if len(df) >= 20:
                df["SMA_20"] = df["Close"].rolling(window=20).mean()
                tech_factors["price_vs_sma20"] = (
                    df["Close"].iloc[-1] > df["SMA_20"].iloc[-1]
                )
                print(
                    f"DEBUG: SMA20 calculado: {df['SMA_20'].iloc[-1]}, Precio: {df['Close'].iloc[-1]}"
                )

            if len(df) >= 50:
                df["SMA_50"] = df["Close"].rolling(window=50).mean()
                tech_factors["price_vs_sma50"] = (
                    df["Close"].iloc[-1] > df["SMA_50"].iloc[-1]
                )
                print(f"DEBUG: SMA50 calculado: {df['SMA_50'].iloc[-1]}")

            if len(df) >= 200:
                df["SMA_200"] = df["Close"].rolling(window=200).mean()
                tech_factors["price_vs_sma200"] = (
                    df["Close"].iloc[-1] > df["SMA_200"].iloc[-1]
                )
                print(f"DEBUG: SMA200 calculado: {df['SMA_200'].iloc[-1]}")

            # Calcular RSI siempre
            if len(df) >= 14:
                delta = df["Close"].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                df["RSI"] = 100 - (100 / (1 + rs))
                tech_factors["rsi"] = float(round(df["RSI"].iloc[-1], 2))
                print(f"DEBUG: RSI calculado: {tech_factors['rsi']}")

            # Calcular MACD siempre
            if len(df) >= 26:
                exp1 = df["Close"].ewm(span=12, adjust=False).mean()
                exp2 = df["Close"].ewm(span=26, adjust=False).mean()
                df["MACD"] = exp1 - exp2
                df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
                tech_factors["macd_signal"] = (
                    "bullish"
                    if df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]
                    else "bearish"
                )
                print(
                    f"DEBUG: MACD calculado: {df['MACD'].iloc[-1]}, Signal: {df['MACD_Signal'].iloc[-1]}"
                )
        except Exception as e:
            print(f"ERROR en cálculo de indicadores: {str(e)}")
            import traceback

            print(traceback.format_exc())

    if not tech_factors:
        st.info("No se encontraron factores técnicos disponibles.")
        return

    st.markdown(
        '<div class="sub-header">Factores Técnicos</div>', unsafe_allow_html=True
    )

    # Crear columnas para mostrar métricas técnicas
    cols = st.columns(5)

    # Convertir valores booleanos en indicadores
    for i, (key, value) in enumerate(
        [
            ("SMA 20", tech_factors.get("price_vs_sma20")),
            ("SMA 50", tech_factors.get("price_vs_sma50")),
            ("SMA 200", tech_factors.get("price_vs_sma200")),
            ("RSI", tech_factors.get("rsi")),
            ("MACD", tech_factors.get("macd_signal")),
        ]
    ):
        with cols[i]:
            if key == "RSI":
                label = key
                if value is None:
                    value_text = "N/A"
                    delta = None
                    delta_color = "off"
                else:
                    try:
                        value_num = (
                            float(value)
                            if isinstance(value, (int, float, str))
                            else None
                        )
                        value_text = (
                            f"{value_num:.1f}" if value_num is not None else "N/A"
                        )

                        # Determinar si está en zona de sobrecompra/sobreventa
                        if value_num is not None:
                            if value_num > 70:
                                delta = "Sobrecompra"
                                delta_color = "inverse"
                            elif value_num < 30:
                                delta = "Sobreventa"
                                delta_color = "normal"
                            else:
                                delta = "Neutral"
                                delta_color = "off"
                        else:
                            delta = None
                            delta_color = "off"
                    except Exception as e:
                        print(f"ERROR procesando RSI: {e}")
                        value_text = str(value) if value is not None else "N/A"
                        delta = None
                        delta_color = "off"

                st.metric(
                    label=label, value=value_text, delta=delta, delta_color=delta_color
                )

            elif key == "MACD":
                if isinstance(value, str):
                    st.metric(
                        label=key,
                        value=value.upper() if value else "N/A",
                        delta=(
                            "Alcista"
                            if value == "bullish"
                            else "Bajista" if value == "bearish" else None
                        ),
                        delta_color=(
                            "normal"
                            if value == "bullish"
                            else "inverse" if value == "bearish" else "off"
                        ),
                    )
                else:
                    st.metric(label=key, value="N/A", delta=None, delta_color="off")

            else:
                # Valores booleanos (SMA)
                if value is True:
                    state = "Por encima"
                    trend = "Alcista"
                    color = "normal"
                elif value is False:
                    state = "Por debajo"
                    trend = "Bajista"
                    color = "inverse"
                else:
                    state = "N/A"
                    trend = None
                    color = "off"

                st.metric(label=key, value=state, delta=trend, delta_color=color)

    # Añadir una sección de análisis profesional de los indicadores
    if tech_factors.get("rsi") is not None:
        rsi_value = tech_factors.get("rsi")
        if isinstance(rsi_value, str) and rsi_value != "N/A":
            try:
                rsi_value = float(rsi_value.replace(",", "."))
            except:
                rsi_value = None

        macd_signal = tech_factors.get("macd_signal", "neutral")
        sma_alignment = (
            tech_factors.get("price_vs_sma20", False)
            and tech_factors.get("price_vs_sma50", False)
            and tech_factors.get("price_vs_sma200", False)
        )

        st.markdown(
            """
            <div class="pro-trading-tip">
                <h4>💡 Análisis Institucional de Indicadores</h4>
            """,
            unsafe_allow_html=True,
        )

        # Análisis de RSI
        if isinstance(rsi_value, (int, float)):
            if rsi_value > 70:
                st.markdown(
                    """
                    <p><strong>RSI > 70:</strong> Condición de sobrecompra que suele preceder correcciones a corto plazo. 
                    Vigilar posibles divergencias con el precio para confirmar debilidad. Zona idónea para estrategias basadas en PUT.</p>
                    """,
                    unsafe_allow_html=True,
                )
            elif rsi_value < 30:
                st.markdown(
                    """
                    <p><strong>RSI < 30:</strong> Condición de sobreventa que indica agotamiento vendedor.
                    Buscar señales de divergencia para confirmar posible inversión. Favorable para estrategias basadas en CALL.</p>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <p><strong>RSI = {rsi_value:.1f}:</strong> Nivel intermedio que no sugiere condiciones extremas.
                    Considerar otros indicadores para confirmar direccionalidad. Entorno favorable para estrategias delta-neutral.</p>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <p><strong>RSI:</strong> Datos no disponibles para este análisis.</p>
                """,
                unsafe_allow_html=True,
            )

        # Análisis de alineación de medias móviles
        if sma_alignment:
            st.markdown(
                """
                <p><strong>Alineación SMA:</strong> Las tres principales medias móviles confirman tendencia alcista.
                Este patrón favorece estrategias direccionales con sesgo alcista y posiciones con delta positivo.</p>
                """,
                unsafe_allow_html=True,
            )
        elif not any(
            [
                tech_factors.get("price_vs_sma20", False),
                tech_factors.get("price_vs_sma50", False),
                tech_factors.get("price_vs_sma200", False),
            ]
        ):
            st.markdown(
                """
                <p><strong>Alineación SMA:</strong> Precio por debajo de todas las medias móviles principales.
                Señal de debilidad persistente que favorece estrategias bajistas y posiciones con delta negativo.</p>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <p><strong>Alineación SMA:</strong> Las medias móviles muestran un patrón mixto.
                Esta divergencia suele indicar un entorno de trading range. Considerar estrategias de rango como Iron Condors.</p>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


def display_sentiment_analysis(recommendation):
    """Muestra análisis de sentimiento con manejo robusto de datos"""
    sentiment = recommendation.get("news_sentiment", {})
    web_analysis = recommendation.get("web_analysis", {})

    if not sentiment and not web_analysis:
        st.info("No se encontró análisis de sentimiento disponible.")
        return

    st.markdown(
        '<div class="sub-header">Análisis de Sentimiento</div>', unsafe_allow_html=True
    )

    # Mostrar sentimiento de noticias
    col1, col2 = st.columns(2)

    with col1:
        if sentiment:
            # Mostrar sentimiento
            sentiment_value = sentiment.get("sentiment", "neutral")
            sentiment_score = sentiment.get("score", 0.5)

            # Validar y convertir score si es necesario
            if isinstance(sentiment_score, str):
                try:
                    sentiment_score = float(sentiment_score.replace(",", "."))
                except:
                    sentiment_score = 0.5

            # Asegurar que el score esté en el rango correcto
            if not isinstance(sentiment_score, (int, float)):
                sentiment_score = 0.5
            sentiment_score = max(0, min(1, sentiment_score))  # Asegurar rango [0,1]

            # Crear medidor
            st.markdown("### Sentimiento de Noticias")

            # Gráfico de gauge simplificado para mayor robustez
            try:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=sentiment_score * 100,
                        title={"text": "Sentimiento"},
                        number={"suffix": "%", "font": {"size": 24}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1},
                            "bar": {"color": "darkblue"},
                            "bgcolor": "white",
                            "steps": [
                                {"range": [0, 40], "color": "lightcoral"},
                                {"range": [40, 60], "color": "lightyellow"},
                                {"range": [60, 100], "color": "lightgreen"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": sentiment_score * 100,
                            },
                        },
                    )
                )

                fig.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=50, b=10),
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Error al generar gráfico de sentimiento: {e}")
                # Fallback simple si falla la visualización
                st.progress(sentiment_score)
                st.write(f"Sentimiento: {sentiment_score*100:.1f}%")

            # Mostrar métricas adicionales
            pos = sentiment.get("positive_mentions", 0)
            neg = sentiment.get("negative_mentions", 0)
            total = sentiment.get("total_analyzed", 0)

            st.markdown(
                f"""
            **Menciones positivas:** {pos}  
            **Menciones negativas:** {neg}  
            **Total noticias analizadas:** {total}
            """
            )

            # Añadir análisis institucional de sentimiento
            if "sector_avg_bullish" in sentiment:
                sector_bullish = sentiment.get("sector_avg_bullish", 0)
                sector_bearish = sentiment.get("sector_avg_bearish", 0)

                # Validar valores del sector
                if isinstance(sector_bullish, str):
                    try:
                        sector_bullish = float(sector_bullish)
                    except:
                        sector_bullish = 0.5

                if isinstance(sector_bearish, str):
                    try:
                        sector_bearish = float(sector_bearish)
                    except:
                        sector_bearish = 0.5

                # Asegurar que los valores están en rango
                sector_bullish = max(0, min(1, sector_bullish))
                sector_bearish = max(0, min(1, sector_bearish))

                st.markdown(
                    f"""
                    <div class="institutional-insight">
                        <h4>Análisis Sectorial</h4>
                        <p>Comparación con el sector: <strong>{'+' if sentiment_score > sector_bullish else '-'}{abs(sentiment_score*100 - sector_bullish*100):.1f}%</strong></p>
                        <p>Media bullish sectorial: <strong>{sector_bullish*100:.1f}%</strong></p>
                        <p>Media bearish sectorial: <strong>{sector_bearish*100:.1f}%</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col2:
        if web_analysis:
            # Mostrar análisis web
            bullish = web_analysis.get("bullish_mentions", 0)
            bearish = web_analysis.get("bearish_mentions", 0)

            # Verificar que son números
            if not isinstance(bullish, (int, float)):
                try:
                    bullish = int(bullish)
                except:
                    bullish = 0

            if not isinstance(bearish, (int, float)):
                try:
                    bearish = int(bearish)
                except:
                    bearish = 0

            st.markdown("### Análisis Web")

            try:
                # Crear un gauge más simple y robusto
                fig = go.Figure()

                # Añadir el indicador de gauge
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=sentiment_score * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Sentimiento", "font": {"size": 24}},
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickwidth": 1,
                                "tickcolor": "darkblue",
                            },
                            "bar": {"color": "darkblue"},
                            "bgcolor": "white",
                            "borderwidth": 2,
                            "bordercolor": "gray",
                            "steps": [
                                {"range": [0, 40], "color": "rgba(255, 0, 0, 0.3)"},
                                {"range": [40, 60], "color": "rgba(255, 255, 0, 0.3)"},
                                {"range": [60, 100], "color": "rgba(0, 255, 0, 0.3)"},
                            ],
                        },
                        number={
                            "font": {"size": 40, "color": "black"},
                            "suffix": "%",
                            "valueformat": ".1f",
                        },
                    )
                )

                # Ajustar el diseño para centrar el valor
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="white",
                    font={"size": 16, "family": "Arial", "color": "black"},
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Error al generar gráfico de análisis web: {e}")
                # Fallback simple
                st.write(f"Menciones alcistas: {bullish}")
                st.write(f"Menciones bajistas: {bearish}")

            # Ratio de sentimiento
            total_mentions = bullish + bearish
            if total_mentions > 0:
                bullish_ratio = bullish / total_mentions * 100
                st.markdown(
                    f"""
                **Ratio alcista:** {bullish_ratio:.1f}%  
                **Fuentes analizadas:** {len(recommendation.get('web_results', []))}
                """
                )
            else:
                st.markdown(
                    "No se encontraron menciones relevantes en el análisis web."
                )


def display_consolidated_report(recommendation, symbol):
    """Muestra una ficha de resumen consolidada con todos los análisis y conclusiones"""
    st.markdown("## 📊 Resumen Consolidado de Análisis")

    # Obtener metadatos y última sincronización
    metadata = recommendation.get("metadata", {})
    last_price = metadata.get("last_price", "N/A")
    data_timestamp = metadata.get("data_timestamp", datetime.now().isoformat())

    if isinstance(data_timestamp, str):
        try:
            timestamp_datetime = datetime.fromisoformat(data_timestamp)
            time_difference = datetime.now() - timestamp_datetime
            sync_mins = int(time_difference.total_seconds() / 60)
            sync_status = f"Última sincronización: hace {sync_mins} minutos"
        except:
            sync_status = "Sincronización: estado desconocido"
    else:
        sync_status = "Sincronización: no disponible"

    # Añadir banner de sincronización
    st.markdown(
        f"""
        <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; margin-bottom: 20px; font-size: 0.9em;">
            <strong>Precio de referencia:</strong> {last_price} | {sync_status} | 
            <span style="color: #721c24;">Nota: Para trading activo, confirme precios con fuentes primarias en tiempo real.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Crear contenedor con estilo
    st.markdown(
        """
        <style>
        .report-container {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #1E88E5;
        }
        .report-header {
            color: #1E88E5;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 10px;
        }
        .report-section {
            margin-top: 15px;
            margin-bottom: 15px;
        }
        .report-section-title {
            font-weight: bold;
            color: #333;
            font-size: 18px;
            margin-bottom: 10px;
        }
        .report-highlight {
            background-color: #e3f2fd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .report-call {
            color: green;
            font-weight: bold;
        }
        .report-put {
            color: red;
            font-weight: bold;
        }
        .report-neutral {
            color: gray;
            font-weight: bold;
        }
        .report-table {
            width: 100%;
            margin: 10px 0;
        }
        .report-table th {
            background-color: #e9ecef;
            padding: 8px;
            text-align: left;
        }
        .report-table td {
            padding: 8px;
            border-bottom: 1px solid #dee2e6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Extraer información clave
    rec_type = recommendation.get("recommendation", "NEUTRAL")
    confidence = recommendation.get("confidence", "baja")
    score = recommendation.get("score", 50)
    timeframe = recommendation.get("timeframe", "corto plazo")

    # Determinar clases CSS para CALL/PUT
    if rec_type == "CALL":
        rec_class = "report-call"
    elif rec_type == "PUT":
        rec_class = "report-put"
    else:
        rec_class = "report-neutral"

    # Obtener datos técnicos y fundamentales
    tech_factors = recommendation.get("technical_factors", {})
    fundamentals = recommendation.get("fundamental_factors", {})
    sentiment = recommendation.get("news_sentiment", {})
    sentiment_score = sentiment.get("score", 0.5)
    if isinstance(sentiment_score, (int, float)):
        sentiment_value = f"{sentiment_score*100:.1f}%"
    else:
        sentiment_value = "N/A"

    # Comenzar el contenedor HTML
    st.markdown(
        f"""
        <div class="report-container">
            <div class="report-header">Análisis Consolidado: {symbol}</div>
            
            <div class="report-section">
                <div class="report-section-title">Recomendación Final</div>
                <div class="report-highlight">
                    <p>En base al análisis técnico, fundamental y de sentimiento, la recomendación es:
                    <span class="{rec_class}">{rec_type}</span> con confianza <strong>{confidence}</strong>.</p>
                    <p>Score: <strong>{score}%</strong> | Horizonte temporal: <strong>{timeframe}</strong></p>
                </div>
            </div>
            
            <div class="report-section">
                <div class="report-section-title">Resumen de Factores</div>
                <table class="report-table">
                    <tr>
                        <th>Factor</th>
                        <th>Valor</th>
                        <th>Interpretación</th>
                    </tr>
        """,
        unsafe_allow_html=True,
    )

    # Generar filas de la tabla con factores técnicos
    factors_html = ""

    # SMAs
    sma20 = tech_factors.get("price_vs_sma20")
    factors_html += f"""
        <tr>
            <td>SMA 20</td>
            <td>{"Por encima" if sma20 is True else "Por debajo" if sma20 is False else "N/A"}</td>
            <td>{"Señal alcista" if sma20 is True else "Señal bajista" if sma20 is False else "Sin datos suficientes"}</td>
        </tr>
    """

    # RSI
    rsi = tech_factors.get("rsi")
    if rsi is not None:
        if isinstance(rsi, (int, float)):
            rsi_interp = (
                "Sobrecompra - Potencial corrección"
                if rsi > 70
                else "Sobreventa - Potencial rebote" if rsi < 30 else "Neutral"
            )
            factors_html += f"""
                <tr>
                    <td>RSI</td>
                    <td>{rsi:.1f}</td>
                    <td>{rsi_interp}</td>
                </tr>
            """

    # MACD
    macd = tech_factors.get("macd_signal")
    if macd:
        factors_html += f"""
            <tr>
                <td>MACD</td>
                <td>{macd.upper() if macd else "N/A"}</td>
                <td>{"Momentum alcista" if macd == "bullish" else "Momentum bajista" if macd == "bearish" else "Sin datos"}</td>
            </tr>
        """

    # P/E Ratio - Fundamental
    pe_ratio = fundamentals.get("pe_ratio", "N/A")
    if pe_ratio != "N/A":
        try:
            pe_float = float(pe_ratio)
            pe_interp = (
                "Potencialmente infravalorado"
                if pe_float < 15
                else (
                    "Potencialmente sobrevalorado"
                    if pe_float > 30
                    else "Valoración razonable"
                )
            )
        except:
            pe_interp = "No analizable"

        factors_html += f"""
            <tr>
                <td>P/E Ratio</td>
                <td>{pe_ratio}</td>
                <td>{pe_interp}</td>
            </tr>
        """

    # Sentimiento
    factors_html += f"""
        <tr>
            <td>Sentimiento</td>
            <td>{sentiment_value}</td>
            <td>{"Muy Positivo" if sentiment_score > 0.7 else "Positivo" if sentiment_score > 0.55 else "Negativo" if sentiment_score < 0.45 else "Muy Negativo" if sentiment_score < 0.3 else "Neutral"}</td>
        </tr>
    """

    # Cerrar tabla y sección
    st.markdown(
        f"""
                {factors_html}
                </table>
            </div>
            
            <div class="report-section">
                <div class="report-section-title">Estrategia Recomendada</div>
                <p>
        """,
        unsafe_allow_html=True,
    )

    # Recomendar estrategia específica basada en el tipo de recomendación
    if rec_type == "CALL":
        st.markdown(
            f"""
                <p><strong>Estrategia Primaria:</strong> Call Debit Spread - Comprar un CALL ATM y vender un CALL OTM con el mismo vencimiento.</p>
                <p><strong>Estrategia Alternativa:</strong> {"Long Call directo" if score > 70 else "Bull Put Spread"}</p>
                <p><strong>Horizonte Óptimo:</strong> {timeframe}</p>
                <p><strong>Gestión de Riesgo:</strong> Stop loss si el precio cae un 5% por debajo del precio actual o rompe la SMA de 20 períodos.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif rec_type == "PUT":
        st.markdown(
            f"""
                <p><strong>Estrategia Primaria:</strong> Put Debit Spread - Comprar un PUT ATM y vender un PUT OTM con el mismo vencimiento.</p>
                <p><strong>Estrategia Alternativa:</strong> {"Long Put directo" if score < 30 else "Bear Call Spread"}</p>
                <p><strong>Horizonte Óptimo:</strong> {timeframe}</p>
                <p><strong>Gestión de Riesgo:</strong> Stop loss si el precio sube un 5% por encima del precio actual o rompe la SMA de 20 períodos al alza.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
                <p><strong>Estrategia Primaria:</strong> Iron Condor - Venta de un Call spread y un Put spread con mismo vencimiento.</p>
                <p><strong>Estrategia Alternativa:</strong> Calendar Spread</p>
                <p><strong>Horizonte Óptimo:</strong> {timeframe}</p>
                <p><strong>Gestión de Riesgo:</strong> Cierre anticipado si el mercado rompe rangos de consolidación o aumenta significativamente su volatilidad.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Cerrar contenedor
    st.markdown("</div>", unsafe_allow_html=True)


def display_news_feed(recommendation):
    """Muestra feed de noticias con manejo mejorado de datos"""
    news = recommendation.get("news", [])

    if not news:
        st.info("No se encontraron noticias recientes.")
        return

    st.markdown(
        '<div class="sub-header">Noticias Recientes</div>', unsafe_allow_html=True
    )

    # Mostrar noticias recientes
    for item in news:
        date_str = item.get("date", "Fecha no disponible")
        title = item.get("title", "Sin título")
        url = item.get("url", "#")

        # Validación básica para evitar problemas de formateo
        if not isinstance(date_str, str):
            date_str = str(date_str)
        if not isinstance(title, str):
            title = str(title)
        if not isinstance(url, str) or not url.startswith(("http://", "https://", "#")):
            url = "#"

        st.markdown(
            f"""
        <div class="news-card">
            <div class="news-date">{date_str}</div>
            <a href="{url}" target="_blank">{title}</a>
        </div>
        """,
            unsafe_allow_html=True,
        )


def display_web_insights(recommendation):
    """Muestra insights de búsqueda web con manejo mejorado de URLs"""
    web_results = recommendation.get("web_results", [])

    if not web_results:
        st.info("No se encontraron resultados de búsqueda web.")
        return

    st.markdown(
        '<div class="sub-header">Insights de Mercado</div>', unsafe_allow_html=True
    )

    # Mostrar resultados de búsqueda web en un expander
    with st.expander("Ver fuentes de análisis"):
        for i, result in enumerate(web_results):
            title = result.get("title", "Sin título")
            content = result.get("content", "Sin contenido")
            source = result.get("source", "Fuente")
            url = result.get("url", "#")

            # Validar URL
            if not isinstance(url, str) or not url.startswith(
                ("http://", "https://", "#")
            ):
                url = "#"

            st.markdown(
                f"""
            #### {title}
            {content}
            
            [Leer más en {source}]({url})
            """
            )

            if i < len(web_results) - 1:
                st.markdown("---")


def display_trading_strategies(recommendation):
    """Muestra estrategias de trading recomendadas basadas en el análisis"""
    rec_type = recommendation.get("recommendation", "NEUTRAL")
    score = recommendation.get("score", 50)

    if rec_type == "NEUTRAL":
        st.markdown(
            """
            <div class="institutional-insight">
                <h4>Posición Neutral - Estrategias Recomendadas</h4>
                <p>En condiciones de mercado laterales, considere estrategias neutral-delta como:</p>
                <ul>
                    <li><strong>Iron Condor:</strong> Venta de un Call spread y un Put spread con mismo vencimiento</li>
                    <li><strong>Calendar Spread:</strong> Compra de opciones de largo plazo y venta de corto plazo</li>
                    <li><strong>Butterfly:</strong> Posición limitada con potencial en un rango específico</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="sub-header">Estrategias Profesionales Recomendadas</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    if rec_type == "CALL":
        with col1:
            st.markdown(
                """
                <div class="strategy-card">
                    <h4>Call Debit Spread</h4>
                    <p><strong>Descripción:</strong> Compra un CALL ATM y vende un CALL OTM con el mismo vencimiento.</p>
                    <p><strong>Beneficio máximo:</strong> Limitado al diferencial entre strikes menos la prima pagada.</p>
                    <p><strong>Pérdida máxima:</strong> Limitada a la prima neta pagada.</p>
                    <p><strong>Volatilidad:</strong> Favorable en entorno de volatilidad baja a moderada.</p>
                    <p><strong>Horizonte:</strong> 2-4 semanas.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="strategy-card">
                    <h4>Bull Put Spread</h4>
                    <p><strong>Descripción:</strong> Vende un PUT OTM y compra un PUT más OTM con el mismo vencimiento.</p>
                    <p><strong>Beneficio máximo:</strong> Limitado a la prima neta recibida.</p>
                    <p><strong>Pérdida máxima:</strong> Diferencia entre strikes menos prima recibida.</p>
                    <p><strong>Volatilidad:</strong> Favorable en entorno de volatilidad alta.</p>
                    <p><strong>Horizonte:</strong> 1-3 semanas.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if score > 75:  # Señal muy alcista
            st.markdown(
                """
                <div class="institutional-insight">
                    <h4>Estrategia Agresiva: Long Call</h4>
                    <p><strong>Implementación:</strong> Comprar CALL ATM con vencimiento de 30-45 días.</p>
                    <p><strong>Ratio Riesgo/Recompensa:</strong> Potencial ilimitado al alza vs pérdida limitada a prima.</p>
                    <p><strong>Nivel de Sofisticación:</strong> Básico</p>
                    <p><strong>Nota de Trading:</strong> Considerar establecer take profit en 100% de ganancia y stop loss en 50% de pérdida.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif rec_type == "PUT":
        with col1:
            st.markdown(
                """
                <div class="strategy-card">
                    <h4>Put Debit Spread</h4>
                    <p><strong>Descripción:</strong> Compra un PUT ATM y vende un PUT OTM con el mismo vencimiento.</p>
                    <p><strong>Beneficio máximo:</strong> Limitado al diferencial entre strikes menos la prima pagada.</p>
                    <p><strong>Pérdida máxima:</strong> Limitada a la prima neta pagada.</p>
                    <p><strong>Volatilidad:</strong> Favorable en entorno de volatilidad baja a moderada.</p>
                    <p><strong>Horizonte:</strong> 2-4 semanas.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="strategy-card">
                    <h4>Bear Call Spread</h4>
                    <p><strong>Descripción:</strong> Vende un CALL OTM y compra un CALL más OTM con el mismo vencimiento.</p>
                    <p><strong>Beneficio máximo:</strong> Limitado a la prima neta recibida.</p>
                    <p><strong>Pérdida máxima:</strong> Diferencia entre strikes menos prima recibida.</p>
                    <p><strong>Volatilidad:</strong> Favorable en entorno de volatilidad alta.</p>
                    <p><strong>Horizonte:</strong> 1-3 semanas.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if score < 25:  # Señal muy bajista
            st.markdown(
                """
                <div class="institutional-insight">
                    <h4>Estrategia Agresiva: Long Put</h4>
                    <p><strong>Implementación:</strong> Comprar PUT ATM con vencimiento de 30-45 días.</p>
                    <p><strong>Ratio Riesgo/Recompensa:</strong> Alto potencial a la baja vs pérdida limitada a prima.</p>
                    <p><strong>Nivel de Sofisticación:</strong> Básico</p>
                    <p><strong>Nota de Trading:</strong> Considerar establecer take profit en 100% de ganancia y stop loss en 50% de pérdida.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def display_cache_stats():
    """Muestra estadísticas de caché con datos detallados"""
    try:
        stats = _data_cache.get_stats()

        st.sidebar.markdown("### 💾 Estadísticas de Caché")
        st.sidebar.text(f"Entradas: {stats['entradas']}")
        st.sidebar.text(f"Hit rate: {stats['hit_rate']}")
        st.sidebar.text(f"Hits/Misses: {stats['hits']}/{stats['misses']}")

        # Botón para limpiar caché
        if st.sidebar.button("🧹 Limpiar Caché"):
            cleared = _data_cache.clear()
            st.sidebar.success(f"Caché limpiado: {cleared} entradas eliminadas")
            st.rerun()
    except Exception as e:
        logger.error(f"Error mostrando estadísticas de caché: {str(e)}")
        st.sidebar.warning("No se pudieron cargar las estadísticas de caché")


def display_disclaimer():
    """Muestra disclaimer legal"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    ### ⚠️ Disclaimer
    
    La información proporcionada tiene fines informativos y educativos únicamente. No constituye asesoramiento financiero ni recomendación para comprar o vender valores.
    
    Este análisis utiliza datos recopilados de fuentes públicas y no garantiza la precisión, integridad o actualidad de la información. El trading de opciones implica riesgos significativos y puede resultar en pérdidas financieras.
    """
    )


def display_login_form():
    """Muestra formulario de login con validación mejorada"""
    st.markdown(
        """
        <div class="login-container">
            <div class="login-header">MarketIntel Trader</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Crear un formulario de login centrado
    login_form = st.form(key="login_form")

    with login_form:
        st.markdown("<h3>Acceso al Sistema</h3>", unsafe_allow_html=True)
        password = st.text_input("Contraseña", type="password")
        submit_button = st.form_submit_button(label="Ingresar")

    if submit_button:
        with st.spinner("Verificando credenciales..."):
            if check_password(password):
                st.session_state.authenticated = True
                st.session_state.last_successful_auth = datetime.now()
                st.success("Autenticación exitosa!")
                time.sleep(1)  # Breve pausa para mostrar mensaje de éxito
                st.rerun()  # Recargar la página para mostrar el contenido principal
            # El mensaje de error lo maneja la función check_password internamente


def display_session_info():
    """Muestra información de sesión en la barra lateral con formato mejorado"""
    try:
        session_info = get_session_info()

        if session_info["authenticated"]:
            st.sidebar.markdown("### 👤 Información de Sesión")
            st.sidebar.text(f"Estado: Autenticado")

            # Mostrar hora de inicio de sesión
            if session_info["last_auth"]:
                login_time = session_info["last_auth"].strftime("%d/%m/%Y %H:%M:%S")
                st.sidebar.text(f"Inicio: {login_time}")

                # Calcular tiempo restante de sesión
                session_expiry = session_info["last_auth"] + timedelta(hours=8)
                remaining = session_expiry - datetime.now()

                if remaining.total_seconds() > 0:
                    remaining_hours = remaining.seconds // 3600
                    remaining_minutes = (remaining.seconds % 3600) // 60

                    progress_value = 1 - (remaining.total_seconds() / (8 * 3600))
                    progress_color = (
                        "green"
                        if progress_value < 0.75
                        else "orange" if progress_value < 0.9 else "red"
                    )

                    st.sidebar.text(
                        f"Tiempo restante: {remaining_hours}h {remaining_minutes}m"
                    )
                    st.sidebar.progress(
                        progress_value, f"Sesión: {int((1-progress_value)*100)}%"
                    )
                else:
                    st.sidebar.warning(
                        "Sesión expirada. Por favor, vuelva a iniciar sesión."
                    )

            # Botón para cerrar sesión
            if st.sidebar.button("🚪 Cerrar Sesión"):
                clear_session()
                st.rerun()
    except Exception as e:
        logger.error(f"Error mostrando información de sesión: {str(e)}")


def main():
    """Función principal mejorada con manejo de errores y optimización del flujo de análisis"""
    # Comprobar si ya existe una sesión autenticada y válida
    if not validate_session():
        display_login_form()
    else:
        try:
            # Inicializar cliente OpenAI
            openai_client, assistant_id = init_openai_client()

            # Mostrar la aplicación principal
            st.markdown(
                '<h1 class="main-header">MarketIntel: Análisis Avanzado de Opciones</h1>',
                unsafe_allow_html=True,
            )

            # Mostrar información de sesión en la barra lateral
            display_session_info()

            with st.sidebar.expander("🔍 Diagnóstico de Datos"):
                st.markdown(
                    """
                ### Orígenes de Datos
                
                **Precios de mercado**:
                - Fuente primaria: Yahoo Finance
                - Fallback: Alpha Vantage, FinnHub
                - Latencia típica: 5-15 minutos
                
                **Datos fundamentales**:
                - Fuente primaria: Yahoo Finance Scraping
                - Fallback: Alpha Vantage
                - Actualización: Diaria
                
                **Noticias y Sentimiento**:
                - Fuentes: Alpha Vantage News, FinViz
                - Análisis: algoritmo propietario
                
                > **Nota para traders profesionales**: Para ejecución, use datos en tiempo real de sus plataformas institucionales. Esta herramienta está optimizada para análisis y planificación, no para timing preciso de entradas/salidas.
                """
                )

            # Sidebar - Configuración
            st.sidebar.title("⚙️ Configuración")

            # Input para API keys con integración de secrets.toml
            with st.sidebar.expander("🔑 Claves API"):
                # Cargar claves desde secrets.toml
                api_keys = get_api_keys_from_secrets()

                # Mostrar estado actual de las claves API
                provider_status = {
                    "YOU API": "you",
                    "Tavily API": "tavily",
                    "Alpha Vantage API": "alpha_vantage",
                    "Finnhub API": "finnhub",
                    "MarketStack API": "marketstack",
                }

                for display_name, key_name in provider_status.items():
                    status = (
                        "✅ Configurada"
                        if key_name in api_keys and api_keys[key_name]
                        else "❌ No configurada"
                    )
                    st.markdown(f"**{display_name}:** {status}")

                # Permitir sobrescribir desde la UI
                st.markdown("---")
                st.markdown("**Sobrescribir claves (opcional):**")

                # Inputs para claves API
                api_inputs = {}
                for display_name, key_name in provider_status.items():
                    api_inputs[key_name] = st.text_input(
                        f"{display_name} Key", type="password"
                    )

                # Sobrescribir si se ingresa algo
                for key_name, value in api_inputs.items():
                    if value:
                        api_keys[key_name] = value

                # Mostrar instrucciones si no hay claves configuradas
                if not any(api_keys.values()):
                    st.info(
                        """
                    Para un análisis más completo, configura tus claves API:
                    1. Edita el archivo `.streamlit/secrets.toml`
                    2. Agrega tus claves en la sección [api_keys]
                    3. O ingresa las claves directamente aquí
                    """
                    )

            # Selección de símbolo por categoría
            st.markdown("### Seleccionar Instrumento")
            col1, col2 = st.columns(2)

            with col1:
                categoria = st.selectbox(
                    "Categoría",
                    options=list(SYMBOLS.keys()),
                    index=1,  # Por defecto selecciona Tecnología
                )

            with col2:
                symbol = st.selectbox(
                    "Símbolo",
                    options=SYMBOLS[categoria],
                    index=0,  # Por defecto selecciona el primer símbolo de la categoría
                )

            # Opción para entrada manual
            usar_simbolo_personalizado = st.checkbox("Usar símbolo personalizado")
            if usar_simbolo_personalizado:
                simbolo_custom = (
                    st.text_input("Ingresa símbolo personalizado", "").strip().upper()
                )
                if simbolo_custom:
                    symbol = simbolo_custom

            # Opciones avanzadas de análisis
            with st.expander("⚙️ Opciones avanzadas de análisis"):
                col1, col2 = st.columns(2)

                with col1:
                    periodo_historico = st.selectbox(
                        "Período histórico",
                        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                        index=2,  # 6mo por defecto
                    )

                with col2:
                    intervalo_datos = st.selectbox(
                        "Intervalo de datos",
                        options=["1d", "1h", "30m", "15m", "5m", "1m", "1wk", "1mo"],
                        index=0,  # 1d por defecto
                    )

                usar_datos_intraday = st.checkbox("Usar datos intraday", value=False)
                if usar_datos_intraday and intervalo_datos == "1d":
                    intervalo_datos = "1h"
                    st.info("Se ha cambiado el intervalo a 1h para datos intraday")

            # Botón de análisis
            analyze_button = st.button(
                "🔍 Analizar Opciones", type="primary", use_container_width=True
            )

            if analyze_button and symbol:
                with st.spinner(f"Analizando {symbol}... Esto puede tomar un momento"):
                    try:
                        # Realizar análisis
                        recommendation = analyze_stock_options(symbol, api_keys)

                        # Intentar sincronizar datos
                        recommendation = synchronize_market_data(recommendation, symbol)

                        if recommendation.get("recommendation") == "ERROR":
                            st.error(
                                f"Error al analizar {symbol}: {recommendation.get('error')}"
                            )
                        else:
                            # 1. Resumen consolidado
                            display_consolidated_report(recommendation, symbol)

                            # 2. Resumen de recomendación
                            display_recommendation_summary(recommendation)

                            # 3. Gráfico técnico
                            st.markdown(
                                '<div class="sub-header">Análisis Técnico</div>',
                                unsafe_allow_html=True,
                            )
                            chart_data = pd.DataFrame(
                                recommendation.get("chart_data", [])
                            )
                            if not chart_data.empty:
                                fig = create_technical_chart(chart_data)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(
                                    "No hay datos técnicos disponibles para visualización."
                                )

                            # 3. Consultar al experto IA
                            if openai_client and assistant_id:
                                with st.spinner("Consultando al experto de mercado..."):
                                    expert_opinion = consult_expert_ia(
                                        openai_client,
                                        assistant_id,
                                        recommendation,
                                        symbol,
                                    )
                                    display_expert_opinion(expert_opinion)
                            else:
                                st.warning(
                                    "El experto IA no está disponible. Verifica las credenciales de OpenAI."
                                )

                            # 4. Estrategias de trading recomendadas
                            display_trading_strategies(recommendation)

                            # Crear pestañas para el resto del análisis
                            tab1, tab2, tab3 = st.tabs(
                                [
                                    "📊 Análisis Fundamental",
                                    "📰 Noticias y Sentimiento",
                                    "🌐 Insights Web",
                                ]
                            )

                            with tab1:
                                col1, col2 = st.columns(2)

                                with col1:
                                    # Factores técnicos
                                    display_technical_factors(recommendation)

                                with col2:
                                    # Factores fundamentales
                                    display_fundamental_factors(recommendation)

                            with tab2:
                                col1, col2 = st.columns(2)

                                with col1:
                                    # Análisis de sentimiento
                                    display_sentiment_analysis(recommendation)

                                with col2:
                                    # Feed de noticias
                                    display_news_feed(recommendation)

                            with tab3:
                                # Insights web
                                display_web_insights(recommendation)

                            # Mostrar fecha de último análisis
                            st.caption(
                                f"Análisis actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
                            )

                    except Exception as e:
                        st.error(f"Error durante el análisis: {str(e)}")
                        logger.error(
                            f"Error durante el análisis de {symbol}: {str(e)}",
                            exc_info=True,
                        )

            # Mostrar estadísticas de caché
            display_cache_stats()

            # Mostrar disclaimer
            display_disclaimer()

            # Pie de página
            st.sidebar.markdown("---")
            st.sidebar.markdown(
                """
            ### 👨‍💻 Desarrollado por
            
            [Trading & Analysis Specialist](https://github.com)
            
            **Versión:** 1.1.0
            """
            )
        except Exception as e:
            st.error(f"Error en la aplicación: {str(e)}")
            logger.error(f"Error crítico en la aplicación: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
