"""
Technical Expert Analyzer
------------------------
Módulo simplificado para análisis técnico con patrones y consulta al experto AI.
Basado en MarketIntel Options Analyzer.
"""

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

# Importar módulos personalizados necesarios
from market_data_engine import (
    fetch_market_data,
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
    detect_candle_patterns,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Clase para manejar la codificación JSON
class NumpyEncoder(json.JSONEncoder):
    """Encoder personalizado para manejar diversos tipos de datos NumPy y Pandas"""

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


# Inicialización del cliente de OpenAI
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


def consult_expert_ia(client, assistant_id, recommendation, symbol):
    """Consulta al experto de IA con análisis detallado"""
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


def display_expert_opinion(expert_opinion):
    """Muestra la opinión del experto IA con manejo mejorado de salida"""
    if not expert_opinion:
        return

    st.markdown("## 🧠 Análisis del Experto")

    # Procesamiento mejorado del texto: buscar secciones clave
    sections = {
        "evaluación": "",
        "contraste": "",
        "recomendación": "",
        "estrategias": "",
        "riesgos": "",
        "proyección": "",
    }

    current_section = None

    try:
        # Intentar identificar secciones en el texto
        lines = expert_opinion.split("\n")
        for line in lines:
            line = line.strip()

            # Detectar secciones por encabezados
            if "EVALUACIÓN INTEGRAL" in line.upper():
                current_section = "evaluación"
                continue
            elif "CONTRASTE NOTICIAS" in line.upper():
                current_section = "contraste"
                continue
            elif "RECOMENDACIÓN DE TRADING" in line.upper():
                current_section = "recomendación"
                continue
            elif "ESTRATEGIAS DE OPCIONES" in line.upper():
                current_section = "estrategias"
                continue
            elif "RIESGOS Y STOP" in line.upper():
                current_section = "riesgos"
                continue
            elif "PROYECCIÓN DE MOVIMIENTO" in line.upper():
                current_section = "proyección"
                continue

            # Agregar línea a la sección actual
            if current_section and line:
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
            st.markdown("### 📊 Evaluación Integral")
            st.markdown(sections["evaluación"])

        if sections["contraste"]:
            st.markdown("### 🔍 Contraste Noticias-Fundamentos")
            st.markdown(sections["contraste"])

        if sections["recomendación"]:
            st.markdown("### 🎯 Recomendación de Trading")
            st.markdown(sections["recomendación"])

        if sections["estrategias"]:
            st.markdown("### 📈 Estrategias con Opciones")
            st.markdown(sections["estrategias"])

        if sections["riesgos"]:
            st.markdown("### ⚠️ Riesgos y Stop Loss")
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


def display_login_form():
    """Muestra formulario de login con validación mejorada"""
    st.markdown(
        """
        <div class="login-container">
            <div class="login-header">Análisis Técnico & Experto</div>
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


def main():
    """Función principal para el Analizador Técnico y Experto"""
    # Configurar página
    st.set_page_config(
        page_title="Análisis Técnico & Experto",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Estilos CSS
    st.markdown(
        """
        <style>
        .main-header {
            color: #1E88E5;
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: bold;
        }
        .sub-header {
            color: #0D47A1;
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid #E0E0E0;
            padding-bottom: 0.5rem;
        }
        .expert-container {
            border: 1px solid #EEEEEE;
            border-radius: 10px;
            padding: 1rem;
            background-color: #FAFAFA;
            margin-top: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .expert-header {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
            border-bottom: 1px solid #EEEEEE;
            padding-bottom: 0.5rem;
        }
        .expert-avatar {
            background-color: #1E88E5;
            color: white;
            font-weight: bold;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1rem;
        }
        .expert-title {
            font-weight: 600;
            font-size: 1.2rem;
            color: #1E88E5;
        }
        .expert-content {
            line-height: 1.6;
        }
        .expert-footer {
            margin-top: 1rem;
            font-size: 0.8rem;
            color: #9E9E9E;
            text-align: right;
            border-top: 1px solid #EEEEEE;
            padding-top: 0.5rem;
        }
        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 2rem 0;
        }
        .login-header {
            font-size: 2.5rem;
            color: #1E88E5;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Comprobar si ya existe una sesión autenticada y válida
    if not validate_session():
        display_login_form()
    else:
        try:
            # Inicializar cliente OpenAI
            openai_client, assistant_id = init_openai_client()

            # Mostrar la aplicación principal
            st.markdown(
                '<h1 class="main-header">Análisis Técnico & Experto</h1>',
                unsafe_allow_html=True,
            )

            # Mostrar información de sesión en la barra lateral
            display_session_info()

            # Universo de Trading
            # Definir un universo de símbolos reducido para simplificar
            SYMBOLS = {
                "Índices": ["SPY", "QQQ", "DIA", "IWM", "EFA", "VWO"],
                "Tecnología": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
                "Finanzas": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
                "Energía": ["XOM", "CVX", "SHEL", "TTE", "COP"],
                "Salud": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY"],
            }

            # Sidebar - Configuración
            st.sidebar.title("⚙️ Configuración")

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
                        options=["1mo", "3mo", "6mo", "1y", "2y"],
                        index=2,  # 6mo por defecto
                    )

                with col2:
                    intervalo_datos = st.selectbox(
                        "Intervalo de datos",
                        options=["1d", "1h", "30m", "15m", "5m"],
                        index=0,  # 1d por defecto
                    )

            # Botón de análisis
            analyze_button = st.button(
                "🔍 Analizar Instrumento", type="primary", use_container_width=True
            )

            if analyze_button and symbol:
                with st.spinner(f"Analizando {symbol}... Esto puede tomar un momento"):
                    try:
                        # Realizar análisis
                        recommendation = analyze_stock_options(symbol)

                        if recommendation.get("recommendation") == "ERROR":
                            st.error(
                                f"Error al analizar {symbol}: {recommendation.get('error')}"
                            )
                        else:
                            # 1. Gráfico técnico
                            st.markdown(
                                '<div class="sub-header">Análisis Técnico con Patrones</div>',
                                unsafe_allow_html=True,
                            )
                            chart_data = pd.DataFrame(
                                recommendation.get("chart_data", [])
                            )
                            if not chart_data.empty:
                                fig = create_technical_chart(chart_data)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Información adicional de patrones
                                    with st.expander(
                                        "📊 Detalles de patrones técnicos detectados"
                                    ):
                                        # Extraer patrones de los datos técnicos
                                        try:
                                            # Crear patrones para mostrar en el análisis del experto
                                            patterns = {}
                                            supports, resistances = (
                                                detect_support_resistance(chart_data)
                                            )
                                            patterns["supports"] = supports
                                            patterns["resistances"] = resistances

                                            bullish_lines, bearish_lines = (
                                                detect_trend_lines(chart_data)
                                            )
                                            patterns["trend_lines"] = {
                                                "bullish": bullish_lines,
                                                "bearish": bearish_lines,
                                            }

                                            # Patrones de velas (últimas 10)
                                            candle_patterns = detect_candle_patterns(
                                                chart_data.tail(20)
                                            )
                                            patterns["candle_patterns"] = (
                                                candle_patterns
                                            )

                                            # Mostrar resumen de patrones
                                            patterns_text = format_patterns_for_prompt(
                                                patterns
                                            )
                                            st.markdown(patterns_text)
                                        except Exception as e:
                                            st.warning(
                                                f"No se pudieron detectar patrones detallados: {str(e)}"
                                            )
                            else:
                                st.warning(
                                    "No hay datos técnicos disponibles para visualización."
                                )

                            # 2. Consultar al experto IA
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

            # Mostrar disclaimer
            display_disclaimer()

            # Pie de página
            st.sidebar.markdown("---")
            st.sidebar.markdown(
                """
            ### 📈 Desarrollado por
            
            Technical Analysis & Expert Systems
            
            **Versión:** 1.0
            """
            )
        except Exception as e:
            st.error(f"Error en la aplicación: {str(e)}")
            logger.error(f"Error crítico en la aplicación: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
