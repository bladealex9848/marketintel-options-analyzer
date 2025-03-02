import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import json
from market_data_engine import (
    analyze_stock_options,
    _data_cache,
    get_api_keys_from_secrets,
)

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

# Estilos personalizados
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6c757d;
    }
    
    .call-badge {
        background-color: rgba(0, 200, 0, 0.2);
        color: #006400;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    
    .put-badge {
        background-color: rgba(200, 0, 0, 0.2);
        color: #8B0000;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    
    .neutral-badge {
        background-color: rgba(128, 128, 128, 0.2);
        color: #696969;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    
    .news-card {
        border-left: 3px solid #1E88E5;
        padding-left: 0.75rem;
        margin-bottom: 0.75rem;
    }
    
    .news-date {
        font-size: 0.75rem;
        color: #6c757d;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: 600;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: 600;
    }
    
    .confidence-low {
        color: #6c757d;
        font-weight: 600;
    }
    
    /* Nuevos estilos para indicadores de análisis institucional */
    .institutional-insight {
        border-left: 4px solid #9C27B0;
        background-color: rgba(156, 39, 176, 0.05);
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        border-radius: 0 0.25rem 0.25rem 0;
    }
    
    .risk-low {
        color: #4CAF50;
        font-weight: 600;
    }
    
    .risk-medium {
        color: #FF9800;
        font-weight: 600;
    }
    
    .risk-high {
        color: #F44336;
        font-weight: 600;
    }
    
    .pro-trading-tip {
        background-color: rgba(33, 150, 243, 0.1);
        border: 1px solid rgba(33, 150, 243, 0.3);
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    
    .strategy-card {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .strategy-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-color: #bbdefb;
    }
</style>
""",
    unsafe_allow_html=True,
)


def create_technical_chart(data):
    """Crea gráfico técnico con indicadores"""
    # Corregido: verificación adecuada de DataFrame vacío
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
        df = data

    # Crear figura con subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("OHLC con Medias Móviles y Bandas Bollinger", "MACD", "RSI"),
    )

    # Añadir Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["Date"] if "Date" in df.columns else df.index,
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
                    x=df["Date"] if "Date" in df.columns else df.index,
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
                    x=df["Date"] if "Date" in df.columns else df.index,
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
                x=df["Date"] if "Date" in df.columns else df.index,
                y=df["MACD"],
                name="MACD",
                line=dict(color="rgba(33, 150, 243, 0.7)", width=1),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["Date"] if "Date" in df.columns else df.index,
                y=df["MACD_Signal"],
                name="Señal MACD",
                line=dict(color="rgba(255, 87, 34, 0.7)", width=1),
            ),
            row=2,
            col=1,
        )

        # Añadir histograma MACD
        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            colors = [
                "rgba(33, 150, 243, 0.7)" if val >= 0 else "rgba(255, 87, 34, 0.7)"
                for val in (df["MACD"] - df["MACD_Signal"])
            ]

            fig.add_trace(
                go.Bar(
                    x=df["Date"] if "Date" in df.columns else df.index,
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
                x=df["Date"] if "Date" in df.columns else df.index,
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
                x0=df["Date"].iloc[0] if "Date" in df.columns else df.index[0],
                x1=df["Date"].iloc[-1] if "Date" in df.columns else df.index[-1],
                y0=level,
                y1=level,
                line=dict(color=color, width=1, dash="dash"),
                row=3,
                col=1,
            )

    # Ajustar layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        title=f"Análisis Técnico",
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
    """Muestra resumen de recomendación"""
    rec_type = recommendation["recommendation"]
    confidence = recommendation["confidence"]
    score = recommendation["score"]

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
    st.markdown(
        f"""
    <div style="margin-top: 1rem; text-align: center;">
        <span style="font-weight: 500;">Horizonte recomendado:</span> {recommendation['timeframe']}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Fecha de análisis
    analysis_date = datetime.fromisoformat(recommendation["analysis_date"])
    st.markdown(
        f"""
    <div style="margin-top: 0.5rem; text-align: center; font-size: 0.8rem; color: #6c757d;">
        Análisis realizado el {analysis_date.strftime('%d/%m/%Y a las %H:%M:%S')}
    </div>
    """,
        unsafe_allow_html=True,
    )

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


def display_fundamental_factors(recommendation):
    """Muestra factores fundamentales"""
    fundamentals = recommendation.get("fundamental_factors", {})
    if not fundamentals:
        st.info("No se encontraron datos fundamentales disponibles.")
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

    for i, (key, field, label) in enumerate(metrics):
        with cols[i % 4]:
            value = fundamentals.get(field, "N/A")
            st.metric(label=label, value=value)

    # Mostrar métricas adicionales avanzadas si existen
    advanced_metrics = [
        ("PEG Ratio", "peg_ratio", "PEG Ratio"),
        ("Price to Book", "price_to_book", "P/B"),
        ("Return on Equity", "roe", "ROE"),
        ("Return on Assets", "roa", "ROA"),
        ("Profit Margin", "profit_margin", "Margen"),
        ("Operating Margin", "operating_margin", "Margen Op."),
    ]

    # Comprobar si hay al menos un métrica avanzada
    has_advanced = any(field in fundamentals for _, field, _ in advanced_metrics)

    if has_advanced:
        st.markdown("##### Métricas Avanzadas")
        adv_cols = st.columns(6)

        for i, (key, field, label) in enumerate(advanced_metrics):
            with adv_cols[i % 6]:
                if field in fundamentals:
                    value = fundamentals.get(field, "N/A")
                    # Convertir a porcentaje si es necesario
                    if field in ["roe", "roa", "profit_margin", "operating_margin"]:
                        try:
                            value = (
                                f"{float(value)*100:.2f}%" if value != "N/A" else "N/A"
                            )
                        except:
                            pass
                    st.metric(label=label, value=value)


def display_technical_factors(recommendation):
    """Muestra factores técnicos"""
    tech_factors = recommendation.get("technical_factors", {})
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
                value_text = f"{value:.1f}" if value is not None else "N/A"

                # Determinar si está en zona de sobrecompra/sobreventa
                if value is not None:
                    if value > 70:
                        delta = "Sobrecompra"
                        delta_color = "inverse"
                    elif value < 30:
                        delta = "Sobreventa"
                        delta_color = "normal"
                    else:
                        delta = "Neutral"
                        delta_color = "off"
                else:
                    delta = None
                    delta_color = "off"

                st.metric(
                    label=label, value=value_text, delta=delta, delta_color=delta_color
                )

            elif key == "MACD":
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
                st.metric(
                    label=key,
                    value=(
                        "Por encima"
                        if value is True
                        else "Por debajo" if value is False else "N/A"
                    ),
                    delta=(
                        "Alcista"
                        if value is True
                        else "Bajista" if value is False else None
                    ),
                    delta_color=(
                        "normal"
                        if value is True
                        else "inverse" if value is False else "off"
                    ),
                )

    # Añadir una sección de análisis profesional de los indicadores
    if tech_factors.get("rsi") is not None:
        rsi_value = tech_factors.get("rsi")
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
    """Muestra análisis de sentimiento"""
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

            # Crear medidor
            st.markdown("### Sentimiento de Noticias")

            # Transformar score a grados (0 = -90°, 0.5 = 0°, 1 = 90°)
            angle = (sentiment_score - 0.5) * 180

            # Crear gráfico gauge con Plotly
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=sentiment_score * 100,
                    title={"text": "Sentimiento"},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": "rgba(0,0,0,0)"},
                        "steps": [
                            {"range": [0, 40], "color": "rgba(255, 87, 34, 0.3)"},
                            {"range": [40, 60], "color": "rgba(158, 158, 158, 0.3)"},
                            {"range": [60, 100], "color": "rgba(76, 175, 80, 0.3)"},
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

            st.markdown("### Análisis Web")

            # Crear gráfico de barras
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=["Alcista", "Bajista"],
                    y=[bullish, bearish],
                    marker_color=["rgba(76, 175, 80, 0.7)", "rgba(255, 87, 34, 0.7)"],
                )
            )

            fig.update_layout(
                title="Menciones en Fuentes Web",
                height=250,
                margin=dict(l=10, r=10, t=50, b=10),
                yaxis_title="Número de menciones",
                xaxis_title="Sentimiento",
            )

            st.plotly_chart(fig, use_container_width=True)

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


def display_news_feed(recommendation):
    """Muestra feed de noticias"""
    news = recommendation.get("news", [])

    if not news:
        st.info("No se encontraron noticias recientes.")
        return

    st.markdown(
        '<div class="sub-header">Noticias Recientes</div>', unsafe_allow_html=True
    )

    # Mostrar noticias recientes
    for item in news:
        st.markdown(
            f"""
        <div class="news-card">
            <div class="news-date">{item.get('date', 'Fecha no disponible')}</div>
            <a href="{item.get('url', '#')}" target="_blank">{item.get('title', 'Sin título')}</a>
        </div>
        """,
            unsafe_allow_html=True,
        )


def display_web_insights(recommendation):
    """Muestra insights de búsqueda web"""
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
            st.markdown(
                f"""
            #### {result.get('title', 'Sin título')}
            {result.get('content', 'Sin contenido')}
            
            [Leer más en {result.get('source', 'Fuente')}]({result.get('url', '#')})
            """
            )

            if i < len(web_results) - 1:
                st.markdown("---")


def display_trading_strategies(recommendation):
    """Muestra estrategias de trading recomendadas"""
    rec_type = recommendation["recommendation"]
    score = recommendation["score"]

    if rec_type == "NEUTRAL":
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
    """Muestra estadísticas de caché"""
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


def display_disclaimer():
    """Muestra disclaimer"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    ### ⚠️ Disclaimer
    
    La información proporcionada tiene fines informativos y educativos únicamente. No constituye asesoramiento financiero ni recomendación para comprar o vender valores.
    
    Este análisis utiliza datos recopilados de fuentes públicas y no garantiza la precisión, integridad o actualidad de la información. El trading de opciones implica riesgos significativos y puede resultar en pérdidas financieras.
    """
    )


def main():
    """Función principal"""
    st.markdown(
        '<h1 class="main-header">MarketIntel: Análisis Avanzado de Opciones</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar - Configuración
    st.sidebar.title("⚙️ Configuración")

    # Input para API keys con integración de secrets.toml
    with st.sidebar.expander("🔑 Claves API"):
        # Cargar claves desde secrets.toml
        api_keys = get_api_keys_from_secrets()

        # Mostrar estado actual de las claves API
        you_status = "✅ Configurada" if "you" in api_keys else "❌ No configurada"
        tavily_status = (
            "✅ Configurada" if "tavily" in api_keys else "❌ No configurada"
        )

        # APIs adicionales
        alpha_vantage_status = (
            "✅ Configurada" if "alpha_vantage" in api_keys else "❌ No configurada"
        )
        finnhub_status = (
            "✅ Configurada" if "finnhub" in api_keys else "❌ No configurada"
        )
        marketstack_status = (
            "✅ Configurada" if "marketstack" in api_keys else "❌ No configurada"
        )

        st.markdown(f"**YOU API:** {you_status}")
        st.markdown(f"**Tavily API:** {tavily_status}")
        st.markdown(f"**Alpha Vantage API:** {alpha_vantage_status}")
        st.markdown(f"**Finnhub API:** {finnhub_status}")
        st.markdown(f"**MarketStack API:** {marketstack_status}")

        # Permitir sobrescribir desde la UI
        st.markdown("---")
        st.markdown("**Sobrescribir claves (opcional):**")
        you_key = st.text_input("YOU API Key", type="password")
        tavily_key = st.text_input("Tavily API Key", type="password")
        alpha_vantage_key = st.text_input("Alpha Vantage API Key", type="password")
        finnhub_key = st.text_input("Finnhub API Key", type="password")

        # Sobrescribir si se ingresa algo
        if you_key:
            api_keys["you"] = you_key
        if tavily_key:
            api_keys["tavily"] = tavily_key
        if alpha_vantage_key:
            api_keys["alpha_vantage"] = alpha_vantage_key
        if finnhub_key:
            api_keys["finnhub"] = finnhub_key

        # Mostrar instrucciones si no hay claves configuradas
        if not api_keys:
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
        simbolo_custom = st.text_input("Ingresa símbolo personalizado", "").upper()
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
            # Realizar análisis
            recommendation = analyze_stock_options(symbol, api_keys)

            if recommendation.get("recommendation") == "ERROR":
                st.error(f"Error al analizar {symbol}: {recommendation.get('error')}")
            else:
                # Mostrar análisis dividido en secciones

                # 1. Resumen de recomendación
                display_recommendation_summary(recommendation)

                # 2. Gráfico técnico
                st.markdown(
                    '<div class="sub-header">Análisis Técnico</div>',
                    unsafe_allow_html=True,
                )
                chart_data = pd.DataFrame(recommendation.get("chart_data", []))
                if not chart_data.empty:
                    fig = create_technical_chart(chart_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay datos técnicos disponibles para visualización.")

                # 3. Estrategias de trading recomendadas (nueva sección)
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
                        # 4. Factores técnicos
                        display_technical_factors(recommendation)

                    with col2:
                        # 5. Factores fundamentales
                        display_fundamental_factors(recommendation)

                with tab2:
                    col1, col2 = st.columns(2)

                    with col1:
                        # 6. Análisis de sentimiento
                        display_sentiment_analysis(recommendation)

                    with col2:
                        # 7. Feed de noticias
                        display_news_feed(recommendation)

                with tab3:
                    # 8. Insights web
                    display_web_insights(recommendation)

                # Mostrar fecha de último análisis
                st.caption(
                    f"Análisis actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
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
    
    **Versión:** 1.0.1
    """
    )


if __name__ == "__main__":
    main()
