# MarketIntel Options Analyzer

Una herramienta completa para analizar opciones de acciones y realizar recomendaciones de trading basado en indicadores técnicos, datos fundamentales, sentimiento de noticias y análisis web.

## 📊 Características

- **Análisis Técnico**: Datos históricos de precios con gráficos OHLC, medias móviles, RSI, MACD y Bandas Bollinger
- **Análisis Fundamental**: Métricas clave como ratio P/E, capitalización de mercado, EPS y rendimiento de dividendos
- **Análisis de Sentimiento de Noticias**: Evaluación de artículos recientes y su sentimiento
- **Insights Web**: Información agregada de sitios financieros y opiniones de analistas
- **Recomendaciones de Opciones**: Recomendaciones basadas en datos para trading de opciones (CALL/PUT/NEUTRAL)
- **Dashboard Interactivo**: Representación visual de todos los datos analíticos

## 🚀 Comenzando

### Prerrequisitos

- Python 3.8+
- Claves API (opcionales pero recomendadas para funcionalidad mejorada):
  - YOU API
  - Tavily API

### Instalación

1. Clona el repositorio:
   ```
   git clone https://github.com/bladealex9848/marketintel-options-analyzer.git
   cd marketintel-options-analyzer
   ```

2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

3. Configura las claves API:
   - Crea un archivo `.streamlit/secrets.toml` con tus claves API
   - O introdúcelas directamente en la barra lateral de la aplicación

### Ejecutando la Aplicación

```
streamlit run stock_options_analyzer.py
```

## 🔍 Uso

1. Ingresa un símbolo de acción en el campo de texto (ej. AAPL, MSFT, TSLA)
2. Haz clic en el botón "Analizar Opciones"
3. Visualiza el análisis completo incluyendo:
   - Indicadores técnicos y gráficos
   - Métricas fundamentales
   - Sentimiento de noticias
   - Recomendación de trading

## 📈 Cómo Funciona

El analizador utiliza múltiples fuentes de datos para proporcionar un análisis completo:

1. **Datos Técnicos**: Datos históricos de precios de Yahoo Finance con indicadores técnicos calculados
2. **Datos Fundamentales**: Métricas financieras clave extraídas de sitios web financieros
3. **Análisis de Noticias**: Noticias recientes sobre la acción con análisis de sentimiento
4. **Búsqueda Web**: Insights agregados de analistas financieros y sitios web

El algoritmo de recomendación pondera todos estos factores para generar una recomendación de trading de opciones:
- **Factores Técnicos** (40%): Tendencias de precio, RSI, MACD, momentum
- **Factores Fundamentales** (30%): Ratio PE, volumen, métricas de mercado
- **Sentimiento de Noticias** (20%): Sentimiento alcista/bajista en noticias recientes
- **Análisis Web** (10%): Opiniones agregadas de analistas y insights de mercado

## 🔧 Configuración Avanzada

La aplicación utiliza un sistema de caché para mejorar el rendimiento y reducir las llamadas a API. Las estadísticas de caché se muestran en la barra lateral, y puedes limpiar el caché utilizando el botón proporcionado.

## ⚠️ Aviso Legal

La información proporcionada por MarketIntel Options Analyzer es solo para fines informativos y educativos. No pretende ser un consejo financiero o una recomendación para comprar o vender valores. El trading de opciones implica un riesgo significativo y puede no ser adecuado para todos los inversores.

## 📝 Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo LICENSE para más detalles.
