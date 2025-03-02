#!/bin/bash

# Nombre y descripción del proyecto
PROJECT_NAME="MarketIntel Options Analyzer"
PROJECT_DESCRIPTION="Una herramienta completa para analizar opciones de acciones y realizar recomendaciones de trading"
CURRENT_DATE=$(date +"%Y-%m-%d")
CURRENT_YEAR=$(date +"%Y")

echo "Creando archivos para $PROJECT_NAME..."

# Crear directorios necesarios
mkdir -p .streamlit

# Crear secret.toml (para secretos de Streamlit)
cat > .streamlit/secrets.toml << EOF
# Configuración de Claves API para $PROJECT_NAME
# Reemplaza con tus claves API reales

[api_keys]
YOU_API_KEY = ""
TAVILY_API_KEY = ""
EOF

# Crear README.md
cat > README.md << EOF
# $PROJECT_NAME

$PROJECT_DESCRIPTION basado en indicadores técnicos, datos fundamentales, sentimiento de noticias y análisis web.

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
   \`\`\`
   git clone https://github.com/tuusuario/marketintel-options-analyzer.git
   cd marketintel-options-analyzer
   \`\`\`

2. Instala las dependencias:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

3. Configura las claves API:
   - Crea un archivo \`.streamlit/secrets.toml\` con tus claves API
   - O introdúcelas directamente en la barra lateral de la aplicación

### Ejecutando la Aplicación

\`\`\`
streamlit run stock_options_analyzer.py
\`\`\`

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
EOF

# Crear CHANGELOG.md
cat > CHANGELOG.md << EOF
# Registro de Cambios

Todos los cambios notables en $PROJECT_NAME serán documentados en este archivo.

## [1.0.0] - $CURRENT_DATE

### Añadido
- Lanzamiento inicial de $PROJECT_NAME
- Análisis técnico con gráficos OHLC, medias móviles, RSI, MACD y Bandas Bollinger
- Datos fundamentales de Yahoo Finance
- Análisis de sentimiento de noticias
- Integración de búsqueda web con múltiples proveedores (YOU API, Tavily, DuckDuckGo)
- Recomendaciones de trading de opciones basadas en análisis multifactorial
- Dashboard interactivo con Streamlit
- Sistema de caché de datos para mejorar el rendimiento

### Características Técnicas
- Motor de datos de mercado con arquitectura modular
- Caché de datos para reducir llamadas a API y mejorar el rendimiento
- Múltiples proveedores de búsqueda web con mecanismos de respaldo
- Cálculos completos de indicadores técnicos
- Análisis de sentimiento para noticias y contenido web
- UI responsiva con visualizaciones Plotly
EOF

# Crear requirements.txt con todas las dependencias externas
cat > requirements.txt << EOF
# Dependencias principales
streamlit>=1.22.0
pandas>=1.5.0
numpy>=1.23.0

# Datos financieros
yfinance>=0.2.12

# Web scraping y búsqueda
requests>=2.28.0
beautifulsoup4>=4.11.0
duckduckgo-search>=2.8.0

# Visualización de datos
plotly>=5.13.0

# Utilidades adicionales
python-dateutil>=2.8.0
pytz>=2022.1
tqdm>=4.64.0
EOF

# Crear .gitignore
cat > .gitignore << EOF
# Secretos de Streamlit
.streamlit/secrets.toml

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Entorno Virtual
venv/
ENV/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Específicos del SO
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
EOF

# Crear un archivo LICENSE simple (Licencia MIT)
cat > LICENSE << EOF
Licencia MIT

Copyright (c) $CURRENT_YEAR $PROJECT_NAME

Por la presente se concede permiso, libre de cargos, a cualquier persona que obtenga una copia
de este software y de los archivos de documentación asociados (el "Software"), a utilizar
el Software sin restricción, incluyendo sin limitación los derechos a usar, copiar, modificar,
fusionar, publicar, distribuir, sublicenciar, y/o vender copias del Software, y a permitir a
las personas a las que se les proporcione el Software a hacer lo mismo, sujeto a las siguientes
condiciones:

El aviso de copyright anterior y este aviso de permiso se incluirán en todas las copias
o partes sustanciales del Software.

EL SOFTWARE SE PROPORCIONA "COMO ESTÁ", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O IMPLÍCITA,
INCLUYENDO PERO NO LIMITADO A GARANTÍAS DE COMERCIALIZACIÓN, IDONEIDAD PARA UN PROPÓSITO
PARTICULAR Y NO INFRACCIÓN. EN NINGÚN CASO LOS AUTORES O TITULARES DEL COPYRIGHT SERÁN
RESPONSABLES DE NINGUNA RECLAMACIÓN, DAÑOS U OTRAS RESPONSABILIDADES, YA SEA EN UNA ACCIÓN
DE CONTRATO, AGRAVIO O CUALQUIER OTRO MOTIVO, QUE SURJA DE O EN CONEXIÓN CON EL SOFTWARE
O EL USO U OTRO TIPO DE ACCIONES EN EL SOFTWARE.
EOF

# Hacer el script ejecutable
chmod +x "$0"

echo "✅ Archivos creados exitosamente para $PROJECT_NAME."
echo "⚠️  Nota importante de seguridad: .streamlit/secrets.toml ha sido añadido a .gitignore para proteger tus claves API."
echo "🚀 Para ejecutar la aplicación: streamlit run stock_options_analyzer.py"