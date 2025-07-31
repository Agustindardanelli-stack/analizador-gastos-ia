# 💰 Analizador de Gastos IA

> Sistema inteligente de análisis financiero personal con Machine Learning y categorización automática

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🎯 Características Principales

- 🤖 **Categorización Automática con IA**: Machine Learning para clasificar gastos inteligentemente
- 📊 **Dashboard Interactivo**: Visualizaciones dinámicas con Plotly y Streamlit
- 📈 **Análisis Predictivo**: Detección de anomalías y patrones de gasto
- 📋 **Reportes Exportables**: CSV, JSON y análisis detallados
- 🔍 **Filtros Avanzados**: Análisis por categoría, fecha y monto
- 🌐 **Interfaz Web Moderna**: Aplicación responsive y fácil de usar

## 🚀 Demo en Vivo

![Demo del Dashboard](https://via.placeholder.com/800x400/1f77b4/ffffff?text=Dashboard+Interactivo)

*Interfaz principal mostrando análisis de gastos categorizado automáticamente*

## 📋 Funcionalidades

### 🤖 Inteligencia Artificial
- **Random Forest Classifier** para categorización automática
- **TF-IDF Vectorization** para procesamiento de texto
- **Detección de anomalías** con Z-score
- **Niveles de confianza** para cada predicción

### 📊 Análisis de Datos
- Procesamiento de múltiples formatos (CSV, Excel, JSON)
- Limpieza automática de datos
- Estadísticas descriptivas completas
- Análisis temporal de tendencias

### 🎨 Visualizaciones
- Gráficos de torta interactivos
- Timeline de evolución de gastos
- Gráficos de barras por categoría
- Tablas dinámicas y filtros en tiempo real

### 💾 Exportación
- Datos procesados con categorías IA
- Reportes completos en JSON
- Análisis por categorías en CSV

## 🛠️ Tecnologías Utilizadas

### Backend
- **Python 3.11+**: Lenguaje principal
- **pandas**: Manipulación de datos
- **scikit-learn**: Machine Learning
- **NLTK**: Procesamiento de lenguaje natural
- **XGBoost**: Algoritmos de gradient boosting

### Frontend
- **Streamlit**: Framework web interactivo
- **Plotly**: Gráficos interactivos
- **HTML/CSS**: Estilos personalizados

### Machine Learning
- **Random Forest**: Clasificación de categorías
- **TF-IDF**: Vectorización de texto
- **Feature Engineering**: Combinación de características

## 📦 Instalación

### Requisitos Previos
- Python 3.11 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/analizador-gastos-ia.git
cd analizador-gastos-ia
```

2. **Crear entorno virtual**
```bash
python -m venv venv_gastos
```

3. **Activar entorno virtual**
```bash
# Windows
venv_gastos\Scripts\activate

# macOS/Linux
source venv_gastos/bin/activate
```

4. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

5. **Configurar NLTK (primera vez)**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

## 🚀 Uso

### Inicio Rápido

1. **Ejecutar la aplicación**
```bash
streamlit run streamlit_app/main.py
```

2. **Abrir navegador**
- Automáticamente se abre en `http://localhost:8501`
- O accede manualmente a la URL

3. **Cargar datos**
- Usa el botón "🧪 Usar Datos de Ejemplo" para probar
- O sube tu propio archivo CSV/Excel

### Formato de Datos

Tu archivo debe tener al menos estas columnas:

```csv
fecha,descripcion,monto
2024-01-15,Supermercado Carrefour Villa Crespo,18500.50
2024-01-16,Uber viaje al centro,2800.00
2024-01-17,Netflix suscripción mensual,2490.00
```

**Columnas soportadas:**
- **Descripción**: `descripcion`, `concepto`, `detalle`, `merchant`
- **Monto**: `monto`, `importe`, `amount`, `precio`, `valor`
- **Fecha**: `fecha`, `date`, `timestamp` (opcional)

### Categorías Detectadas

El sistema categoriza automáticamente en:

- 🍽️ **Alimentación**: Supermercados, restaurantes, delivery
- 🚗 **Transporte**: Uber, combustible, peajes, transporte público
- ⚡ **Servicios**: Electricidad, internet, telefonía, expensas
- 🎬 **Entretenimiento**: Netflix, cines, juegos, streaming
- 🏥 **Salud**: Farmacias, médicos, laboratorios
- 👔 **Ropa**: Indumentaria, calzado, accesorios
- 🏠 **Hogar**: Muebles, decoración, electrodomésticos
- 📚 **Educación**: Cursos, universidades, libros
- 📦 **Otros**: Categoría general para gastos no clasificados

## 🧠 Cómo Funciona la IA

### 1. Procesamiento de Texto
- Limpieza y normalización de descripciones
- Remoción de stop words en español e inglés
- Tokenización inteligente

### 2. Feature Engineering
- **TF-IDF Vectorization**: Convierte texto en características numéricas
- **Keyword Matching**: Patrones específicos por categoría
- **Weighted Features**: Combinación ponderada de características

### 3. Modelo de Clasificación
```python
# Arquitectura del modelo
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
```

### 4. Evaluación
- Accuracy score en conjunto de prueba
- Matriz de confusión por categoría
- Niveles de confianza por predicción

## 📊 Ejemplos de Uso

### Análisis Personal
```python
# Procesar archivo personal
from src.data_processor import ExpenseDataProcessor

processor = ExpenseDataProcessor()
df, insights = processor.process_file("mis_gastos.csv")
print(f"Total gastado: ${insights['monto_total']:,.2f}")
```

### Categorización Automática
```python
# Usar categorizador entrenado
from src.categorizer import CategoryTrainer

trainer = CategoryTrainer()
trainer.train_with_sample_data()

# Categorizar nuevos gastos
predictions = trainer.categorizer.predict([
    "Supermercado Disco compra semanal",
    "Uber viaje al aeropuerto"
])
# Resultado: ['Alimentación', 'Transporte']
```

## 📈 Casos de Uso

### 👤 Personal
- Análisis de gastos mensuales
- Identificación de patrones de consumo
- Detección de gastos inusuales
- Planificación presupuestaria

### 💼 Empresarial
- Análisis de gastos corporativos
- Auditoría de expenses
- Reportes para contabilidad
- Dashboard ejecutivo

### 🔬 Investigación
- Análisis de comportamiento financiero
- Estudios de patrones de consumo
- Investigación en FinTech

## 🗂️ Estructura del Proyecto

```
analizador-gastos-ia/
├── src/                          # Código fuente principal
│   ├── data_processor.py         # Procesamiento de datos
│   ├── categorizer.py           # Sistema de categorización IA
│   └── utils.py                 # Utilidades y configuración
├── streamlit_app/               # Aplicación web
│   └── main.py                  # Interfaz principal Streamlit
├── data/                        # Datos del proyecto
│   ├── examples/                # Datos de ejemplo
│   ├── processed/               # Datos procesados
│   └── temp/                    # Archivos temporales
├── models/                      # Modelos entrenados
│   └── categorizer_trained.pkl  # Modelo de categorización
├── config/                      # Configuración
├── tests/                       # Tests del sistema
├── requirements.txt             # Dependencias Python
├── README.md                    # Documentación
└── .gitignore                   # Archivos ignorados por Git
```

## 🧪 Testing

### Ejecutar Tests
```bash
# Test completo del sistema
python test_complete.py

# Verificar instalación
python verify_python311.py
```

### Datos de Prueba
El sistema incluye 30+ transacciones de ejemplo que cubren todas las categorías.

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una branch para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Áreas de Mejora
- 🌐 Integración con APIs bancarias
- 🔮 Modelos predictivos avanzados
- 📱 Aplicación móvil
- 🌍 Soporte multi-idioma
- ☁️ Deploy en la nube

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 👤 Autor

**Tu Nombre**
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- LinkedIn: [Tu Perfil](https://linkedin.com/in/tu-perfil)
- Email: tu.email@ejemplo.com

## 🙏 Agradecimientos

- [Streamlit](https://streamlit.io) por el framework web
- [scikit-learn](https://scikit-learn.org) por las herramientas de ML
- [Plotly](https://plotly.com) por las visualizaciones
- Comunidad de Python por las librerías open source

## 📊 Estadísticas del Proyecto

![GitHub repo size](https://img.shields.io/github/repo-size/tu-usuario/analizador-gastos-ia)
![GitHub last commit](https://img.shields.io/github/last-commit/tu-usuario/analizador-gastos-ia)
![GitHub issues](https://img.shields.io/github/issues/tu-usuario/analizador-gastos-ia)
![GitHub stars](https://img.shields.io/github/stars/tu-usuario/analizador-gastos-ia?style=social)

---

<div align="center">

**⭐ Si te gusta este proyecto, ¡no olvides darle una estrella! ⭐**

</div>