"""
Analizador de Gastos IA - Aplicación Streamlit
Interfaz web interactiva para análisis de gastos personales
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import numpy as np

# Configurar el path CORRECTAMENTE
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Importar nuestros módulos (SIN src. y nombres corregidos)
try:
    from data_processor import ExpenseDataProcessor
    from categorizer import CategoryTrainer, IntelligentCategorizer
    from utils import ConfigManager, CurrencyFormatter, ExpenseAnalyzer
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    st.stop()

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Gastos IA",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .category-chip {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 2rem;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    
    .success-box {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Funciones auxiliares
@st.cache_data
def load_and_process_data(uploaded_file):
    """Cargar y procesar archivo subido"""
    try:
        # Crear directorio temporal
        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo temporalmente
        temp_path = temp_dir / uploaded_file.name
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Procesar con nuestro sistema
        processor = ExpenseDataProcessor()
        df_processed, insights = processor.process_file(str(temp_path))
        
        return df_processed, insights, None
        
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def get_trained_categorizer():
    """Obtener categorizador entrenado"""
    model_path = Path("models/categorizer_trained.pkl")
    
    if model_path.exists():
        try:
            categorizer = IntelligentCategorizer()
            categorizer.load_model(str(model_path))
            return categorizer
        except:
            pass
    
    # Entrenar nuevo modelo
    trainer = CategoryTrainer()
    trainer.train_with_sample_data()
    
    # Crear directorio models si no existe
    model_path.parent.mkdir(exist_ok=True)
    trainer.categorizer.save_model(str(model_path))
    
    return trainer.categorizer

def create_category_pie_chart(df):
    """Crear gráfico de torta por categorías"""
    category_totals = df.groupby('categoria_ia')['monto'].sum().reset_index()
    
    fig = px.pie(
        category_totals, 
        values='monto', 
        names='categoria_ia',
        title='💸 Distribución de Gastos por Categoría',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.3
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Monto: $%{value:,.0f}<br>Porcentaje: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        font=dict(size=14),
        showlegend=True,
        height=500
    )
    
    return fig

def create_timeline_chart(df):
    """Crear gráfico de línea temporal"""
    if 'fecha' not in df.columns:
        return None
    
    # Agrupar por fecha
    daily_expenses = df.groupby(df['fecha'].dt.date)['monto'].sum().reset_index()
    daily_expenses['fecha'] = pd.to_datetime(daily_expenses['fecha'])
    
    fig = px.line(
        daily_expenses,
        x='fecha',
        y='monto',
        title='📈 Evolución de Gastos en el Tiempo',
        markers=True,
        line_shape='spline'
    )
    
    fig.update_traces(
        line=dict(width=3, color='#1f77b4'),
        marker=dict(size=8)
    )
    
    fig.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Monto ($)',
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_category_bar_chart(df):
    """Crear gráfico de barras por categoría"""
    category_stats = df.groupby('categoria_ia').agg({
        'monto': ['sum', 'count', 'mean']
    }).round(2)
    
    category_stats.columns = ['Total', 'Cantidad', 'Promedio']
    category_stats = category_stats.reset_index()
    category_stats = category_stats.sort_values('Total', ascending=True)
    
    fig = px.bar(
        category_stats,
        x='Total',
        y='categoria_ia',
        orientation='h',
        title='💰 Total de Gastos por Categoría',
        color='Total',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        yaxis_title='Categoría',
        xaxis_title='Monto Total ($)'
    )
    
    return fig

def create_sample_data():
    """Crear datos de ejemplo si no hay archivo"""
    sample_data = {
        'fecha': pd.date_range('2024-01-01', periods=30, freq='D'),
        'descripcion': [
            'Supermercado Carrefour', 'Uber viaje', 'Netflix', 'Farmacia', 'Restaurant',
            'YPF Combustible', 'Gym', 'Zara ropa', 'Edenor luz', 'Cine'
        ] * 3,
        'monto': np.random.uniform(1000, 25000, 30)
    }
    return pd.DataFrame(sample_data)

def main():
    """Función principal de la aplicación"""
    
    # Header principal con estilo
    st.markdown('<h1 class="main-header">💰 Analizador de Gastos IA</h1>', unsafe_allow_html=True)
    
    # Subtítulo
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
    🤖 Análisis inteligente de gastos personales con Machine Learning
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar mejorado
    with st.sidebar:
        st.title("🎛️ Panel de Control")
        
        st.markdown("### 🚀 Funcionalidades")
        st.markdown("""
        - 📊 **Análisis automático** de gastos
        - 🤖 **Categorización IA** inteligente  
        - 📈 **Visualizaciones** interactivas
        - ⚠️ **Detección de anomalías**
        - 📋 **Reportes** exportables
        - 🎯 **Predicciones** de patrones
        """)
        
        st.markdown("### 📁 Formatos Soportados")
        st.markdown("- CSV (.csv)")
        st.markdown("- Excel (.xlsx, .xls)")
        st.markdown("- JSON (.json)")
        
        st.markdown("---")
        st.markdown("### ℹ️ Sobre el Sistema")
        st.info("Desarrollado con Streamlit + Scikit-learn + Plotly")
    
    # Tabs principales con iconos
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Cargar Datos", 
        "📊 Dashboard", 
        "🔍 Análisis Detallado", 
        "📋 Reportes"
    ])
    
    with tab1:
        
        st.header("💰 ¿Cómo quieres agregar tus gastos?")
        
        # Selector de método de entrada
        input_method = st.radio(
            "Elige tu método preferido:",
            [
                "✍️ Agregar uno por uno",
                "📝 Lista rápida (copiar/pegar)",
                "📄 Subir archivo",
                "🧪 Usar datos de ejemplo"
            ],
            horizontal=True
        )
        
        st.markdown("---")
        
        # MÉTODO 1: Entrada individual
        if input_method == "✍️ Agregar uno por uno":
            st.markdown("### ✍️ Agregar Gasto Individual")
            st.markdown("*Perfecto para ir agregando gastos conforme los haces*")
            
            # Inicializar lista de gastos en session_state
            if 'gastos_individuales' not in st.session_state:
                st.session_state.gastos_individuales = []
            
            with st.form("gasto_individual", clear_on_submit=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    descripcion = st.text_input(
                        "Descripción del gasto",
                        placeholder="Ej: Supermercado Carrefour, Uber al trabajo, Netflix...",
                        help="Describe tu gasto lo más específico posible"
                    )
                
                with col2:
                    monto = st.number_input(
                        "Monto ($)",
                        min_value=0.0,
                        step=0.01,
                        format="%.2f"
                    )
                
                col3, col4 = st.columns(2)
                
                with col3:
                    fecha = st.date_input(
                        "Fecha",
                        value=datetime.now().date(),
                        help="¿Cuándo hiciste este gasto?"
                    )
                
                with col4:
                    categoria_manual = st.selectbox(
                        "Categoría (opcional)",
                        ["🤖 Dejar que la IA decida", "🍽️ Alimentación", "🚗 Transporte", "⚡ Servicios", 
                        "🎬 Entretenimiento", "🏥 Salud", "👔 Ropa", "🏠 Hogar", "📚 Educación", "📦 Otros"],
                        help="Si sabes la categoría, selecciónala. Si no, la IA la predicirá."
                    )
                
                submitted = st.form_submit_button("🎯 Categorizar y Agregar", use_container_width=True)
                
                if submitted and descripcion and monto > 0:
                    # Predecir categoría si no se especificó
                    if categoria_manual == "🤖 Dejar que la IA decida":
                        categorizer = get_trained_categorizer()
                        prediction = categorizer.predict_with_confidence([descripcion])[0]
                        categoria_ia = prediction['categoria']
                        confianza = prediction['confidence']
                    else:
                        categoria_ia = categoria_manual.split(' ', 1)[1]  # Remover emoji
                        confianza = 1.0
                    
                    # Agregar a la lista
                    nuevo_gasto = {
                        'fecha': pd.to_datetime(fecha),
                        'descripcion': descripcion,
                        'monto': monto,
                        'categoria_ia': categoria_ia,
                        'confianza_ia': confianza
                    }
                    
                    st.session_state.gastos_individuales.append(nuevo_gasto)
                    
                    # Mostrar confirmación
                    st.success(f"✅ Agregado: {descripcion} - ${monto:,.2f} → **{categoria_ia}** ({confianza:.1%} confianza)")
            
            # Mostrar gastos de la sesión
            if st.session_state.gastos_individuales:
                st.markdown("### 📋 Gastos Agregados en Esta Sesión")
                
                df_session = pd.DataFrame(st.session_state.gastos_individuales)
                
                # Métricas rápidas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total gastos", len(df_session))
                with col2:
                    st.metric("💰 Total monto", f"${df_session['monto'].sum():,.0f}")
                with col3:
                    st.metric("📈 Promedio", f"${df_session['monto'].mean():,.0f}")
                
                # Tabla de gastos
                display_df = df_session.copy()
                display_df['fecha'] = display_df['fecha'].dt.strftime('%Y-%m-%d')
                display_df['monto'] = display_df['monto'].apply(lambda x: f"${x:,.2f}")
                display_df['confianza_ia'] = display_df['confianza_ia'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    display_df[['fecha', 'descripcion', 'monto', 'categoria_ia', 'confianza_ia']],
                    use_container_width=True,
                    height=300
                )
                
                # Botones de acción
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Analizar Gastos", type="primary", use_container_width=True):
                        st.session_state['df_data'] = df_session
                        st.session_state['data_loaded'] = True
                        st.success("✅ ¡Datos listos para analizar! Ve a la pestaña Dashboard")
                
                with col2:
                    if st.button("🗑️ Limpiar Lista", type="secondary", use_container_width=True):
                        st.session_state.gastos_individuales = []
                        st.rerun()
                
                with col3:
                    # Exportar gastos de sesión
                    csv_session = df_session.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Exportar CSV",
                        data=csv_session,
                        file_name=f"gastos_sesion_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        # MÉTODO 2: Lista rápida
        elif input_method == "📝 Lista rápida (copiar/pegar)":
            st.markdown("### 📝 Lista Rápida de Gastos")
            st.markdown("*Copia y pega desde WhatsApp, notas, o escribe varios gastos de una vez*")
            
            # Mostrar ejemplos de formato
            with st.expander("📋 Ver ejemplos de formato"):
                st.markdown("""
                **Formato simple** (descripción, monto):
                ```
                Supermercado Carrefour, 18500
                Uber viaje trabajo, 2800
                Netflix, 2490
                ```
                
                **Con fechas** (fecha, descripción, monto):
                ```
                2024-01-15, Supermercado Carrefour, 18500
                2024-01-16, Uber viaje trabajo, 2800
                2024-01-17, Netflix, 2490
                ```
                
                **Formato libre** (la IA entiende):
                ```
                - Supermercado: $18.500
                - Uber: $2.800 (ayer)
                - Netflix: $2.490
                Cena restaurant: 15000 pesos
                ```
                """)
            
            texto_gastos = st.text_area(
                "Pega o escribe tus gastos aquí:",
                placeholder="Supermercado Carrefour, 18500\nUber viaje trabajo, 2800\nNetflix, 2490",
                height=200,
                help="Un gasto por línea. La IA es inteligente y entiende varios formatos."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                fecha_base = st.date_input(
                    "Fecha base (si no especificas fechas)",
                    value=datetime.now().date()
                )
            
            with col2:
                auto_categorizar = st.checkbox(
                    "Categorizar automáticamente",
                    value=True,
                    help="La IA categorizará todos los gastos automáticamente"
                )
            
            if st.button("🚀 Procesar Lista", type="primary", use_container_width=True):
                if texto_gastos.strip():
                    with st.spinner("🤖 Procesando lista de gastos..."):
                        # Procesar texto de gastos
                        gastos_procesados = []
                        lineas = [linea.strip() for linea in texto_gastos.split('\n') if linea.strip()]
                        
                        for linea in lineas:
                            # Limpiar línea
                            linea = linea.replace('$', '').replace('.', '').replace('pesos', '').strip()
                            
                            # Intentar parsear diferentes formatos
                            gasto_parseado = None
                            
                            # Formato: fecha, descripcion, monto
                            if linea.count(',') >= 2:
                                partes = [p.strip() for p in linea.split(',')]
                                try:
                                    fecha_gasto = pd.to_datetime(partes[0])
                                    descripcion = partes[1]
                                    monto = float(partes[2])
                                    gasto_parseado = (fecha_gasto, descripcion, monto)
                                except:
                                    pass
                            
                            # Formato: descripcion, monto
                            if not gasto_parseado and linea.count(',') == 1:
                                partes = [p.strip() for p in linea.split(',')]
                                try:
                                    descripcion = partes[0]
                                    monto = float(partes[1])
                                    gasto_parseado = (pd.to_datetime(fecha_base), descripcion, monto)
                                except:
                                    pass
                            
                            # Formato libre con ':'
                            if not gasto_parseado and ':' in linea:
                                partes = linea.split(':')
                                if len(partes) == 2:
                                    try:
                                        descripcion = partes[0].replace('-', '').strip()
                                        monto_str = partes[1].strip()
                                        # Extraer números
                                        import re
                                        numeros = re.findall(r'\d+', monto_str)
                                        if numeros:
                                            monto = float(''.join(numeros))
                                            gasto_parseado = (pd.to_datetime(fecha_base), descripcion, monto)
                                    except:
                                        pass
                            
                            if gasto_parseado:
                                gastos_procesados.append({
                                    'fecha': gasto_parseado[0],
                                    'descripcion': gasto_parseado[1],
                                    'monto': gasto_parseado[2]
                                })
                        
                        if gastos_procesados:
                            df_lista = pd.DataFrame(gastos_procesados)
                            
                            # Categorizar con IA si está habilitado
                            if auto_categorizar:
                                categorizer = get_trained_categorizer()
                                predictions = categorizer.predict_with_confidence(
                                    df_lista['descripcion'].tolist()
                                )
                                
                                df_lista['categoria_ia'] = [p['categoria'] for p in predictions]
                                df_lista['confianza_ia'] = [p['confidence'] for p in predictions]
                            else:
                                df_lista['categoria_ia'] = 'Sin categorizar'
                                df_lista['confianza_ia'] = 0.0
                            
                            # Mostrar resultados
                            st.success(f"✅ {len(df_lista)} gastos procesados exitosamente!")
                            
                            # Vista previa
                            st.markdown("### 👀 Vista Previa")
                            display_df = df_lista.copy()
                            display_df['fecha'] = display_df['fecha'].dt.strftime('%Y-%m-%d')
                            display_df['monto'] = display_df['monto'].apply(lambda x: f"${x:,.0f}")
                            if auto_categorizar:
                                display_df['confianza_ia'] = display_df['confianza_ia'].apply(lambda x: f"{x:.1%}")
                            
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Botones de acción
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if st.button("📊 Analizar Estos Gastos", type="primary", use_container_width=True):
                                    st.session_state['df_data'] = df_lista
                                    st.session_state['data_loaded'] = True
                                    st.success("✅ ¡Datos listos! Ve a la pestaña Dashboard")
                            
                            with col2:
                                csv_data = df_lista.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="💾 Descargar como CSV",
                                    data=csv_data,
                                    file_name=f"gastos_lista_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                        else:
                            st.error("❌ No se pudieron procesar los gastos. Verifica el formato.")
                else:
                    st.warning("⚠️ Por favor ingresa algunos gastos para procesar.")
        
        # MÉTODO 3: Archivo (código existente mejorado)
        elif input_method == "📄 Subir archivo":
            st.markdown("### 📄 Subir Archivo de Gastos")
            st.markdown("*Para cuando tienes muchos gastos en Excel, CSV o exportación del banco*")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Área de carga con estilo (código existente)
                st.markdown('<div class="upload-section">', unsafe_allow_html=True)
                st.markdown("### 🎯 Arrastra tu archivo aquí")
                
                uploaded_file = st.file_uploader(
                    "Selecciona tu archivo de gastos",
                    type=['csv', 'xlsx', 'xls', 'json'],
                    help="Formatos: CSV, Excel, JSON"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Procesar archivo si se subió
                if uploaded_file is not None:
                    with st.spinner("🔄 Procesando archivo..."):
                        df_processed, insights, error = load_and_process_data(uploaded_file)
                        
                        if error:
                            st.error(f"❌ Error: {error}")
                        elif df_processed is not None and len(df_processed) > 0:
                            # Categorización IA
                            with st.spinner("🤖 Categorizando con IA..."):
                                categorizer = get_trained_categorizer()
                                predictions = categorizer.predict_with_confidence(
                                    df_processed['descripcion'].tolist()
                                )
                                
                                df_processed['categoria_ia'] = [p['categoria'] for p in predictions]
                                df_processed['confianza_ia'] = [p['confidence'] for p in predictions]
                            
                            # Guardar en session state
                            st.session_state['df_data'] = df_processed
                            st.session_state['data_loaded'] = True
                            
                            # Mostrar éxito (usar código existente)
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown("### ✅ ¡Archivo procesado exitosamente!")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Métricas rápidas (usar código existente)
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("📊 Transacciones", f"{len(df_processed):,}")
                            
                            with col2:
                                total = df_processed['monto'].sum()
                                st.metric("💰 Total", f"${total:,.0f}")
                            
                            with col3:
                                promedio = df_processed['monto'].mean()
                                st.metric("📈 Promedio", f"${promedio:,.0f}")
                            
                            with col4:
                                categorias = df_processed['categoria_ia'].nunique()
                                st.metric("🏷️ Categorías", categorias)
                            
                            # Vista previa
                            st.markdown("### 👀 Vista Previa")
                            preview_cols = ['fecha', 'descripcion', 'monto', 'categoria_ia', 'confianza_ia']
                            available_cols = [col for col in preview_cols if col in df_processed.columns]
                            st.dataframe(
                                df_processed[available_cols].head(10),
                                use_container_width=True
                            )
            
            with col2:
                # Información de formato (código existente)
                st.markdown("### 📋 Formato Requerido")
                st.markdown("""
                **Columnas necesarias:**
                - **Descripción**: detalle del gasto
                - **Monto**: valor del gasto  
                - **Fecha**: fecha del gasto (opcional)
                """)
                
                # Ejemplo visual
                st.markdown("### 🎯 Ejemplo")
                example_df = pd.DataFrame({
                    'fecha': ['2024-01-15', '2024-01-16'],
                    'descripcion': ['Supermercado Carrefour', 'Uber centro'],
                    'monto': [15500.50, 2800.00]
                })
                st.dataframe(example_df, use_container_width=True)
                
                # Plantillas descargables
                st.markdown("### 📥 Plantillas")
                
                template_data = pd.DataFrame({
                    'fecha': ['2024-01-01', '2024-01-02', '2024-01-03'],
                    'descripcion': ['Supermercado ejemplo', 'Transporte ejemplo', 'Servicio ejemplo'],
                    'monto': [10000.00, 2500.00, 5000.00]
                })
                
                csv_template = template_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Plantilla CSV",
                    data=csv_template,
                    file_name="plantilla_gastos.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # MÉTODO 4: Datos de ejemplo
        elif input_method == "🧪 Usar datos de ejemplo":
            st.markdown("### 🧪 Datos de Ejemplo")
            st.markdown("*Perfecto para probar el sistema sin tener que subir tus propios datos*")
            
            st.info("Los datos de ejemplo incluyen gastos típicos argentinos con diferentes categorías para que puedas probar todas las funcionalidades del sistema.")
            
            # Vista previa de datos de ejemplo
            sample_preview = pd.DataFrame({
                'fecha': ['2024-01-15', '2024-01-16', '2024-01-17'],
                'descripcion': ['Supermercado Carrefour Villa Crespo', 'Uber viaje centro', 'Netflix Suscripción'],
                'monto': [18500.50, 2800.00, 2490.00],
                'categoria': ['Alimentación', 'Transporte', 'Entretenimiento']
            })
            
            st.markdown("### 👀 Vista Previa de Datos de Ejemplo")
            st.dataframe(sample_preview, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                cantidad_gastos = st.slider(
                    "🎚️ Cantidad de gastos de ejemplo",
                    min_value=10,
                    max_value=100,
                    value=35,
                    help="Más gastos = análisis más completo"
                )
            
            with col2:
                periodo_dias = st.slider(
                    "📅 Período (días)",
                    min_value=7,
                    max_value=90,
                    value=30,
                    help="Gastos distribuidos en este período"
                )
            
            if st.button("🎯 Generar y Usar Datos de Ejemplo", type="primary", use_container_width=True):
                with st.spinner("🤖 Generando datos de ejemplo..."):
                    # Generar datos de ejemplo más realistas
                    import random
                    
                    gastos_ejemplo = [
                        ('Supermercado Carrefour Villa Crespo', 'Alimentación', (8000, 25000)),
                        ('Panaderia La Esquina', 'Alimentación', (1500, 5000)),
                        ('McDonalds Palermo', 'Alimentación', (3000, 8000)),
                        ('Restaurant Don Julio', 'Alimentación', (15000, 35000)),
                        ('Verduleria Central', 'Alimentación', (2000, 8000)),
                        ('Uber viaje centro', 'Transporte', (1500, 5000)),
                        ('YPF Combustible', 'Transporte', (8000, 20000)),
                        ('SUBE recarga', 'Transporte', (1000, 3000)),
                        ('Taxi Aeropuerto', 'Transporte', (5000, 12000)),
                        ('Peaje Autopista', 'Transporte', (300, 1000)),
                        ('Edenor Factura Electricidad', 'Servicios', (8000, 18000)),
                        ('Movistar Plan Celular', 'Servicios', (6000, 12000)),
                        ('Fibertel Internet', 'Servicios', (4000, 8000)),
                        ('Metrogas Factura', 'Servicios', (10000, 20000)),
                        ('Expensas Edificio', 'Servicios', (25000, 60000)),
                        ('Netflix Suscripcion', 'Entretenimiento', (2000, 3000)),
                        ('Cine Hoyts Palermo', 'Entretenimiento', (2500, 5000)),
                        ('Spotify Premium', 'Entretenimiento', (1000, 2000)),
                        ('Bar Antares Cerveza', 'Entretenimiento', (5000, 15000)),
                        ('Steam Videojuego', 'Entretenimiento', (8000, 25000)),
                        ('Farmacity Medicamentos', 'Salud', (3000, 8000)),
                        ('Dr. Martinez Consulta', 'Salud', (10000, 25000)),
                        ('Dentista Limpieza', 'Salud', (15000, 30000)),
                        ('Laboratorio Analisis', 'Salud', (5000, 15000)),
                        ('Zara Camisa Trabajo', 'Ropa', (8000, 20000)),
                        ('Nike Zapatillas Running', 'Ropa', (25000, 50000)),
                        ('H&M Pantalon Jean', 'Ropa', (6000, 15000)),
                        ('Easy Herramientas', 'Hogar', (3000, 10000)),
                        ('Sodimac Pintura', 'Hogar', (5000, 15000)),
                        ('IKEA Escritorio', 'Hogar', (30000, 80000)),
                        ('Universidad UTN Cuota', 'Educación', (15000, 35000)),
                        ('Udemy Curso Python', 'Educación', (8000, 20000)),
                        ('Libros Amazon', 'Educación', (5000, 15000))
                    ]
                    
                    # Generar gastos aleatorios
                    datos_ejemplo = []
                    fecha_inicio = datetime.now() - timedelta(days=periodo_dias)
                    
                    for i in range(cantidad_gastos):
                        gasto_info = random.choice(gastos_ejemplo)
                        descripcion = gasto_info[0]
                        categoria = gasto_info[1]
                        rango_monto = gasto_info[2]
                        
                        # Fecha aleatoria en el período
                        dias_random = random.randint(0, periodo_dias)
                        fecha = fecha_inicio + timedelta(days=dias_random)
                        
                        # Monto aleatorio en el rango
                        monto = random.uniform(rango_monto[0], rango_monto[1])
                        
                        datos_ejemplo.append({
                            'fecha': fecha,
                            'descripcion': descripcion,
                            'monto': round(monto, 2),
                            'categoria_ia': categoria,
                            'confianza_ia': random.uniform(0.85, 0.98)  # Alta confianza para ejemplos
                        })
                    
                    df_ejemplo = pd.DataFrame(datos_ejemplo)
                    df_ejemplo = df_ejemplo.sort_values('fecha')
                    
                    # Guardar en session state
                    st.session_state['df_data'] = df_ejemplo
                    st.session_state['data_loaded'] = True
                    
                    st.success(f"✅ {len(df_ejemplo)} gastos de ejemplo generados y listos para analizar!")
                    
                    # Métricas rápidas
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📊 Transacciones", f"{len(df_ejemplo):,}")
                    
                    with col2:
                        total = df_ejemplo['monto'].sum()
                        st.metric("💰 Total", f"${total:,.0f}")
                    
                    with col3:
                        promedio = df_ejemplo['monto'].mean()
                        st.metric("📈 Promedio", f"${promedio:,.0f}")
                    
                    with col4:
                        categorias = df_ejemplo['categoria_ia'].nunique()
                        st.metric("🏷️ Categorías", categorias)
                    
                    st.info("🎉 ¡Perfecto! Ahora ve a la pestaña **Dashboard** para ver el análisis completo.")

def show_quick_actions():
    """Mostrar acciones rápidas cuando hay datos cargados"""
    if st.session_state.get('data_loaded', False):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚡ Acciones Rápidas")
        
        df = st.session_state['df_data']
        
        # Métricas en sidebar
        st.sidebar.metric("📊 Total gastos", len(df))
        st.sidebar.metric("💰 Total", f"${df['monto'].sum():,.0f}")
        
        # Gasto único rápido
        with st.sidebar.expander("➕ Agregar Gasto Rápido"):
            with st.form("gasto_rapido_sidebar"):
                desc_rapido = st.text_input("Descripción", placeholder="Ej: Café Starbucks")
                monto_rapido = st.number_input("Monto", min_value=0.0, step=1.0)
                
                if st.form_submit_button("Agregar"):
                    if desc_rapido and monto_rapido > 0:
                        categorizer = get_trained_categorizer()
                        prediction = categorizer.predict_with_confidence([desc_rapido])[0]
                        
                        nuevo_gasto = pd.DataFrame([{
                            'fecha': pd.to_datetime(datetime.now().date()),
                            'descripcion': desc_rapido,
                            'monto': monto_rapido,
                            'categoria_ia': prediction['categoria'],
                            'confianza_ia': prediction['confidence']
                        }])
                        
                        st.session_state['df_data'] = pd.concat([df, nuevo_gasto], ignore_index=True)
                        st.success(f"✅ Agregado: {prediction['categoria']}")
                        st.rerun()    
if __name__ == "__main__":
    # Inicializar session state
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    
    if 'use_example' not in st.session_state:
        st.session_state['use_example'] = False
    
    main()