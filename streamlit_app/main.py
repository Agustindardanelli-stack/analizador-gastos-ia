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
    
    # TAB 1: Cargar datos
    with tab1:
        st.header("📤 Cargar tus Datos de Gastos")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Área de carga con estilo
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.markdown("### 🎯 Arrastra tu archivo aquí")
            
            uploaded_file = st.file_uploader(
                "Selecciona tu archivo de gastos",
                type=['csv', 'xlsx', 'xls', 'json'],
                help="Formatos: CSV, Excel, JSON"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Botones de acción
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🧪 Usar Datos de Ejemplo", type="secondary", use_container_width=True):
                    st.session_state['use_example'] = True
            
            with col_btn2:
                if st.button("🔄 Limpiar Datos", type="secondary", use_container_width=True):
                    st.session_state.clear()
                    st.rerun()
        
        with col2:
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
        
        # Procesar datos
        if uploaded_file is not None or st.session_state.get('use_example', False):
            
            with st.spinner("🔄 Procesando datos..."):
                
                if st.session_state.get('use_example', False):
                    # Usar datos de ejemplo
                    df_processed = create_sample_data()
                    error = None
                else:
                    # Procesar archivo subido
                    df_processed, insights, error = load_and_process_data(uploaded_file)
                
                if error:
                    st.error(f"❌ Error: {error}")
                    return
                
                if df_processed is not None and len(df_processed) > 0:
                    
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
                    
                    # Mostrar éxito
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("### ✅ ¡Datos procesados exitosamente!")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Métricas rápidas
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
                
                else:
                    st.error("❌ No se pudieron procesar los datos")
    
    # TAB 2: Dashboard
    with tab2:
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Primero carga tus datos en la pestaña 'Cargar Datos'")
            return
        
        df = st.session_state['df_data']
        
        st.header("📊 Dashboard Interactivo")
        
        # Métricas principales con estilo
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = df['monto'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Total Gastado</h3>
                <h2>${total:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            promedio = df['monto'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Gasto Promedio</h3>
                <h2>${promedio:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            max_gasto = df['monto'].max()
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Gasto Máximo</h3>
                <h2>${max_gasto:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            analyzer = ExpenseAnalyzer()
            df_anomalies = analyzer.detect_anomalies(df.copy())
            anomalies_count = df_anomalies['is_anomaly'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚠️ Gastos Anómalos</h3>
                <h2>{anomalies_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Gráficos principales
        col1, col2 = st.columns(2)
        
        with col1:
            pie_chart = create_category_pie_chart(df)
            st.plotly_chart(pie_chart, use_container_width=True)
        
        with col2:
            bar_chart = create_category_bar_chart(df)
            st.plotly_chart(bar_chart, use_container_width=True)
        
        # Timeline si hay fechas
        if 'fecha' in df.columns:
            timeline_chart = create_timeline_chart(df)
            if timeline_chart:
                st.plotly_chart(timeline_chart, use_container_width=True)
        
        # Top gastos
        st.markdown("### 🏆 Top 10 Gastos")
        top_gastos = df.nlargest(10, 'monto')[['descripcion', 'monto', 'categoria_ia']]
        st.dataframe(top_gastos, use_container_width=True)
    
    # TAB 3: Análisis detallado
    with tab3:
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Primero carga tus datos en la pestaña 'Cargar Datos'")
            return
        
        df = st.session_state['df_data']
        st.header("🔍 Análisis Detallado")
        
        # Filtros en sidebar
        with st.sidebar:
            st.markdown("### 🔧 Filtros de Análisis")
            
            # Filtro por categoría
            categorias = ['Todas'] + sorted(df['categoria_ia'].unique().tolist())
            categoria_seleccionada = st.selectbox("📂 Categoría", categorias)
            
            # Filtro por monto
            min_val, max_val = float(df['monto'].min()), float(df['monto'].max())
            rango_montos = st.slider(
                "💰 Rango de Montos",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                format="$%.0f"
            )
        
        # Aplicar filtros
        df_filtered = df.copy()
        
        if categoria_seleccionada != 'Todas':
            df_filtered = df_filtered[df_filtered['categoria_ia'] == categoria_seleccionada]
        
        df_filtered = df_filtered[
            (df_filtered['monto'] >= rango_montos[0]) & 
            (df_filtered['monto'] <= rango_montos[1])
        ]
        
        # Mostrar resultados filtrados
        st.subheader(f"📋 Resultados Filtrados ({len(df_filtered):,} registros)")
        
        if len(df_filtered) > 0:
            # Estadísticas del filtro
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💰 Total Filtrado", f"${df_filtered['monto'].sum():,.0f}")
            with col2:
                st.metric("📊 Promedio", f"${df_filtered['monto'].mean():,.0f}")
            with col3:
                percentage = (len(df_filtered) / len(df)) * 100
                st.metric("📈 % del Total", f"{percentage:.1f}%")
            
            # Tabla de datos
            st.dataframe(
                df_filtered.sort_values('monto', ascending=False),
                use_container_width=True,
                height=400
            )
            
            # Análisis de anomalías
            st.subheader("⚠️ Detección de Anomalías")
            
            analyzer = ExpenseAnalyzer()
            df_anomalies = analyzer.detect_anomalies(df_filtered.copy())
            anomalies = df_anomalies[df_anomalies['is_anomaly'] == True]
            
            if len(anomalies) > 0:
                st.warning(f"🚨 {len(anomalies)} gastos anómalos detectados")
                st.dataframe(
                    anomalies[['descripcion', 'monto', 'categoria_ia', 'z_score']].sort_values('monto', ascending=False),
                    use_container_width=True
                )
            else:
                st.success("✅ No se detectaron anomalías en los datos filtrados")
        
        else:
            st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados")
    
    # TAB 4: Reportes
    with tab4:
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Primero carga tus datos en la pestaña 'Cargar Datos'")
            return
        
        df = st.session_state['df_data']
        st.header("📋 Reportes y Exportación")
        
        # Resumen ejecutivo
        st.subheader("📊 Resumen Ejecutivo")
        
        formatter = CurrencyFormatter()
        analyzer = ExpenseAnalyzer()
        stats = analyzer.generate_summary_stats(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            ### 💰 Estadísticas Financieras
            
            - **Total gastado**: ${stats['total_amount']:,.2f}
            - **Promedio por transacción**: ${stats['average_transaction']:,.2f}
            - **Mediana**: ${stats['median_transaction']:,.2f}
            - **Gasto máximo**: ${stats['max_transaction']:,.2f}
            - **Gasto mínimo**: ${stats['min_transaction']:,.2f}
            - **Desviación estándar**: ${stats['std_transaction']:,.2f}
            """)
        
        with col2:
            # Distribución por categorías
            st.markdown("### 🏷️ Distribución por Categorías")
            
            for categoria in df['categoria_ia'].value_counts().head(5).index:
                cat_data = df[df['categoria_ia'] == categoria]
                total = cat_data['monto'].sum()
                count = len(cat_data)
                percentage = (total / df['monto'].sum()) * 100
                
                st.markdown(f"""
                <div class="category-chip">
                    {categoria}: ${total:,.0f} ({count} gastos - {percentage:.1f}%)
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Análisis detallado por categorías
        st.subheader("📈 Análisis Detallado por Categorías")
        
        category_analysis = df.groupby('categoria_ia').agg({
            'monto': ['sum', 'count', 'mean', 'std'],
            'confianza_ia': 'mean'
        }).round(2)
        
        category_analysis.columns = ['Total ($)', 'Cantidad', 'Promedio ($)', 'Desv. Est. ($)', 'Confianza IA']
        category_analysis['% del Total'] = ((category_analysis['Total ($)'] / df['monto'].sum()) * 100).round(1)
        category_analysis = category_analysis.sort_values('Total ($)', ascending=False)
        
        st.dataframe(category_analysis, use_container_width=True)
        
        # Exportación
        st.subheader("💾 Exportar Datos y Reportes")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Exportar datos procesados
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Descargar Datos (CSV)",
                data=csv_data,
                file_name=f"gastos_analizados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Exportar reporte completo
            reporte = {
                'fecha_reporte': datetime.now().isoformat(),
                'resumen': {
                    'total_transacciones': len(df),
                    'monto_total': float(df['monto'].sum()),
                    'promedio': float(df['monto'].mean()),
                    'periodo_analizado': f"{df['fecha'].min()} a {df['fecha'].max()}" if 'fecha' in df.columns else 'No especificado'
                },
                'estadisticas': {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in stats.items()},
                'categorias': category_analysis.to_dict(),
                'top_gastos': df.nlargest(10, 'monto')[['descripcion', 'monto', 'categoria_ia']].to_dict('records')
            }
            
            json_data = json.dumps(reporte, indent=2, ensure_ascii=False, default=str).encode('utf-8')
            st.download_button(
                label="📊 Descargar Reporte (JSON)",
                data=json_data,
                file_name=f"reporte_completo_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Exportar solo categorías
            category_csv = category_analysis.to_csv().encode('utf-8')
            st.download_button(
                label="🏷️ Análisis Categorías (CSV)",
                data=category_csv,
                file_name=f"analisis_categorias_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    # Inicializar session state
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    
    if 'use_example' not in st.session_state:
        st.session_state['use_example'] = False
    
    main()