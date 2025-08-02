"""
Utilidades generales para el Analizador de Gastos IA
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from typing import Dict, List, Any, Union, Optional, Callable  # ← AGREGAR ESTA LÍNEA
import locale
import psutil
import time
from sklearn.ensemble import IsolationForest
from concurrent.futures import ThreadPoolExecutor
import threading
import gc
# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Maneja la configuración del sistema"""
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Cargar configuración desde archivo"""
        default_config = {
            "categories": {
                "Alimentación": {"color": "#FF6B6B", "icon": "🍽️"},
                "Transporte": {"color": "#4ECDC4", "icon": "🚗"},
                "Servicios": {"color": "#45B7D1", "icon": "⚡"},
                "Entretenimiento": {"color": "#96CEB4", "icon": "🎬"},
                "Salud": {"color": "#FFEAA7", "icon": "🏥"},
                "Ropa": {"color": "#DDA0DD", "icon": "👔"},
                "Hogar": {"color": "#98D8C8", "icon": "🏠"},
                "Educación": {"color": "#F7DC6F", "icon": "📚"},
                "Otros": {"color": "#BDC3C7", "icon": "📦"}
            },
            "currency": "ARS",
            "currency_symbol": "$",
            "date_format": "%Y-%m-%d",
            "decimal_places": 2,
            "insights": {
                "anomaly_threshold": 2.0,
                "trend_periods": 30,
                "top_expenses_count": 10
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge con configuración por defecto
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Error cargando configuración: {e}")
        
        return default_config
    
    def save_config(self):
        """Guardar configuración actual"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def get(self, key: str, default=None):
        """Obtener valor de configuración"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Establecer valor de configuración"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

class CurrencyFormatter:
    """Formatear valores monetarios"""
    
    def __init__(self, currency: str = "ARS", symbol: str = "$"):
        self.currency = currency
        self.symbol = symbol
        
        # Configurar locale para Argentina
        try:
            if currency == "ARS":
                locale.setlocale(locale.LC_ALL, 'es_AR.UTF-8')
        except:
            pass
    
    def format_amount(self, amount: float, decimals: int = 2) -> str:
        """Formatear monto con moneda"""
        if pd.isna(amount):
            return f"{self.symbol}0.00"
        
        # Formatear número con separadores de miles
        formatted = f"{amount:,.{decimals}f}"
        return f"{self.symbol}{formatted}"
    
    def format_amount_short(self, amount: float) -> str:
        """Formatear monto en forma abreviada (K, M)"""
        if pd.isna(amount):
            return f"{self.symbol}0"
        
        if amount >= 1_000_000:
            return f"{self.symbol}{amount/1_000_000:.1f}M"
        elif amount >= 1_000:
            return f"{self.symbol}{amount/1_000:.1f}K"
        else:
            return f"{self.symbol}{amount:.0f}"
    
    def parse_amount(self, amount_str: str) -> float:
        """Parsear string de monto a float"""
        if pd.isna(amount_str):
            return 0.0
        
        # Remover símbolo de moneda y espacios
        cleaned = str(amount_str).replace(self.symbol, '').strip()
        cleaned = cleaned.replace(',', '').replace(' ', '')
        
        try:
            return float(cleaned)
        except:
            return 0.0

class DateHelper:
    """Utilidades para manejo de fechas"""
    
    @staticmethod
    def get_date_range_options() -> Dict[str, Dict]:
        """Obtener opciones predefinidas de rangos de fechas"""
        today = datetime.now()
        
        return {
            "Última semana": {
                "start": today - timedelta(days=7),
                "end": today
            },
            "Último mes": {
                "start": today - timedelta(days=30),
                "end": today
            },
            "Últimos 3 meses": {
                "start": today - timedelta(days=90),
                "end": today
            },
            "Último año": {
                "start": today - timedelta(days=365),
                "end": today
            },
            "Este año": {
                "start": datetime(today.year, 1, 1),
                "end": today
            },
            "Año pasado": {
                "start": datetime(today.year - 1, 1, 1),
                "end": datetime(today.year - 1, 12, 31)
            }
        }
    
    @staticmethod
    def format_date_range(start_date: datetime, end_date: datetime) -> str:
        """Formatear rango de fechas para mostrar"""
        if start_date.year == end_date.year:
            if start_date.month == end_date.month:
                return f"{start_date.day}-{end_date.day} {start_date.strftime('%B %Y')}"
            else:
                return f"{start_date.strftime('%B')} - {end_date.strftime('%B %Y')}"
        else:
            return f"{start_date.strftime('%B %Y')} - {end_date.strftime('%B %Y')}"
    
    @staticmethod
    def get_period_name(date: datetime) -> str:
        """Obtener nombre del período (Ene 2024, etc.)"""
        return date.strftime('%b %Y')

class DataValidator:
    """Validar calidad de los datos"""
    
    @staticmethod
    def validate_expense_data(df: pd.DataFrame) -> Dict:
        """Validar datos de gastos"""
        issues = []
        warnings = []
        
        # Verificar columnas requeridas
        required_columns = ['descripcion', 'monto']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issues.append(f"Columnas faltantes: {missing_columns}")
        
        # Verificar datos vacíos
        if 'descripcion' in df.columns:
            empty_descriptions = df['descripcion'].isna().sum()
            if empty_descriptions > 0:
                warnings.append(f"Descripciones vacías: {empty_descriptions}")
        
        if 'monto' in df.columns:
            invalid_amounts = df['monto'].isna().sum()
            negative_amounts = (df['monto'] < 0).sum()
            zero_amounts = (df['monto'] == 0).sum()
            
            if invalid_amounts > 0:
                warnings.append(f"Montos inválidos: {invalid_amounts}")
            if negative_amounts > 0:
                warnings.append(f"Montos negativos: {negative_amounts}")
            if zero_amounts > 0:
                warnings.append(f"Montos en cero: {zero_amounts}")
        
        # Verificar fechas
        if 'fecha' in df.columns:
            invalid_dates = df['fecha'].isna().sum()
            if invalid_dates > 0:
                warnings.append(f"Fechas inválidas: {invalid_dates}")
        
        # Verificar duplicados
        if len(df) > 0:
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                warnings.append(f"Registros duplicados: {duplicates}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'row_count': len(df),
            'data_quality_score': max(0, 100 - len(issues) * 20 - len(warnings) * 5)
        }

class ExpenseAnalyzer:
    """Analizador de patrones de gastos"""
    
    def __init__(self, config: ConfigManager = None):
        self.config = config or ConfigManager()
        self.currency_formatter = CurrencyFormatter(
            self.config.get('currency', 'ARS'),
            self.config.get('currency_symbol', '$')
        )
    
    def detect_anomalies(self, df: pd.DataFrame, column: str = 'monto') -> pd.DataFrame:
        """Detectar gastos anómalos"""
        if column not in df.columns:
            return df
        
        threshold = self.config.get('insights.anomaly_threshold', 2.0)
        
        # Calcular Z-score
        mean_val = df[column].mean()
        std_val = df[column].std()
        
        if std_val == 0:
            df['is_anomaly'] = False
            return df
        
        df['z_score'] = np.abs((df[column] - mean_val) / std_val)
        df['is_anomaly'] = df['z_score'] > threshold
        
        return df
    
    def calculate_trends(self, df: pd.DataFrame) -> Dict:
        """Calcular tendencias de gastos"""
        if 'fecha' not in df.columns:
            return {}
        
        # Gastos por día
        daily_expenses = df.groupby(df['fecha'].dt.date)['monto'].sum().reset_index()
        daily_expenses['fecha'] = pd.to_datetime(daily_expenses['fecha'])
        
        # Gastos por mes
        monthly_expenses = df.groupby([df['fecha'].dt.year, df['fecha'].dt.month])['monto'].sum()
        
        # Calcular tendencia (últimos vs anteriores)
        periods = self.config.get('insights.trend_periods', 30)
        cutoff_date = df['fecha'].max() - timedelta(days=periods)
        
        recent_expenses = df[df['fecha'] > cutoff_date]['monto'].sum()
        previous_expenses = df[df['fecha'] <= cutoff_date]['monto'].sum()
        
        trend_percentage = 0
        if previous_expenses > 0:
            trend_percentage = ((recent_expenses - previous_expenses) / previous_expenses) * 100
        
        return {
            'daily_expenses': daily_expenses.to_dict('records'),
            'monthly_expenses': monthly_expenses.to_dict(),
            'trend_percentage': trend_percentage,
            'recent_total': recent_expenses,
            'previous_total': previous_expenses
        }
    
    def get_category_insights(self, df: pd.DataFrame) -> Dict:
        """Obtener insights por categoría"""
        if 'categoria' not in df.columns:
            return {}
        
        insights = {}
        
        for category in df['categoria'].unique():
            cat_data = df[df['categoria'] == category]
            
            insights[category] = {
                'total_amount': cat_data['monto'].sum(),
                'transaction_count': len(cat_data),
                'average_amount': cat_data['monto'].mean(),
                'percentage_of_total': (cat_data['monto'].sum() / df['monto'].sum()) * 100,
                'top_expenses': cat_data.nlargest(3, 'monto')[['descripcion', 'monto']].to_dict('records')
            }
        
        return insights
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generar estadísticas resumidas"""
        stats = {
            'total_transactions': len(df),
            'total_amount': df['monto'].sum(),
            'average_transaction': df['monto'].mean(),
            'median_transaction': df['monto'].median(),
            'max_transaction': df['monto'].max(),
            'min_transaction': df['monto'].min(),
            'std_transaction': df['monto'].std()
        }
        
        # Formatear montos
        for key in ['total_amount', 'average_transaction', 'median_transaction', 
                   'max_transaction', 'min_transaction']:
            if key in stats:
                stats[f"{key}_formatted"] = self.currency_formatter.format_amount(stats[key])
        
        return stats

class FileHelper:
    """Utilidades para manejo de archivos"""
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """Obtener formatos de archivo soportados"""
        return ['.csv', '.xlsx', '.xls', '.json', '.txt']
    
    @staticmethod
    def validate_file_format(file_path: str) -> bool:
        """Validar que el formato de archivo es soportado"""
        file_path = Path(file_path)
        return file_path.suffix.lower() in FileHelper.get_supported_formats()
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict:
        """Obtener información del archivo"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'exists': False}
        
        stat = file_path.stat()
        
        return {
            'exists': True,
            'name': file_path.name,
            'size': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'extension': file_path.suffix.lower(),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'is_supported': FileHelper.validate_file_format(str(file_path))
        }

# Funciones de utilidad globales
def setup_directories():
    """Crear directorios necesarios para el proyecto"""
    directories = [
        'data/raw',
        'data/processed',
        'data/examples',
        'models',
        'config',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio creado/verificado: {directory}")

def create_sample_config():
    """Crear archivo de configuración de ejemplo"""
    config_manager = ConfigManager()
    config_manager.save_config()
    logger.info("Archivo de configuración creado")

if __name__ == "__main__":
    print("🛠️ Utilidades - Analizador de Gastos IA")
    
    # Configurar directorios
    setup_directories()
    
    # Crear configuración
    create_sample_config()
    
    # Ejemplo de uso
    config = ConfigManager()
    currency_formatter = CurrencyFormatter()
    
    print(f"\n💰 Ejemplo de formateo de moneda:")
    print(f"   {currency_formatter.format_amount(1234.56)}")
    print(f"   {currency_formatter.format_amount_short(1500000)}")
    
    print(f"\n📅 Rangos de fecha disponibles:")
    for name, range_info in DateHelper.get_date_range_options().items():
        print(f"   {name}: {DateHelper.format_date_range(range_info['start'], range_info['end'])}")
    
    print(f"\n✅ Configuración lista para usar")

class MemoryMonitor:
    """Monitor de uso de memoria del sistema"""
    
    def __init__(self):
        self.baseline_memory = self.get_current_memory()
        self.peak_memory = self.baseline_memory
        
    def get_current_memory(self) -> Dict:
        """Obtener uso actual de memoria"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                'process_mb': memory_info.rss / (1024 * 1024),
                'process_percent': process.memory_percent(),
                'system_total_mb': system_memory.total / (1024 * 1024),
                'system_available_mb': system_memory.available / (1024 * 1024),
                'system_percent': system_memory.percent
            }
        except:
            return {'error': 'No se pudo obtener información de memoria'}
    
    def check_memory_usage(self) -> Dict:
        """Verificar uso de memoria y detectar problemas"""
        current = self.get_current_memory()
        
        if 'error' not in current:
            # Actualizar pico
            if current['process_mb'] > self.peak_memory['process_mb']:
                self.peak_memory = current.copy()
            
            # Calcular incremento desde baseline
            memory_increase = current['process_mb'] - self.baseline_memory['process_mb']
            
            # Generar alertas
            alerts = []
            if current['process_percent'] > 50:
                alerts.append(f"⚠️ Alto uso de memoria del proceso: {current['process_percent']:.1f}%")
            if current['system_percent'] > 85:
                alerts.append(f"⚠️ Memoria del sistema baja: {current['system_percent']:.1f}%")
            if memory_increase > 500:  # 500MB
                alerts.append(f"⚠️ Incremento significativo de memoria: +{memory_increase:.1f}MB")
            
            return {
                'current': current,
                'baseline': self.baseline_memory,
                'peak': self.peak_memory,
                'memory_increase_mb': memory_increase,
                'alerts': alerts,
                'status': 'warning' if alerts else 'ok'
            }
        
        return current

# 3. MEJORAR ExpenseAnalyzer CON DETECCIÓN DE ANOMALÍAS:

class EnhancedExpenseAnalyzer(ExpenseAnalyzer):
    """Analizador mejorado con detección de anomalías avanzada"""
    
    def __init__(self, config: ConfigManager = None):
        super().__init__(config)
        self.anomaly_detector = None
        self.memory_monitor = MemoryMonitor()
    
    def detect_advanced_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detectar anomalías usando múltiples métodos"""
        df_anomalies = df.copy()
        
        if len(df) < 10:
            df_anomalies['is_anomaly'] = False
            return df_anomalies
        
        # Método 1: Isolation Forest
        try:
            features = []
            if 'monto' in df.columns:
                features.append(df['monto'].values.reshape(-1, 1))
            
            if 'fecha' in df.columns:
                df_temp = df.copy()
                df_temp['fecha'] = pd.to_datetime(df_temp['fecha'])
                temporal_features = np.column_stack([
                    df_temp['fecha'].dt.hour.values,
                    df_temp['fecha'].dt.dayofweek.values
                ])
                features.append(temporal_features)
            
            if features:
                combined_features = np.hstack(features)
                
                # Entrenar detector si no existe
                if self.anomaly_detector is None:
                    self.anomaly_detector = IsolationForest(
                        contamination=0.1, 
                        random_state=42,
                        n_estimators=50  # Reducido para mejor performance
                    )
                    self.anomaly_detector.fit(combined_features)
                
                # Detectar anomalías
                anomaly_labels = self.anomaly_detector.predict(combined_features)
                df_anomalies['is_ml_anomaly'] = anomaly_labels == -1
            else:
                df_anomalies['is_ml_anomaly'] = False
        
        except Exception as e:
            logger.warning(f"Error en detección ML de anomalías: {e}")
            df_anomalies['is_ml_anomaly'] = False
        
        # Método 2: Reglas de negocio (del método original)
        df_anomalies = super().detect_anomalies(df_anomalies)
        
        # Método 3: Detección por categoría
        if 'categoria' in df.columns and 'monto' in df.columns:
            category_anomalies = []
            for _, row in df_anomalies.iterrows():
                cat_data = df_anomalies[df_anomalies['categoria'] == row['categoria']]
                if len(cat_data) > 3:
                    cat_mean = cat_data['monto'].mean()
                    cat_std = cat_data['monto'].std()
                    z_score = abs((row['monto'] - cat_mean) / cat_std) if cat_std > 0 else 0
                    category_anomalies.append(z_score > 2.5)
                else:
                    category_anomalies.append(False)
            
            df_anomalies['is_category_anomaly'] = category_anomalies
        
        # Combinar todas las detecciones
        df_anomalies['is_any_anomaly'] = (
            df_anomalies.get('is_anomaly', False) |
            df_anomalies.get('is_ml_anomaly', False) |
            df_anomalies.get('is_category_anomaly', False)
        )
        
        return df_anomalies
    
    def get_performance_insights(self, df: pd.DataFrame) -> Dict:
        """Obtener insights de performance del análisis"""
        
        insights = super().get_category_insights(df)
        
        # Agregar métricas de memoria
        memory_status = self.memory_monitor.check_memory_usage()
        insights['system_performance'] = {
            'memory_status': memory_status,
            'data_size_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'processing_recommendations': self._get_processing_recommendations(df, memory_status)
        }
        
        return insights
    
    def _get_processing_recommendations(self, df: pd.DataFrame, memory_status: Dict) -> List[str]:
        """Generar recomendaciones de procesamiento"""
        recommendations = []
        
        # Recomendaciones por tamaño de datos
        if len(df) > 10000:
            recommendations.append("💡 Dataset grande - considerar procesamiento por lotes")
        
        # Recomendaciones por memoria
        if memory_status['status'] == 'warning':
            recommendations.append("⚠️ Alto uso de memoria - optimizar procesamiento")
            
        if memory_status['current'].get('process_mb', 0) > 1000:
            recommendations.append("🔧 Usar técnicas de reducción de memoria")
        
        # Recomendaciones por categorías
        if 'categoria' in df.columns:
            unique_categories = df['categoria'].nunique()
            if unique_categories > 20:
                recommendations.append("📊 Muchas categorías - revisar clasificación")
        
        return recommendations

# 4. AGREGAR CLASE DE PROCESAMIENTO ASÍNCRONO:

class AsyncProcessor:
    """Procesador asíncrono para tareas pesadas"""
    
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = {}
        
    def process_async(self, func, *args, **kwargs) -> str:
        """Ejecutar función de forma asíncrona"""
        task_id = f"task_{int(time.time() * 1000)}"
        
        future = self.executor.submit(func, *args, **kwargs)
        self.active_tasks[task_id] = {
            'future': future,
            'start_time': time.time(),
            'function': func.__name__
        }
        
        return task_id
    
    def get_result(self, task_id: str, timeout: float = None) -> Optional[any]:
        """Obtener resultado de tarea asíncrona"""
        if task_id not in self.active_tasks:
            return None
        
        task = self.active_tasks[task_id]
        try:
            result = task['future'].result(timeout=timeout)
            # Limpiar tarea completada
            del self.active_tasks[task_id]
            return result
        except Exception as e:
            logger.error(f"Error en tarea {task_id}: {e}")
            return None
    
    def get_task_status(self, task_id: str) -> Dict:
        """Obtener estado de tarea"""
        if task_id not in self.active_tasks:
            return {'status': 'not_found'}
        
        task = self.active_tasks[task_id]
        elapsed_time = time.time() - task['start_time']
        
        return {
            'status': 'completed' if task['future'].done() else 'running',
            'function': task['function'],
            'elapsed_time': elapsed_time,
            'is_done': task['future'].done()
        }
    
    def shutdown(self):
        """Cerrar executor"""
        self.executor.shutdown(wait=True)

# 5. FUNCIÓN DE UTILIDAD PARA OPTIMIZACIÓN:

def optimize_dataframe_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Optimizar DataFrame para análisis más eficiente"""
    df_optimized = df.copy()
    
    # Optimizar tipos de datos
    for column in df_optimized.columns:
        if df_optimized[column].dtype == 'object':
            # Intentar conversión a categoría si hay pocos valores únicos
            unique_ratio = df_optimized[column].nunique() / len(df_optimized)
            if unique_ratio < 0.5:
                df_optimized[column] = df_optimized[column].astype('category')
        
        elif 'int' in str(df_optimized[column].dtype):
            # Downcast enteros
            df_optimized[column] = pd.to_numeric(df_optimized[column], downcast='integer')
        
        elif 'float' in str(df_optimized[column].dtype):
            # Downcast floats
            df_optimized[column] = pd.to_numeric(df_optimized[column], downcast='float')
    
    return df_optimized

# 6. MEJORAR ConfigManager CON CONFIGURACIONES DE PERFORMANCE:

# Agregar esta función al ConfigManager existente:
def add_performance_config(self) -> Dict:
    """Agregar configuraciones de performance al config existente"""
    performance_config = {
        "performance": {
            "cache_enabled": True,
            "cache_size": 1000,
            "batch_size": 50,
            "memory_threshold_mb": 1000,
            "async_processing": True,
            "max_workers": 2,
            "anomaly_detection": {
                "enabled": True,
                "contamination": 0.1,
                "methods": ["isolation_forest", "statistical", "category_based"]
            }
        }
    }
    
    self.config.update(performance_config)
    return performance_config

# Agregar este método a ConfigManager:
ConfigManager.add_performance_config = add_performance_config    