"""
Utilidades generales para el Analizador de Gastos IA
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from typing import Dict, List, Any, Union
import locale

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