"""
Procesador de Datos para Analizador de Gastos IA
Maneja la carga, limpieza y procesamiento de datos de gastos
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Union
import logging
from pathlib import Path
import chardet

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpenseDataProcessor:
    """
    Procesador principal para datos de gastos personales
    """
    
    def __init__(self):
        self.data = None
        self.raw_data = None
        self.processed_data = None
        
        # Patrones comunes para categorización
        self.category_patterns = {
            'Alimentación': ['supermercado', 'grocery', 'restaurant', 'comida', 'food', 'mercado', 
                           'carrefour', 'dia', 'coto', 'la anonima', 'mcdonalds', 'burger', 'pizza'],
            'Transporte': ['uber', 'taxi', 'subte', 'colectivo', 'bus', 'combustible', 'nafta',
                         'peaje', 'estacionamiento', 'ypf', 'shell', 'axion'],
            'Servicios': ['electricidad', 'gas', 'agua', 'internet', 'telefono', 'cable',
                        'netflix', 'spotify', 'gym', 'gimnasio'],
            'Entretenimiento': ['cine', 'teatro', 'bar', 'disco', 'spotify', 'netflix',
                              'steam', 'playstation', 'xbox', 'juego'],
            'Salud': ['farmacia', 'medico', 'doctor', 'hospital', 'clinica', 'medicamento',
                    'dentista', 'oculista'],
            'Ropa': ['ropa', 'zapatillas', 'zapatos', 'camisa', 'pantalon', 'vestido',
                   'nike', 'adidas', 'zara', 'h&m'],
            'Hogar': ['muebles', 'decoracion', 'ferreteria', 'limpieza', 'detergente',
                    'easy', 'sodimac'],
            'Educación': ['universidad', 'curso', 'libro', 'escuela', 'colegio', 'capacitacion'],
            'Otros': []  # Categoría por defecto
        }
        
        # Columnas esperadas (flexible)
        self.expected_columns = {
            'fecha': ['fecha', 'date', 'timestamp', 'day'],
            'descripcion': ['descripcion', 'description', 'concepto', 'detalle', 'merchant'],
            'monto': ['monto', 'amount', 'importe', 'valor', 'price'],
            'tipo': ['tipo', 'type', 'debito', 'credito', 'debit', 'credit']
        }
    
    def detect_encoding(self, file_path: str) -> str:
        """Detectar encoding del archivo"""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    def load_data(self, file_path: str, file_type: str = None) -> pd.DataFrame:
        """
        Cargar datos desde diferentes formatos de archivo
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        # Determinar tipo de archivo si no se especifica
        if file_type is None:
            file_type = file_path.suffix.lower()
        
        logger.info(f"Cargando archivo: {file_path} (tipo: {file_type})")
        
        try:
            if file_type in ['.csv', '.txt']:
                # Detectar encoding
                encoding = self.detect_encoding(file_path)
                
                # Intentar diferentes separadores
                separators = [',', ';', '\t', '|']
                
                for sep in separators:
                    try:
                        df = pd.read_csv(file_path, sep=sep, encoding=encoding, 
                                       low_memory=False, parse_dates=True)
                        if len(df.columns) > 1:  # Archivo válido con múltiples columnas
                            break
                    except:
                        continue
                else:
                    # Último intento con separador automático
                    df = pd.read_csv(file_path, sep=None, engine='python', 
                                   encoding=encoding, low_memory=False)
                    
            elif file_type in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, parse_dates=True)
                
            elif file_type == '.json':
                df = pd.read_json(file_path)
                
            else:
                raise ValueError(f"Formato de archivo no soportado: {file_type}")
            
            logger.info(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
            self.raw_data = df.copy()
            return df
            
        except Exception as e:
            logger.error(f"Error cargando archivo: {e}")
            raise
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estandarizar nombres de columnas
        """
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.lower().str.strip()
        
        # Mapear columnas a nombres estándar
        column_mapping = {}
        
        for standard_name, variants in self.expected_columns.items():
            for col in df_clean.columns:
                if any(variant in col for variant in variants):
                    column_mapping[col] = standard_name
                    break
        
        df_clean = df_clean.rename(columns=column_mapping)
        
        # Verificar columnas mínimas necesarias
        required_cols = ['descripcion', 'monto']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        
        if missing_cols:
            logger.warning(f"Columnas faltantes: {missing_cols}")
            # Intentar inferir columnas faltantes
            if 'descripcion' not in df_clean.columns:
                # Buscar columna más likely para descripcion
                text_cols = df_clean.select_dtypes(include=['object']).columns
                if len(text_cols) > 0:
                    df_clean['descripcion'] = df_clean[text_cols[0]]
                    
            if 'monto' not in df_clean.columns:
                # Buscar columna numérica para monto
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df_clean['monto'] = df_clean[numeric_cols[0]]
        
        return df_clean
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpiar y preparar los datos
        """
        logger.info("Iniciando limpieza de datos...")
        df_clean = df.copy()
        
        # 1. Limpiar columna de descripción
        if 'descripcion' in df_clean.columns:
            df_clean['descripcion'] = df_clean['descripcion'].astype(str)
            df_clean['descripcion'] = df_clean['descripcion'].str.strip()
            df_clean['descripcion'] = df_clean['descripcion'].str.lower()
            # Remover caracteres especiales pero mantener espacios
            df_clean['descripcion'] = df_clean['descripcion'].str.replace(r'[^\w\s]', ' ', regex=True)
            df_clean['descripcion'] = df_clean['descripcion'].str.replace(r'\s+', ' ', regex=True)
        
        # 2. Limpiar columna de monto
        if 'monto' in df_clean.columns:
            # Manejar diferentes formatos de números
            df_clean['monto'] = df_clean['monto'].astype(str)
            df_clean['monto'] = df_clean['monto'].str.replace('$', '')
            df_clean['monto'] = df_clean['monto'].str.replace(',', '.')
            df_clean['monto'] = df_clean['monto'].str.replace(' ', '')
            
            # Convertir a numérico
            df_clean['monto'] = pd.to_numeric(df_clean['monto'], errors='coerce')
            df_clean['monto'] = df_clean['monto'].abs()  # Valores absolutos
        
        # 3. Procesar fechas
        if 'fecha' in df_clean.columns:
            df_clean['fecha'] = pd.to_datetime(df_clean['fecha'], errors='coerce', infer_datetime_format=True)
        else:
            # Si no hay fecha, usar fecha actual
            df_clean['fecha'] = datetime.now()
        
        # 4. Remover filas inválidas
        initial_rows = len(df_clean)
        
        # Remover filas sin descripción o monto
        df_clean = df_clean.dropna(subset=['descripcion', 'monto'])
        df_clean = df_clean[df_clean['monto'] > 0]  # Solo gastos positivos
        df_clean = df_clean[df_clean['descripcion'].str.len() > 0]
        
        # Remover duplicados
        df_clean = df_clean.drop_duplicates(subset=['descripcion', 'monto'], keep='first')
        
        final_rows = len(df_clean)
        logger.info(f"Limpieza completada: {initial_rows} -> {final_rows} filas ({initial_rows-final_rows} removidas)")
        
        return df_clean
    
    def categorize_expenses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorizar gastos automáticamente basado en patrones
        """
        logger.info("Categorizando gastos...")
        df_categorized = df.copy()
        
        def get_category(description: str) -> str:
            """Determinar categoría basada en descripción"""
            description = str(description).lower()
            
            for category, keywords in self.category_patterns.items():
                if category == 'Otros':
                    continue
                    
                for keyword in keywords:
                    if keyword in description:
                        return category
            
            return 'Otros'
        
        # Aplicar categorización
        df_categorized['categoria'] = df_categorized['descripcion'].apply(get_category)
        
        # Estadísticas de categorización
        category_counts = df_categorized['categoria'].value_counts()
        logger.info(f"Categorías asignadas:\n{category_counts}")
        
        return df_categorized
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agregar características temporales
        """
        df_time = df.copy()
        
        if 'fecha' in df_time.columns:
            df_time['año'] = df_time['fecha'].dt.year
            df_time['mes'] = df_time['fecha'].dt.month
            df_time['dia'] = df_time['fecha'].dt.day
            df_time['dia_semana'] = df_time['fecha'].dt.dayofweek
            df_time['dia_semana_nombre'] = df_time['fecha'].dt.day_name()
            df_time['mes_nombre'] = df_time['fecha'].dt.month_name()
            df_time['trimestre'] = df_time['fecha'].dt.quarter
            df_time['es_fin_semana'] = df_time['dia_semana'].isin([5, 6])
        
        return df_time
    
    def generate_insights(self, df: pd.DataFrame) -> Dict:
        """
        Generar insights básicos de los datos
        """
        insights = {}
        
        # Estadísticas básicas
        insights['total_gastos'] = len(df)
        insights['monto_total'] = df['monto'].sum()
        insights['monto_promedio'] = df['monto'].mean()
        insights['monto_mediano'] = df['monto'].median()
        
        # Por categoría
        insights['gastos_por_categoria'] = df.groupby('categoria')['monto'].agg(['sum', 'count', 'mean']).to_dict()
        
        # Por tiempo (si hay fechas)
        if 'fecha' in df.columns:
            insights['periodo'] = {
                'fecha_inicio': df['fecha'].min(),
                'fecha_fin': df['fecha'].max(),
                'dias_total': (df['fecha'].max() - df['fecha'].min()).days
            }
            
            # Gastos por mes
            monthly_expenses = df.groupby([df['fecha'].dt.year, df['fecha'].dt.month])['monto'].sum()
            insights['gastos_mensuales'] = monthly_expenses.to_dict()
        
        # Top gastos
        insights['top_gastos'] = df.nlargest(10, 'monto')[['descripcion', 'monto', 'categoria']].to_dict('records')
        
        return insights
    
    def process_file(self, file_path: str, file_type: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Procesar archivo completo: cargar, limpiar, categorizar y generar insights
        """
        logger.info(f"Procesando archivo: {file_path}")
        
        # 1. Cargar datos
        df = self.load_data(file_path, file_type)
        
        # 2. Estandarizar columnas
        df = self.standardize_columns(df)
        
        # 3. Limpiar datos
        df = self.clean_data(df)
        
        # 4. Categorizar gastos
        df = self.categorize_expenses(df)
        
        # 5. Agregar características temporales
        df = self.add_time_features(df)
        
        # 6. Generar insights
        insights = self.generate_insights(df)
        
        # Guardar datos procesados
        self.processed_data = df
        
        logger.info("Procesamiento completado exitosamente")
        return df, insights
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """
        Guardar datos procesados
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif output_path.suffix in ['.xlsx', '.xls']:
            df.to_excel(output_path, index=False)
        else:
            df.to_csv(output_path.with_suffix('.csv'), index=False, encoding='utf-8')
        
        logger.info(f"Datos guardados en: {output_path}")

# Función de utilidad para uso rápido
def process_expense_file(file_path: str, output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Función de conveniencia para procesar un archivo de gastos
    """
    processor = ExpenseDataProcessor()
    df, insights = processor.process_file(file_path)
    
    if output_path:
        processor.save_processed_data(df, output_path)
    
    return df, insights

if __name__ == "__main__":
    # Ejemplo de uso
    print("🚀 Procesador de Datos - Analizador de Gastos IA")
    print("Ejemplo de uso del procesador...")
    
    # Crear datos de ejemplo
    sample_data = {
        'fecha': pd.date_range('2024-01-01', periods=50, freq='D'),
        'descripcion': [
            'Supermercado Carrefour', 'Uber viaje centro', 'Netflix suscripcion',
            'Farmacia medicamento', 'Restaurant pizza', 'Combustible YPF',
            'Gym mensualidad', 'Cine entrada', 'Supermercado Coto',
            'Taxi aeropuerto'
        ] * 5,
        'monto': np.random.uniform(100, 5000, 50)
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('data/examples/gastos_ejemplo.csv', index=False)
    
    # Procesar el archivo de ejemplo
    try:
        processor = ExpenseDataProcessor()
        df_processed, insights = processor.process_file('data/examples/gastos_ejemplo.csv')
        
        print(f"\n✅ Procesamiento exitoso!")
        print(f"📊 Total gastos: {insights['total_gastos']}")
        print(f"💰 Monto total: ${insights['monto_total']:,.2f}")
        print(f"📈 Promedio: ${insights['monto_promedio']:,.2f}")
        
        print(f"\n🏷️ Categorías encontradas:")
        for categoria, data in insights['gastos_por_categoria'].items():
            print(f"   {categoria}: ${data['sum']:,.2f} ({data['count']} gastos)")
            
    except Exception as e:
        print(f"❌ Error: {e}")