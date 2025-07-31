"""
Test Completo del Sistema - Lunes (VERSIÓN CORREGIDA)
Prueba todos los componentes desarrollados
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Agregar src al path
sys.path.append('src')

def test_system():
    """Probar todo el sistema desarrollado"""
    
    print("🚀 ANALIZADOR DE GASTOS IA - TEST COMPLETO")
    print("="*60)
    
    # Paso 1: Crear datos de prueba
    print("\n📊 PASO 1: Creando datos de prueba...")
    
    # Datos realistas para Argentina
    gastos_ejemplo = [
        # Alimentación
        ('2024-01-15', 'Supermercado Carrefour Villa Crespo', 18500.50),
        ('2024-01-16', 'Panaderia La Esquina', 3200.00),
        ('2024-01-17', 'McDonalds Palermo', 4800.75),
        ('2024-01-18', 'Restaurant Don Julio', 25000.00),
        ('2024-01-19', 'Verduleria Central', 5500.30),
        
        # Transporte
        ('2024-01-20', 'Uber viaje centro', 2800.00),
        ('2024-01-21', 'YPF Combustible', 15000.00),
        ('2024-01-22', 'SUBE recarga', 2000.00),
        ('2024-01-23', 'Taxi Aeropuerto', 8500.00),
        ('2024-01-24', 'Peaje Autopista', 650.00),
        
        # Servicios
        ('2024-01-25', 'Edenor Factura Electricidad', 12000.00),
        ('2024-01-26', 'Movistar Plan Celular', 8900.00),
        ('2024-01-27', 'Fibertel Internet', 6500.00),
        ('2024-01-28', 'Metrogas Factura', 15000.00),
        ('2024-01-29', 'Expensas Edificio', 45000.00),
        
        # Entretenimiento
        ('2024-01-30', 'Netflix Suscripcion', 2490.00),
        ('2024-02-01', 'Cine Hoyts Palermo', 3500.00),
        ('2024-02-02', 'Spotify Premium', 1290.00),
        ('2024-02-03', 'Bar Antares Cerveza', 8500.00),
        ('2024-02-04', 'Steam Videojuego', 12000.00),
        
        # Salud
        ('2024-02-05', 'Farmacity Medicamentos', 4500.00),
        ('2024-02-06', 'Dr. Martinez Consulta', 15000.00),
        ('2024-02-07', 'Dentista Limpieza', 18000.00),
        ('2024-02-08', 'Laboratorio Analisis', 8500.00),
        ('2024-02-09', 'Farmacia del Pueblo', 2800.00),
        
        # Ropa
        ('2024-02-10', 'Zara Camisa Trabajo', 12500.00),
        ('2024-02-11', 'Nike Zapatillas Running', 35000.00),
        ('2024-02-12', 'H&M Pantalon Jean', 8900.00),
        ('2024-02-13', 'Adidas Campera', 28000.00),
        
        # Hogar
        ('2024-02-14', 'Easy Herramientas', 5500.00),
        ('2024-02-15', 'Sodimac Pintura', 8900.00),
        ('2024-02-16', 'Detergente Skip', 1200.00),
        ('2024-02-17', 'IKEA Escritorio', 45000.00),
        
        # Educación
        ('2024-02-18', 'Universidad UTN Cuota', 25000.00),
        ('2024-02-19', 'Udemy Curso Python', 15000.00),
        ('2024-02-20', 'Libros Amazon', 8500.00)
    ]
    
    # Crear DataFrame
    df_gastos = pd.DataFrame(gastos_ejemplo, columns=['fecha', 'descripcion', 'monto'])
    df_gastos['fecha'] = pd.to_datetime(df_gastos['fecha'])
    
    print(f"✅ Creados {len(df_gastos)} gastos de ejemplo")
    
    # Guardar datos de ejemplo
    Path('data/examples').mkdir(parents=True, exist_ok=True)
    df_gastos.to_csv('data/examples/gastos_test.csv', index=False)
    print("✅ Datos guardados en: data/examples/gastos_test.csv")
    
    # Paso 2: Probar el Data Processor
    print("\n📈 PASO 2: Probando Data Processor...")
    
    try:
        from data_proccers import ExpenseDataProcessor
        
        processor = ExpenseDataProcessor()
        df_processed, insights = processor.process_file('data/examples/gastos_test.csv')
        
        print(f"✅ Datos procesados exitosamente")
        print(f"   📊 Total gastos: {insights['total_gastos']}")
        print(f"   💰 Monto total: ${insights['monto_total']:,.2f}")
        print(f"   📈 Promedio: ${insights['monto_promedio']:,.2f}")
        
        # Mostrar categorías detectadas (versión corregida)
        print(f"\n   🏷️ Categorías detectadas:")
        category_counts = df_processed['categoria'].value_counts()
        category_totals = df_processed.groupby('categoria')['monto'].sum()
        
        for categoria in category_counts.index:
            count = category_counts[categoria]
            total = category_totals[categoria]
            print(f"      {categoria}: ${total:,.2f} ({count} gastos)")
            
    except Exception as e:
        print(f"❌ Error en Data Processor: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Paso 3: Probar el Categorizador Inteligente
    print("\n🤖 PASO 3: Probando Categorizador IA...")
    
    try:
        from categorizer import CategoryTrainer, IntelligentCategorizer
        
        # Entrenar modelo
        trainer = CategoryTrainer()
        training_results = trainer.train_with_sample_data()
        
        print(f"✅ Modelo entrenado: {training_results['status']}")
        if training_results['status'] == 'trained':
            print(f"   🎯 Accuracy: {training_results['accuracy']:.3f}")
            print(f"   📚 Muestras entrenamiento: {training_results['samples']}")
        
        # Categorizar nuestros datos
        predictions = trainer.categorizer.predict_with_confidence(
            df_processed['descripcion'].tolist()
        )
        
        print(f"\n   🏷️ Ejemplos de categorización IA:")
        for i, pred in enumerate(predictions[:10]):  # Mostrar primeros 10
            emoji = "🟢" if pred['confidence'] > 0.7 else "🟡" if pred['confidence'] > 0.4 else "🔴"
            print(f"      {emoji} {pred['categoria']:12} | {pred['descripcion'][:40]:<40} | {pred['confidence']:.2f}")
        
        # Guardar modelo
        Path('models').mkdir(parents=True, exist_ok=True)
        trainer.categorizer.save_model('models/categorizer_trained.pkl')
        print(f"✅ Modelo guardado en: models/categorizer_trained.pkl")
        
    except Exception as e:
        print(f"❌ Error en Categorizador: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Paso 4: Probar Utilidades
    print("\n🛠️ PASO 4: Probando Utilidades...")
    
    try:
        from utils import ConfigManager, CurrencyFormatter, ExpenseAnalyzer
        
        # Config Manager
        config = ConfigManager()
        print(f"✅ Configuración cargada")
        
        # Currency Formatter
        formatter = CurrencyFormatter()
        ejemplo_monto = 1234567.89
        print(f"✅ Formateo de moneda: {formatter.format_amount(ejemplo_monto)}")
        print(f"   Forma corta: {formatter.format_amount_short(ejemplo_monto)}")
        
        # Expense Analyzer
        analyzer = ExpenseAnalyzer()
        df_with_categories = df_processed.copy()
        
        # Usar categorías del categorizador IA
        df_with_categories['categoria'] = [p['categoria'] for p in predictions]
        
        # Detectar anomalías
        df_anomalies = analyzer.detect_anomalies(df_with_categories)
        anomalies_count = df_anomalies['is_anomaly'].sum()
        print(f"✅ Anomalías detectadas: {anomalies_count}")
        
        # Insights por categoría
        category_insights = analyzer.get_category_insights(df_with_categories)
        print(f"✅ Insights por categoría generados")
        
        # Estadísticas generales
        stats = analyzer.generate_summary_stats(df_with_categories)
        print(f"✅ Estadísticas: Total {stats['total_amount_formatted']}, "
              f"Promedio {stats['average_transaction_formatted']}")
        
    except Exception as e:
        print(f"❌ Error en Utilidades: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Paso 5: Generar reporte final
    print("\n📋 PASO 5: Generando reporte final...")
    
    try:
        # Crear reporte completo
        reporte = {
            'fecha_reporte': datetime.now().isoformat(),
            'total_gastos': len(df_with_categories),
            'monto_total': float(df_with_categories['monto'].sum()),
            'periodo': {
                'desde': df_with_categories['fecha'].min().isoformat(),
                'hasta': df_with_categories['fecha'].max().isoformat()
            },
            'categorias': {},
            'top_gastos': df_with_categories.nlargest(5, 'monto')[['descripcion', 'monto', 'categoria']].to_dict('records'),
            'anomalias': int(anomalies_count),
            'modelo_accuracy': float(training_results.get('accuracy', 0))
        }
        
        # Estadísticas por categoría
        for categoria in df_with_categories['categoria'].unique():
            cat_data = df_with_categories[df_with_categories['categoria'] == categoria]
            reporte['categorias'][categoria] = {
                'cantidad': len(cat_data),
                'total': float(cat_data['monto'].sum()),
                'promedio': float(cat_data['monto'].mean()),
                'porcentaje': float((cat_data['monto'].sum() / df_with_categories['monto'].sum()) * 100)
            }
        
        # Guardar reporte
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        with open('data/processed/reporte_completo.json', 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Reporte guardado en: data/processed/reporte_completo.json")
        
        # Guardar datos procesados
        df_with_categories.to_csv('data/processed/gastos_procesados.csv', index=False)
        print(f"✅ Datos procesados guardados en: data/processed/gastos_procesados.csv")
        
    except Exception as e:
        print(f"❌ Error generando reporte: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Resultado final
    print("\n" + "="*60)
    print("🎉 ¡SISTEMA COMPLETAMENTE FUNCIONAL!")
    print("="*60)
    
    print(f"\n📊 RESUMEN DEL SISTEMA:")
    print(f"   💾 Datos procesados: {len(df_with_categories)} transacciones")
    print(f"   🏷️ Categorías detectadas: {len(df_with_categories['categoria'].unique())}")
    print(f"   🤖 Modelo accuracy: {training_results.get('accuracy', 0):.1%}")
    print(f"   💰 Total analizado: ${df_with_categories['monto'].sum():,.2f}")
    print(f"   ⚠️ Gastos anómalos: {anomalies_count}")
    
    print(f"\n📂 ARCHIVOS GENERADOS:")
    print(f"   📄 data/examples/gastos_test.csv")
    print(f"   📄 data/processed/gastos_procesados.csv") 
    print(f"   📄 data/processed/reporte_completo.json")
    print(f"   🤖 models/categorizer_trained.pkl")
    
    print(f"\n🚀 PRÓXIMOS PASOS:")
    print(f"   📅 Martes: Desarrollar modelos ML avanzados")
    print(f"   📅 Miércoles: Optimizar y validar modelos")
    print(f"   📅 Jueves: Crear interfaz Streamlit")
    print(f"   📅 Viernes: Pulir UI/UX")
    
    return True

if __name__ == "__main__":
    success = test_system()
    
    if success:
        print(f"\n✅ ¡Lunes completado exitosamente!")
        print(f"🎯 El sistema está listo para continuar con el desarrollo de ML avanzado")
    else:
        print(f"\n❌ Hubo algunos errores. Revisa los logs arriba.")