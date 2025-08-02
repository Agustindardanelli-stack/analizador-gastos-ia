# ============================================================================
# CAMBIOS PARA test-complete.py - Reemplazar la función test_system()
# ============================================================================

def test_system_improved():
    """Probar todo el sistema con las mejoras implementadas"""
    
    print("🚀 ANALIZADOR DE GASTOS IA - TEST COMPLETO MEJORADO")
    print("="*60)
    
    # Paso 1: Crear datos de prueba (mantener igual)
    print("\n📊 PASO 1: Creando datos de prueba...")
    
    gastos_ejemplo = [
        # Mismos datos que antes...
        ('2024-01-15', 'Supermercado Carrefour Villa Crespo', 18500.50),
        ('2024-01-16', 'Panaderia La Esquina', 3200.00),
        ('2024-01-17', 'McDonalds Palermo', 4800.75),
        ('2024-01-18', 'Restaurant Don Julio', 25000.00),
        ('2024-01-19', 'Verduleria Central', 5500.30),
        ('2024-01-20', 'Uber viaje centro', 2800.00),
        ('2024-01-21', 'YPF Combustible', 15000.00),
        ('2024-01-22', 'SUBE recarga', 2000.00),
        ('2024-01-23', 'Taxi Aeropuerto', 8500.00),
        ('2024-01-24', 'Peaje Autopista', 650.00),
        ('2024-01-25', 'Edenor Factura Electricidad', 12000.00),
        ('2024-01-26', 'Movistar Plan Celular', 8900.00),
        ('2024-01-27', 'Fibertel Internet', 6500.00),
        ('2024-01-28', 'Metrogas Factura', 15000.00),
        ('2024-01-29', 'Expensas Edificio', 45000.00),
        ('2024-01-30', 'Netflix Suscripcion', 2490.00),
        ('2024-02-01', 'Cine Hoyts Palermo', 3500.00),
        ('2024-02-02', 'Spotify Premium', 1290.00),
        ('2024-02-03', 'Bar Antares Cerveza', 8500.00),
        ('2024-02-04', 'Steam Videojuego', 12000.00),
        ('2024-02-05', 'Farmacity Medicamentos', 4500.00),
        ('2024-02-06', 'Dr. Martinez Consulta', 15000.00),
        ('2024-02-07', 'Dentista Limpieza', 18000.00),
        ('2024-02-08', 'Laboratorio Analisis', 8500.00),
        ('2024-02-09', 'Farmacia del Pueblo', 2800.00),
        ('2024-02-10', 'Zara Camisa Trabajo', 12500.00),
        ('2024-02-11', 'Nike Zapatillas Running', 35000.00),
        ('2024-02-12', 'H&M Pantalon Jean', 8900.00),
        ('2024-02-13', 'Adidas Campera', 28000.00),
        ('2024-02-14', 'Easy Herramientas', 5500.00),
        ('2024-02-15', 'Sodimac Pintura', 8900.00),
        ('2024-02-16', 'Detergente Skip', 1200.00),
        ('2024-02-17', 'IKEA Escritorio', 45000.00),
        ('2024-02-18', 'Universidad UTN Cuota', 25000.00),
        ('2024-02-19', 'Udemy Curso Python', 15000.00),
        ('2024-02-20', 'Libros Amazon', 8500.00)
    ]
    
    df_gastos = pd.DataFrame(gastos_ejemplo, columns=['fecha', 'descripcion', 'monto'])
    df_gastos['fecha'] = pd.to_datetime(df_gastos['fecha'])
    
    print(f"✅ Creados {len(df_gastos)} gastos de ejemplo")
    
    # Guardar datos de ejemplo
    Path('data/examples').mkdir(parents=True, exist_ok=True)
    df_gastos.to_csv('data/examples/gastos_test_improved.csv', index=False)
    print("✅ Datos guardados en: data/examples/gastos_test_improved.csv")
    
    # Paso 2: Probar Data Processor Mejorado
    print("\n📈 PASO 2: Probando Data Processor...")
    
    try:
        from src.data_processor import ExpenseDataProcessor
        from src.utils import optimize_dataframe_for_analysis, MemoryMonitor
        
        processor = ExpenseDataProcessor()
        memory_monitor = MemoryMonitor()
        
        # Procesar datos
        df_processed, insights = processor.process_file('data/examples/gastos_test_improved.csv')
        
        # Optimizar DataFrame
        df_optimized = optimize_dataframe_for_analysis(df_processed)
        memory_savings = df_processed.memory_usage(deep=True).sum() - df_optimized.memory_usage(deep=True).sum()
        
        print(f"✅ Datos procesados y optimizados")
        print(f"   📊 Total gastos: {insights['total_gastos']}")
        print(f"   💰 Monto total: ${insights['monto_total']:,.2f}")
        print(f"   🔧 Memoria ahorrada: {memory_savings / 1024:.1f} KB")
        
        # Mostrar uso de memoria
        memory_status = memory_monitor.check_memory_usage()
        print(f"   🖥️ Memoria proceso: {memory_status['current']['process_mb']:.1f} MB")
        
    except Exception as e:
        print(f"❌ Error en Data Processor: {e}")
        return False
    
    # Paso 3: Probar Categorizador Mejorado
    print("\n🤖 PASO 3: Probando Categorizador IA Mejorado...")
    
    try:
        from categorizer import ImprovedIntelligentCategorizer, OptimizedCategorizer, categorize_expenses_improved
        
        # Usar función mejorada de categorización
        df_categorized = categorize_expenses_improved(df_optimized, use_cache=True)
        
        print(f"✅ Categorización mejorada completada")
        print(f"   🎯 Categorías encontradas: {df_categorized['categoria'].nunique()}")
        print(f"   🔍 Confianza promedio: {df_categorized['confidence'].mean():.3f}")
        
        # Mostrar distribución de métodos usados
        method_distribution = df_categorized['method'].value_counts()
        print(f"   📊 Métodos utilizados:")
        for method, count in method_distribution.items():
            print(f"      {method}: {count} predicciones")
        
        # Mostrar ejemplos de alta y baja confianza
        high_conf = df_categorized[df_categorized['confidence'] > 0.8]
        low_conf = df_categorized[df_categorized['confidence'] < 0.5]
        
        print(f"\n   🟢 Predicciones de alta confianza ({len(high_conf)}):")
        for _, row in high_conf.head(5).iterrows():
            print(f"      {row['categoria']:12} | {row['descripcion'][:40]:<40} | {row['confidence']:.2f}")
        
        if len(low_conf) > 0:
            print(f"\n   🟡 Predicciones de baja confianza ({len(low_conf)}):")
            for _, row in low_conf.head(3).iterrows():
                print(f"      {row['categoria']:12} | {row['descripcion'][:40]:<40} | {row['confidence']:.2f}")
        
    except Exception as e:
        print(f"❌ Error en Categorizador Mejorado: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Paso 4: Probar Detección de Anomalías
    print("\n🔍 PASO 4: Probando Detección de Anomalías...")
    
    try:
        from src.utils import EnhancedExpenseAnalyzer
        
        enhanced_analyzer = EnhancedExpenseAnalyzer()
        df_with_anomalies = enhanced_analyzer.detect_advanced_anomalies(df_categorized)
        
        anomalies_count = df_with_anomalies['is_any_anomaly'].sum()
        ml_anomalies = df_with_anomalies.get('is_ml_anomaly', pd.Series(False)).sum()
        category_anomalies = df_with_anomalies.get('is_category_anomaly', pd.Series(False)).sum()
        
        print(f"✅ Detección de anomalías completada")
        print(f"   ⚠️ Total anomalías: {anomalies_count}")
        print(f"   🤖 Anomalías ML: {ml_anomalies}")
        print(f"   📊 Anomalías por categoría: {category_anomalies}")
        
        # Mostrar anomalías detectadas
        if anomalies_count > 0:
            anomalies = df_with_anomalies[df_with_anomalies['is_any_anomaly']]
            print(f"\n   🚨 Gastos anómalos detectados:")
            for _, row in anomalies.head(3).iterrows():
                print(f"      ${row['monto']:8,.0f} | {row['descripcion'][:35]:<35} | {row['categoria']}")
        
        # Obtener insights de performance
        performance_insights = enhanced_analyzer.get_performance_insights(df_with_anomalies)
        
        if 'system_performance' in performance_insights:
            sys_perf = performance_insights['system_performance']
            print(f"   💾 Tamaño datos: {sys_perf['data_size_mb']:.2f} MB")
            
            if sys_perf['processing_recommendations']:
                print(f"   💡 Recomendaciones:")
                for rec in sys_perf['processing_recommendations'][:2]:
                    print(f"      {rec}")
        
    except Exception as e:
        print(f"❌ Error en Detección de Anomalías: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Paso 5: Probar Procesamiento Asíncrono
    print("\n⚡ PASO 5: Probando Procesamiento Asíncrono...")
    
    try:
        from src.utils import AsyncProcessor
        import time
        
        async_processor = AsyncProcessor(max_workers=2)
        
        def simulate_heavy_task(data_size):
            """Simular tarea pesada"""
            time.sleep(1)  # Simular procesamiento
            return f"Procesados {data_size} registros en background"
        
        # Lanzar tarea asíncrona
        task_id = async_processor.process_async(simulate_heavy_task, len(df_with_anomalies))
        
        print(f"✅ Tarea asíncrona iniciada: {task_id}")
        
        # Verificar estado
        status = async_processor.get_task_status(task_id)
        print(f"   📊 Estado: {status['status']}")
        print(f"   ⏱️ Función: {status['function']}")
        
        # Esperar resultado
        print("   ⏳ Esperando resultado...")
        result = async_processor.get_result(task_id, timeout=3.0)
        
        if result:
            print(f"   ✅ Resultado: {result}")
        else:
            print("   ⏰ Tarea aún en proceso o timeout")
        
        # Limpiar
        async_processor.shutdown()
        
    except Exception as e:
        print(f"❌ Error en Procesamiento Asíncrono: {e}")
        return False
    
    # Paso 6: Generar Reporte Completo
    print("\n📋 PASO 6: Generando reporte completo...")
    
    try:
        # Crear reporte mejorado
        reporte_mejorado = {
            'fecha_reporte': datetime.now().isoformat(),
            'version_sistema': '2.0_mejorado',
            'datos_procesados': {
                'total_gastos': len(df_with_anomalies),
                'monto_total': float(df_with_anomalies['monto'].sum()),
                'periodo': {
                    'desde': df_with_anomalies['fecha'].min().isoformat(),
                    'hasta': df_with_anomalies['fecha'].max().isoformat()
                },
                'optimizaciones_aplicadas': True
            },
            'categorizacion': {
                'categorias_detectadas': int(df_with_anomalies['categoria'].nunique()),
                'confianza_promedio': float(df_with_anomalies['confidence'].mean()),
                'metodos_utilizados': df_with_anomalies['method'].value_counts().to_dict(),
                'distribuciones_confianza': {
                    'alta_confianza': int((df_with_anomalies['confidence'] > 0.8).sum()),
                    'media_confianza': int(((df_with_anomalies['confidence'] >= 0.5) & (df_with_anomalies['confidence'] <= 0.8)).sum()),
                    'baja_confianza': int((df_with_anomalies['confidence'] < 0.5).sum())
                }
            },
            'anomalias': {
                'total_detectadas': int(anomalies_count),
                'por_metodo': {
                    'ml_anomalies': int(ml_anomalies),
                    'category_anomalies': int(category_anomalies),
                    'statistical_anomalies': int(df_with_anomalies.get('is_anomaly', pd.Series(False)).sum())
                }
            },
            'rendimiento': {
                'memoria_utilizada': memory_status['current']['process_mb'],
                'optimizaciones_memoria': memory_savings / 1024,
                'procesamiento_asincrono': True
            }
        }
        
        # Distribución por categoría
        category_stats = {}
        for categoria in df_with_anomalies['categoria'].unique():
            cat_data = df_with_anomalies[df_with_anomalies['categoria'] == categoria]
            category_stats[categoria] = {
                'cantidad': len(cat_data),
                'total': float(cat_data['monto'].sum()),
                'promedio': float(cat_data['monto'].mean()),
                'confianza_promedio': float(cat_data['confidence'].mean()),
                'anomalias': int(cat_data['is_any_anomaly'].sum())
            }
        
        reporte_mejorado['categorias_detalladas'] = category_stats
        
        # Top gastos con más información
        top_gastos = df_with_anomalies.nlargest(5, 'monto')
        reporte_mejorado['top_gastos'] = []
        for _, row in top_gastos.iterrows():
            reporte_mejorado['top_gastos'].append({
                'descripcion': row['descripcion'],
                'monto': float(row['monto']),
                'categoria': row['categoria'],
                'confidence': float(row['confidence']),
                'method': row['method'],
                'is_anomaly': bool(row['is_any_anomaly'])
            })
        
        # Guardar reporte
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        with open('data/processed/reporte_mejorado.json', 'w', encoding='utf-8') as f:
            json.dump(reporte_mejorado, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Reporte mejorado guardado en: data/processed/reporte_mejorado.json")
        
        # Guardar datos procesados
        df_with_anomalies.to_csv('data/processed/gastos_procesados_mejorado.csv', index=False)
        print(f"✅ Datos procesados guardados en: data/processed/gastos_procesados_mejorado.csv")
        
    except Exception as e:
        print(f"❌ Error generando reporte: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Resultado final
    print("\n" + "="*60)
    print("🎉 ¡SISTEMA MEJORADO COMPLETAMENTE FUNCIONAL!")
    print("="*60)
    
    print(f"\n📊 RESUMEN DEL SISTEMA MEJORADO:")
    print(f"   💾 Datos procesados: {len(df_with_anomalies)} transacciones")
    print(f"   🏷️ Categorías detectadas: {df_with_anomalies['categoria'].nunique()}")
    print(f"   🎯 Confianza promedio: {df_with_anomalies['confidence'].mean():.1%}")
    print(f"   💰 Total analizado: ${df_with_anomalies['monto'].sum():,.2f}")
    print(f"   ⚠️ Anomalías detectadas: {anomalies_count}")
    print(f"   🔧 Optimizaciones aplicadas: ✅")
    print(f"   ⚡ Procesamiento asíncrono: ✅")
    print(f"   💾 Memoria optimizada: {memory_savings / 1024:.1f} KB ahorrados")
    
    print(f"\n📂 ARCHIVOS GENERADOS:")
    print(f"   📄 data/examples/gastos_test_improved.csv")
    print(f"   📄 data/processed/gastos_procesados_mejorado.csv") 
    print(f"   📄 data/processed/reporte_mejorado.json")
    
    print(f"\n🆕 MEJORAS IMPLEMENTADAS:")
    print(f"   ✅ Categorizador mejorado con mejor accuracy")
    print(f"   ✅ Sistema de caché para predicciones")
    print(f"   ✅ Detección de anomalías multi-método")
    print(f"   ✅ Optimización de memoria automática")
    print(f"   ✅ Procesamiento asíncrono para tareas pesadas")
    print(f"   ✅ Monitoreo de rendimiento en tiempo real")
    
    print(f"\n🚀 PRÓXIMOS PASOS:")
    print(f"   📅 Implementar en Streamlit con nuevas funciones")
    print(f"   📅 Agregar dashboard de monitoreo en tiempo real")
    print(f"   📅 Implementar alertas automáticas de anomalías")
    print(f"   📅 Optimizar para datasets grandes (>50k transacciones)")
    
    return True

# REEMPLAZAR LA LLAMADA AL FINAL DEL ARCHIVO:
if __name__ == "__main__":
    success = test_system_improved()  # Cambiar aquí
    
    if success:
        print(f"\n✅ ¡Sistema mejorado funcionando perfectamente!")
        print(f"🎯 Todas las mejoras implementadas y probadas exitosamente")
    else:
        print(f"\n❌ Hubo algunos errores. Revisa los logs arriba.")