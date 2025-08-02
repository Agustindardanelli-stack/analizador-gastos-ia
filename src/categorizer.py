"""
Sistema de Categorización Inteligente para Gastos
Utiliza Machine Learning para categorizar gastos automáticamente
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from typing import List, Dict, Tuple, Optional
import logging
import time
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentCategorizer:
    """
    Categorizador inteligente que aprende de los datos para clasificar gastos
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Manejaremos stop words manualmente
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Stop words en español e inglés
        self.stop_words = set([
            'de', 'la', 'el', 'en', 'y', 'a', 'que', 'es', 'se', 'no', 'te', 'lo', 'le',
            'da', 'su', 'por', 'son', 'con', 'para', 'del', 'las', 'un', 'al', 'como',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        ])
        
        # Categorías predefinidas con palabras clave
        self.category_keywords = {
            'Alimentación': {
                'keywords': ['supermercado', 'restaurant', 'comida', 'food', 'grocery', 'mercado',
                           'carrefour', 'coto', 'dia', 'jumbo', 'disco', 'la anonima', 'vea',
                           'mcdonalds', 'burger', 'pizza', 'cafe', 'bar', 'panaderia', 'almacen',
                           'verduleria', 'carniceria', 'rotiseria', 'heladeria', 'confiteria'],
                'weight': 2.0
            },
            'Transporte': {
                'keywords': ['uber', 'taxi', 'cabify', 'subte', 'colectivo', 'bus', 'tren',
                           'combustible', 'nafta', 'gasolina', 'peaje', 'estacionamiento',
                           'ypf', 'shell', 'axion', 'puma', 'viaje', 'transporte', 'sube',
                           'remis', 'transfer', 'avion', 'micro', 'cochera'],
                'weight': 2.0
            },
            'Servicios': {
                'keywords': ['electricidad', 'gas', 'agua', 'internet', 'telefono', 'celular',
                           'cable', 'directv', 'telecentro', 'fibertel', 'movistar', 'personal',
                           'claro', 'servicio', 'factura', 'expensas', 'edenor', 'edesur',
                           'aysa', 'metrogas', 'wifi', 'plan'],
                'weight': 1.8
            },
            'Entretenimiento': {
                'keywords': ['cine', 'teatro', 'netflix', 'spotify', 'amazon', 'disney',
                           'steam', 'playstation', 'xbox', 'juego', 'pelicula', 'musica',
                           'disco', 'bar', 'pub', 'entretenimiento', 'show', 'concierto',
                           'boliche', 'after', 'recital', 'streaming'],
                'weight': 1.5
            },
            'Salud': {
                'keywords': ['farmacia', 'medico', 'doctor', 'hospital', 'clinica', 'medicamento',
                           'dentista', 'oculista', 'laboratorio', 'analisis', 'consulta',
                           'farmacity', 'dr', 'salud', 'kinesiologo', 'psicologo', 'traumatologo',
                           'dermatologo', 'ginecologo', 'cardiologo', 'pediatra'],
                'weight': 2.2
            },
            'Ropa': {
                'keywords': ['ropa', 'vestimenta', 'zapatillas', 'zapatos', 'camisa', 'pantalon',
                           'vestido', 'nike', 'adidas', 'zara', 'h&m', 'forever21', 'moda',
                           'jean', 'campera', 'remera', 'buzo', 'pollera', 'calzado',
                           'underwear', 'medias', 'abrigo'],
                'weight': 1.7
            },
            'Hogar': {
                'keywords': ['muebles', 'decoracion', 'ferreteria', 'limpieza', 'detergente',
                           'easy', 'sodimac', 'hogar', 'casa', 'electrodomestico', 'ikea',
                           'bazar', 'cocina', 'baño', 'living', 'dormitorio', 'jardin',
                           'pintura', 'herramientas', 'bombita', 'lampara'],
                'weight': 1.6
            },
            'Educación': {
                'keywords': ['universidad', 'curso', 'libro', 'escuela', 'colegio', 'capacitacion',
                           'educacion', 'estudio', 'academia', 'instituto', 'idiomas',
                           'utn', 'uba', 'udemy', 'coursera', 'maestria', 'doctorado'],
                'weight': 1.9
            }
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocesar texto para mejorar la categorización
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remover caracteres especiales pero mantener espacios
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remover números (opcional, puede ser útil mantenerlos)
        # text = re.sub(r'\d+', '', text)
        
        # Remover espacios extra
        text = re.sub(r'\s+', ' ', text)
        
        # Remover stop words
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def create_enhanced_features(self, descriptions: List[str]) -> np.ndarray:
        """
        Crear características mejoradas combinando TF-IDF con palabras clave
        """
        # Preprocesar descripciones
        processed_descriptions = [self.preprocess_text(desc) for desc in descriptions]
        
        # Características TF-IDF
        tfidf_features = self.vectorizer.fit_transform(processed_descriptions)
        
        # Características basadas en palabras clave
        keyword_features = []
        
        for desc in processed_descriptions:
            features = []
            for category, info in self.category_keywords.items():
                # Contar matches de palabras clave
                matches = sum(1 for keyword in info['keywords'] if keyword in desc)
                # Aplicar peso
                weighted_score = matches * info['weight']
                features.append(weighted_score)
            
            keyword_features.append(features)
        
        keyword_features = np.array(keyword_features)
        
        # Combinar características
        if hasattr(tfidf_features, 'toarray'):
            tfidf_array = tfidf_features.toarray()
        else:
            tfidf_array = tfidf_features
        
        combined_features = np.hstack([tfidf_array, keyword_features])
        
        return combined_features
    
    def generate_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generar datos de entrenamiento usando reglas predefinidas y datos existentes
        """
        logger.info("Generando datos de entrenamiento...")
        
        training_data = []
        
        # Usar categorización por palabras clave como base
        for _, row in df.iterrows():
            description = row['descripcion']
            processed_desc = self.preprocess_text(description)
            
            # Encontrar la mejor categoría basada en palabras clave
            best_category = 'Otros'
            best_score = 0
            
            for category, info in self.category_keywords.items():
                score = 0
                for keyword in info['keywords']:
                    if keyword in processed_desc:
                        score += info['weight']
                
                if score > best_score:
                    best_score = score
                    best_category = category
            
            training_data.append({
                'descripcion': description,
                'categoria': best_category,
                'confidence': best_score
            })
        
        training_df = pd.DataFrame(training_data)
        
        # Filtrar solo casos con alta confianza para entrenamiento
        high_confidence = training_df[training_df['confidence'] > 1.0]
        
        logger.info(f"Datos de entrenamiento generados: {len(high_confidence)} casos de alta confianza")
        
        return high_confidence
    
    def train(self, df: pd.DataFrame, target_column: str = None) -> Dict:
        """
        Entrenar el modelo de categorización
        """
        logger.info("Iniciando entrenamiento del modelo...")
        
        if target_column and target_column in df.columns:
            # Usar categorías existentes si están disponibles
            training_data = df[['descripcion', target_column]].copy()
            training_data = training_data.rename(columns={target_column: 'categoria'})
        else:
            # Generar datos de entrenamiento automáticamente
            training_data = self.generate_training_data(df)
        
        # Remover filas con categorías faltantes
        training_data = training_data.dropna(subset=['descripcion', 'categoria'])
        
        if len(training_data) < 10:
            logger.warning("Pocos datos de entrenamiento disponibles. Usando categorización por reglas.")
            self.is_trained = False
            return {'status': 'rule_based', 'samples': len(training_data)}
        
        # Preparar características
        X = self.create_enhanced_features(training_data['descripcion'].tolist())
        
        # Preparar etiquetas
        y = self.label_encoder.fit_transform(training_data['categoria'])
        
        # Dividir datos
        if len(training_data) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Entrenar modelo
        self.classifier.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Obtener importancia de características
        feature_importance = self.classifier.feature_importances_
        
        self.is_trained = True
        
        training_results = {
            'status': 'trained',
            'accuracy': accuracy,
            'samples': len(training_data),
            'categories': self.label_encoder.classes_.tolist(),
            'feature_importance_mean': np.mean(feature_importance)
        }
        
        logger.info(f"Modelo entrenado - Accuracy: {accuracy:.3f}, Muestras: {len(training_data)}")
        
        return training_results
    
    def predict(self, descriptions: List[str]) -> List[str]:
        """
        Predecir categorías para nuevas descripciones
        """
        if not descriptions:
            return []
        
        if not self.is_trained:
            # Usar categorización por reglas si el modelo no está entrenado
            return self._rule_based_prediction(descriptions)
        
        try:
            # Crear características
            X = self.create_enhanced_features(descriptions)
            
            # Predecir
            y_pred = self.classifier.predict(X)
            
            # Convertir de vuelta a etiquetas
            categories = self.label_encoder.inverse_transform(y_pred)
            
            return categories.tolist()
            
        except Exception as e:
            logger.error(f"Error en predicción ML: {e}")
            return self._rule_based_prediction(descriptions)
    
    def _rule_based_prediction(self, descriptions: List[str]) -> List[str]:
        """
        Categorización basada en reglas como fallback
        """
        categories = []
        
        for desc in descriptions:
            processed_desc = self.preprocess_text(desc)
            
            best_category = 'Otros'
            best_score = 0
            
            for category, info in self.category_keywords.items():
                score = 0
                for keyword in info['keywords']:
                    if keyword in processed_desc:
                        score += info['weight']
                
                if score > best_score:
                    best_score = score
                    best_category = category
            
            categories.append(best_category)
        
        return categories
    
    def predict_with_confidence(self, descriptions: List[str]) -> List[Dict]:
        """
        Predecir categorías con nivel de confianza
        """
        if not descriptions:
            return []
        
        results = []
        
        if self.is_trained:
            try:
                X = self.create_enhanced_features(descriptions)
                y_pred = self.classifier.predict(X)
                y_proba = self.classifier.predict_proba(X)
                
                for i, desc in enumerate(descriptions):
                    predicted_category = self.label_encoder.inverse_transform([y_pred[i]])[0]
                    confidence = np.max(y_proba[i])
                    
                    results.append({
                        'descripcion': desc,
                        'categoria': predicted_category,
                        'confidence': confidence,
                        'method': 'ml'
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"Error en predicción con confianza: {e}")
        
        # Fallback a reglas
        for desc in descriptions:
            processed_desc = self.preprocess_text(desc)
            
            best_category = 'Otros'
            best_score = 0
            total_possible_score = sum(info['weight'] * len(info['keywords']) 
                                     for info in self.category_keywords.values())
            
            for category, info in self.category_keywords.items():
                score = 0
                for keyword in info['keywords']:
                    if keyword in processed_desc:
                        score += info['weight']
                
                if score > best_score:
                    best_score = score
                    best_category = category
            
            # Calcular confianza normalizada
            confidence = min(best_score / 5.0, 1.0) if best_score > 0 else 0.1
            
            results.append({
                'descripcion': desc,
                'categoria': best_category,
                'confidence': confidence,
                'method': 'rules'
            })
        
        return results
    
    def save_model(self, model_path: str):
        """
        Guardar modelo entrenado
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained,
            'category_keywords': self.category_keywords
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Modelo guardado en: {model_path}")
    
    def load_model(self, model_path: str):
        """
        Cargar modelo previamente entrenado
        """
        try:
            model_data = joblib.load(model_path)
            
            self.vectorizer = model_data['vectorizer']
            self.classifier = model_data['classifier']
            self.label_encoder = model_data['label_encoder']
            self.is_trained = model_data['is_trained']
            
            if 'category_keywords' in model_data:
                self.category_keywords = model_data['category_keywords']
            
            logger.info(f"Modelo cargado desde: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return False
    
    def evaluate_categorization(self, df: pd.DataFrame, true_column: str) -> Dict:
        """
        Evaluar la calidad de la categorización
        """
        if true_column not in df.columns:
            raise ValueError(f"Columna {true_column} no encontrada")
        
        descriptions = df['descripcion'].tolist()
        true_categories = df[true_column].tolist()
        
        predicted_categories = self.predict(descriptions)
        
        # Calcular métricas
        accuracy = accuracy_score(true_categories, predicted_categories)
        
        # Reporte detallado
        report = classification_report(true_categories, predicted_categories, 
                                     output_dict=True, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': list(zip(descriptions, true_categories, predicted_categories))
        }
    
    def get_category_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Obtener distribución de categorías
        """
        if 'categoria' not in df.columns:
            # Predecir categorías si no existen
            descriptions = df['descripcion'].tolist()
            categories = self.predict(descriptions)
            df_temp = df.copy()
            df_temp['categoria'] = categories
        else:
            df_temp = df
        
        distribution = df_temp['categoria'].value_counts().to_dict()
        percentages = df_temp['categoria'].value_counts(normalize=True).to_dict()
        
        return {
            'counts': distribution,
            'percentages': {k: v*100 for k, v in percentages.items()}
        }

class CategoryTrainer:
    """
    Clase auxiliar para entrenar el categorizador con datos específicos
    """
    
    def __init__(self):
        self.categorizer = IntelligentCategorizer()
    
    def create_sample_training_data(self) -> pd.DataFrame:
        """
        Crear datos de entrenamiento de muestra
        """
        sample_data = [
            # Alimentación
            ('supermercado carrefour compra semanal', 'Alimentación'),
            ('restaurant pizza delivery', 'Alimentación'),
            ('panaderia pan dulce', 'Alimentación'),
            ('mercado frutas verduras', 'Alimentación'),
            ('mcdonalds hamburguesa', 'Alimentación'),
            ('bar cafe cortado', 'Alimentación'),
            ('heladeria cono vainilla', 'Alimentación'),
            ('rotiseria pollo asado', 'Alimentación'),
            ('almacen fiambre queso', 'Alimentación'),
            ('confiteria facturas', 'Alimentación'),
            
            # Transporte
            ('uber viaje centro ciudad', 'Transporte'),
            ('taxi aeropuerto', 'Transporte'),
            ('combustible ypf nafta', 'Transporte'),
            ('peaje autopista', 'Transporte'),
            ('colectivo sube recarga', 'Transporte'),
            ('estacionamiento shopping', 'Transporte'),
            ('remis hospital', 'Transporte'),
            ('tren mitre boleto', 'Transporte'),
            ('micro larga distancia', 'Transporte'),
            ('cochera mensual', 'Transporte'),
            
            # Servicios
            ('factura electricidad edenor', 'Servicios'),
            ('internet fibertel mensual', 'Servicios'),
            ('telefono movistar factura', 'Servicios'),
            ('gas metrogas factura', 'Servicios'),
            ('agua aysa servicio', 'Servicios'),
            ('expensas edificio', 'Servicios'),
            ('cable directv mensual', 'Servicios'),
            ('celular personal plan', 'Servicios'),
            ('wifi claro hogar', 'Servicios'),
            ('servicio tecnico aire', 'Servicios'),
            
            # Entretenimiento
            ('netflix suscripcion mensual', 'Entretenimiento'),
            ('cine entrada pelicula', 'Entretenimiento'),
            ('spotify premium musica', 'Entretenimiento'),
            ('teatro entrada show', 'Entretenimiento'),
            ('steam juego videojuego', 'Entretenimiento'),
            ('disco boliche entrada', 'Entretenimiento'),
            ('recital banda rock', 'Entretenimiento'),
            ('streaming disney plus', 'Entretenimiento'),
            ('bar copas amigos', 'Entretenimiento'),
            ('bowling partida', 'Entretenimiento'),
            
            # Salud
            ('farmacity medicamento', 'Salud'),
            ('medico consulta particular', 'Salud'),
            ('dentista limpieza dental', 'Salud'),
            ('laboratorio analisis sangre', 'Salud'),
            ('hospital clinica', 'Salud'),
            ('kinesiologo sesion', 'Salud'),
            ('oculista lentes contacto', 'Salud'),
            ('psicologo terapia', 'Salud'),
            ('dermatologo consulta', 'Salud'),
            ('farmacia vitaminas', 'Salud'),
            
            # Ropa
            ('zara camisa nueva', 'Ropa'),
            ('nike zapatillas running', 'Ropa'),
            ('h&m pantalon jean', 'Ropa'),
            ('adidas campera deportiva', 'Ropa'),
            ('forever21 vestido', 'Ropa'),
            ('calzado zapatos cuero', 'Ropa'),
            ('underwear ropa interior', 'Ropa'),
            ('medias algodon pack', 'Ropa'),
            ('abrigo invierno', 'Ropa'),
            ('jean pantalon azul', 'Ropa'),
            
            # Hogar
            ('easy ferreteria tornillos', 'Hogar'),
            ('ikea mueble escritorio', 'Hogar'),
            ('detergente limpieza', 'Hogar'),
            ('sodimac pintura pared', 'Hogar'),
            ('bazar cocina utensilios', 'Hogar'),
            ('electrodomestico microondas', 'Hogar'),
            ('lampara living led', 'Hogar'),
            ('herramientas destornillador', 'Hogar'),
            ('jardin plantas macetas', 'Hogar'),
            ('bombita luz led', 'Hogar'),
            
            # Educación
            ('universidad cuota mensual', 'Educación'),
            ('curso online udemy', 'Educación'),
            ('libro estudio amazon', 'Educación'),
            ('instituto idiomas', 'Educación'),
            ('utn matricula', 'Educación'),
            ('academia computacion', 'Educación'),
            ('capacitacion laboral', 'Educación'),
            ('maestria posgrado', 'Educación'),
            ('coursera certificacion', 'Educación'),
            ('material estudio', 'Educación')
        ]
        
        return pd.DataFrame(sample_data, columns=['descripcion', 'categoria'])
    
    def train_with_sample_data(self) -> Dict:
        """
        Entrenar el modelo con datos de muestra
        """
        sample_df = self.create_sample_training_data()
        results = self.categorizer.train(sample_df, 'categoria')
        return results
    
    def improve_with_user_data(self, df: pd.DataFrame) -> Dict:
        """
        Mejorar el modelo con datos del usuario
        """
        # Combinar datos de muestra con datos del usuario
        sample_df = self.create_sample_training_data()
        
        # Si el usuario ya tiene categorías, usarlas
        if 'categoria' in df.columns:
            user_training = df[['descripcion', 'categoria']].copy()
            combined_df = pd.concat([sample_df, user_training], ignore_index=True)
        else:
            combined_df = sample_df
        
        results = self.categorizer.train(combined_df, 'categoria')
        return results

# Funciones de utilidad
def categorize_expenses_smart(df: pd.DataFrame, model_path: str = None) -> pd.DataFrame:
    """
    Función de conveniencia para categorizar gastos inteligentemente
    """
    categorizer = IntelligentCategorizer()
    
    # Cargar modelo si existe
    if model_path and Path(model_path).exists():
        categorizer.load_model(model_path)
    else:
        # Entrenar con datos actuales
        trainer = CategoryTrainer()
        if 'categoria' in df.columns:
            trainer.improve_with_user_data(df)
        else:
            trainer.train_with_sample_data()
        categorizer = trainer.categorizer
    
    # Categorizar
    descriptions = df['descripcion'].tolist()
    categories = categorizer.predict(descriptions)
    
    df_result = df.copy()
    df_result['categoria'] = categories
    
    return df_result

if __name__ == "__main__":
    # Ejemplo de uso
    print("🤖 Categorizador Inteligente - Analizador de Gastos IA")
    
    # Crear datos de ejemplo
    sample_expenses = pd.DataFrame({
        'descripcion': [
            'supermercado carrefour compra',
            'uber viaje trabajo',
            'netflix suscripcion',
            'farmacia aspirinas',
            'restaurant italiano cena',
            'combustible ypf ruta',
            'gym cuota mensual',
            'zara camisa nueva',
            'electricidad edenor factura',
            'cine entrada avengers'
        ]
    })
    
    # Entrenar y categorizar
    trainer = CategoryTrainer()
    print("\n📚 Entrenando modelo...")
    results = trainer.train_with_sample_data()
    print(f"✅ Entrenamiento completado: {results['status']}")
    
    if results['status'] == 'trained':
        print(f"   Accuracy: {results['accuracy']:.3f}")
        print(f"   Muestras: {results['samples']}")
        print(f"   Categorías: {results['categories']}")
    
    # Categorizar gastos
    print("\n🏷️ Categorizando gastos...")
    predictions = trainer.categorizer.predict_with_confidence(
        sample_expenses['descripcion'].tolist()
    )
    
    print("\n📊 Resultados:")
    for pred in predictions:
        confidence_emoji = "🟢" if pred['confidence'] > 0.7 else "🟡" if pred['confidence'] > 0.4 else "🔴"
        print(f"   {confidence_emoji} '{pred['descripcion']}' -> {pred['categoria']} "
              f"(confianza: {pred['confidence']:.2f}, método: {pred['method']})")
    
    # Guardar modelo
    Path('models').mkdir(exist_ok=True)
    trainer.categorizer.save_model('models/categorizer_model.pkl')
    print(f"\n💾 Modelo guardado en: models/categorizer_model.pkl")
    
    # Estadísticas de distribución
    sample_expenses['categoria'] = [p['categoria'] for p in predictions]
    distribution = trainer.categorizer.get_category_distribution(sample_expenses)
    
    print(f"\n📈 Distribución de categorías:")
    for categoria, count in distribution['counts'].items():
        percentage = distribution['percentages'][categoria]
        print(f"   {categoria}: {count} gastos ({percentage:.1f}%)")


class ImprovedIntelligentCategorizer(IntelligentCategorizer):
    """
    Versión mejorada del categorizador que soluciona el problema de features
    """
    
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.feature_names = None
        self.model_metadata = {}
        
        # Configuración mejorada del vectorizador
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # Reducido para evitar el error de features
            stop_words=None,
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,  # Palabras deben aparecer al menos 2 veces
            max_df=0.95,  # Ignorar palabras muy frecuentes
            sublinear_tf=True
        )
        
        # Clasificador mejorado
        self.classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'  # Para datos desbalanceados
        )
    
    def create_consistent_features(self, descriptions: List[str], fit: bool = False) -> np.ndarray:
        """
        Crear características de manera consistente - SOLUCIONA EL ERROR DE FEATURES
        """
        # Preprocesar descripciones
        processed_descriptions = [self.preprocess_text(desc) for desc in descriptions]
        
        if fit:
            # Durante entrenamiento: fit el vectorizador
            tfidf_features = self.vectorizer.fit_transform(processed_descriptions)
        else:
            # Durante predicción: solo transform
            tfidf_features = self.vectorizer.transform(processed_descriptions)
        
        # Convertir a array denso
        if hasattr(tfidf_features, 'toarray'):
            tfidf_features = tfidf_features.toarray()
        
        # Características de palabras clave (siempre mismo tamaño)
        keyword_features = []
        for desc in processed_descriptions:
            features = []
            for category, info in self.category_keywords.items():
                matches = sum(1 for keyword in info['keywords'] if keyword in desc)
                weighted_score = matches * info['weight']
                features.append(weighted_score)
            keyword_features.append(features)
        
        keyword_features = np.array(keyword_features)
        
        # Combinar características
        combined_features = np.hstack([tfidf_features, keyword_features])
        
        # Guardar nombres de características para debugging
        if fit:
            self.feature_names = (
                [f"tfidf_{i}" for i in range(tfidf_features.shape[1])] +
                [f"keyword_{cat}" for cat in self.category_keywords.keys()]
            )
            logger.info(f"Features creadas: {combined_features.shape[1]} total")
        
        return combined_features
    
    def train_improved(self, df: pd.DataFrame, target_column: str = None) -> Dict:
        """
        Entrenamiento mejorado que soluciona problemas de features
        """
        logger.info("🚀 Iniciando entrenamiento mejorado...")
        
        if target_column and target_column in df.columns:
            training_data = df[['descripcion', target_column]].copy()
            training_data = training_data.rename(columns={target_column: 'categoria'})
        else:
            training_data = self.generate_training_data(df)
        
        training_data = training_data.dropna(subset=['descripcion', 'categoria'])
        
        if len(training_data) < 10:
            logger.warning("Pocos datos de entrenamiento. Usando método de reglas.")
            self.is_trained = False
            return {'status': 'rule_based', 'samples': len(training_data)}
        
        # Preparar datos
        X_descriptions = training_data['descripcion'].tolist()
        y = self.label_encoder.fit_transform(training_data['categoria'])
        
        # Crear características de forma consistente
        X = self.create_consistent_features(X_descriptions, fit=True)
        
        # Dividir datos si hay suficientes
        if len(training_data) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Entrenar modelo
        start_time = time.time()
        self.classifier.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluar
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Guardar metadata
        self.model_metadata = {
            'training_date': datetime.now().isoformat(),
            'training_samples': len(training_data),
            'feature_count': X.shape[1],
            'categories': self.label_encoder.classes_.tolist(),
            'accuracy': accuracy,
            'training_time': training_time
        }
        
        self.is_trained = True
        
        results = {
            'status': 'trained_improved',
            'accuracy': accuracy,
            'samples': len(training_data),
            'features': X.shape[1],
            'categories': self.label_encoder.classes_.tolist(),
            'training_time': training_time
        }
        
        logger.info(f"✅ Entrenamiento completado - Accuracy: {accuracy:.3f}, Features: {X.shape[1]}")
        
        return results
    
    def predict_improved(self, descriptions: List[str]) -> List[str]:
        """
        Predicción mejorada que usa features consistentes
        """
        if not descriptions:
            return []
        
        if not self.is_trained:
            return self._rule_based_prediction(descriptions)
        
        try:
            # Crear características consistentes (sin fit)
            X = self.create_consistent_features(descriptions, fit=False)
            
            # Predecir
            y_pred = self.classifier.predict(X)
            
            # Convertir a categorías
            categories = self.label_encoder.inverse_transform(y_pred)
            
            return categories.tolist()
            
        except Exception as e:
            logger.error(f"Error en predicción mejorada: {e}")
            return self._rule_based_prediction(descriptions)
    
    def predict_with_confidence_improved(self, descriptions: List[str]) -> List[Dict]:
        """
        Predicción con confianza mejorada
        """
        if not descriptions:
            return []
        
        results = []
        
        if self.is_trained:
            try:
                X = self.create_consistent_features(descriptions, fit=False)
                y_pred = self.classifier.predict(X)
                y_proba = self.classifier.predict_proba(X)
                
                for i, desc in enumerate(descriptions):
                    predicted_category = self.label_encoder.inverse_transform([y_pred[i]])[0]
                    confidence = np.max(y_proba[i])
                    
                    results.append({
                        'descripcion': desc,
                        'categoria': predicted_category,
                        'confidence': confidence,
                        'method': 'ml_improved',
                        'feature_count': X.shape[1]
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"Error en predicción con confianza: {e}")
        
        # Fallback a reglas
        for desc in descriptions:
            processed_desc = self.preprocess_text(desc)
            
            best_category = 'Otros'
            best_score = 0
            
            for category, info in self.category_keywords.items():
                score = sum(info['weight'] for keyword in info['keywords'] if keyword in processed_desc)
                if score > best_score:
                    best_score = score
                    best_category = category
            
            confidence = min(best_score / 5.0, 1.0) if best_score > 0 else 0.1
            
            results.append({
                'descripcion': desc,
                'categoria': best_category,
                'confidence': confidence,
                'method': 'rules_fallback'
            })
        
        return results        
    


class PredictionCache:
    """Caché simple para predicciones"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, description: str) -> Optional[Dict]:
        key = description.lower().strip()
        if key in self.cache:
            self.hits += 1
            return self.cache[key].copy()
        self.misses += 1
        return None
    
    def set(self, description: str, prediction: Dict):
        if len(self.cache) >= self.max_size:
            # Remover el más antiguo (simple)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = description.lower().strip()
        self.cache[key] = prediction.copy()
    
    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0
        }    
    

class OptimizedCategorizer:
    """Categorizador optimizado con caché"""
    
    def __init__(self, base_categorizer: IntelligentCategorizer = None):
        self.base_categorizer = base_categorizer or ImprovedIntelligentCategorizer()
        self.cache = PredictionCache()
        self.performance_stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }
    
    def predict_single_cached(self, description: str) -> Dict:
        """Predicción individual con caché"""
        start_time = time.time()
        
        # Verificar caché
        cached_result = self.cache.get(description)
        if cached_result:
            self.performance_stats['cache_hits'] += 1
            self.performance_stats['total_predictions'] += 1
            return cached_result
        
        # Predicción nueva
        if hasattr(self.base_categorizer, 'predict_with_confidence_improved'):
            result = self.base_categorizer.predict_with_confidence_improved([description])[0]
        else:
            result = self.base_categorizer.predict_with_confidence([description])[0]
        
        # Guardar en caché
        self.cache.set(description, result)
        
        # Estadísticas
        self.performance_stats['total_predictions'] += 1
        self.performance_stats['total_time'] += time.time() - start_time
        
        return result
    
    def get_performance_stats(self) -> Dict:
        stats = self.performance_stats.copy()
        cache_stats = self.cache.get_stats()
        
        if stats['total_predictions'] > 0:
            stats['avg_time_per_prediction'] = stats['total_time'] / stats['total_predictions']
        
        stats['cache_stats'] = cache_stats
        return stats

# 5. FUNCIÓN DE UTILIDAD MEJORADA:

def categorize_expenses_improved(df: pd.DataFrame, model_path: str = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Función mejorada para categorizar gastos con mejor rendimiento
    """
    # Usar categorizador mejorado
    base_categorizer = ImprovedIntelligentCategorizer()
    
    # Cargar modelo si existe
    if model_path and Path(model_path).exists():
        if base_categorizer.load_model(model_path):
            logger.info(f"✅ Modelo cargado desde: {model_path}")
        else:
            logger.warning("❌ Error cargando modelo, entrenando nuevo")
            base_categorizer.train_improved(df)
    else:
        # Entrenar nuevo modelo
        results = base_categorizer.train_improved(df)
        logger.info(f"✅ Nuevo modelo entrenado: {results['status']}")
    
    # Usar categorizador optimizado si se requiere caché
    if use_cache:
        categorizer = OptimizedCategorizer(base_categorizer)
        descriptions = df['descripcion'].tolist()
        predictions = [categorizer.predict_single_cached(desc) for desc in descriptions]
        
        # Mostrar estadísticas de caché
        stats = categorizer.get_performance_stats()
        logger.info(f"📊 Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
    else:
        descriptions = df['descripcion'].tolist()
        if hasattr(base_categorizer, 'predict_with_confidence_improved'):
            predictions = base_categorizer.predict_with_confidence_improved(descriptions)
        else:
            predictions = base_categorizer.predict_with_confidence(descriptions)
    
    # Preparar resultado
    df_result = df.copy()
    df_result['categoria'] = [p['categoria'] for p in predictions]
    df_result['confidence'] = [p['confidence'] for p in predictions]
    df_result['method'] = [p['method'] for p in predictions]
    
    return df_result    