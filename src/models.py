import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.data_proccers import clean_and_preprocess_data # Importa desde tu otro archivo

def train_and_save_model(df, preprocessor, model_filepath='models/expense_classifier_model.joblib'):
    """
    Entrena un modelo de clasificación de gastos y lo guarda.
    Asume que 'Categoria' es la variable objetivo.
    """
    if df.empty or preprocessor is None:
        print("No hay datos o preprocesador para entrenar el modelo.")
        return None, None

    # Asegurarse de que el preprocessor está 'fitteado' con los datos
    # Esto es crucial para que OneHotEncoder sepa todas las categorías existentes
    # antes de transformar los datos para el entrenamiento.
    X = df[['Monto', 'Categoria', 'Dia_Semana', 'Mes']] # Características para el modelo
    y = df['Categoria'] # Variable objetivo

    # Fitear el preprocessor con todos los datos disponibles
    # Esto asegura que todas las posibles categorías sean aprendidas
    preprocessor.fit(X)
    X_processed = preprocessor.transform(X)

    # Convertir X_processed de sparse matrix a array si es necesario para el modelo
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entrenar el modelo (usamos RandomForestClassifier como ejemplo)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0) # zero_division para evitar warnings

    print(f"Precisión del modelo: {accuracy:.2f}")
    print("Reporte de Clasificación:\n", report)

    # Guardar el modelo entrenado y el preprocessor
    joblib.dump(model, model_filepath)
    joblib.dump(preprocessor, model_filepath.replace('.joblib', '_preprocessor.joblib')) # Guarda también el preprocessor
    print(f"Modelo y preprocesador guardados en '{model_filepath}' y '{model_filepath.replace('.joblib', '_preprocessor.joblib')}'")

    return model, preprocessor

def load_model_and_preprocessor(model_filepath='models/expense_classifier_model.joblib'):
    """
    Carga un modelo y su preprocesador guardados.
    """
    try:
        model = joblib.load(model_filepath)
        preprocessor = joblib.load(model_filepath.replace('.joblib', '_preprocessor.joblib'))
        print(f"Modelo y preprocesador cargados desde '{model_filepath}'")
        return model, preprocessor
    except FileNotFoundError:
        print(f"Error: Modelo o preprocesador no encontrados en {model_filepath}. Entrenando uno nuevo...")
        return None, None
    except Exception as e:
        print(f"Error al cargar el modelo o preprocesador: {e}")
        return None, None

def predict_category(model, preprocessor, monto, fecha, descripcion):
    """
    Realiza una predicción de categoría para un nuevo gasto.
    """
    if model is None or preprocessor is None:
        return "Modelo no disponible. Entrene el modelo primero."

    # Crear un DataFrame con los datos de entrada para el preprocesamiento
    input_df = pd.DataFrame([{
        'Monto': monto,
        'Fecha': pd.to_datetime(fecha),
        'Descripcion': descripcion, # Descripción no se usa directamente en el modelo de ejemplo, pero se incluye para consistencia
        'Categoria': 'Desconocido' # Poner una categoría dummy para que el preprocesador no falle en el OHE
    }])

    # Asegurarse de que las columnas de fecha se generen
    input_df['Dia_Semana'] = input_df['Fecha'].dt.dayofweek
    input_df['Mes'] = input_df['Fecha'].dt.month
    input_df['Anio'] = input_df['Fecha'].dt.year # No usada en preprocessor de ejemplo, pero es una feature potencial

    # Seleccionar solo las características que el preprocesador espera
    features_for_prediction = input_df[['Monto', 'Categoria', 'Dia_Semana', 'Mes']]

    # Transformar los datos de entrada usando el preprocesador fitteado
    transformed_input = preprocessor.transform(features_for_prediction)

    # Convertir a array si es una sparse matrix
    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    # Realizar la predicción
    predicted_category = model.predict(transformed_input)[0]
    return predicted_category