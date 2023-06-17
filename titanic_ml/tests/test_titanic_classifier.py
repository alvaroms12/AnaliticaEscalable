import pytest
import pandas as pd
from sklearn.datasets import load_iris

from titanic_ml.titanic_classifier import TitanicClassifier

@pytest.fixture
def titanic_classifier():
    return TitanicClassifier()

def test_fit_and_predict(titanic_classifier):
    # Cargar los datos del archivo CSV
    df = pd.read_csv('titanic.csv', sep=";")
    X = df.drop('survived', axis=1)
    y = df['survived']

    # Ajustar el clasificador
    titanic_classifier.fit(X, y)

    # Realizar predicciones
    predictions = titanic_classifier.predict(X)

    # Convertir las predicciones a una lista
    predictions = predictions.tolist()

    # Verificar que las predicciones sean una lista
    assert isinstance(predictions, list)

    # Verificar que las predicciones tengan la misma longitud que los datos de entrada
    assert len(predictions) == len(X)

    # Comparar las predicciones con las etiquetas reales
    correct_predictions = sum(predictions == y)
    total_examples = len(y)
    accuracy = correct_predictions / total_examples

    # Verificar que la precisión sea mayor a un umbral determinado (por ejemplo, 0.8)
    assert accuracy > 0.8
