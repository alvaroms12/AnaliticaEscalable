from flask import Flask, jsonify, request
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

app = Flask(__name__)

@app.route('/')
def predict():
    df = pd.read_csv('titanic.csv', sep=";")

    # Imputar los valores faltantes en la columna 'Age' con la media
    imputer = SimpleImputer(strategy='mean')
    df['age'] = imputer.fit_transform(df[['age']])

    # Dividir los datos en características (X) y variable objetivo (y)
    X = df.drop('survived', axis=1)
    y = df['survived']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Definir las columnas numéricas y categóricas
    numeric_features = ['age', 'fare']
    categorical_features = ['sex', 'pclass', 'embarked']

    # Crear transformers para escalado y codificación one-hot
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    # Combinar transformers en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
		
    # Crear el clasificador
    classifier = RandomForestClassifier()

    # Crear el pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', classifier)])
    # Entrenar el pipeline
    pipeline.fit(X_train, y_train)

    # Evaluar el desempeño en el conjunto de prueba
    accuracy = pipeline.score(X_test, y_test)

    # Devolver el resultado en formato JSON
    return jsonify({'accuracy': accuracy})

if __name__ == '__main__':
    app.run(port=8080)

