# Imagen base de Python
FROM python:3.9

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY requirements.txt .
COPY app.py .
COPY titanic.csv .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 5000
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]

