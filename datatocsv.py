import ast
import pandas as pd

# Leer el archivo .txt
with open("data.txt", "r") as file:
    content = file.read()

# Convertir texto a diccionario
data_dict = ast.literal_eval(content.strip().split("=", 1)[1].strip())

# Crear DataFrame
df = pd.DataFrame(data_dict)

# Guardar a CSV
df.to_csv("datos_publicidad_ventas.csv", index=False)
print("Archivo CSV generado: datos_publicidad_ventas.csv")