# U. Internacional de Aguascalientes
# Doctorado en Tecnologías de la Transformación Digital
# Materia: Estadística
# Tutor: Dr. Jonás Velasco Álvarez
# Alumno: Luis Alejandro Santana Valadez
# Trabajo: Análisis de Regresión Simple
# -------------------------------------------------------------------------------------------------------
# Caso de Análisis de Deportes (Béisbol)
# - Análisis de la relación de bateos de jugadores de equipos de béisbol y el número de runs producidas.
# - Se aplica el modelo de regresión lineal simple con python
# - Se calculan las variables y coeficientes del modelo para preparar las predicciones
# - Se generan predicciones a traves del modelo de regresión lineal
# - Se aplican métricas para evaluar la efectividad del modelo (varianza ANOVA, R², R² ajustada y MSE)
# - Se grafica el modelo de regresión lineal y las métricas

import math
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# ----------------------------------------------------------
# (1) - Planteamiento, datos base, variables y coeficientes
# ----------------------------------------------------------

# Datos del archivo deportes.csv
# Lectura del archivo CSV
df = pd.read_csv('deportes.csv')

# Definición de variables del modelo
# Variable dependiente (Y): Runs (dato que se quiere predecir).
# Variable independiente (X): Bateos (dato que afecta el resultado de predicción).
X = df[['Bateos']]  # Variable independiente
y = df['Runs']      # Variable dependiente

# ----------------------------------------------------------------------
# (2) - Aplicar regresión lineal, cálculo de variables y coeficientes
# ----------------------------------------------------------------------
model = LinearRegression()
model.fit(X, y)
# Variable de predicción
# Cálculo de variables ajustadas y los residuales por la predicción
y_pred = model.predict(X)       # Variables ajustadas
residuales = y - y_pred         # Residuales (diferencia en datos observados y datos ajustados)
print(f"Datos ajustados: {y_pred}")
print(f"Residuales: {residuales}")

# Mostrar coeficientes
print(f"Intercepto (b0): {model.intercept_:.3f}")
print(f"Coeficiente (b1): {model.coef_[0]:.3f}")
# La ecuación principal Y = b0 + b1*X
# Runs = b0 + b1 × Bateos
print("La ecuación de predicción resultante es: Runs = -2789.243 + 0.631 * Bateos")

# -----------------------------------------------------------------------------------------
# Con la ecuación de predicción, ya se pueden hacer ejercicios de predicción para saber 
# quiene puede ganar un partido si se enfrentan dos equipos
# -----------------------------------------------------------------------------------------
# Ejemplo de predicción:
# Partido: Texas (5659 bateos) vs. Kansas (5672)
runsTexas = model.intercept_ + model.coef_[0] * 5659
runsKansas = model.intercept_ + model.coef_[0] * 5672
if runsTexas > runsKansas:
    print("Predicción: Texas ganará el partido por la predicción de bateos")
else:
    print("Predicción: Kansas ganará el partido por la predicción de bateos")
   

# ---------------------------------------------------------------
# (3) - Cálculo de las métricas R² y MSE para evaluar el modelo.
# ---------------------------------------------------------------

# Análisis de Varianza (ANOVA)
X_const = sm.add_constant(X)
# Modelo de regresión con fórmula
modelo_formula = smf.ols('Runs ~ Bateos', data=df).fit()
# ANOVA
anova_table = sm.stats.anova_lm(modelo_formula, typ=2)
print("ANÁLISIS DE VARIANZA (ANOVA):")
print(anova_table)

# Calcular R²
r2 = r2_score(y, y_pred)
print(f"Coeficiente de determinación R^2: {r2:.4f}")

# R² ajustado
print("\nR² ajustado:", modelo_formula.rsquared_adj)
# Resumen del modelo
print("\nResumen del modelo:")
print(modelo_formula.summary())

# Calcular el Error Cuadrático Medio (MSE)
mse = mean_squared_error(y, y_pred)
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
r2mse = math.sqrt(mse)
print(f"Raiz cuadrada de MSE: {r2mse:.2f}")

# ---------------




# -----------------------------------------------------------------------------------
# (4) - Graficación del modelo de regresión lineal y gráficas de residuos y métricas
# -----------------------------------------------------------------------------------

# Grafica de regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_pred, color='red', label='Línea de regresión')
plt.title('Relación entre Bateos y Runs')
plt.xlabel('Bateos')
plt.ylabel('Runs')
plt.legend()
plt.grid(True)
plt.show()

# gráfica de residuos 
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuales, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Valores de predicción (Runs)')
plt.ylabel('Residuales')
plt.title('Gráfico de Residuales')
plt.grid(True)
plt.show()

# Histograma de residuales
plt.figure(figsize=(8,5))
sns.histplot(residuales, kde=True, color='green', bins=10)
plt.title('Histograma de Residuales')
plt.xlabel('Residuales')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Gráfica: ANOVA (Suma de cuadrados)
anova_table[['sum_sq']].plot(kind='bar', legend=False)
plt.title("Suma de cuadrados - ANOVA")
plt.ylabel("Suma de cuadrados")
plt.grid(True)
plt.tight_layout()
plt.show()