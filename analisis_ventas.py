# U. Internacional de Aguascalientes
# Doctorado en Tecnologías de la Transformación Digital
# Materia: Estadística
# Tutor: Dr. Jonás Velasco Álvarez
# Alumno: Luis Alejandro Santana Valadez
# Trabajo: Análisis de Regresión Múltiple
# -------------------------------------------------------------------------------------------------------
# Caso de Análisis de Ventas
# - Análisis de la relación de la publicidad utilizada en canales (tx, radio, periódico) en las ventas de 200 regiones.
# - Se aplica el modelo de regresión lineal múltiple con python
# - Se calculan las variables y coeficientes del modelo para preparar las predicciones
# - Se generan predicciones a traves del modelo de regresión lineal
# - Se aplican métricas para evaluar la efectividad del modelo (varianza ANOVA, R², R² ajustada y MSE)
# - Se grafica el modelo de regresión lineal y las métricas

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------
# (1) - Planteamiento, datos base, variables y coeficientes
# ----------------------------------------------------------
# Almacén de datos (tv, radio, periodico, ventas)
df = pd.read_csv('publicidad_ventas.csv')

# Definición de variables del modelo
# Separar variables independientes (X) y dependiente (Y)
X = df[['tv', 'radio', 'periodico']]    # Variables independientes
Y = df['ventas']                        # Variable dependiente

# ----------------------------------------------------------------------
# (2) - Aplicar regresión lineal, cálculo de variables y coeficientes
# ----------------------------------------------------------------------
# Ajustar el modelo
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
# Variable de predicción
# Cálculo de variables ajustadas y los residuales por la predicción
Y_pred = lin_reg.predict(X)
print(f"Datos ajustados: {Y_pred}")
# Calcular residuos
residuales = Y - Y_pred

# Coeficientes y ecuación del modelo
intercept = lin_reg.intercept_      
coefficients = lin_reg.coef_        # Se generan los 3 coeficientes b1, b2 y b3

# Mostrar coeficientes
print(f"Intercepto (b0): {lin_reg.intercept_:.4f}")
print(f"Coeficiente (b1): {lin_reg.coef_[0]:.4f}")
print(f"Coeficiente (b2): {lin_reg.coef_[1]:.4f}")
print(f"Coeficiente (b3): {lin_reg.coef_[2]:.4f}")

# Añadir constante para modelo OLS
X_ols = sm.add_constant(X)
modelo_ols = sm.OLS(Y, X_ols).fit()

# Ecuación del modelo
# La ecuación principal Y = b0 + b1*X + b2*X + b3*X
# ventas = b0 + b1 × tv + b2 × radio + b3 × periodico
# ventas = 2.9389 + 0.0458*tv + 0.1885*radio − 0.0010*periodico
equation = f"ventas = {intercept:.4f} + ({coefficients[0]:.4f} * tv) + ({coefficients[1]:.4f} * radio) + ({coefficients[2]:.4f} * periodico)"
print(equation)


# -----------------------------------------------------------------------------------------
# Con la ecuación de predicción, ya se pueden hacer ejercicios de predicción para saber 
# cuál es la influencia de la publicidad de un canal en las ventas
# -----------------------------------------------------------------------------------------
# Ejemplo de predicción:

# Una empresa desea saber cuánto podrían aumentar sus ventas si decide incrementar el 
# presupuesto en publicidad de TV. Se usará el modelo de regresión múltiple generado previamente:

# Predecir las ventas con el siguiente presupuesto:
# TV:        250 millones
# Radio:     20 millones
# Periódico: 30 millones
# ventas = 2.9389 + 0.0458 * tv + 0.1885 * radio − 0.0010 * periodico
#        = 2.9389 + 11.45 + 3.77 − 0.03
#        = 18.13 millones en ventas

# ---------------------------------------------------------------
# (3) - Cálculo de las métricas R² y MSE para evaluar el modelo.
# ---------------------------------------------------------------

# la salida del modelo incluye estadísticas R cuadrado múltiple y el error estándar residual.

# Calcular el Error Cuadrático Medio (MSE)
mse = mean_squared_error(Y, Y_pred)
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
r2mse = math.sqrt(mse)
print(f"Raiz cuadrada de MSE: {r2mse:.2f}")

# === DISTANCIA DE COOK ===
influencia = modelo_ols.get_influence()
cooks_d = influencia.cooks_distance[0]

# Calcular umbral sugerido
n = len(df)
umbral_cook = 4 / n

# Crear DataFrame con distancias de Cook
df_cook = pd.DataFrame({
    "Índice": np.arange(n),
    "Distancia_Cook": cooks_d
})

# Filtrar las observaciones influyentes
influencias = df_cook[df_cook["Distancia_Cook"] > umbral_cook].sort_values(by="Distancia_Cook", ascending=False)

# Mostrar resultados
print(f"\nObservaciones con Distancia de Cook mayor a {umbral_cook:.4f} (umbral 4/n):\n")
print(influencias)


# === MULTICOLINEALIDAD - VIF ===
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("=== FACTOR DE INFLACIÓN DE VARIANZA (VIF) ===")
print(vif_data)

# === R CUADRADO Y R CUADRADO AJUSTADO ===
r2 = modelo_ols.rsquared
r2_adj = modelo_ols.rsquared_adj

print("\nR²: {:.4f}".format(r2))
print("R² ajustado: {:.4f}".format(r2_adj))

# === R CUADRADO MÚLTIPLE Y ERROR ESTÁNDAR RESIDUAL ===
residuales = modelo_ols.resid
mse = np.mean(residuales**2)
rmse = np.sqrt(mse)

print("\nError estándar residual (RMSE): {:.4f}".format(rmse))


# ------------------------------------------------------------------------------------
# (4) - Graficación del modelo de regresión lineal y gráficas de residuos y métricas
# ------------------------------------------------------------------------------------

# Gráfica de dispersión: valores reales vs. predichos
plt.figure(figsize=(8, 6))
sns.scatterplot(x=Y, y=Y_pred, color='blue')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', linestyle='--')
plt.xlabel('Ventas reales')
plt.ylabel('Ventas predichas')
plt.title('Ventas reales vs. Ventas predichas')
plt.grid(True)
plt.tight_layout()
plt.show()

# === NORMALIDAD DE LOS ERRORES ===

# Histograma de residuos
plt.figure(figsize=(6, 5))
# plt.subplot(1, 2, 1)
sns.histplot(residuales, kde=True, color="skyblue")
plt.title("Histograma de residuales (normalidad)")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# Q-Q plot
plt.figure(figsize=(6, 5))
sm.qqplot(residuales, line='45', fit=True)
plt.title("Q-Q plot de residuales")
plt.xlabel("Quantiles teóricos")
plt.ylabel("Quantiles observados")
plt.tight_layout()
plt.show()

# === HOMOCEDASTICIDAD ===
# Residuos vs. valores predichos
plt.figure(figsize=(6, 5))
sns.scatterplot(x=Y_pred, y=residuales, color="purple")
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuales vs. Ventas predichas")
plt.xlabel("Ventas predichas")
plt.ylabel("Residuales")
plt.grid(True)
plt.show()

# Gráfico de barras "R²" y "R² Ajustado
plt.figure(figsize=(5, 4))
plt.bar(["R²", "R² Ajustado"], [r2, r2_adj], color=['blue', 'green'])
plt.title("Coeficientes de determinación")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Gráfica de distancia de Cook
plt.figure(figsize=(10, 4))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.title("Distancia de Cook para cada observación")
plt.xlabel("Índice de observación")
plt.ylabel("Distancia de Cook")
plt.axhline(4 / len(df), color='red', linestyle='--', label="Umbral sugerido: 4/n")
plt.legend()
plt.tight_layout()
plt.show()




