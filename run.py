import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Generar un dataset de juguete
X, y = make_classification(
    n_classes=2, 
    class_sep=1.5, 
    weights=[0.8, 0.2], 
    n_informative=2, 
    n_redundant=0, 
    flip_y=0.05, 
    n_features=2, 
    n_clusters_per_class=1, 
    n_samples=1000, 
    random_state=42
)

# Visualizar los datos originales
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Clase mayoritaria", alpha=0.6)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Clase minoritaria", alpha=0.6, color='red')
plt.title("Datos originales")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Aplicar Enlaces de Tomek
tomek = TomekLinks()
X_res, y_res = tomek.fit_resample(X, y)

# Visualizar los datos después de eliminar los enlaces de Tomek
plt.figure(figsize=(8, 6))
plt.scatter(X_res[y_res == 0][:, 0], X_res[y_res == 0][:, 1], label="Clase mayoritaria", alpha=0.6)
plt.scatter(X_res[y_res == 1][:, 0], X_res[y_res == 1][:, 1], label="Clase minoritaria", alpha=0.6, color='red')
plt.title("Datos después de eliminar enlaces de Tomek")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Entrenar un modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Predecir y evaluar el modelo
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))