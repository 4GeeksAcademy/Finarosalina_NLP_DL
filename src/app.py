import pandas as pd
import regex as re
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from pickle import dump

url= 'https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv'

total_data = pd.read_csv(url)
total_data.head()

total_data['is_spam'] = total_data['is_spam'].astype(int)

total_data.head()

print(total_data.shape)
print(f"Spam: {len(total_data.loc[total_data.is_spam == 1])}")
print(f"No spam: {len(total_data.loc[total_data.is_spam == 0])}")

total_data = total_data.drop_duplicates()
total_data = total_data.reset_index(inplace = False, drop = True)
total_data.shape

def preprocess_text(text):
    # Eliminar cualquier caracter que no sea una letra (a-z) o un espacio en blanco ( )
    text = re.sub(r'[^a-z ]', " ", text)
    
    # Eliminar espacios en blanco
    text = re.sub(r'\s+[a-zA-Z]\s+', " ", text)
    text = re.sub(r'\^[a-zA-Z]\s+', " ", text)

    # Reducir espacios en blanco múltiples a uno único
    text = re.sub(r'\s+', " ", text.lower())

    # Eliminar tags
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)

    return text.split()

total_data["url"] = total_data["url"].apply(preprocess_text)
total_data.head()

download("wordnet")
lemmatizer = WordNetLemmatizer()

download("stopwords")
stop_words = stopwords.words("english")

def lemmatize_text(words, lemmatizer = lemmatizer):
    tokens = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 3]
    return tokens

total_data["url"] = total_data["url"].apply(lemmatize_text)
total_data.head()

wordcloud = WordCloud(width = 800, height = 800, background_color = "black", max_words = 1000, min_font_size = 20, random_state = 42)\
    .generate(str(total_data["url"]))

fig = plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

tokens_list = total_data["url"]
tokens_list = [" ".join(tokens) for tokens in tokens_list]

vectorizer = TfidfVectorizer(max_features = 5000, max_df = 0.8, min_df = 5)
X = vectorizer.fit_transform(tokens_list).toarray()
y = total_data["is_spam"]

X[:5]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = SVC(kernel = "linear", random_state = 42)
model.fit(X_train, y_train)

# Guardar el modelo entrenado (sin optimización)
dump(model, open("/workspaces/Finarosalina_NLP_DL/models/svc_classifier_linear_42.sav", "wb"))


y_pred = model.predict(X_test)
y_pred

accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

# optimización de parametros para tratar de mejorar el accuracy

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=3)
grid.fit(X_train, y_train)

print("Mejores parámetros:", grid.best_params_)


# Evaluar el modelo con test con mejores parametros

best_model = grid.best_estimator_

# Ahora sí puedes hacer predicciones con él
y_pred_best = best_model.predict(X_test)

# Evaluar el modelo optimizado


print("Resultados con el modelo optimizado:")
print(classification_report(y_test, y_pred_best))
print("Accuracy:", accuracy_score(y_test, y_pred_best))




param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

svc = SVC(class_weight='balanced')
grid = GridSearchCV(svc, param_grid, cv=3, scoring='recall')  #  'f1'  balance entre precision y recall
grid.fit(X_train, y_train)

print("Mejores parámetros:", grid.best_params_)

# Evaluación
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, scoring='precision', cv=3)
grid.fit(X_train, y_train)
print("Mejores parámetros:", grid.best_params_)


param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, scoring='precision', cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Mejores parámetros:", grid.best_params_)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


from imblearn.over_sampling import SMOTE

# 1. Split de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Aplicar SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 3. Entrenar modelo SVM
model = SVC(kernel="linear", random_state=42)
model.fit(X_train_smote, y_train_smote)

# 4. Predecir y evaluar
y_pred = model.predict(X_test)

# 5. Resultados
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print("Accuracy:", accuracy_score(y_test, y_pred))

from imblearn.over_sampling import SMOTE

# 1. Split de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# 2. Aplicar SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 3. Entrenar modelo SVM
model = SVC(kernel="linear", random_state=42)
model.fit(X_train_smote, y_train_smote)

# 4. Predecir y evaluar
y_pred = model.predict(X_test)

# 5. Resultados
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print("Accuracy:", accuracy_score(y_test, y_pred))

from joblib import load

# Cargar el modelo guardado
model_loaded = load("/workspaces/Finarosalina_NLP_DL/models/svc_classifier_linear_42.sav")

# Evaluar el modelo cargado
y_pred = model_loaded.predict(X_test)

# Mostrar las métricas para verificar
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print("Accuracy del modelo cargado:", accuracy_score(y_test, y_pred))



import json

# Ruta de los archivos
notebook_path = '/workspaces/Finarosalina_NLP_DL/src/explore.ipynb'
python_file_path = '/workspaces/Finarosalina_NLP_DL/src/app.py'

# Leer el notebook (archivo JSON)
with open(notebook_path, 'r') as f:
    notebook_content = json.load(f)

# Extraer las celdas de código del notebook
code_cells = []
for cell in notebook_content['cells']:
    if cell['cell_type'] == 'code':
        code = ''.join(cell['source'])  # Unir las líneas de código en una sola cadena
        code_cells.append(code)

# Escribir las celdas de código en el archivo Python
with open(python_file_path, 'w') as f:
    for code in code_cells:
        f.write(code + '\n\n')  # Escribir cada celda de código en el archivo

print(f'El contenido del notebook se ha copiado a {python_file_path}')


