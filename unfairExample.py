import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Dati di esempio con nuova variabile
data = {
    'Genere': ['Maschio', 'Femmina', 'Maschio', 'Femmina', 'Maschio'],
    'Tempo_di_studio': [10, 8, 5, 3, 7],
    'Partecipazione_attività_extracurricolari': [1, 1, 0, 1, 0],  # 1 se partecipa, 0 altrimenti
    'Performance': ['Bravo', 'Bravo', 'Meno bravo', 'Meno bravo', 'Bravo']
}

# Creazione del dataframe
df = pd.DataFrame(data)

# Mappatura del genere a valori numerici
df['Genere'] = df['Genere'].map({'Maschio': 0, 'Femmina': 1})

# Split dei dati in training e test set
X = df[['Genere', 'Tempo_di_studio', 'Partecipazione_attività_extracurricolari']]
y = df['Performance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione del Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Valutazione del modello
y_pred_dt = dt_classifier.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred_dt)}')
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_dt))
print('Classification Report:\n', classification_report(y_test, y_pred_dt))



# Creazione del dataframe (già mappato per il genere)
data = {
    'Genere': [0, 1, 0, 1, 0],
    'Tempo_di_studio': [10, 8, 5, 3, 7],
    'Performance': ['Bravo', 'Bravo', 'Meno bravo', 'Meno bravo', 'Bravo']
}
df = pd.DataFrame(data)

# Aggiunta di bias nella Regressione Logistica verso il genere femminile
lr_classifier = LogisticRegression(class_weight={'Bravo': 0.5, 'Meno bravo': 0.5})
lr_classifier.fit(X_train, y_train)

# Valutazione del modello
y_pred_lr = lr_classifier.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred_lr)}')
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_lr))
print('Classification Report:\n', classification_report(y_test, y_pred_lr))
