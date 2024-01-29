import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tabulate import tabulate

# Carica i dati dal file CSV
file_path = 'students_dataset.csv'
data = pd.read_csv(file_path, sep=',')

# Data analisys
print(data.info())
print ('=' * 80)
print()
print ('=' * 80)
print ("Variabile risposta GIUDIZIO-2 positivo se media(votop) per studente>= 210")
print ('=' * 80)

mm = data.loc[data['sesso'] == 'M', 'voto'].mean()
mf = data.loc[data['sesso'] == 'F', 'voto'].mean()

mpm = data.loc[data['sesso'] == 'M', 'media pesata'].mean()
mpf = data.loc[data['sesso'] == 'F', 'media pesata'].mean()

print (f'Media voti maschi \t{mm:.2f}')
print (f'Media voti femmine \t{mf:.2f}')
print ('-' * 80)

print (f'Media voti pesati maschi \t{mpm:.2f}')
print (f'Media voti pesati femmine \t{mpf:.2f}')
print ('=' * 80)
print()
print('Analisi: previsione del giudizio in base agli altri parametri escluso il sesso')
print ('=' * 80)

# Seleziona le colonne delle features
selected_columns = ['ID', 'codne', 'voto', 'votop']
X = data[selected_columns]

# Seleziona la colonna target
y = data['giud2']

# Suddividi i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento del modello HistGradientBoosting
hgb_model = HistGradientBoostingClassifier()
hgb_model.fit(X_train, y_train)
hgb_predictions = hgb_model.predict(X_test)
hgb_accuracy = accuracy_score(y_test, hgb_predictions)
hgb_cm = confusion_matrix(y_test, hgb_predictions)
hgb_cr = classification_report(y_test, hgb_predictions)

# Addestramento del modello KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_cm = confusion_matrix(y_test, knn_predictions)
knn_cr = classification_report(y_test, knn_predictions)

# Addestramento del modello XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_cm = confusion_matrix(y_test, xgb_predictions)
xgb_cr = classification_report(y_test, xgb_predictions)

print()
print ("Accuracy dei diversi algoritmi")
print(f"HGB Accuracy: {hgb_accuracy}")
print(f"KNN Accuracy: {knn_accuracy}")
print(f"XGB Accuracy: {xgb_accuracy}")
print()
print ("Matrici di confusione")
print(f'HGB confusion matrix: \n{tabulate(hgb_cm)}\n')
print(f'KNN confusion matrix: \n{tabulate(knn_cm)}\n')
print(f'XGB confusion matrix: \n{tabulate(xgb_cm)}\n')
print()
print ("Report")
print(f'HGB report: \n{hgb_cr}\n')
print(f'KNN report: \n{knn_cr}\n')
print(f'XGB report: \n{xgb_cr}\n')

print()
print('Analisi: previsione del giudizio in base agli altri parametri compreso il sesso')
print ('=' * 80)

# Seleziona le colonne delle features
selected_columns = ['ID', 'codne', 'voto', 'votop', 'sesson']
X = data[selected_columns]

# Seleziona la colonna target
y = data['giud2']

# Suddividi i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento del modello Random Forest
hgb_model = HistGradientBoostingClassifier()
hgb_model.fit(X_train, y_train)
hgb_predictions = hgb_model.predict(X_test)
hgb_accuracy = accuracy_score(y_test, hgb_predictions)
hgb_cm = confusion_matrix(y_test, hgb_predictions)
hgb_cr = classification_report(y_test, hgb_predictions)

# Addestramento del modello KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_cm = confusion_matrix(y_test, knn_predictions)
knn_cr = classification_report(y_test, knn_predictions)

# Addestramento del modello XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_cm = confusion_matrix(y_test, xgb_predictions)
xgb_cr = classification_report(y_test, xgb_predictions)


print()
print("Accuracy dei diversi algoritmi")
print(f"HGB Accuracy: {hgb_accuracy}")
print(f"KNN Accuracy: {knn_accuracy}")
print(f"XGB Accuracy: {xgb_accuracy}")

print()
print ("Matrici di confusione")
print(f'HGB confusion matrix: \n{tabulate(hgb_cm, showindex="always")}\n')
print(f'KNN confusion matrix: \n{tabulate(knn_cm, showindex="always")}\n')
print(f'XGB confusion matrix: \n{tabulate(xgb_cm, showindex="always")}\n')

print()
print ("Report")
print(f'HGB report: \n{hgb_cr}\n')
print(f'KNN report: \n{knn_cr}\n')
print(f'XGB report: \n{xgb_cr}\n')


# Aggiunge le previsioni ai dati
data['prediction'] = knn_model.predict(X)


# Epsilon for fairness measures
eps = 0.01

males = data[data['sesson'] == 1]
females = data[data['sesson'] == 0]

tm = len(males)
tf = len(females)

sm = males['giud2'].sum()
sf = females['giud2'].sum()

#
# Fairness measures
#

# Independence measures
yhat_female = females['prediction'].sum()
yhat_male = males['prediction'].sum()
female_prob = yhat_female / tf
male_prob = yhat_male / tm

# Statistical parity
statistical_parity = (abs(female_prob - male_prob) < eps)
print (f"Statistical Parity fairness is {statistical_parity}")

# Disparate Impact
disparate_impact = (abs(female_prob / male_prob - 1) < eps)
print (f"Disparate Impact fairness is {disparate_impact}")

# Separation measures
# Confusion matrix
female_TP = len(females[(females['prediction'] == 1) & (females['giud2'] == 1)])
female_FP = len(females[(females['prediction'] == 1) & (females['giud2'] == 0)])
female_TN = len(females[(females['prediction'] == 0) & (females['giud2'] == 0)])
female_FN = len(females[(females['prediction'] == 0) & (females['giud2'] == 1)])

male_TP = len(males[(males['prediction'] == 1) & (males['giud2'] == 1)])
male_FP = len(males[(males['prediction'] == 1) & (males['giud2'] == 0)])
male_TN = len(males[(males['prediction'] == 0) & (males['giud2'] == 0)])
male_FN = len(males[(males['prediction'] == 0) & (males['giud2'] == 1)])

female_TP_prob = female_TP / sf
female_FP_prob = female_FP / (tf - sf)
male_TP_prob = male_TP / sm
male_FP_prob = male_FP / (tm - sm)

# Equal opportunity
equal_opportunity = ((abs(female_TP_prob - male_TP_prob) < eps))
print (f"Equal opportunity fairness is {equal_opportunity}")

# Equalized odds
equalized_odds = (abs(female_TP_prob - male_TP_prob) < eps) & (abs(female_FP_prob - male_FP_prob) < eps)
print (f"Equalized odds fairness is {equalized_odds}")

# Total accuracy
female_acc = (female_TP + female_TN) / tf
male_acc = (male_TP + male_TN) / tm

total_accuracy = (abs(female_acc - male_acc) < eps)
print (f"Total accuracy fairness is {total_accuracy}")

# Individual fairness
#
# GEI Index
alpha = 0.5
n = len(data)
mu = (data["prediction"].sum() - data["giud2"].sum() + n) / n
sum = 0

for index, row in data.iterrows():
    bi = row["prediction"] - row["giud2"] + 1
    sum = sum + (bi / mu) ** alpha - 1

gei = (1 / ((n * alpha) * (alpha - 1))) * sum
print (f"Generalized Entropy Index is {gei}")

# Theil index
sum = 0
for index, row in data.iterrows():
    bi = row["prediction"] - row["giud2"] + 1
    pbi = bi / mu
    if pbi != 0:
        sum = sum + pbi * np.log(pbi)

theil = sum / n
print (f"Theil Index is {theil}")