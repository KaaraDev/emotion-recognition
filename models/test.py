import pandas as pd

df = pd.read_csv("features_case/combined.csv.gz")
pd.set_option("display.max_columns", None)  # keine Begrenzung
pd.set_option("display.width", 200)         # breite Ausgabe
print(df.head(3))                           # zeigt alle Spalten der ersten 3 Zeilen

