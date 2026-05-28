import pandas as pd

from src.preprocesamiento import detectar_outliers
from src.sentimientos import analizar_sentimiento

df = pd.read_csv("data/playas_prueba.csv")

comentarios = df["comentarios"].tolist()

print("\nCOMENTARIOS:")
print(comentarios)

print("\nOUTLIERS:")
print(detectar_outliers(comentarios))

print("\nSENTIMIENTOS:")

for comentario in comentarios:
    print(comentario, "->", analizar_sentimiento(comentario))
