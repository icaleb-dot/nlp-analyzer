import pandas as pd
from src.cli import obtener_argumentos
from src.preprocesamiento import cargar_modelo_spacy, preprocesar_dataframe

from src.topicos import AnalizadorSemantico 

def main():
    #CLI

    args = obtener_argumentos()
    
    print(f"\nLeyendo el archivo: {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error crítico al abrir el archivo CSV: {e}")
        return

    print(f"Cargando el pipeline de lenguaje para el idioma: '{args.idioma}'...")
    nlp = cargar_modelo_spacy(args.idioma)

    # Preprocesamiento

    df_limpio = preprocesar_dataframe(df, args.columna, nlp)
    

    # outliers y Sentimientos (Integrante 2)

    df_positivos = df_limpio.copy() 
    df_negativos = df_limpio.copy() 


    # modelado de tópicos y Semántica
    print("\nIniciando análisis semántico y modelado de tópicos...")
    
    # pasándole el idioma del CLI
    analizador = AnalizadorSemantico(idioma=args.idioma, umbral_minimo=30)
    
    # Análisis de particiones 
    print("-> Ejecutando modelado de tópicos por particiones de sentimiento...")
    res_pos = analizador.procesar_particion(df_positivos, col_limpia='texto_limpio', col_original=args.columna)
    res_neg = analizador.procesar_particion(df_negativos, col_limpia='texto_limpio', col_original=args.columna)
        
    # análisis específico de precio utilizando similitud de coseno
    print("-> Analizando semántica de Precio/Valor/Costo...")
    top_precios = analizador.analizar_precio_valor(df_limpio, col_limpia='texto_limpio', col_original=args.columna)
    
 
    print("\n=========================================")
    print("Top 5 Comentarios sobre Precio/Valor:")
    print("=========================================")
    for item in top_precios:
        print(f"Similitud {item['similitud']:.4f}: {item['comentario_original']}")
    print("=========================================\n")

    # visualizaciones (Integrante 4)
    # ...

if __name__ == '__main__':
    main()