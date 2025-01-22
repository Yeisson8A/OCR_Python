import cv2
import easyocr
import pandas as pd

def extract_text_ocr(image, lang = 'en'):
    # Configurar el lector OCR
    reader = easyocr.Reader([lang], gpu=False)

    # Extraer texto de la imagen
    result = reader.readtext(image, paragraph=False)

    color = (0, 51, 176)
    text_color = (255, 255, 255)
    text_list = [{}]

    # Recorrer textos extraidos
    for res in result:
        print(res)
        # Obtener coordenadas donde se encuentra el texto extraido
        pt0 = tuple([int(value) for value in res[0][0]])
        pt1 = tuple([int(value) for value in res[0][1]])
        pt2 = tuple([int(value) for value in res[0][2]])
        pt3 = tuple([int(value) for value in res[0][3]])
        text_list.append({'pt0': pt0, 'pt1': pt1, 'words': res[1], 'confidence interval': res[2]})

        # Dibujar rectangulo con las coordenadas del texto extraido
        cv2.rectangle(image, pt0, (pt1[0], pt1[1] - 23), color, -1)
        cv2.putText(image, res[1], (pt0[0], pt0[1] -3), 2, 0.8, text_color, 1)
        cv2.rectangle(image, pt0, pt2, color, 2)
    
    # Retornar lista con texto leido
    return text_list

def get_text_sorted(text_list):
    # Convertir lista con texto leido a dataframe
    df = pd.DataFrame(text_list)
    df = df.dropna(how='all')

    # Ordenar por 'y' (orden vertical) y luego por 'x' (orden horizontal)
    df['y0'] = df['pt0'].apply(lambda pt: pt[1])  # Extraer coordenada y de pt0
    df['x0'] = df['pt0'].apply(lambda pt: pt[0])  # Extraer coordenada x de pt0
    sorted_df = df.sort_values(by=['y0', 'x0']).reset_index(drop=True)
    # Guardar texto ordenado en archivo plano
    file = open("ocr_on_image.txt", 'w')
    file.write(" ".join(sorted_df['words']))
    file.close()

# Leer imagen
image = cv2.imread("./Data/Imagen_6.jpg")

# Llamar función para extraer texto de la imagen
text_list = extract_text_ocr(image, 'es')
# Llamar función para obtener texto extraido ordenado
get_text_sorted(text_list)

# Mostrar imagen
image_scale = cv2.resize(image, (540, 540))
cv2.imshow("Image", image_scale)
cv2.waitKey(0)

# Liberar recursos
cv2.destroyAllWindows()