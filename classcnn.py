from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json

# Carregar o classificador
arquivo = open('classificador_bh.json', 'r')
estrutura_classificador = arquivo.read()
arquivo.close()
classificador_carregado = model_from_json(estrutura_classificador)
classificador_carregado.load_weights(r"classificador_bh.h5")
classificador = classificador_carregado

#### Classificação por uma imagem ####
image_teste = image.load_img(r'plantas/treino/saudavel/saudavel9.jpg',
                             target_size=(128, 128))
image_teste = image.img_to_array(image_teste)
image_teste /= 255
image_teste = np.expand_dims(image_teste, axis=0)
previsao = classificador.predict(image_teste)

# {'doente': 0, 'saudavel': 1}
if previsao > 0.5:
    print('Planta saudavel')
else:
    print('Planta doente')  
