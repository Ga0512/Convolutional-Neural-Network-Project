from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

classificador = Sequential()
classificador.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))
classificador.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Flatten())

classificador.add(Dense(units=600, activation='relu'))  # ((64*64 imagem):(3*3 kernel):3 rgb)
classificador.add(Dropout(0.1))
classificador.add(Dense(units=600, activation='relu'))
classificador.add(Dropout(0.1))
classificador.add(Dense(units=1, activation='sigmoid'))

classificador.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale=1. / 255,
                                         rotation_range=7,
                                         horizontal_flip=True,
                                         shear_range=0.2,
                                         height_shift_range=0.07,
                                         zoom_range=0.2)
gerador_teste = ImageDataGenerator(rescale=1. / 255)

base_treinamento = gerador_treinamento.flow_from_directory('plantas/treino',
                                                           target_size=(128, 128),
                                                           batch_size=1,
                                                           class_mode='binary')
base_teste = gerador_teste.flow_from_directory('plantas/teste',
                                               target_size=(128, 128),
                                               batch_size=1,
                                               class_mode='binary')

classificador.fit(base_treinamento, steps_per_epoch=40,
                            epochs=1000, validation_data=base_teste,
                            validation_steps=10)

# Salvar o classificador
classificador_json = classificador.to_json()
with open("classificador_bh.json", "w") as json_file:
    json_file.write(classificador_json)
classificador.save_weights("classificador_bh.h5")

a = base_treinamento.class_indices
print(a)
