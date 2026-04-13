from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train = datagen.flow_from_directory(
        'data/train',
        target_size=(128,128),
        batch_size=16,
        class_mode='binary',
        subset='training'
    )

    val = datagen.flow_from_directory(
        'data/train',
        target_size=(128,128),
        batch_size=16,
        class_mode='binary',
        subset='validation'
    )

    test = datagen.flow_from_directory(
        'data/test',
        target_size=(128,128),
        batch_size=16,
        class_mode='binary'
    )

    return train, val, test