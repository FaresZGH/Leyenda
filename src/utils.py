def get_data_augmentation(image_h: int, image_w: int):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError()

    return tf.keras.Sequential(
        [
            # Déplacement de l'image,
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(height_factor=(0.0, 0.1))
        ]
    )