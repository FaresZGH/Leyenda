import tensorflow as tf
import numpy as np
import datetime

from src.utils import get_data_augmentation

class ModelLoader:
    def __init__(self, model_name, model_weights_path=None, model_logs_path=None):
        self.model_name: str = None

        if model_weights_path is not None:
            self.model_weights_path: str = model_weights_path
        else:
            self.model_weights_path: str = f"./../models/weights/{model_name}.weights.h5"

        if model_logs_path is not None:
            self.model_logs_path: str = model_logs_path
        else:
            self.model_logs_path: str = f"./../logs/fit/{model_name}" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def get_early_stopping(self, patience: int =5, restore_best_weights: bool =True, value_to_monitor: str = "val_loss"):
        try:
            from tensorflow.python.keras.callbacks import EarlyStopping
        except ImportError:
            raise ImportError()

        early_stop = EarlyStopping(
            monitor=value_to_monitor,  # Ce qu’on surveille (val_loss ou val_accuracy)
            patience=patience,  # Nombre d’époques sans amélioration avant arrêt
            restore_best_weights=restore_best_weights  # Recharger les meilleurs poids à la fin
        )

        return early_stop

    def get_tensorboard_callback(self, log_dir: str = None):
        if log_dir is None:
            log_dir = self.model_logs_path

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
        )

        return tensorboard_callback

    def create_model_resnet50(self, show_summary: bool = True, init_weigths_path: str = None, num_classes: int = 1):
        try:
            from tensorflow.keras.applications import ResNet50
        except ModuleNotFoundError:
            raise ModuleNotFoundError()

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            get_data_augmentation(image_h=256, image_w=256),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation='sigmoid'),
        ])
        if show_summary:
            model.summary()

        loss_fn = tf.keras.losses.BinaryCrossentropy() if num_classes == 1 else tf.keras.losses.CategoricalCrossentropy()

        model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])

        if init_weigths_path is not None:
            model.add_weights(tf.keras.models.load_weights(init_weigths_path))

        return model

    def create_model_CNN_simple(self, show_summary: bool = True, init_weigths_path: str = None, num_classes: int = 1):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(256, 256, 3)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation='sigmoid')
        ])
        if show_summary:
            model.summary()

        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True) if num_classes == 1 else tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])

        if init_weigths_path is not None:
            model.add_weights(tf.keras.models.load_weights(init_weigths_path))

        return model

    def create_model_CNN_hard(self, show_summary: bool = True, init_weigths_path: str = None, num_classes: int = 1):
        model = tf.keras.Sequential([
            get_data_augmentation(image_h=256, image_w=256),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            # tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='sigmoid')  # Classification binaire
        ])
        if show_summary:
            model.summary()

        loss_fn = tf.keras.losses.BinaryCrossentropy() if num_classes == 1 else tf.keras.losses.CategoricalCrossentropy()
        model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])

        if init_weigths_path is not None:
            model.add_weights(tf.keras.models.load_weights(init_weigths_path))

        return model


    def create_model_with_inception(self, show_summary: bool = True, init_weigths_path: str = None, num_classes: int = 1):
        try:
            from tensorflow.keras.applications import InceptionV3
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
        except:
            raise ModuleNotFoundError()

        input_tensor = Input(shape=(256, 256, 3))
        base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)

        base_model.trainable = False  # Freeze base model

        x = base_model.output
        x = get_data_augmentation(image_h=256, image_w=256)(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
        predictions = Dense(num_classes, activation=output_activation)(x)

        model = Model(inputs=input_tensor, outputs=predictions)

        if show_summary:
            model.summary()

        loss_fn = tf.keras.losses.BinaryCrossentropy() if num_classes == 1 else tf.keras.losses.CategoricalCrossentropy()
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

        if init_weigths_path is not None:
            model.load_weights(init_weigths_path)

        return model
