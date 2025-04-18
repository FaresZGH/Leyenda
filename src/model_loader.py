import os

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
            self.model_weights_path: str = f"./../models/weights/{model_name}/"

        if model_logs_path is not None:
            self.model_logs_path: str = model_logs_path
        else:
            self.model_logs_path: str = f"./../logs/fit/{model_name}"

    def get_early_stopping(self, patience: int =4, restore_best_weights: bool =False, value_to_monitor: str = "val_loss"):
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

    def get_model_checkpoint(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError()

        return tf.keras.callbacks.ModelCheckpoint(
            self.model_weights_path + f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.weights.h5",
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            save_freq="epoch"
        )

    def get_tensorboard_callback(self, log_dir: str = None):
        if log_dir is None:
            log_dir = self.model_logs_path

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir + f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
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

        output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            get_data_augmentation(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation=output_activation),
        ])
        if show_summary:
            model.summary()

        loss_fn = tf.keras.losses.BinaryCrossentropy() if num_classes == 1 else tf.keras.losses.SparseCategoricalCrossentropy()

        model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])

        if init_weigths_path is not None:
            model.build()
            model.load_weights(init_weigths_path)

        return model

    def create_model_CNN_simple(self, show_summary: bool = True, init_weigths_path: str = None, num_classes: int = 1):
        output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation=output_activation)
        ])
        if show_summary:
            model.summary()

        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True) if num_classes == 1 else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])

        if init_weigths_path is not None:
            model.build(input_shape=(1, 256, 256, 3))
            model.load_weights(init_weigths_path)

        return model

    def create_model_CNN_hard(self, show_summary: bool = True, init_weigths_path: str = None, num_classes: int = 1):
        output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
        model = tf.keras.Sequential([
            get_data_augmentation(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation=output_activation)  # Classification binaire
        ])
        if show_summary:
            model.summary()

        loss_fn = tf.keras.losses.BinaryCrossentropy() if num_classes == 1 else tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])

        if init_weigths_path is not None:
            model.build(input_shape=(1, 256, 256, 3))
            model.load_weights(init_weigths_path)

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
        x = get_data_augmentation()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
        predictions = Dense(num_classes, activation=output_activation)(x)

        model = Model(inputs=input_tensor, outputs=predictions)

        if show_summary:
            model.summary()

        loss_fn = tf.keras.losses.BinaryCrossentropy() if num_classes == 1 else tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

        if init_weigths_path is not None:
            model.build(input_shape=(1, 256, 256, 3))
            model.load_weights(init_weigths_path)

        return model


    def create_weighted_models(self):
        # Fonction pour charger les poids de chaque modèle pour le tri des photos / non photos
        # Cela prend les poids avec binary_cw, binary_nocw, multiclass_cw, multiclass_nocw comme nom de fichier
        # dans les dossiers liés aux modèles
        try:
            from os import listdir
        except:
            raise ModuleNotFoundError()

        datasets_name = ['binary_cw', 'binary_nocw', 'multiclass_cw', 'multiclass_nocw']
        all_model_cnn = {}
        all_model_resnet = {}
        all_model_inception = {}

        for repertory in os.listdir('./../models/weights/'):

            if repertory == "CNN_HARD":
                print(f"\nLoading models from {repertory}...")
                for dataset_name in datasets_name:
                    model = None
                    folder_path = f'./../models/weights/{repertory}/'
                    if dataset_name == "binary_cw":
                        weight_path = folder_path + [f for f in os.listdir(folder_path) if f.startswith(dataset_name)][0]
                        model = self.create_model_CNN_hard(show_summary=False, init_weigths_path=weight_path ,num_classes=1)
                    elif dataset_name == "binary_nocw":
                        weight_path = folder_path + [f for f in os.listdir(folder_path) if f.startswith(dataset_name)][0]
                        model = self.create_model_CNN_hard(show_summary=False, init_weigths_path=weight_path ,num_classes=1)
                    elif dataset_name == "multiclass_cw":
                        weight_path = folder_path + [f for f in os.listdir(folder_path) if f.startswith(dataset_name)][0]
                        model = self.create_model_CNN_hard(show_summary=False, init_weigths_path=weight_path ,num_classes=5)
                    elif dataset_name == "multiclass_nocw":
                        weight_path = folder_path + [f for f in os.listdir(folder_path) if f.startswith(dataset_name)][0]
                        model = self.create_model_CNN_hard(show_summary=False, init_weigths_path=weight_path ,num_classes=5)
                    else:
                        raise ValueError("Dataset name not recognized")
                    if model is not None:
                        all_model_cnn[f'{dataset_name}'] = model

                    print(f"{dataset_name}...Done")


            if repertory == "INCEPTION":
                print(f"\nLoading models from {repertory}...")
                model = None
                for dataset_name in datasets_name:
                    folder_path = f'./../models/weights/{repertory}/'
                    if dataset_name == "binary_cw":
                        weight_path = folder_path + [f for f in os.listdir(folder_path) if f.startswith(dataset_name)][0]
                        model = self.create_model_with_inception(show_summary=False, init_weigths_path=weight_path ,num_classes=1)
                    elif dataset_name == "binary_nocw":
                        weight_path = folder_path + [f for f in os.listdir(folder_path) if f.startswith(dataset_name)][0]
                        model = self.create_model_with_inception(show_summary=False, init_weigths_path=weight_path ,num_classes=1)
                    elif dataset_name == "multiclass_cw":
                        weight_path = folder_path + [f for f in os.listdir(folder_path) if f.startswith(dataset_name)][0]
                        model = self.create_model_with_inception(show_summary=False, init_weigths_path=weight_path ,num_classes=5)
                    elif dataset_name == "multiclass_nocw":
                        weight_path = folder_path + [f for f in os.listdir(folder_path) if f.startswith(dataset_name)][0]
                        model = self.create_model_with_inception(show_summary=False, init_weigths_path=weight_path ,num_classes=5)
                    else:
                        raise ValueError("Dataset name not recognized")

                    if model is not None:
                        all_model_inception[f'{dataset_name}'] = model

                    print(f"{dataset_name}...Done")


            if repertory == "RES_NET":
                print(f"\nLoading models from {repertory}...")
                for dataset_name in datasets_name:
                    model = None
                    folder_path = f'./../models/weights/{repertory}/'
                    if dataset_name == "binary_cw":
                        weight_path = folder_path + [f for f in os.listdir(folder_path) if f.startswith(dataset_name)][0]
                        model = self.create_model_resnet50(show_summary=False, init_weigths_path=weight_path ,num_classes=1)
                    elif dataset_name == "binary_nocw":
                        weight_path = folder_path + [f for f in os.listdir(folder_path) if f.startswith(dataset_name)][0]
                        model = self.create_model_resnet50(show_summary=False, init_weigths_path=weight_path ,num_classes=1)
                    elif dataset_name == "multiclass_cw":
                        weight_path = folder_path + [f for f in os.listdir(folder_path) if f.startswith(dataset_name)][0]
                        model = self.create_model_resnet50(show_summary=False, init_weigths_path=weight_path ,num_classes=5)
                    elif dataset_name == "multiclass_nocw":
                        weight_path = folder_path + [f for f in os.listdir(folder_path) if f.startswith(dataset_name)][0]
                        model = self.create_model_resnet50(show_summary=False, init_weigths_path=weight_path ,num_classes=5)
                    else:
                        raise ValueError("Dataset name not recognized")
                    if model is not None:
                        all_model_resnet[f'{dataset_name}'] = model

                    print(f"{dataset_name}...Done")

        return all_model_cnn, all_model_inception, all_model_resnet
    
    def create_skip_layer_autoencoder(self, show_summary: bool = True, init_weigths_path: str = None):

        try:
            import tensorflow as tf
            from keras.layers import Conv2D, Conv2DTranspose, Concatenate, Input, BatchNormalization, LeakyReLU
            from keras.models import Model
        except ImportError:
            raise ImportError()

        encoder_inputs = Input(shape=(256, 256, 3))  # Image RGB 256x256

        # --- Block 1 ---
        x1 = Conv2D(32, (3, 3), strides=2, padding='same')(encoder_inputs)  # (128, 128, 32)
        x1 = BatchNormalization()(x1)
        x1 = LeakyReLU()(x1)

        # --- Block 2 ---
        x2 = Conv2D(64, (3, 3), strides=2, padding='same')(x1)  # (64, 64, 64)
        x2 = BatchNormalization()(x2)
        x2 = LeakyReLU()(x2)

        # --- Block 3 ---
        x3 = Conv2D(128, (3, 3), strides=2, padding='same')(x2)  # (32, 32, 128)
        x3 = BatchNormalization()(x3)
        x3 = LeakyReLU()(x3)

        # --- Block 4 ---
        x4 = Conv2D(256, (3, 3), strides=2, padding='same')(x3)  # (16, 16, 256)
        x4 = BatchNormalization()(x4)
        x4 = LeakyReLU()(x4)

        # --- Block 5 ---
        x5 = Conv2D(256, (3, 3), strides=2, padding='same')(x4)  # (8, 8, 256)
        x5 = BatchNormalization()(x5)
        x5 = LeakyReLU()(x5)

        # Sorties : encoder_output + features pour skip connections
        encoder = Model(inputs=encoder_inputs, outputs=[x5, x4, x3, x2, x1], name="encoder")

        if show_summary:
            encoder.summary()

        print("Encoder created successfully.")


        # Entrées : latent + les 4 couches skip
        latent_input = Input(shape=(8, 8, 256))     # Sortie finale de l'encodeur
        skip4 = Input(shape=(16, 16, 256))
        skip3 = Input(shape=(32, 32, 128))
        skip2 = Input(shape=(64, 64, 64))
        skip1 = Input(shape=(128, 128, 32))

        x = latent_input

        # Decode Block 1
        x = Conv2DTranspose(256, (3, 3), strides=2, padding='same')(x)  # (16, 16, 256)
        x = Concatenate()([x, skip4])
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Decode Block 2
        x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)  # (32, 32, 128)
        x = Concatenate()([x, skip3])
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Decode Block 3
        x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)  # (64, 64, 64)
        x = Concatenate()([x, skip2])
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Decode Block 4
        x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)  # (128, 128, 32)
        x = Concatenate()([x, skip1])
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Decode Block 5 - reconstruction finale
        x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)  # (256, 256, 32)
        decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        # Modèle
        decoder = Model(inputs=[latent_input, skip4, skip3, skip2, skip1], outputs=decoder_output, name="decoder")

        if show_summary:
            decoder.summary()

        print("Decoder created successfully.")

        encoder_outputs = encoder(encoder_inputs)
        decoded_img = decoder(encoder_outputs)

        autoencoder = Model(inputs=encoder_inputs, outputs=decoded_img)

        if init_weigths_path is not None:
            autoencoder.build(input_shape=(1, 256, 256, 3))
            autoencoder.load_weights(init_weigths_path)

        def ssim_metric(y_true, y_pred):
            # SSIM attend des valeurs entre [0, 1], donc il est important de s'assurer que les images sont dans cette plage.
            y_true = tf.clip_by_value(y_true, 0.0, 1.0)  # Pour être sûr que les valeurs sont entre [0, 1]
            y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)  # Pour être sûr que les valeurs sont entre [0, 1]
            
            # Calcul du SSIM sur les images RGB
            return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))  # max_val=1.0 pour les images normalisées

        def psnr_metric(y_true, y_pred):
            # PSNR attend des valeurs entre [0, 1], donc on s'assure que les images sont dans cette plage.
            y_true = tf.clip_by_value(y_true, 0.0, 1.0)  # Normalisation entre [0, 1]
            y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)  # Normalisation entre [0, 1]
            
            # Calcul du PSNR sur les images RGB
            return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))  # max_val=1.0 pour les images normalisées
    
        # Compilation de l'autoencodeur
        autoencoder.compile(optimizer="adam", loss="mae", metrics=['mae', ssim_metric, psnr_metric])

        if show_summary:
            print("\nAutoencoder Summary:")
            autoencoder.summary()

        print("Autoencoder created successfully.")

        return encoder, decoder, autoencoder
    
    def create_base_autoencoder(self, show_summary: bool = True, init_weigths_path: str = None):
        try:
            import tensorflow as tf
            from keras.layers import Conv2D, Conv2DTranspose, Concatenate, Input, BatchNormalization, LeakyReLU
            from keras.models import Model
        except ImportError:
            raise ImportError()
        
        encoder_inputs = Input(shape=(256, 256, 3))  # Image RGB 256x256

        # Conv Block 1
        x = Conv2D(32, (3, 3), strides=2, padding='same')(encoder_inputs)  # (128, 128, 32)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Conv Block 2
        x = Conv2D(64, (3, 3), strides=2, padding='same')(x)  # (64, 64, 64)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Conv Block 3
        x = Conv2D(128, (3, 3), strides=2, padding='same')(x)  # (32, 32, 128)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Conv Block 4
        x = Conv2D(256, (3, 3), strides=2, padding='same')(x)  # (16, 16, 256)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Conv Block 5
        x = Conv2D(256, (3, 3), strides=2, padding='same')(x)  # (8, 8, 256)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        encoder_output = x  # Pas de flatten, on garde (8, 8, 256)

        encoder = Model(encoder_inputs, encoder_output, name="encoder")
        if show_summary:
            encoder.summary()

        print("Encoder created successfully.")

        # --- Decoder simplifié ---
        decoder_inputs = Input(shape=(8, 8, 256))  # Sortie directe de l'encodeur

        # DeConv1
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(decoder_inputs)  # (16, 16, 128)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # DeConv2
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)  # (32, 32, 128)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # DeConv3
        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)  # (64, 64, 64)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # DeConv4
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)  # (128, 128, 32)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # DeConv5 - reconstruction finale
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)  # (256, 256, 32)
        decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Sortie finale RGB

        # Modèle decoder
        decoder = Model(decoder_inputs, decoder_output, name="decoder")

        if show_summary:
            decoder.summary()

        # On récupère la sortie de l'encodeur (le "latent space")
        latent_space = encoder(encoder_inputs)

        # On passe cette sortie directement dans le décodeur
        outputs = decoder(latent_space)

        # Maintenant, on crée l'autoencodeur avec l'encodeur et le décodeur
        autoencoder = Model(encoder_inputs, outputs, name="autoencoder")

        if init_weigths_path is not None:   
            autoencoder.build(input_shape=(1, 256, 256, 3))
            autoencoder.load_weights(init_weigths_path)


        def ssim_metric(y_true, y_pred):
            # SSIM attend des valeurs entre [0, 1], donc il est important de s'assurer que les images sont dans cette plage.
            y_true = tf.clip_by_value(y_true, 0.0, 1.0)  # Pour être sûr que les valeurs sont entre [0, 1]
            y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)  # Pour être sûr que les valeurs sont entre [0, 1]
            
            # Calcul du SSIM sur les images RGB
            return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))  # max_val=1.0 pour les images normalisées

        def psnr_metric(y_true, y_pred):
            # PSNR attend des valeurs entre [0, 1], donc on s'assure que les images sont dans cette plage.
            y_true = tf.clip_by_value(y_true, 0.0, 1.0)  # Normalisation entre [0, 1]
            y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)  # Normalisation entre [0, 1]
            
            # Calcul du PSNR sur les images RGB
            return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))  # max_val=1.0 pour les images normalisées
        
        # Compilation de l'autoencodeur
        autoencoder.compile(optimizer="adam", loss="mae", metrics=["mae", ssim_metric, psnr_metric])

        if show_summary:
            print("\nAutoencoder Summary:")
            autoencoder.summary()

        return autoencoder
        




            