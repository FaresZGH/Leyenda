import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorboard.compat.tensorflow_stub.errors import InvalidArgumentError


class DataLoader:
    def __init__(self, base_path="/mnt/c/Users/fares/OneDrive/CESI/2025/DataScience/git/datasets", image_size=(256, 256), batch_size=32, test_ratio=0.2, validation_ratio=0.2, seed=42):
        self.base_path = base_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.seed = seed
        self.valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

    def list_datasets(self):
        """Liste les sous-dossiers disponibles dans datasets/"""
        return [f.name for f in os.scandir(self.base_path) if f.is_dir()]

    def create_dataframe_for_class(self, folder_name, label):
        folder = os.path.join(self.base_path, folder_name)
        files = [f for f in os.listdir(folder) if f.lower().endswith(self.valid_ext)]
        paths = [os.path.join(folder, f) for f in files]
        return pd.DataFrame({"path": paths, "label": label})

    def load_binary_dataset(self, positive_class, negative_classes, class_weights: bool=False):
        """Crée un dataset binaire équilibré (1 vs 0)"""
        df_pos = self.create_dataframe_for_class(positive_class, label=1)
        dfs_neg = [self.create_dataframe_for_class(cls, label=0) for cls in negative_classes]
        df_neg = pd.concat(dfs_neg, ignore_index=True)
        
        # Équilibrage des classes
        if class_weights is False:
            n = min(len(df_pos), len(df_neg))
            df_pos = df_pos.sample(n, random_state=self.seed)
            df_neg = df_neg.sample(n, random_state=self.seed)
            df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=self.seed).reset_index(drop=True)
        else:
            df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=self.seed).reset_index(drop=True)

        print(f"Binary Dataset size: {len(df)}")

        return self._create_tf_datasets(df)

    def load_multiclass_dataset(self, class_folders, class_weights: bool= False):
        """Crée un dataset multi-classes équilibré à partir d'une liste de dossiers"""
        dfs = []
        min_count = float('inf')

        # Charger les données et trouver la taille minimale des classes
        for idx, folder in enumerate(class_folders):
            df = self.create_dataframe_for_class(folder, label=idx)
            dfs.append(df)
            if len(df) < min_count:
                min_count = len(df)

        # Équilibrage
        if class_weights is False:
            balanced_dfs = [df_.sample(min_count, random_state=self.seed) for df_ in dfs]
            dfs = pd.concat(balanced_dfs).sample(frac=1, random_state=self.seed).reset_index(drop=True)
        else:
            dfs = [df_.sample(frac=1, random_state=self.seed) for df_ in dfs]
            dfs = pd.concat(dfs).sample(frac=1, random_state=self.seed).reset_index(drop=True)

        print(f"Multiclass Dataset size: {len(dfs)}")
        
        return self._create_tf_datasets(dfs)

    def _create_tf_datasets(self, df):
        train_val_df, test_df = train_test_split(df, test_size=self.test_ratio, stratify=df["label"], random_state=self.seed)
        train_df, val_df = train_test_split(train_val_df, test_size=self.validation_ratio, stratify=train_val_df["label"], random_state=self.seed)

        def df_to_dataset(df):
            path_ds = tf.data.Dataset.from_tensor_slices(df["path"].values)
            label_ds = tf.data.Dataset.from_tensor_slices(df["label"].values)

            def load_image(path):
                img = tf.io.read_file(path)
                img = tf.image.decode_image(img, channels=3, expand_animations=False)
                img = tf.image.resize(img, self.image_size)
                img = tf.cast(img, tf.float32) / 255.0
                return img

            img_ds = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
            return tf.data.Dataset.zip((img_ds, label_ds)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return df_to_dataset(train_df), df_to_dataset(val_df), df_to_dataset(test_df)