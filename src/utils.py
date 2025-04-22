def get_data_augmentation():
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError()

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(height_factor=(0.0, 0.1))
        ]
    )

def get_classification_report(model, model_name: str, test_ds, is_binary: bool=True):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import classification_report
    except ImportError:
        raise ImportError()

    y_true, y_pred = [], []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        if is_binary:
            y_pred.extend((preds > 0.5).astype(int).flatten())
        elif is_binary is False:
            y_pred.extend(np.argmax(preds, axis=1))

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred)
    labels = np.unique(y_true)

    plt.figure(figsize=(10, 6))
    x = range(len(labels))
    plt.bar(x, precision, width=0.2, label='Precision', align='center')
    plt.bar([p + 0.2 for p in x], recall, width=0.2, label='Recall', align='center')
    plt.bar([p + 0.4 for p in x], f1_score, width=0.2, label='F1-score', align='center')
    plt.xticks([p + 0.2 for p in x], labels)
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Classification Report')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./../figures/{model_name}_classification_report_{datetime.now().strftime("%Y%m%d-%H%M%S")}.png')
    plt.close()

    return classification_report(y_true, y_pred)

def get_confusion_matrix(model, model_name: str, test_ds, is_binary: bool=True):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        from datetime import datetime
    except ImportError:
        raise ImportError()

    y_true, y_pred = [], []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        if is_binary:
            y_pred.extend((preds > 0.5).astype(int).flatten())
        elif is_binary is False:
            y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'./../figures/{model_name}_classification_report_{datetime.now().strftime("%Y%m%d-%H%M%S")}.png')
    plt.close()

    return cm

def get_class_weigths(test_ds, is_binary: bool=True):
    try:
        import numpy as np
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError:
        raise ImportError()

    y_true = []
    for images, labels in test_ds:
        y_true.extend(labels.numpy())

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_true),
        y=y_true
    )

    return dict(enumerate(class_weights))

def showTrainingHistory(history):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError()

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def show_clean_and_noisy_images(clean_dataset, noisy_dataset, max_images=10, batch_index=0):
    import matplotlib.pyplot as plt

    # Si ce sont des tf.data.Dataset, les convertir temporairement en liste pour accéder à un batch précis
    if hasattr(clean_dataset, 'skip'):
        clean_batch = list(clean_dataset.skip(batch_index).take(1))[0]
    else:
        clean_batch = clean_dataset[batch_index]

    if hasattr(noisy_dataset, 'skip'):
        noisy_batch = list(noisy_dataset.skip(batch_index).take(1))[0]
    else:
        noisy_batch = noisy_dataset[batch_index]

    clean_images, _ = clean_batch
    noisy_images, _ = noisy_batch

    plt.figure(figsize=(15, 5))
    for i in range(min(max_images, len(clean_images))):
        # Image originale
        ax = plt.subplot(2, max_images, i + 1)
        plt.imshow(clean_images[i].numpy())
        plt.title("Originale")
        plt.axis("off")

        # Image bruitée
        ax = plt.subplot(2, max_images, max_images + i + 1)
        plt.imshow(noisy_images[i].numpy())
        plt.title("Bruitée")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


 

def show_original_vs_decoded(noisy_dataset, clean_dataset, model, n=10):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError()
    
    noisy_images, _ = next(iter(noisy_dataset))
    clean_images, _ = next(iter(clean_dataset))

    decoded_images = model.predict(noisy_images)

    plt.figure(figsize=(15, 6))
    for i in range(n):

        # Image bruitée (entrée)
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(noisy_images[i].numpy())
        plt.title("Bruitée")
        plt.axis("off")

        #  Image originale non bruitée 
        ax = plt.subplot(3, n, n + i + 1)
        plt.imshow(clean_images[i].numpy())
        plt.title("Originale")
        plt.axis("off")

        # Image reconstruite
        ax = plt.subplot(3, n, 2 * n + i + 1)
        plt.imshow(decoded_images[i])
        plt.title("Reconstruite")
        plt.axis("off")

    plt.tight_layout()
    plt.show()



def filter_by_custom_binary_model(model, input_folder, output_folder,
                                  image_size=(256, 256), threshold=0.5, max_images=None, verbose=True):
    
    try:
        from tensorflow.keras.preprocessing import image
        import numpy as np
        import os
        import shutil
        from tqdm import tqdm
    except ImportError:
        raise ImportError()
    """
    Utilise un modèle binaire (photo vs non-photo) pour filtrer les vraies photos depuis un dossier d'images.
    """

    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if max_images:
        image_files = image_files[:max_images]

    kept = 0
    print(f"Analyse de {len(image_files)} images...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(input_folder, img_name)

        try:
            img = image.load_img(img_path, target_size=image_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0  

            pred = model.predict(x, verbose=0)[0][0]  
            if verbose:
                print(f"{img_name} → {pred:.2f}")

            if pred > threshold:
                shutil.copy(img_path, os.path.join(output_folder, img_name))
                kept += 1

        except Exception as e:
            print(f"Erreur avec {img_name} : {e}")

    print(f"{kept}/{len(image_files)} images conservées dans : {output_folder}")

    def save_tokenizer(tokenizer, path_to_token = "./../models/weights/captioning_token/"):
    token_json = tokenizer.to_json()

    with open(path_to_token + "captioning_tokenizer_excellent.json", "w", encoding="utf-8") as f:
        f.write(token_json)

def load_tokenizer(path_to_token = "./../models/weights/captioning_token/captioning_tokenizer_excellent.json"):
    try:
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
    except ImportError:
        raise ImportError()

    with open(path_to_token, "r", encoding="utf-8") as f:
        token_json = f.read()

    return tokenizer_from_json(token_json)