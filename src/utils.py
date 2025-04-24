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

def load_caption_dataset(annotation_path, image_folder, nb_images=2000):
    try:
        import json
        import collections
        import os
        # Charger les annotations TRAIN (et uniquement celles-là !)
    except ImportError:
        raise ImportError()
    
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    # Grouper les annotations par image_id
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = '<start> ' + val['caption'] + ' <end>'
        image_path = os.path.join(image_folder, 'COCO_train2014_' + '%012d.jpg' % val['image_id'])
        image_path_to_caption[image_path].append(caption)

    # Prendre un sous-ensemble aléatoire d'images
    image_paths = list(image_path_to_caption.keys())
    train_image_paths = image_paths[:nb_images]  # tu peux ajuster ce nombre

    print(f"{len(train_image_paths)} images sélectionnées.")

    # Créer les listes d'annotations et de chemins d'images correspondantes
    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        captions = image_path_to_caption[image_path]
        train_captions.extend(captions)
        img_name_vector.extend([image_path] * len(captions))

    print(f"Total de {len(train_captions)} légendes pour {len(train_image_paths)} images.")

    return train_captions, img_name_vector

def preprocess_captionning_image(img_name_vector):
    """
    Charge et prétraite une image à partir d'un chemin donné.
    """
    try:
        import tensorflow as tf
        import numpy as np
        from tqdm import tqdm
        import os
    except ImportError:
        raise ImportError()

    # Telechargement du modèle InceptionV3 pré-entrainé avec la classification sur ImageNet
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')

    # Création d'une variable qui sera l'entrée du nouveau modèle de pré-traitement d'images
    new_input = image_model.input

    # Récupérer la dernière couche cachée qui contient l'image en représentation compacte
    hidden_layer = image_model.layers[-1].output

    # Modèle qui calcule une représentation dense des images avec InceptionV3
    image_features_extract_model = tf.keras.Model(inputs=new_input, outputs=hidden_layer)
    
    # Pré-traitement des images
    # Prendre les noms des images
    encode_train = sorted(set(img_name_vector))

    # Creation d'une instance de "tf.data.Dataset" partant des noms des images 
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    # Division du données en batchs après application du pré-traitement fait par load_image
    image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    # Parcourir le dataset batch par batch pour effectuer le pré-traitement d'InceptionV3
    for img, path in tqdm(image_dataset):
        # Passage des images dans le CNN pour récupérer les features
        batch_features = image_features_extract_model(img)

        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        # Sauvegarde de chaque image prétraitée individuellement
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            # Ajout de l'extension .npy pour le fichier sauvegardé
            if not os.path.exists(path_of_feature + ".npy"):
                np.save(path_of_feature, bf.numpy())
    
    return image_features_extract_model

def preprocess_caption_annotation(train_captions, is_training: bool =False, tokenizer_path=(), nb_top_word : int = 5000):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError()
    
    def calc_max_length(tensor):
            return max(len(t) for t in tensor)
    
    if is_training:
        # Trouver la taille maximale 
        

        # Chosir les 5000 mots les plus frequents du vocabulaire
        top_k = nb_top_word
        #La classe Tokenizer permet de faire du pre-traitement de texte pour reseau de neurones 
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                        oov_token="<unk>",
                                                        filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        # Construit un vocabulaire en se basant sur la liste train_captions
        tokenizer.fit_on_texts(train_captions)

        # Créer le token qui sert à remplir les annotations pour égaliser leurs longueurs
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'

        # Création des vecteurs (liste de token entiers) à partir des annotations (liste de mots)
        train_seqs = tokenizer.texts_to_sequences(train_captions)

        # Remplir chaque vecteur jusqu'à la longueur maximale des annotations
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

        # Calcule la longueur maximale qui est utilisée pour stocker les poids d'attention 
        # Elle servira plus tard pour l'affichage lors de l'évaluation
        max_length = calc_max_length(train_seqs)

        print("Création du Tokenizer et du vecteur d'annotations terminée.")
        print("Taille du vocabulaire : ", len(tokenizer.word_index))
        print("Taille maximale des annotations : ", max_length)

        return cap_vector, tokenizer, max_length
    else :
        reloaded_tokenizer = load_tokenizer(tokenizer_path)
        train_seqs = reloaded_tokenizer.texts_to_sequences(train_captions)
        max_length = calc_max_length(train_seqs)
        print("Chargement du Tokenizer terminé.")
        print("Taille du vocabulaire : ", len(reloaded_tokenizer.word_index))
        print("Taille maximale des annotations : ", reloaded_tokenizer.num_words)
        return reloaded_tokenizer, reloaded_tokenizer.num_words, max_length
    
def load_image(image_path, image_size=(256, 256)):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError()

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    return img, image_path

def create_caption_dataset(caption_vector, img_name_vector, tokenizer):
    try:
        import collections
        import random
        import numpy as np
        import tensorflow as tf

    except ImportError:
        raise ImportError()
    
    def map_func(img_name, cap):
        img_path = img_name.decode('utf-8') + '.npy'
        img_tensor = np.load(img_path).astype(np.float32)

        return img_tensor, cap

    # tf wrapper de map_func
    def tf_map_func(img, cap):
        img_tensor, cap = tf.numpy_function(
            map_func,
            [img, cap],
            [tf.float32, tf.int32]
        )
        img_tensor.set_shape((64, 2048))   
        cap.set_shape([None])             
        return img_tensor, cap

    BATCH_SIZE = 64 # taille du batch
    BUFFER_SIZE = 1000 # taille du buffer pour melanger les donnes

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, caption_vector):
        img_to_cap_vector[img].append(cap)

    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys) * 0.8)
    img_name_train_keys = img_keys[:slice_index]
    img_name_val_keys = img_keys[slice_index:]

    img_name_train, cap_train = [], []
    img_name_val, cap_val = [], []

    for imgt in img_name_train_keys:
        img_name_train.extend([imgt] * len(img_to_cap_vector[imgt]))
        cap_train.extend(img_to_cap_vector[imgt])

    for imgv in img_name_val_keys:
        img_name_val.extend([imgv] * len(img_to_cap_vector[imgv]))
        cap_val.extend(img_to_cap_vector[imgv])

    # Affichage des tailles finales
    len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)

    num_steps = len(img_name_train) // BATCH_SIZE
    val_num_steps = len(img_name_val) // BATCH_SIZE

    # ========================= Train ==========================

    # Tokenisation + padding après split
    train_seqs = tokenizer.texts_to_sequences(cap_train)
    cap_train_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post', dtype='int32')

    # Dataset correct
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train_vector))

    # Appliquer map + shuffle + batch
    dataset = dataset.map(tf_map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # ========================= Validation ==========================
    # Tokenisation + padding après split
    val_seqs = tokenizer.texts_to_sequences(cap_val)
    cap_val_vector = tf.keras.preprocessing.sequence.pad_sequences(val_seqs, padding='post', dtype='int32')

    # Dataset correct
    dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val_vector))

    dataset_val = dataset_val.map(tf_map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_val = dataset_val.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"Train Dataset size: {len(img_name_train)}")
    print(f"Validation Dataset size: {len(img_name_val)}")
    
    return dataset, img_name_train, cap_train, dataset_val, img_name_val, cap_val, num_steps, val_num_steps


def captionning_evaluate(image, encoder, decoder, real_caption_tokens=None, max_length=None, image_features_extract_model=None, tokenizer=None, is_gru: bool = False):
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Please install nltk to compute BLEU score.")
        return None, None, None
    
    smoothie = SmoothingFunction().method4

    # La forme du vecteur extrait à partir d'InceptionV3 est (64, 2048)
    attention_features_shape = 64
    attention_plot = np.zeros((max_length, attention_features_shape))


    # Charger et encoder l'image
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    if is_gru:
        hidden = tf.zeros((1, decoder.units))
    else:
        hidden, cell_state = decoder.reset_states(batch_size=1)

    for i in range(max_length):
        if is_gru:
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        else:  
            predictions, hidden, cell_state, attention_weights = decoder(dec_input, features, hidden, cell_state)
            
        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = tokenizer.index_word.get(predicted_id, '<unk>')
        result.append(predicted_word)

        if predicted_word == '<end>':
            break

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]

    # === Calcul BLEU si légende réelle fournie ===
    bleu_score = None
    if real_caption_tokens:
        # Nettoyer <start>, <end> et <pad>
        ref = [word for word in real_caption_tokens if word not in ['<start>', '<end>', '<pad>']]
        hyp = [word for word in result if word not in ['<start>', '<end>', '<pad>']]
        if len(hyp) > 0 and len(ref) > 0:
            bleu_score = sentence_bleu([ref], hyp, smoothing_function=smoothie)
            print(f"🟦 BLEU score: {bleu_score:.4f}")

    return result, attention_plot, bleu_score


def plot_attention(image, result, attention_plot, bleu_score=None, real_caption=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    temp_image = np.array(Image.open(image))
    len_result = len(result)

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle("🖼️ Attention Visualization", fontsize=16)

    # ==== Affichage de l'image originale ====
    ax = fig.add_subplot((len_result + 1) // 2 + 2, 2, 1)
    ax.set_title("Original Image")
    ax.imshow(temp_image)
    ax.axis("off")

    # ==== Affichage des textes ====
    text_str = ""
    if real_caption:
        text_str += f"📜 Real Caption:\n{' '.join(real_caption)}\n\n"
    text_str += f"🔮 Predicted:\n{' '.join(result)}\n"
    if bleu_score is not None:
        text_str += f"\n🟦 BLEU Score: {bleu_score:.4f}"

    # Affichage dans une cellule vide
    ax = fig.add_subplot((len_result + 1) // 2 + 2, 2, 2)
    ax.text(0.5, 0.5, text_str, wrap=True, fontsize=12, ha='center', va='center')
    ax.axis("off")

    # ==== Affichage des attentions ====
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot((len_result + 1) // 2 + 2, 2, l + 3)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
