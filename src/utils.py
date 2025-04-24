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

def load_tokenizer(path_to_token = "./../models/weights/captioning_token/captioning_tokenizer_GRU.json"):
    try:
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
    except ImportError:
        raise ImportError()

    with open(path_to_token, 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
        
    return tokenizer


def is_valid_image(path):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError()

    try:
        img_bytes = tf.io.read_file(path)
        decoded_img = tf.io.decode_image(img_bytes)
        return True
    except tf.errors.InvalidArgumentError as e:
        return False

def clean_invalid_images(datasets_base_path):
    try:
        import os
    except ImportError:
        raise ImportError()

    for root, dirs, files in os.walk(datasets_base_path):
        for file in files:
            if file.lower().endswith((".png", ".jpeg", ".png", ".bmp")):
                image_path = os.path.join(root, file)
                if not is_valid_image(image_path):
                    print(f"Removing invalid image: {image_path}")
                    os.remove(image_path)

def is_noisy_image(img_array, threshold=0.05):
    import cv2
    import numpy as np

    # Convertir en niveaux de gris
    gray = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Appliquer un flou gaussien pour lisser les détails
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Calculer l'image de différence
    diff = cv2.absdiff(gray, blurred)

    # Normaliser et calculer la variance de la différence
    diff = diff.astype(np.float32) / 255.0
    noise_score = np.std(diff)

    return noise_score > threshold

def denoise_images(input_dir, output_dir, model, target_size=(256, 256), check_noise=False):
    import os
    import numpy as np
    import shutil
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                # Chargement et préparation de l'image
                img = load_img(input_path, target_size=target_size)
                img_array = img_to_array(img) / 255.0

                if not check_noise or is_noisy_image(img_array):
                    img_array_exp = np.expand_dims(img_array, axis=0)
                    denoised = model.predict(img_array_exp)
                    denoised_img = array_to_img(denoised[0])
                    save_img(output_path, denoised_img)
                    print(f"Image débruitée : {filename}")
                else:
                    shutil.copy(input_path, output_path)
                    print(f"Image non bruitée copiée : {filename}")
            except Exception as e:
                print(f"Erreur lors du traitement de {filename} : {e}")


def load_caption_dataset(annotation_path, image_folder, nb_images=None):
    try:
        import json
        import collections
        import os

    except ImportError:
        raise ImportError()
    
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = '<start> ' + val['caption'] + ' <end>'
        image_path = os.path.join(image_folder, 'COCO_train2014_' + '%012d.jpg' % val['image_id'])
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    train_image_paths = image_paths[:nb_images] 
    print(f"{len(train_image_paths)} images sélectionnées.")

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        captions = image_path_to_caption[image_path]
        train_captions.extend(captions)
        img_name_vector.extend([image_path] * len(captions))

    print(f"Total de {len(train_captions)} légendes pour {len(train_image_paths)} images.")

    return train_captions, img_name_vector

def load_captionning_image(image_path):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError()
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def build_image_model():
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError()
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = base_model.input
    hidden_layer = base_model.layers[-1].output
    return tf.keras.Model(new_input, hidden_layer)

    
def preprocess_caption_annotation(train_captions=None, is_training: bool =False, tokenizer_path=(), nb_top_word : int = 5000):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError()
    
    def calc_max_length(tensor):
            return max(len(t) for t in tensor)
    
    if is_training:

        top_k = nb_top_word

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                        oov_token="<unk>",
                                                        filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        tokenizer.fit_on_texts(train_captions)

        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'

        train_seqs = tokenizer.texts_to_sequences(train_captions)

        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
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
    
def load_image(image_path):
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

    BATCH_SIZE = 64
    BUFFER_SIZE = 1000 

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
    
    return dataset, img_name_train, cap_train, dataset_val, img_name_val, cap_val, num_steps, val_num_steps, cap_val_vector

def captionning_evaluate(image_path, encoder, decoder, max_length=None, tokenizer=None, is_gru: bool = True, image_features_extract_model=None):
    try:
    
        import numpy as np
        import tensorflow as tf
    except ImportError:
        return None, None, None

    attention_plot = np.zeros((max_length, 64))
    hidden = decoder.reset_state(batch_size=1)

    img, _ = load_image(image_path)
    temp_input = tf.expand_dims(img, 0)
    img_tensor_val = image_features_extract_model(temp_input)
    print("Inception output:", img_tensor_val.shape)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    print("Reshaped:", img_tensor_val.shape)
    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        top_k = tf.nn.top_k(predictions[0], k=5)
        top_ids = top_k.indices.numpy()
        top_scores = top_k.values.numpy()

        for rank, (word_id, score) in enumerate(zip(top_ids, top_scores), start=1):
            word = tokenizer.index_word.get(word_id, "<unk>")
            # print(f"   {rank}. {word} (id: {word_id}) — score: {score:.4f}")

        predicted_id = top_ids[0]
        predicted_word = tokenizer.index_word.get(predicted_id, '<unk>')
        result.append(predicted_word)

        if predicted_word == '<end>':
            # print("✅ Fin de génération détectée avec <end>")
            break

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def run_captioning_on_folder(folder_path, encoder, decoder, tokenizer, max_length, image_features_extract_model, is_gru=True, max_images=10):
    try:
        import os
        import random
    except ImportError:
        raise ImportError()
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    selected_images = random.sample(image_files, min(max_images, len(image_files)))

    for fname in selected_images:
        img_path = os.path.join(folder_path, fname)
        print(f"\{fname}")
        result, attn = captionning_evaluate(
            img_path, encoder, decoder,
            max_length=max_length,
            tokenizer=tokenizer,
            is_gru=is_gru,
            image_features_extract_model=image_features_extract_model
        )
        plot_attention(img_path, result, attn)

def plot_attention(image_path, result, attention_plot):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
    except ImportError:
        raise ImportError()
    
    temp_image = np.array(Image.open(image_path))
    fig = plt.figure(figsize=(12, 12))
    len_result = len(result)
    fig.suptitle("Prediction: " + " ".join(result), fontsize=14)

    ax = fig.add_subplot((len_result + 1) // 2 + 1, 2, 1)
    ax.set_title("Original")
    ax.imshow(temp_image)
    ax.axis("off")

    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot((len_result + 1) // 2 + 1, 2, l + 2)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def add_noise_to_images_in_folder(input_dir, output_dir, noise_factor_range=(0.1, 0.5), pixel_noise_prob=0.4, target_size=(128, 128)):
    try:
        import os
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
    except ImportError as e:
        print(f"Erreur d'importation : {e}")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def apply_noise(img):
        noise_factor = np.random.uniform(*noise_factor_range)
        noise_type = np.random.randint(0, 3)
        mask = np.random.rand(*img.shape) < pixel_noise_prob

        if noise_type == 0:
            # Gaussian noise
            noise = np.random.normal(0, 1, img.shape)
            noisy = img + noise_factor * noise * mask
        elif noise_type == 1:
            # Speckle noise
            noise = np.random.normal(0, 1, img.shape)
            noisy = img + img * noise * noise_factor * mask
        else:
            # Salt and pepper noise
            rnd = np.random.rand(*img.shape)
            salt = (rnd < (pixel_noise_prob * noise_factor / 2)).astype(np.float32)
            pepper = (rnd > (1.0 - pixel_noise_prob * noise_factor / 2)).astype(np.float32)
            noisy = img * (1.0 - salt - pepper) + salt

        return np.clip(noisy, 0.0, 1.0)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                img = load_img(input_path, target_size=target_size)
                img_array = img_to_array(img) / 255.0  # Normalisation
                noisy_img_array = apply_noise(img_array)
                noisy_img = array_to_img(noisy_img_array)
                save_img(output_path, noisy_img)
                print(f"Image bruitée enregistrée : {filename}")
            except Exception as e:
                print(f"Erreur sur {filename} : {e}")
