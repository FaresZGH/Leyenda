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

    return class_weights

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