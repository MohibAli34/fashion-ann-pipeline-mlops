import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    x_test = np.load('data/processed/x_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    model = tf.keras.models.load_model('models/model.h5')
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    metrics = {
        'test_loss': float(loss),
        'test_accuracy': float(accuracy)
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Predict and create confusion matrix
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('models/confusion_matrix.png')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Metrics and confusion matrix saved.")

if __name__ == '__main__':
    main()
