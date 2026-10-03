import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
        
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Metrics and confusion matrix saved.")

if __name__ == '__main__':
    main()
