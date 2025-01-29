import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score

class CustomLogger(Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Calculate metrics
        f1_score_value = self.calculate_f1_score()
        precision_value = self.calculate_precision()
        recall_value = self.calculate_recall()
        learning_rate = self.get_learning_rate()
        batch_size = self.params.get('batch_size', 'Unknown')
        epochs = self.params.get('epochs', 'Unknown')
        input_data_stats = self.get_input_data_stats()

        # Log metrics to file
        with open(self.log_file, 'a') as f:
            f.write(
                f"Epoch: {epoch + 1}, "
                f"Training Loss: {logs.get('loss')}, "
                f"Validation Loss: {logs.get('val_loss')}, "
                f"Training Accuracy: {logs.get('accuracy')}, "
                f"Validation Accuracy: {logs.get('val_accuracy')}, "
                f"F1 Score: {f1_score_value}, "
                f"Precision: {precision_value}, "
                f"Recall: {recall_value}, "
                f"Learning Rate: {learning_rate}, "
                f"Batch Size: {batch_size}, "
                f"Epochs: {epochs}, "
                f"Input Data Stats: {input_data_stats}, "
                f"Loss Function Values: {logs.get('loss')}\n"
            )

    def calculate_f1_score(self):
        y_true = np.argmax(self.validation_data[1], axis=1)
        y_pred = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        return f1_score(y_true, y_pred, average='weighted')

    def calculate_precision(self):
        y_true = np.argmax(self.validation_data[1], axis=1)
        y_pred = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        return precision_score(y_true, y_pred, average='weighted')

    def calculate_recall(self):
        y_true = np.argmax(self.validation_data[1], axis=1)
        y_pred = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        return recall_score(y_true, y_pred, average='weighted')

    def get_learning_rate(self):
        return self.model.optimizer.learning_rate.numpy()

    def get_gradient_norms(self):
        gradients = self.model.optimizer.get_gradients(self.model.total_loss, self.model.trainable_variables)
        norms = [tf.norm(grad).numpy() for grad in gradients]
        return norms

    def get_gradient_magnitudes(self):
        gradients = self.model.optimizer.get_gradients(self.model.total_loss, self.model.trainable_variables)
        magnitudes = [tf.norm(grad).numpy() for grad in gradients]
        return magnitudes


    def get_input_data_stats(self):
        # Implement a way to get input data statistics if necessary
        return "Data stats not implemented"
