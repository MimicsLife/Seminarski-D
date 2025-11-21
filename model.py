import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

class TraditionalModels:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Trenira Logistic Regression model"""
        print("Treniranje Logistic Regression modela...")
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs'
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['logistic_regression'] = {
            'model': model,
            'accuracy': accuracy
        }
        
        if accuracy > self.best_score:
            self.best_score = accuracy
            self.best_model = model
        
        print(f"Logistic Regression tačnost: {accuracy:.4f}")
        return model, accuracy
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Trenira Random Forest model"""
        print("Treniranje Random Forest modela...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['random_forest'] = {
            'model': model,
            'accuracy': accuracy
        }
        
        if accuracy > self.best_score:
            self.best_score = accuracy
            self.best_model = model
        
        print(f"Random Forest tačnost: {accuracy:.4f}")
        return model, accuracy
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Trenira SVM model"""
        print("Treniranje SVM modela...")
        model = SVC(
            kernel='linear',
            random_state=42,
            probability=True
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['svm'] = {
            'model': model,
            'accuracy': accuracy
        }
        
        if accuracy > self.best_score:
            self.best_score = accuracy
            self.best_model = model
        
        print(f"SVM tačnost: {accuracy:.4f}")
        return model, accuracy
    
    def compare_models(self):
        """Poređenje performansi tradicionalnih modela"""
        print("\n" + "="*60)
        print("POREDENJE TRADICIONALNIH MODELA")
        print("="*60)
        
        for name, result in self.models.items():
            print(f"{name.replace('_', ' ').title():<20} | Tačnost: {result['accuracy']:.4f}")
        
        best_model_name = [k for k, v in self.models.items() if v['accuracy'] == self.best_score][0]
        print(f"\nNajbolji model: {best_model_name.replace('_', ' ').title()} (tačnost: {self.best_score:.4f})")

class NeuralNetworkModels:
    def __init__(self, num_classes, vocab_size=10000, sequence_length=200):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
    
    def build_cnn_model(self, embedding_dim=100):
        """Gradi CNN model za klasifikaciju teksta"""
        model = Sequential([
            Embedding(self.vocab_size, embedding_dim, input_length=self.sequence_length),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_lstm_model(self, embedding_dim=100):
        """Gradi LSTM model za klasifikaciju teksta"""
        model = Sequential([
            Embedding(self.vocab_size, embedding_dim, input_length=self.sequence_length),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_dense_model(self, embedding_dim=50):
        """Gradi jednostavan Dense model"""
        model = Sequential([
            Embedding(self.vocab_size, embedding_dim, input_length=self.sequence_length),
            tf.keras.layers.Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_cnn(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Trenira CNN model"""
        print("Treniranje CNN modela...")
        model = self.build_cnn_model()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        val_accuracy = max(history.history['val_accuracy'])
        
        self.models['cnn'] = {
            'model': model,
            'history': history,
            'accuracy': val_accuracy
        }
        
        if val_accuracy > self.best_score:
            self.best_score = val_accuracy
            self.best_model = model
        
        print(f"CNN validation tačnost: {val_accuracy:.4f}")
        return model, history
    
    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Trenira LSTM model"""
        print("Treniranje LSTM modela...")
        model = self.build_lstm_model()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        val_accuracy = max(history.history['val_accuracy'])
        
        self.models['lstm'] = {
            'model': model,
            'history': history,
            'accuracy': val_accuracy
        }
        
        if val_accuracy > self.best_score:
            self.best_score = val_accuracy
            self.best_model = model
        
        print(f"LSTM validation tačnost: {val_accuracy:.4f}")
        return model, history
    
    def train_dense(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Trenira Dense model"""
        print("Treniranje Dense modela...")
        model = self.build_dense_model()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        val_accuracy = max(history.history['val_accuracy'])
        
        self.models['dense'] = {
            'model': model,
            'history': history,
            'accuracy': val_accuracy
        }
        
        if val_accuracy > self.best_score:
            self.best_score = val_accuracy
            self.best_model = model
        
        print(f"Dense validation tačnost: {val_accuracy:.4f}")
        return model, history
    
    def compare_models(self):
        """Poređenje performansi neuronskih mreža"""
        print("\n" + "="*60)
        print("POREDENJE NEURONSKIH MODELA")
        print("="*60)
        
        for name, result in self.models.items():
            print(f"{name.upper():<10} | Validation tačnost: {result['accuracy']:.4f}")
        
        best_model_name = [k for k, v in self.models.items() if v['accuracy'] == self.best_score][0]
        print(f"\nNajbolji neuronski model: {best_model_name.upper()} (tačnost: {self.best_score:.4f})")