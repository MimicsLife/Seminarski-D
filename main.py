import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import DataProcessor, load_sample_data, evaluate_model
from model import TraditionalModels, NeuralNetworkModels

def main():
    print("DETEKCIJA JEZIKA TEKSTA - KOMPARATIVNA ANALIZA")
    print("=" * 60)
    
    # 1. Učitavanje i priprema podataka
    print("\n1. UČITAVANJE I PRIREMA PODATAKA")
    print("-" * 40)
    
    df = load_sample_data()
    print(f"Dataset: {len(df)} primera, {len(df['language'].unique())} jezika")
    print("Jezici:", df['language'].unique())
    
    processor = DataProcessor()
    
    # Podela na train i test
    X = df['text'].values
    y = df['language'].values
    
    # Enkodiranje labela
    y_encoded = processor.encode_labels(y)
    
    # Podela na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # Dodatna podela za neuronske mreže (train/val)
    # Proračunaj odgovarajuću veličinu validacione grupe tako da stratifikacija
    # obezbedi bar po jedan primer iz svake klase u oba skupa. Ako to nije
    # moguće, padamo na nestratifikovanu podelu.
    n_classes_train = len(np.unique(y_train))
    # želimo približno 20% za validaciju, ali najmanje po jedan primer po klasi
    desired_val = int(np.ceil(len(X_train) * 0.2))
    val_count = max(desired_val, n_classes_train)

    # Ako posle izdvajanja validacione grupe u treningu ostane manje primera
    # nego broja klasa, ne možemo stratifikovati — uradi nestratifikovanu podelu
    if len(X_train) - val_count < n_classes_train:
        X_train_neural, X_val_neural, y_train_neural, y_val_neural = train_test_split(
            X_train, y_train, test_size=val_count, random_state=42
        )
    else:
        X_train_neural, X_val_neural, y_train_neural, y_val_neural = train_test_split(
            X_train, y_train, test_size=val_count, random_state=42, stratify=y_train
        )
    
    # 2. TRADICIONALNI MODELI
    print("\n2. TRENIRANJE TRADICIONALNIH MODELA")
    print("-" * 40)
    
    # Priprema feature za tradicionalne modele
    X_train_traditional = processor.prepare_traditional_features(X_train, fit=True)
    X_test_traditional = processor.prepare_traditional_features(X_test, fit=False)
    
    traditional_models = TraditionalModels()
    
    # Treniranje tradicionalnih modela
    lr_model, lr_acc = traditional_models.train_logistic_regression(
        X_train_traditional, y_train, X_test_traditional, y_test
    )
    
    # print("\nEvaluacija Logistic Regression modela:")
    # evaluate_model(lr_model, X_test_traditional, y_test, 'traditional')

    rf_model, rf_acc = traditional_models.train_random_forest(
        X_train_traditional, y_train, X_test_traditional, y_test
    )
    
    #print("\nEvaluacija Random Forest modela:")
    #evaluate_model(rf_model, X_test_traditional, y_test, 'traditional')

    svm_model, svm_acc = traditional_models.train_svm(
        X_train_traditional, y_train, X_test_traditional, y_test
    )
    
    #print("\nEvaluacija SVM modela:")
    #evaluate_model(svm_model, X_test_traditional, y_test, 'traditional')

    traditional_models.compare_models()
    
    # 3. NEURONSKE MREŽE
    print("\n3. TRENIRANJE NEURONSKIH MREŽA")
    print("-" * 40)
    
    # Priprema feature za neuronske mreže
    X_train_neural_features = processor.prepare_neural_features(X_train_neural, fit=True)
    X_val_neural_features = processor.prepare_neural_features(X_val_neural, fit=False)
    X_test_neural_features = processor.prepare_neural_features(X_test, fit=False)
    
    # Konvertuj labele u kategorijalne
    num_classes = len(np.unique(y_encoded))
    y_train_categorical = tf.keras.utils.to_categorical(y_train_neural, num_classes)
    y_val_categorical = tf.keras.utils.to_categorical(y_val_neural, num_classes)
    y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # Odredi sigurnu veličinu vokabulara za embedding tako da pokrije najveći index
    # koji se pojavljuje u sekvencama (ponekad tokenizer.word_index može biti
    # manji/različit od najvećeg indeksiranog tokena u sekvencama).
    max_token_index = int(np.max(X_train_neural_features)) if X_train_neural_features.size > 0 else 0
    inferred_vocab_size = max(len(processor.tokenizer.word_index) + 1, max_token_index + 1)

    neural_models = NeuralNetworkModels(
        num_classes=num_classes,
        vocab_size=inferred_vocab_size,
        sequence_length=processor.max_sequence_length
    )
    
    # Treniranje neuronskih modela
    cnn_model, cnn_history = neural_models.train_cnn(
        X_train_neural_features, y_train_categorical,
        X_val_neural_features, y_val_categorical,
        epochs=30
    )
    
    #print("\nEvaluacija CNN modela na test podacima:")
    #evaluate_model(cnn_model, X_test_neural_features, y_test_categorical, 'neural')

    lstm_model, lstm_history = neural_models.train_lstm(
        X_train_neural_features, y_train_categorical,
        X_val_neural_features, y_val_categorical,
        epochs=30
    )

    #print("\nEvaluacija LSTM modela na test podacima:")
    #evaluate_model(lstm_model, X_test_neural_features, y_test_categorical, 'neural')

    dense_model, dense_history = neural_models.train_dense(
        X_train_neural_features, y_train_categorical,
        X_val_neural_features, y_val_categorical,
        epochs=30
    )
    
    #print("\nEvaluacija Dense modela na test podacima:")
    #evaluate_model(dense_model, X_test_neural_features, y_test_categorical, 'neural')

    neural_models.compare_models()
    
    # 4. KOMPARATIVNA ANALIZA
    print("\n4. KOMPARATIVNA ANALIZA SVIH MODELA")
    print("-" * 40)
    
    # Prikupljanje rezultata
    results = {
        'Logistic Regression': lr_acc,
        'Random Forest': rf_acc,
        'SVM': svm_acc,
        'CNN': neural_models.models['cnn']['accuracy'],
        'LSTM': neural_models.models['lstm']['accuracy'],
        'Dense NN': neural_models.models['dense']['accuracy']
    }

    #print(f"\nEvaluacija najboljeg tradicionalnog modela:")
    #best_trad_name = [k for k, v in traditional_models.models.items() if v['accuracy'] == traditional_models.best_score][0]
    #print(f"Model: {best_trad_name.replace('_', ' ').title()}")
    #evaluate_model(traditional_models.best_model, X_test_traditional, y_test, 'traditional')
 
    #print(f"\nEvaluacija najboljeg neuronskog modela:")
    #best_neural_name = [k for k, v in neural_models.models.items() if v['accuracy'] == neural_models.best_score][0]
    #print(f"Model: {best_neural_name.upper()}")
    #evaluate_model(neural_models.best_model, X_test_neural_features, y_test_categorical, 'neural')
    
    # Prikaz rezultata
    print("\n" + "="*50)
    print("FINALNA KOMPARACIJA")
    print("="*50)
    
    for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:<20} | Tačnost: {accuracy:.4f}")
    
    # Vizuelizacija
    plt.figure(figsize=(12, 6))
    
    # Grafik tačnosti
    plt.subplot(1, 2, 1)
    models_names = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(models_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink', 'wheat'])
    plt.title('Uporedna tačnost modela', fontsize=14, fontweight='bold')
    plt.xlabel('Modeli')
    plt.ylabel('Tačnost')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Dodaj vrednosti na grafiku
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Grafik konvergencije neuronskih mreža
    plt.subplot(1, 2, 2)
    plt.plot(cnn_history.history['val_accuracy'], label='CNN', linewidth=2)
    plt.plot(lstm_history.history['val_accuracy'], label='LSTM', linewidth=2)
    plt.plot(dense_history.history['val_accuracy'], label='Dense NN', linewidth=2)
    plt.title('Konvergencija neuronskih mreža', fontsize=14, fontweight='bold')
    plt.xlabel('Epoha')
    plt.ylabel('Validation Tačnost')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. TESTIRANJE NA PRIMERIMA
    print("\n5. TESTIRANJE NA PRIMERIMA")
    print("-" * 40)
    
    test_texts = [
        # Engleski - duži test primer
        "Language detection models work better when provided with sentences that include a range of vocabulary and sentence structures. "
        "This example paragraph mentions data processing, model evaluation, and visualization so the classifier sees diverse lexical cues across multiple sentences. "
        "It is intended to represent a typical excerpt from a technical blog or article.",

        # Srpski - duži test primer
        "Ovaj duži primer teksta na srpskom sadrži nekoliko rečenica koje opisuju proces obrade podataka i primenu modela u praksi. "
        "U tekstu se pominju alati, koraci preprocesiranja i kratki primeri upotrebe, čime se pruža bogatiji kontekst za detekciju jezika.",

        # Nemački - duži test primer
        "Dieses längere Beispiel erklärt Konzepte aus dem Bereich Datenanalyse und maschinelles Lernen in mehreren Sätzen, um dem Klassifikator mehr Kontext zu geben. "
        "Es enthält Fachbegriffe, erläuternde Nebensätze und gebräuchliche Ausdrücke.",

        # Francuski - duži test primer
        "Ce passage en français décrit de manière concise des étapes typiques d'un projet de science des données, incluant la préparation des données, l'entraînement d'un modèle et l'interprétation des résultats. "
        "La variété lexicale et syntaxique aide à l'identification correcte de la langue.",

        # Španski - duži test primer
        "Este texto en español presenta varias oraciones que comentan sobre técnicas de análisis de datos, la evaluación de modelos y la presentación de resultados. "
        "Incluye términos técnicos y frases coloquiales para ofrecer un contexto más amplio al clasificador.",
        
        # Hrvatski - duži test primer
        "Ovaj tekst na hrvatskom sadrži nekoliko rečenica o obradi podataka, modeliranju i evaluaciji. "
        "Sadrži tehničke termine i primjere upotrebe koji pomažu modelu u prepoznavanju karakterističnih leksičkih obrazaca. "
        "Namijenjen je kao reprezentativan uzorak za testiranje prepoznavanja jezika.",

        # Bosanski - duži test primer
        "Ovaj primjer teksta na bosanskom opisuje korake u pripremi podataka i treniranju modela za klasifikaciju. "
        "Sadrži mješavinu formalnog i neformalnog jezika, tehničke izraze i uobičajene fraze kako bi se model testirao u realnim scenarijima. "
        "Cilj je obezbijediti dovoljan kontekst za pouzdano prepoznavanje jezika.",
    ]
    
    # Koristimo najbolji tradicionalni i najbolji neuronski model
    best_traditional = traditional_models.best_model
    best_neural = neural_models.best_model
    
    print("\nPredviđanja na test primerima:")
    print("-" * 50)
    
    for i, text in enumerate(test_texts):
        # Tradicionalni model (transformacija pomoću već istreniranog vektorizatora)
        text_features = processor.prepare_traditional_features([text], fit=False)
        trad_pred = best_traditional.predict(text_features)[0]
        trad_lang = processor.decode_labels([trad_pred])[0]
        
        # Neuronski model
        text_neural_features = processor.prepare_neural_features([text], fit=False)
        neural_pred = best_neural.predict(text_neural_features)
        neural_lang = processor.decode_labels([np.argmax(neural_pred[0])])[0]
        
        print(f"\nTekst {i+1}: '{text}'")
        print(f"  Tradicionalni model: {trad_lang}")
        print(f"  Neuronski model: {neural_lang}")
    
    # 6. ZAKLJUČAK
    print("\n6. ZAKLJUČAK")
    print("-" * 40)
    
    best_trad_name = [k for k, v in traditional_models.models.items() if v['accuracy'] == traditional_models.best_score][0]
    best_neural_name = [k for k, v in neural_models.models.items() if v['accuracy'] == neural_models.best_score][0]
    
    print(f"Najbolji tradicionalni model: {best_trad_name.replace('_', ' ').title()}")
    print(f"Najbolji neuronski model: {best_neural_name.upper()}")
    print(f"Ukupno najbolji model: {max(results, key=results.get)}")
    
    # Preporuka
    best_overall_model = max(results, key=results.get)
    best_accuracy = results[best_overall_model]
    
    print(f"\nPREPORUKA: Koristiti {best_overall_model} model")
    print(f"   sa tačnošću od {best_accuracy:.1%}")

if __name__ == "__main__":
    main()