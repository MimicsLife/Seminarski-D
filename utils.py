import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataProcessor:
    def __init__(self):
        self.vectorizer = None
        self.label_encoder = None
        self.tokenizer = None
        self.max_sequence_length = 0
    
    def clean_text(self, text):
        """Čisti tekst - uklanja specijalne karaktere i višestruke razmake"""
        if isinstance(text, str):
            # Ukloni specijalne karaktere, ostavi slova, brojeve i osnovne interpunkcije
            text = re.sub(r'[^\w\s]', ' ', text)
            # Ukloni višestruke razmake
            text = re.sub(r'\s+', ' ', text)
            # Skini razmake sa početka i kraja
            text = text.strip()
            return text.lower()
        return ""
    
    def prepare_traditional_features(self, texts, method='tfidf', max_features=5000, fit=True):
        """Priprema feature za tradicionalne modele"""
        texts_clean = [self.clean_text(text) for text in texts]
        
        if fit or self.vectorizer is None:
            if method == 'tfidf':
                self.vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words=None,  # Radi sa više jezika
                    analyzer='char_wb',  # Radi sa karakterima umesto rečima
                    ngram_range=(2, 4)  # Bigrami, trigrami i quadgrami karaktera
                )
            else:  # bow
                self.vectorizer = CountVectorizer(
                    max_features=max_features,
                    stop_words=None,
                    analyzer='char_wb',
                    ngram_range=(2, 4)
                )

            features = self.vectorizer.fit_transform(texts_clean)
        else:
            # transform using already-fitted vectorizer
            features = self.vectorizer.transform(texts_clean)

        return features
    
    def prepare_neural_features(self, texts, max_words=10000, max_len=200, fit=True):
        """Priprema feature za neuronske mreže"""
        texts_clean = [self.clean_text(text) for text in texts]
        if fit or self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts_clean)

            sequences = self.tokenizer.texts_to_sequences(texts_clean)

            # Pronađi maksimalnu dužinu sekvence
            self.max_sequence_length = max(len(seq) for seq in sequences)
            self.max_sequence_length = min(self.max_sequence_length, max_len)

            features = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        else:
            # transform using existing tokenizer and max_sequence_length
            sequences = self.tokenizer.texts_to_sequences(texts_clean)
            features = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')

        return features
    
    def encode_labels(self, labels):
        """Enkodira labele"""
        self.label_encoder = LabelEncoder()
        return self.label_encoder.fit_transform(labels)
    
    def decode_labels(self, encoded_labels):
        """Dekodira labele"""
        return self.label_encoder.inverse_transform(encoded_labels)

def load_sample_data():
    """
    Kreira sample dataset za demonstraciju.
    U realnom slučaju, ovo bi bila zamena sa pravim datasetom.
    """
    data = {
        'text': [
            # Engleski - duži paragrafi
            "Language identification is a common preprocessing step in multilingual NLP pipelines. "
            "This paragraph discusses how short samples may be ambiguous, so longer context often helps models disambiguate closely related languages by providing syntactic and lexical cues. "
            "The goal of the dataset here is to provide richer examples that include varied vocabulary, punctuation and multiple sentences.",

            "In real-world applications, text can come from articles, user comments, or technical documentation where sentence structure and domain-specific words give important signals for classification. "
            "Models that see longer contiguous text often learn more robust features and generalize better to unseen examples.",

            "Natural language has idioms, named entities and collocations that are distributed unevenly across languages; including multi-sentence samples increases the chance those features appear and improves detection accuracy. "
            "These longer samples include examples of such patterns to simulate realistic input.",

            "This paragraph contains sample content about programming and data science, referencing libraries, tooling and common workflows that are useful for model training. "
            "It mentions terms like data preprocessing, model evaluation and visualization to provide lexical variety.",

            "Finally, longer English passages can show formal and informal styles, which further challenges classifiers but helps them learn more robust representations when present in the training set.",

            # Srpski - duži paragrafi
            "Detekcija jezika u stvarnim aplikacijama često zahteva analizu nekoliko rečenica kako bi se uočile karakteristične gramatičke strukture i reči. "
            "Ovaj duži primer sadrži više rečenica koje opisuju svakodnevne scenarije, tehničke teme i opšte komentare kako bi model dobio bogatiji kontekst. "
            "Uključene su uobičajene fraze, skraćenice i imena koja se pojavljuju u pisanom jeziku.",

            "Na primer, u tekstu o programiranju i analizi podataka nalaze se izrazi koji ukazuju na domenu problema: učenje mašina, preprocesiranje podataka, vizualizacija i biblioteke koje se koriste. "
            "Takvi primeri pomažu modelima da nauče terminologiju specifičnu za srpski jezik i regionalne varijante.",

            "Ovaj odlomak takođe sadrži opise svakodnevnih aktivnosti, kratke narativne sekvence i upotrebu interpunkcije koja može biti korisna kod izdvajanja karakterističnih uzoraka jezika. "
            "Cilj je da set sadrži tekstove različitih registara i stilova.",

            "Kombinacija formalnog i razgovornog jezika unutar uzoraka povećava raznovrsnost i pomaže u treniranju robusnijih klasifikatora koji se lakše prilagođavaju stvarnim podacima.",

            "Na kraju, primeri uključuju deskriptivne rečenice o nauci podataka i programskim alatima, što omogućava modelima da nauče i tehničku leksiku.",

            # Nemački - duži paragrafi
            "Spracherkennung ist ein wichtiger Schritt in vielen mehrsprachigen Anwendungen, besonders wenn Texte aus unterschiedlichen Domänen stammen. "
            "Längere Abschnitte ermöglichen es Modellen, morphologische und syntaktische Hinweise zu erfassen, die in kurzen Phrasen nicht sichtbar sind. "
            "Dieser Text enthält Fachbegriffe aus dem Bereich Datenverarbeitung sowie alltägliche Ausdrücke.",

            "Beispielsweise können Artikel über Programmierung oder maschinelles Lernen spezifische Phrasen und Fachtermini enthalten, die einem Modell helfen, die richtige Sprache zu identifizieren. "
            "Variationen in Stil und Satzbau werden ebenfalls abgebildet.",

            "Längere Beispiele, die Erklärungen, Nebensätze und Listen von Konzepten enthalten, sind nützlich, damit Modelle robuste Repräsentationen lernen, die nicht nur auf wenigen Signalen beruhen.",

            "Dieser Paragraph kombiniert narrative, erklärende und technische Sätze, um die Vielfalt der natürlichen Sprache in deutschen Texten zu repräsentieren.",

            # Francuski - duži paragrafi
            "La détection de la langue est essentielle pour le traitement automatique des langues, en particulier dans des environnements multilingues. "
            "Des exemples de texte plus longs fournissent un contexte grammatical et lexical qui aide à distinguer des langues proches. "
            "Ce passage inclut des termes liés à l'informatique, aux workflows de données et à des expressions courantes.",

            "Par exemple, un article qui décrit les étapes d'un projet de science des données contient du vocabulaire technique et des tournures de phrase spécifiques qui aident le modèle à apprendre. "
            "Nous incluons aussi des phrases conversationnelles afin de couvrir différents registres.",

            "Les textes plus détaillés permettent de capturer des indices morphologiques, des accords et des constructions syntaxiques caractéristiques du français écrit.",

            # Španski - duži paragrafi
            "La identificación del idioma es un componente clave en sistemas que procesan contenido multilingüe, como noticias o redes sociales. "
            "Un texto más extenso ofrece mayor contexto léxico y sintáctico, lo que facilita la distinción entre idiomas cercanos. "
            "Este ejemplo incluye descripciones técnicas, oraciones coloquiales y vocabulario de dominio específico.",

            "Por ejemplo, un párrafo que describe un flujo de trabajo de ciencia de datos contiene términos y expresiones que ayudan al clasificador a reconocer patrones propios del español. "
            "También se han añadido variaciones estilísticas para cubrir registros formales e informales.",

            "En conjunto, los pasajes largos aumentan la probabilidad de que aparezcan entidades y colocaciones informativas que mejoren la capacidad de los modelos para generalizar."
        ],
        'language': [
            # English (5)
            'english', 'english', 'english', 'english', 'english',
            # Serbian (5)
            'serbian', 'serbian', 'serbian', 'serbian', 'serbian',
            # German (4)
            'german', 'german', 'german', 'german',
            # French (3)
            'french', 'french', 'french',
            # Spanish (3)
            'spanish', 'spanish', 'spanish'
        ]
    }
    
    return pd.DataFrame(data)

def evaluate_model(model, X_test, y_test, model_type):
    """Evaluacija modela i prikaz metrika"""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    y_pred = model.predict(X_test)
    
    if model_type == 'neural':
        y_pred = np.argmax(y_pred, axis=1)
        if len(y_test.shape) > 1:
            y_test = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"EVALUACIJA {model_type.upper()} MODELA")
    print(f"{'='*50}")
    print(f"Tačnost: {accuracy:.4f}")
    print(f"\nDetaljni izveštaj:")
    print(classification_report(y_test, y_pred))
    
    # Konfuzione matrice
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Konfuziona matrica - {model_type} model')
    plt.ylabel('Stvarna vrednost')
    plt.xlabel('Predviđena vrednost')
    plt.show()
    
    return accuracy