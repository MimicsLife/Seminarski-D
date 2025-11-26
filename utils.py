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
            else:  # bag-of-words (BOW)
                self.vectorizer = CountVectorizer(
                    max_features=max_features,
                    stop_words=None,
                    analyzer='char_wb',
                    ngram_range=(2, 4)
                )

            features = self.vectorizer.fit_transform(texts_clean)
        else:
            # transformacija pomoću već istreniranog vektorizatora
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
            # transformacija pomoću postojećeg tokenizera i vrednosti `max_sequence_length`
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
            # Engleski
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

            "Machine learning algorithms require careful tuning of hyperparameters to achieve optimal performance. "
            "Researchers often use techniques such as cross-validation and grid search to find the best configuration. "
            "The choice of hyperparameters can significantly impact the model's ability to generalize to new data.",

            "Deep learning frameworks like TensorFlow and PyTorch have become industry standards for building neural networks. "
            "These frameworks provide high-level APIs that simplify model development while maintaining flexibility. "
            "Understanding the underlying mathematics is crucial for debugging and optimizing deep learning systems.",

            "Cloud computing platforms such as AWS, Google Cloud, and Azure offer scalable solutions for machine learning. "
            "These services handle infrastructure management, allowing data scientists to focus on model development. "
            "Cost optimization is an important consideration when using cloud resources for intensive computational tasks.",

            "Natural language processing has applications in sentiment analysis, machine translation, and question answering. "
            "Recent advances in transformer architectures have led to state-of-the-art results across many NLP benchmarks. "
            "Transfer learning from pre-trained models has become a standard practice in the field.",

            "Data quality is paramount in machine learning projects, as biased or noisy data can lead to poor results. "
            "Data scientists spend significant time on exploratory data analysis and data cleaning before training. "
            "Proper data validation and testing procedures help ensure the reliability of deployed systems.",

            # Srpski
            "Detekcija jezika u stvarnim aplikacijama često zahteva analizu nekoliko rečenica kako bi se uočile karakteristične gramatičke strukture i reči. "
            "Ovaj duži primer sadrži više rečenica koje opisuju svakodnevne scenarije, tehničke teme i opšte komentare kako bi model dobio bogatiji kontekst. "
            "Uključene su uobičajene fraze, skraćenice i imena koja se pojavljuju u pisanom jeziku.",

            "Na primer, u tekstu o programiranju i analizi podataka nalaze se izrazi koji ukazuju na domenu problema: učenje mašina, preprocesiranje podataka, vizualizacija i biblioteke koje se koriste. "
            "Takvi primeri pomažu modelima da nauče terminologiju specifičnu za srpski jezik i regionalne varijante.",

            "Ovaj odlomak takođe sadrži opise svakodnevnih aktivnosti, kratke narativne sekvence i upotrebu interpunkcije koja može biti korisna kod izdvajanja karakterističnih uzoraka jezika. "
            "Cilj je da set sadrži tekstove različitih registara i stilova.",

            "Kombinacija formalnog i razgovornog jezika unutar uzoraka povećava raznovrsnost i pomaže u treniranju robusnijih klasifikatora koji se lakše prilagođavaju stvarnim podacima.",

            "Na kraju, primeri uključuju deskriptivne rečenice o nauci podataka i programskim alatima, što omogućava modelima da nauče i tehničku leksiku.",

            "Matematika je osnova svih algoritama maskinskog ucenja i dubokog ucenja. "
            "Razumevanje linearne algebre, teorije verovatnoce i analize je kljucno za razvoj efikasnih modela. "
            "Inzenjeri koji rade na ovom polju cesto koriste Numpy i druge biblioteke za numericke proracune.",

            "Baze podataka su kriticne za skladistenje i upravljanje velikim kolicinama podataka. "
            "Relacione baze kao sto su SQL, kao i NoSQL sistemi, imaju svoje prednosti i nedostatke. "
            "Izbor odgovarajuce baze zavisi od specificnih zahteva projekta i ocekivanog volumena podataka.",

            "Bezbednost podataka je od presudnog znacaja u svim IT sistemima. "
            "Sifiranje, autentifikacija i autorizacija su osnovni mehanizmi za zastitu podataka. "
            "Redovne provere bezbednosti i azuriranja sigurnosnih zakrpa su neophodne za odrzavanje sistema.",

            "Industija tehnologije se brzo menja sa novim tehnologijama i trendovima koji se pojavljuju. "
            "Fleksibilnost i sposobnost za ucenje su kljucne karakteristike uspesnih tehnicara. "
            "Networking i ucesce u zajednicama omogućava stalno usavrsavanje i razmenu iskustva.",

            "Razvoj softvera zahteva kombinaciju tehnickog znanja i razumevanja poslovnih zahteva. "
            "Kvalitetna dokumentacija i jasna komunikacija timova su kljucne za uspesne projekte. "
            "Pregled koda od strane kolega sprečava greške i poboljsava kvalitetu koda.",

            # Nemački
            "Spracherkennung ist ein wichtiger Schritt in vielen mehrsprachigen Anwendungen, besonders wenn Texte aus unterschiedlichen Domänen stammen. "
            "Längere Abschnitte ermöglichen es Modellen, morphologische und syntaktische Hinweise zu erfassen, die in kurzen Phrasen nicht sichtbar sind. "
            "Dieser Text enthält Fachbegriffe aus dem Bereich Datenverarbeitung sowie alltägliche Ausdrücke.",

            "Beispielsweise können Artikel über Programmierung oder maschinelles Lernen spezifische Phrasen und Fachtermini enthalten, die einem Modell helfen, die richtige Sprache zu identifizieren. "
            "Variationen in Stil und Satzbau werden ebenfalls abgebildet.",

            "Längere Beispiele, die Erklärungen, Nebensätze und Listen von Konzepten enthalten, sind nützlich, damit Modelle robuste Repräsentationen lernen, die nicht nur auf wenigen Signalen beruhen.",

            "Dieser Paragraph kombiniert narrative, erklärende und technische Sätze, um die Vielfalt der natürlichen Sprache in deutschen Texten zu repräsentieren.",

            "Kuenstliche Intelligenz und maschinelles Lernen haben revolutionaere Veraenderungen in vielen Industriebranchen bewirkt. "
            "Die Anwendung von neuronalen Netzen ermoeglicht es Computern, komplexe Muster in grossen Datenmengen zu erkennen. "
            "Forscher arbeiten kontinuierlich an der Verbesserung von Algorithmen und Trainingsmethoden.",

            "Software-Entwicklung erfordert nicht nur technisches Wissen, sondern auch Faehigkeiten in Projektmanagement und Teamwork. "
            "Agile Methoden wie Scrum und Kanban haben sich in der modernen Softwareentwicklung etabliert. "
            "Code-Qualitaet und Wartbarkeit sind wichtige Faktoren fuer den langfristigen Erfolg von Projekten.",

            "Datenbanken sind das Rueckgrat vieler moderner Anwendungen und speichern kritische Geschaeftsinformationen. "
            "Die Wahl zwischen relationalen und nicht-relationalen Datenbanken haengt von den spezifischen Anforderungen ab. "
            "Indexierung und Query-Optimierung sind wichtige Techniken zur Verbesserung der Datenbankleistung.",

            "Cybersecurity ist in einer zunehmend vernetzten Welt von grosser Bedeutung. "
            "Verschiedene Bedrohungen wie Malware, Phishing und Ransomware erfordern mehrschichtige Schutzmassnahmen. "
            "Regelmaessige Sicherheitstrainings und Updates sind essentiell fuer den Schutz von Systemen.",

            "Webentwicklung hat sich mit neuen Frameworks und Technologien dramatisch weiterentwickelt. "
            "Single-Page Applications und Progressive Web Apps bieten verbesserte Benutzererfahrungen. "
            "Die Kombination von Frontend- und Backend-Technologien ermoeglicht die Erstellung komplexer Web-Anwendungen.",

            "Mobile Anwendungsentwicklung ist heute ein entscheidender Aspekt der digitalen Strategie von Unternehmen. "
            "Plattformen wie iOS und Android erfordern unterschiedliche Entwicklungsansaetze und Optimierungen. "
            "Cross-Platform-Entwicklungsframeworks bieten Effizienzgewinne beim Schreiben von Code fuer mehrere Plattformen.",

            # Francuski
            "La détection de la langue est essentielle pour le traitement automatique des langues, en particulier dans des environnements multilingues. "
            "Des exemples de texte plus longs fournissent un contexte grammatical et lexical qui aide à distinguer des langues proches. "
            "Ce passage inclut des termes liés à l'informatique, aux workflows de données et à des expressions courantes.",

            "Par exemple, un article qui décrit les étapes d'un projet de science des données contient du vocabulaire technique et des tournures de phrase spécifiques qui aident le modèle à apprendre. "
            "Nous incluons aussi des phrases conversationnelles afin de couvrir différents registres.",

            "Les textes plus détaillés permettent de capturer des indices morphologiques, des accords et des constructions syntaxiques caractéristiques du français écrit.",

            "L'apprentissage automatique transforme la maniere dont les entreprises exploitent leurs donnees. "
            "Les algorithmes de classification, de regression et de clustering sont utilises pour resoudre une variete de problemes pratiques. "
            "La validation croisee et les tests d'hypotheses statistiques sont essentiels pour evaluer la performance des modeles.",

            "Les reseaux de neurones artificiels imitent le fonctionnement du cerveau humain. "
            "L'architecture des reseaux, y compris le nombre de couches et de neurones, affecte grandement la capacite d'apprentissage. "
            "L'optimisation du taux d'apprentissage et d'autres hyperparametres est crucial pour la convergence.",

            "Les systemes de recommandation sont omnipresents dans les applications modernes. "
            "Les approches collaboratives et basees sur le contenu offrent differents avantages selon le contexte. "
            "La personalisation amelioree l'experience utilisateur et augmente l'engagement.",

            "Le traitement du langage naturel comprend des taches telles que l'analyse syntaxique, la reconnaissance d'entites nommees et l'extraction de relations. "
            "Les modeles transformer comme BERT et GPT ont revolutionne le domaine en atteignant des performances surhumaines. "
            "L'apprentissage par transfert permet de reutiliser les connaissances acquises sur de grands corpus.",

            "L'informatique en nuage offre une flexibilite et une scalabilite inegalees pour les applications d'entreprise. "
            "Les services de calcul sans serveur reduisent la complexite de la gestion de l'infrastructure. "
            "La facturation a l'usage permet aux organisations de controler leurs couts informatiques.",

            "Les approches agiles et DevOps ont ameliore l'efficacite du developpement logiciel. "
            "L'integration continue et le deploiement continu automatisent les etapes de test et de mise en production. "
            "La collaboration etroite entre les equipes de developpement et d'exploitation ameliore la qualite et la fiabilite des services.",

            "La securite des donnees est un defi majeur pour les organisations modernes. "
            "Les methodes de chiffrement et les protocols de securite doivent etre constamment mises a jour. "
            "La formation des utilisateurs sur les bonnes pratiques de securite est tout aussi importante que la technologie.",

            # Španski
            "La identificación del idioma es un componente clave en sistemas que procesan contenido multilingüe, como noticias o redes sociales. "
            "Un texto más extenso ofrece mayor contexto léxico y sintáctico, lo que facilita la distinción entre idiomas cercanos. "
            "Este ejemplo incluye descripciones técnicas, oraciones coloquiales y vocabulario de dominio específico.",

            "Por ejemplo, un párrafo que describe un flujo de trabajo de ciencia de datos contiene términos y expresiones que ayudan al clasificador a reconocer patrones propios del español. "
            "También se han añadido variaciones estilísticas para cubrir registros formales e informales.",

            "En conjunto, los pasajes largos aumentan la probabilidad de que aparezcan entidades y colocaciones informativas que mejoren la capacidad de los modelos para generalizar.",

            "Los algoritmos de aprendizaje automatico requieren ajuste cuidadoso de parametros para lograr rendimiento optimo. "
            "La validacion cruzada y la busqueda en cuadricula son tecnicas comunes para encontrar la mejor configuracion. "
            "La eleccion de hiperparametros puede impactar significativamente en la generalizacion del modelo.",

            "Las redes neuronales profundas han revolucionado muchas aplicaciones en vision por computadora y procesamiento de lenguaje natural. "
            "Las arquitecturas como CNN y RNN tienen fortalezas especificas para diferentes tipos de datos. "
            "El entrenamiento de redes profundas requiere grandes cantidades de datos y recursos computacionales.",

            "La ingenieria de caracteristicas es un arte crucial que mejora significativamente el rendimiento del modelo. "
            "La seleccion de caracteristicas relevantes reduce la dimensionalidad y mejora la interpretabilidad. "
            "La creacion de nuevas caracteristicas a partir de datos crudos requiere conocimiento del dominio.",

            "Los pipelines de datos modernos integran multiples componentes para extraer, transformar y cargar informacion. "
            "Las herramientas como Apache Spark y Flink permiten procesamiento distribuido de grandes volumenes de datos. "
            "La orquestacion de flujos de trabajo complejos requiere soluciones como Apache Airflow.",

            "La explicabilidad y la interpretabilidad de los modelos son cada vez mas importantes en aplicaciones criticas. "
            "Las tecnicas como LIME y SHAP ayudan a entender como los modelos hacen predicciones. "
            "La transparencia genera confianza en los sistemas de inteligencia artificial.",

            "Las pruebas unitarias y la integracion continua garantizan la calidad del codigo. "
            "La cobertura de codigo y el analisis estatico detectan problemas potenciales tempranamente. "
            "Las practicas de desarrollo seguro protegen contra vulnerabilidades comunes.",

            "La gestion de datos y la proteccion de la privacidad son esenciales en proyectos de aprendizaje automatico. "
            "Buenas practicas de anonimización y cumplimiento normativo reducen riesgos legales y aumentan la confianza de los usuarios. "
            "Una gobernanza de datos clara mejora la calidad y la reutilizacion de los conjuntos de entrenamiento.",

            # Hrvatski
            "Prepoznavanje jezika kritičan je korak u obradi prirodnog jezika, posebno u aplikacijama koje rade s različitim jezicima. "
            "Dulji tekstovi pružaju bogatiji kontekst koji pomaže modelima da razlikuju jezike koji su međusobno slični. "
            "Ovaj primjer sadrži tehničke izraze, svakodnevne fraze i stručnu terminologiju iz različitih domena.",

            "Na primjer, tekst koji opisuje proces obrade podataka i analize sadrži specifične izraze i rječnik koji pomažu modelima da nauče karakteristične elemente hrvatskog jezika. "
            "Uključeni su i formalni i neformalni registri kako bi se obuhvatila raznolikost prirodnog jezika.",

            "Dulji primjeri omogućuju modelima da nauče morfološke, sintaktičke i leksičke karakteristike koje su specifične za hrvatski jezik.",

            "Kombinacija različitih stilova pisanja, tehnička terminologija i opisi svakodnevnih scenarija čine ovaj primjer reprezentativnim za stvarne tekstove.",

            "Razvoj softvera zahtijeva kombinaciju tehničkog znanja i razumijevanja poslovnih zahtjeva. "
            "Kvalitetna dokumentacija i jasna komunikacija unutar tima ključne su za uspješne projekte. "
            "Pregled koda od strane kolega sprječava pogreške i poboljšava kvalitetu koda.",

            "Arhitektura softvera određuje kako su komponente organizirane i kako međusobno komuniciraju. "
            "Arhitektura mikroservisa omogućuje skalabilnost i fleksibilnost, ali povećava složenost. "
            "Odabir prave arhitekture ovisi o zahtjevima, veličini tima i proračunu projekta.",

            "Testiranje je sastavni dio životnog ciklusa razvoja softvera. "
            "Automatizirani testovi osiguravaju da kod ostaje funkcionalan tijekom izmjena. "
            "Jedinični testovi, integracijski testovi i testovi prihvaćanja pokrivaju različite razine.",

            "Upravljanje inačicama pomoću Gita standard je u industriji. "
            "Grananje, spajanje i rebase operacije omogućuju učinkovitu suradnju među programerima. "
            "Zahtjevi za povlačenjem omogućuju pregled koda prije nego što se promjene integriraju.",

            "Sigurnost aplikacija zahtijeva višeslojni pristup od dizajna do implementacije. "
            "Validacija unosa, zaštita od SQL ubrizgavanja i međustraničnog skriptiranja temeljne su mjere. "
            "Redovite sigurnosne provjere i testiranje prodiranja identificiraju ranjivosti.",

            "Održavanje performansi i skalabilnost sustava zahtijevaju kontinuirano praćenje i optimizaciju. "
            "Optimizacija upita, predmemoriranje i horizontalno skaliranje pomažu u postizanju pouzdanih SLA-ova. "
            "Alati za nadgledanje i uzbunjivanje omogućuju brzo reagiranje na degradacije performansi.",

            # Bosanski
            "Prepoznavanje jezika je važna komponenta u obradi prirodnog jezika, naročito u višejezičnim okruženjima. "
            "Duži tekstovi pružaju bolji kontekst koji olakšava razlikovanje između sličnih jezika. "
            "Ovaj primjer sadrži tehničke termine, uobičajene izraze i specijalizovanu leksiku iz različitih oblasti.",

            "Na primjer, tekst koji opisuje proces rada sa podacima i mašinskim učenjem sadrži termine i fraze specifične za bosanski jezik. "
            "Različiti stilovi pisanja, od formalnog do neformalne komunikacije, uključeni su u primjere.",

            "Duži tekstualni primjeri omogućavaju modelima da nauče karakteristične osobine bosanskog jezika, uključujući gramatičke strukture i leksiku.",

            "Kombinacija formalnih i neformalnih registara, tehničke terminologije i općenamjenskih tekstova čini ovaj skup podataka reprezentativnim za realne scenarije.",

            "Industrija tehnologije se brzo mijenja sa novim tehnologijama i trendovima koji se pojavljuju. "
            "Fleksibilnost i sposobnost za učenje su ključne karakteristike uspješnih tehničara. "
            "Umrežavanje i učešće u zajednicama omogućava stalno usavršavanje i razmjenu iskustava.",

            "Vještačka inteligencija i mašinsko učenje otvaraju nove mogućnosti u raznim domenima. "
            "Od zdravstvene zaštite do finansija, primjene AI-ja imaju duboke uticaje. "
            "Etička razmatranja o pristrasnosti i pravičnosti su sve važnija.",

            "Cloud infrastruktura omogućava preduzećima da se fokusiraju na poslovnu logiku umjesto na upravljanje serverima. "
            "Skaliranje aplikacija je olakšano zbog elastičnosti oblaka. "
            "Troškovi se mogu kontrolisati pažljivim monitoringom i optimizacijom resursa.",

            "Timski rad je suština modernog razvoja softvera. "
            "Alati kao što su Slack, Jira i Confluence poboljšavaju komunikaciju i produktivnost. "
            "Daljinski rad je postao norma u mnogim IT kompanijama.",

            "Inovacija zahtijeva eksperimentisanje i toleranciju na neuspjehe. "
            "Fail fast principi u startup kulturi promovišu brzo učenje. "
            "Balansiranje između inovacije i stabilnosti je ključni izazov za kompanije.",

            "Razvoj aplikacija zahtijeva konstantno ulaganje u kvalitetu i testiranje. "
            "Automatizovani testovi i kontinuirana integracija smanjuju rizik od greške. "
            "Monitoring u produkciji i brzo reagovanje na probleme su ključni za zadržavanje korisnika.",

        ],
        'language': [
            # Engleski (10)
            'english', 'english', 'english', 'english', 'english', 'english', 'english', 'english', 'english', 'english',
            # Srpski (10)
            'serbian', 'serbian', 'serbian', 'serbian', 'serbian', 'serbian', 'serbian', 'serbian', 'serbian', 'serbian',
            # Nemacki (10)
            'german', 'german', 'german', 'german', 'german', 'german', 'german', 'german', 'german', 'german',
            # Francuski (10)
            'french', 'french', 'french', 'french', 'french', 'french', 'french', 'french', 'french', 'french',
            # Spanski (10)
            'spanish', 'spanish', 'spanish', 'spanish', 'spanish', 'spanish', 'spanish', 'spanish', 'spanish', 'spanish',
            # Hrvatski (10)
            'croatian', 'croatian', 'croatian', 'croatian', 'croatian', 'croatian', 'croatian', 'croatian', 'croatian', 'croatian',
            # Bosanski (10)
            'bosnian', 'bosnian', 'bosnian', 'bosnian', 'bosnian', 'bosnian', 'bosnian', 'bosnian', 'bosnian', 'bosnian',
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