import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import FunctionTransformer
import joblib

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were","hi","new","help"
    "be", "to", "of", "in", "on", "for", "with", "at", "by", "from", "as",
    "this", "that", "these", "those", "it", "its", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "them", "my", "your", "our", "their",
    "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "so", "not", "no", "up", "out", "now", "then", "than", "too", "very",
}

RANDOM_STATE = 42
CSV_PATH   = r"D:\Projects\models\it_support_tickets.csv"
MODEL_PATH = r"D:\Projects\models\ticket_predict\ticket_model.joblib"
pd.set_option("display.max_colwidth", 120)


class Model():

    def __init__(self,model_path=MODEL_PATH,csv_path=CSV_PATH,random_state=RANDOM_STATE):
        self.model_path = model_path
        self.csv_path = csv_path
        self.random_state = RANDOM_STATE

    def clean_series(self,texts):
        # Aplica clean_text pe fiecare element dintr-o lista
        # Necesar deoarece Pipeline lucreaza cu liste, nu cu un singur string
        return [self.clean_text(t) for t in texts]

    def clean_text(self,text: str) -> str:
        # Pas 1: transforma totul in litere mici
        text = text.lower()

        # Pas 2: sterge orice nu e litera sau spatiu (cifre, punctuatie, simboluri ca "%", "-", ".")
        text = re.sub(r"[^a-z\s]", " ", text)

        # Pas 3: taie stringul in cuvinte individuale si sterge cuvintele din lista STOPWORDS
        tokens = [t for t in text.split() if t not in STOPWORDS]

        # Reuneste cuvintele ramase intr-un singur string curat
        return " ".join(tokens)

    def clean_series(self,texts):
        # Aplica clean_text pe fiecare element dintr-o lista
        # Necesar deoarece Pipeline lucreaza cu liste, nu cu un singur string
        return [self.clean_text(t) for t in texts]

    def top_words(self,df,n=20):

        def get_words(text):
            return re.findall(r'\b\w+\b', text.lower())

        all_words = df['description'].apply(get_words).sum()
        Counter(all_words).most_common(n)

        # Per clasă - vezi ce cuvinte sunt caracteristice fiecărei clase
        for label in df['assigned_team'].unique():
            words = df[df['assigned_team']==label]['description'].apply(get_words).sum()
            print(label, Counter(words).most_common(10))

    def load_data(self):
        # Citeste CSV-ul ca un tabel (DataFrame)
        df = pd.read_csv(self.csv_path)

        df.shape # câte rânduri, câte coloane
        df.head(10) # primele rânduri
        df.info() # tipuri de date, valori non-null
        df.describe(include='all') # statistici de bază

        # Top words pentru a vedea daca trebuie sterse anumite cuvinte des intalnite
        print("<<<<<<<<<<<<<<<<================= TOP WORDS ===========================>>>>>>>>>>>>>>>\n")
        self.top_words(df,25)
        print("<<<<<<<<<<<<<<<<============================================>>>>>>>>>>>>>>>\n")

        print(f"tag-uri HTML : {df['description'].str.contains(r'<[^>]+>').sum()}")
        print(f"caractere non-ASCII : {df['description'].str.contains(r'[^\x00-\x7F]').sum()}")

        # Extrage toate caracterele non-ASCII unice din dataset
        non_ascii_chars = set()
        for text in df['description']:
            non_ascii_chars.update(re.findall(r'[^\x00-\x7F]', text))

        print("Lista non asci:",sorted(non_ascii_chars),"\n\n\n")

        # Ticket-uri identice (exact acelasi text) -- verificam inainte de split ca sa nu ajunga aceeasi propozitie si in train si in test (data leakage)
        n_dupes = df.duplicated(subset=["description"]).sum()
        if n_dupes:
            print(f"[WARN] {n_dupes} duplicate description(s) found -- dropping.")

        # Cazul si mai grav: acelasi text, dar cu echipe diferite atribuite
        conflicts = (
            df.groupby("description")["assigned_team"]
            .nunique()
            .loc[lambda s: s > 1]
        )
        if len(conflicts):
            print(f"[WARN] {len(conflicts)} description(s) with conflicting team labels.")

        df = df.drop_duplicates(subset=["description"], keep="first")
        df['description'] = df['description'].str.encode('ascii', errors='ignore').str.decode('ascii')

        # Cazul acesta este putin mai specific deoarece am vazut ca pe categoria Telephony aveam :
        # precision    recall  f1-score   support
        #      1.00      1.00      1.00        37
        # Parea cam am un data leakage dar se intampla asta din cauza cuvintelor cheie: "phone", "voicemail", "extension". As fi putut face va mai mult?

        tel = df[df["assigned_team"] == "Telephony"]
        print("Duplicate Telephony descriptions:", tel.duplicated(subset=["description"]).sum())
        print("Unique Telephony descriptions:", tel["description"].nunique(), "/", len(tel))

        return df["description"], df["assigned_team"]

    def build_pipeline(self, clf) -> Pipeline:
        return Pipeline([

            # Statia 1: curatare text
            ("clean", FunctionTransformer(self.clean_series)),

            # Statia 2: TF-IDF — transforma textul in numere pe care algoritmul le poate citi
            #
            # Un calculator nu poate lucra cu stringuri. TF-IDF rezolva asta:
            # pentru fiecare ticket construieste un dictionar de tipul:
            #   { "kubernetes": 0.8, "crashloopbackoff": 0.9, "pod": 0.6, ... }
            # unde valoarea reprezinta cat de "important/specific" e cuvantul in acel ticket.
            #
            # ngram_range=(1, 2): ia in considerare si perechi de cuvinte consecutive
            #   ex: "null pointer" e mai specific decat "null" si "pointer" separat
            #
            # max_features=50_000: pastreaza doar cele mai frecvente 50.000 de cuvinte/perechi
            #   (filtru de performanta — un vocabular prea mare incetineste antrenarea)
            #
            # sublinear_tf=True: un cuvant care apare de 10 ori nu e de 10x mai important
            #   decat unul care apare o data — aplicam log() ca sa atenuam aceasta diferenta
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
            )),

            # Statia 3: Logistic Regression — algoritmul care invata sa clasifice
            #
            # Dupa antrenare, stie ca: daca vectorul contine "kubernetes", "pod", "helm"
            # cu valori mari -> probabilitate mare ca echipa sa fie DevOps.
            # Functioneaza ca un if/else foarte complex invatat din date, nu scris manual.
            #
            # C=5.0: cat de strict e algoritmul. Valoare mica = mai permisiv cu greselile
            #   pe datele de antrenare, dar generalizeaza mai bine pe date noi.
            #   Valoare mare = incearca sa fie perfect pe antrenare, risc de over-fitting.
            #
            # max_iter=1000: numarul maxim de incercari de ajustare a raspunsurilor inainte
            #   de a se opri (ca un numar maxim de retry-uri)
            #
            # solver="lbfgs": algoritmul intern de optimizare — alegerea recomandata
            #   pentru probleme cu mai multe clase (10 echipe in cazul nostru)
            ("clf", clf),
        ])

    def plot_confusion_matrix(self, y_test, y_pred):
        # Ia lista unica de echipe, sortata alfabetic (pentru axele tabelului)
        labels = sorted(set(y_test))

        # Construieste tabelul NxN cu numarul de raspunsuri pentru fiecare combinatie
        # cm[i][j] = "de cate ori echipa i a fost prezisa ca echipa j"
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        plt.figure(figsize=(12, 9))
        sns.heatmap(
            cm,
            annot=True,       # scrie numarul efectiv in fiecare celula
            fmt="d",          # format numar intreg (fara .0)
            cmap="Blues",     # culoare: albastru mai inchis = numar mai mare
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.5,
        )
        plt.title("Confusion Matrix -- Ticket Team Classifier")
        plt.xlabel("Predicted")   # ce a zis modelul
        plt.ylabel("Actual")      # ce era corect
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(r"D:\Projects\models\ticket_predict\confusion_matrix.png", dpi=150)
        plt.show()

    def train(self):
        # 1. Citeste datele din CSV
        X, y = self.load_data()

        # 2. Imparte datele in doua seturi:
        #    - X_train / y_train (75%): "lectia" — datele pe care modelul le vede si invata din ele
        #    - X_test  / y_test  (25%): "examenul" — date noi, ascunse in timpul antrenarii
        #    stratify=y: asigura ca fiecare echipa apare proportional in ambele seturi
        #    (altfel am putea nimeri sa punem toata echipa DevOps doar in test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
        )

        # 3. Porneste banda de productie si antreneaza modelul pe datele de "lectie"
        pipeline = self.build_pipeline(LogisticRegression(
            max_iter=1000, C=5.0, solver="lbfgs",
        ))

        dummy_pipeline = self.build_pipeline(DummyClassifier(
            strategy="most_frequent", random_state=RANDOM_STATE,
        ))

        pipeline.fit(X_train, y_train)
        dummy_pipeline.fit(X_train, y_train)

        # 4. Ruleaza modelul pe datele de "examen" (pe care nu le-a vazut niciodata)
        y_pred = pipeline.predict(X_test)
        y_dummy_pred = dummy_pipeline.predict(X_test)
        # 5. Calculeaza scorurile de performanta:
        #
        #    accuracy  = din toate raspunsurile, ce procent a fost corect?
        #                ex: 0.92 = a ghicit corect echipa in 92% din cazuri
        #
        #    precision = din toate ticketele pe care le-a trimis la "DevOps",
        #                ce procent chiar apartineau lui "DevOps"?
        #                (masoara cat de des trimite la echipa gresita)
        #
        #    recall    = din toate ticketele care chiar apartineau lui "DevOps",
        #                ce procent le-a identificat corect?
        #                (masoara cat de des rateaza tickete dintr-o echipa)
        #
        #    f1        = media armonica intre precision si recall
        #                (un singur numar care combina ambele aspecte)
        #
        #    average="macro": calculeaza scorul pentru fiecare echipa separat,
        #    apoi face media — astfel toate echipele conteaza egal, indiferent de marime
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec  = recall_score(y_test, y_pred, average="macro")
        f1   = f1_score(y_test, y_pred, average="macro")

        dummy_acc  = accuracy_score(y_test, y_dummy_pred)
        dummy_prec = precision_score(y_test, y_dummy_pred, average="macro", zero_division=0)
        dummy_rec  = recall_score(y_test, y_dummy_pred, average="macro", zero_division=0)
        dummy_f1   = f1_score(y_test, y_dummy_pred, average="macro", zero_division=0)

        # 6. Afiseaza un tabel rezumat cu toate scorurile
        summary = pd.DataFrame([
            {"model": "LogisticRegression", "accuracy": round(acc, 3),       "precision": round(prec, 3),       "recall": round(rec, 3),       "f1": round(f1, 3)},
            {"model": "DummyClassifier",    "accuracy": round(dummy_acc, 3), "precision": round(dummy_prec, 3), "recall": round(dummy_rec, 3), "f1": round(dummy_f1, 3)},
        ])
        print(summary.to_string(index=False))


        # 7. Afiseaza scorul detaliat pentru fiecare echipa in parte
        print(classification_report(y_test, y_pred))

        # 8. Deseneaza matricea de confuzie (explicata in functia de mai jos)
        self.plot_confusion_matrix(y_test, y_pred)

        # 9. Salveaza intregul lant (curatare + TF-IDF + model) ca fisier pe disc
        #    La fel ca serializing un obiect — poate fi reincarcat mai tarziu fara reantrenare
        joblib.dump(pipeline, self.model_path)
        print(f"Model saved to {self.model_path}")
        return pipeline

    def parameter_calibration(self):
        # Foloseste acelasi dataset ca train(): self.csv_path (setat la construirea Model(csv_path=...))
        X, y = self.load_data()

        # Refolosim build_pipeline ca sa includem si pasul de clean_series in grid search
        pipeline = self.build_pipeline(LogisticRegression(max_iter=1000))

        # Note the naming convention: step__param

        param_grid = [
            {
                'tfidf__max_features': [100,500,1000 ,5000, 10000, None], #Limits the vocabulary to the top N words (ranked by frequency across the corpus). 100 keeps only the 100 most frequent terms (very restrictive, may lose signal); None keeps every word that appears (no cap). Smaller values reduce overfitting and speed things up but risk dropping useful rare/domain-specific words (e.g. "voicemail" for Telephony).
                'tfidf__ngram_range': [(1, 1), (1, 2)], # Controls whether features are single words or also include word pairs. (1, 1) = unigrams only (e.g. "phone", "not", "working"). (1, 2) = unigrams + bigrams (e.g. also "phone not", "not working"). Bigrams can capture phrases/context but multiply the vocabulary size a lot.
                'tfidf__min_df': [1, 2],  # Minimum number of documents a word must appear in to be kept as a feature. 1 = keep every word, even ones appearing in just a single ticket (noisy, includes typos/rare terms). 2 = word must appear in at least 2 tickets to count — filters out one-off noise.
                'clf__C': [0.01, 0.1, 1, 10], # Inverse of regularization strength for LogisticRegression. Lower C (e.g. 0.01) = stronger regularization = simpler model, less likely to overfit but may underfit. Higher C (e.g. 10) = weaker regularization = model fits training data more closely, higher overfitting risk. It's the main "how flexible should the model be" knob.
                'clf__class_weight': [None, 'balanced'], # How to handle class imbalance. None = all classes treated equally regardless of how many samples each has. 'balanced' = automatically upweights minority classes (inversely proportional to their frequency) so the model doesn't just favor the majority class. Since your classes are already fairly balanced (37-38 each), this may have little effect, but it's cheap to test.
            },
            {
                'tfidf__max_features': [100,500,1000 ,5000, 10000, None],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'tfidf__min_df': [1, 2],
                'clf__C': [0.01, 0.1, 1, 10],
                'clf__class_weight': [None, 'balanced']
            }
        ]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=skf,
            scoring='f1_macro',
            n_jobs=-1,       # parallelize across param combos (safe here, not inside an estimator)
            verbose=2
        )

        grid.fit(X, y)

        print("Best params:", grid.best_params_)
        print("Best CV score:", grid.best_score_)

        return grid

    def predict(self, texts: list[str]) -> pd.DataFrame:
        """
        Primeste o lista de texte de tickete si returneaza un DataFrame cu 3 coloane:
          - ticket     : textul original trimis
          - team       : echipa recomandata de model (sau "No team recommended")
          - confidence : un numar intre 0 si 1 — cat de sigur e modelul
                         (0.9 = foarte sigur, 0.3 = aproape ghicit la intamplare)
        """

        # Incarca modelul salvat anterior de pe disc
        pipeline = joblib.load(self.model_path)

        # Ruleaza banda de productie: clean -> tfidf -> clf
        # Rezultat: o lista cu echipa prezisa pentru fiecare ticket
        labels = pipeline.predict(texts)

        # predict_proba returneaza pentru fiecare ticket probabilitatile pentru TOATE echipele
        # ex: [DevOps=0.7, QA=0.1, Security=0.05, ...]
        # .max(axis=1) ia doar valoarea cea mai mare (echipa castigatoare)
        # Aceasta valoare maxima devine "confidence" — cat de sigur e modelul
        proba      = pipeline.predict_proba(texts)
        confidence = proba.max(axis=1)

        conf_rounded = [round(float(c), 3) for c in confidence]

        df = pd.DataFrame({
            "ticket": texts,
            # Daca modelul e nesigur (confidence < 0.5 = sub 50%) nu recomandam nicio echipa
            # Pragul 0.5 inseamna: modelul trebuie sa fie mai sigur decat un "flip de moneda"
            "team":       [t if c >= 0.5 else "No team recommended" for t, c in zip(labels, conf_rounded)],
            "confidence": conf_rounded,
        })

        return df