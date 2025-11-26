**Project Overview**
Ovaj projekat je jednostavan demo za prepoznavanje jezika teksta i komparativnu analizu tradicionalnih mašinskih modela (Logistic Regression, Random Forest, SVM) i neuronskih mreža (CNN, LSTM, Dense). Koristi se za obrazovanje i eksperimentisanje sa malim uzorkom podataka.

**Brzi pregled fajlova**
- `main.py`: Glavni skript koji pokreće učitavanje podataka, trening modela, evaluaciju i testiranje na primerima.
- `utils.py`: Pomoćne funkcije, uključujući `load_sample_data()`, čišćenje teksta, i pripremu feature-a za tradicionalne i neuronske modele.
- `model.py`: Implementacija tradicionalnih i neuronskih modela i funkcije za njihovo treniranje.
- `requirements.txt`: Zavisnosti koje su korišćene u ovom projektu.

**Instalacija zavisnosti**
Aktivirajte virtuelno okruženje i instalirajte zavisnosti iz `requirements.txt`:

```powershell
# Aktivacija venv (PowerShell)
{YourPath}/Seminarski-D-main/.venv/Scripts/Activate.ps1

# Instalacija
{YourPath}/Seminarski-D-main/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Napomena: `tensorflow` je velik paket i preuzimanje/instalacija može potrajati.

**Kako pokrenuti projekat**
```powershell
{YourPath}/Seminarski-D-main/.venv/Scripts/python.exe main.py
```

**Šta očekivati**
- Skript će ispisati poruke o učitavanju podataka, rezultatima treninga za svaki model, poređenje performansi i prikaz grafika. Na kraju takođe ispisuje predikcije na nekoliko test primera.

**Kako raditi izmene i eksperimente**
- Da dodate ili promenite uzorke za trening, uredite `load_sample_data()` u `utils.py`.
- Da promenite uzorke za testiranje, uredite `test_texts` u `main.py`.

