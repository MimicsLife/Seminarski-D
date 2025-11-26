from utils import load_sample_data

df = load_sample_data()
print(f'Broj tekstova: {len(df)}')
print(f'\nRasporedlabela:')
print(df['language'].value_counts().sort_index())
