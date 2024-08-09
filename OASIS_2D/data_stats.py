import os

for label in ['disease', 'healthy']:
    files = os.listdir(f'{label}')
    patients = []
    for file in files:
        name = file.split('/')[-1].split('.')[0]
        patient_id = name.split('_')[1]
        patients.append(patient_id)
        
    print(f'Label {label}: Total {len(patients)} patients, unique {len(set(patients))}.')