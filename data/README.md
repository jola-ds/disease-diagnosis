# Nigerian Disease Synthetic Dataset Generation

This dataset is generated using a **custom synthetic data generator** that simulates patient profiles and disease occurrences in Nigeria, capturing demographic distributions, disease prevalence, symptom patterns, seasonal effects, and comorbidities.

- For domain reference, see [References](../citations.md).
- For the generator script see [Data Generator](src/data_generator.py).

## 1. Patient Demographics

Each patient record includes the following demographic features, sampled according to Nigerian population distributions:

| Feature      | Possible Values                          | Notes |
|--------------|-----------------------------------------|-------|
| `age_band`   | 0–4, 5–14, 15–24, 25–44, 45–64, 65+   | Reflects Nigerian age distribution |
| `gender`     | male, female                             | Slight male predominance (51%) |
| `setting`    | urban, rural                             | Slight urban predominance (52%) |
| `region`     | north, middle_belt, south               | Regional population weights applied |
| `season`     | dry, rainy, transition                   | Represents typical yearly distribution |

These demographics are encapsulated in the `PatientProfile` dataclass.

## 2. Disease Prevalence

The generator models **10 common diseases in Nigeria**, plus a "healthy" class for undiagnosed or healthy patients. Base prevalence rates (before demographic adjustments) are:

| Disease          | Base Prevalence |
|-----------------|----------------|
| malaria          | 22%            |
| typhoid          | 18%            |
| peptic_ulcer     | 10%            |
| tuberculosis     | 12%            |
| measles          | 8%             |
| gastroenteritis  | 19%            |
| pneumonia        | 20%            |
| diabetes         | 10%            |
| hypertension     | 15%            |
| HIV              | 6%             |
| healthy          | adjustable, default 18% |

> The `healthy_share` parameter allows control over the proportion of healthy individuals (default 18%).

## 3. Disease-Specific Demographic Nudges

Each disease’s base probability is **adjusted according to demographic factors**:

- **Age band:** e.g., malaria is more prevalent in children.  
- **Gender:** some diseases have slight gender skew (e.g., HIV higher in females).  
- **Setting (urban/rural):** typhoid favors rural settings; diabetes favors urban.  
- **Region:** regional variations are applied, reflecting epidemiology.  
- **Season:** malaria peaks in rainy season; TB and measles have dry-season patterns.  

These adjustments ensure that generated patient profiles follow realistic patterns.

## 4. Symptom Generation

Each disease has a **predefined symptom probability profile**, e.g., fever is highly probable for malaria, while abdominal pain is common for peptic ulcer.  

The generator applies:

1. **Age-specific modifiers:** certain age groups may have higher or lower symptom probabilities.  
2. **Seasonal effects:** e.g., rainy season increases probabilities of malaria-related symptoms.  
3. **Symptom correlations:** e.g., malaria patients with fever and chills are more likely to also have sweats.  
4. **Comorbidities:** some diseases often co-occur (e.g., diabetes + hypertension, TB + HIV).
5. **Mutually exclusive adjustments:** Conflicting or overlapping symptoms are handled to avoid unrealistic combinations (e.g., chronic cough reduces likelihood of regular cough).  

Symptoms are then **sampled as binary indicators** (present/absent).

## 5. Disease Assignment

Disease is assigned per patient using the following steps:

1. Compute **adjusted probabilities** for each disease based on demographics.  
2. Normalize probabilities to sum to 1.  
3. Apply the `healthy_share`, scaling other probabilities accordingly.  
4. Sample a disease based on the final probability distribution.  

## 6. Dataset Generation

The generator can produce:

- A dataset of arbitrary size (`n_patients`) using `generate_dataset()`.  
- Optional **balanced disease datasets**, ensuring roughly equal representation of all diseases.
- Metadata inclusion: Patient ID and generation timestamp can be included for traceability.
- Shuffling: Patients are randomly shuffled to avoid systematic ordering.

Each patient record contains:

- Demographics (`age_band`, `gender`, `setting`, `region`, `season`)  
- Disease diagnosis (`diagnosis`)  
- Binary symptom indicators for 15 common symptoms (fever, headache, cough, fatigue, body_ache, chills, sweats, nausea, vomiting, diarrhea, abdominal_pain, loss_of_appetite, sore_throat, runny_nose, dysuria)

## 7. Statistics

After generation, the dataset can be analyzed to verify realism:

- Disease distribution and percentages
- Demographic patterns by disease (age, gender, setting, region, season)
- Symptom prevalence and identification of pathognomonic symptoms (>70% prevalence and disease-specific)
- Symptom correlations for clinically meaningful pairs  

The `get_disease_statistics()` method provides a statistical summary of the dataset.

## 8. Key Design Highlights

- Reproducible: Seeded random generator for consistent results.

- Epidemiologically informed: Disease probabilities, demographic nudges, and comorbidities mirror real-world patterns in Nigeria.

- Flexible: Supports balanced or unbalanced datasets, adjustable healthy share, and dataset size.
