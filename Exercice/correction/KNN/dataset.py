import pandas as pd
import numpy as np
from random import randint
# Set the random seed for reproducibility
np.random.seed(42)

# Generate data for 200 rows
study_hours = np.random.uniform(5.0, 30.0, 750)  # Study hours between 5 and 30 hours per week
sleep_hours = np.random.uniform(4.0, 10.0, 750)  # Sleep hours between 4 and 10 hours per night
extra_curricular = np.random.uniform(0.0, 5.0, 750)  # 0 for no participation, 1 for participation

# Generate the label (Pass/Fail) based on study and sleep hours, and participation
# Let's say:
# - A student passes if they study more than 20 hours, sleep more than 6 hours, and participate in extracurricular activities
# - Otherwise, they fail
label = []
for i in range(750):
    if (study_hours[i] > 20 and sleep_hours[i] > 6 and extra_curricular[i] > 2) or randint(0, 100) < 5:
        label.append('Pass')
    else:
        label.append('Fail')

# Create the DataFrame
df = pd.DataFrame({
    'Study Hours': study_hours,
    'Sleep Hours': sleep_hours,
    'Extra-curricular Participation': extra_curricular,
    'Label': label
})
df.to_csv('student_performance_dataset.csv', sep='\t', index=False)
# Show the first few rows of the dataset
df.head()