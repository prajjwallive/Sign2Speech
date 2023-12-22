import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Pad sequences to the maximum length
max_length = max(len(lst) for lst in data_dict['data'])
padded_data = [lst + [0] * (max_length - len(lst)) for lst in data_dict['data']]

data = np.asarray(padded_data)
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_predict, y_test)
print(f'{accuracy * 100:.4f}% of samples were classified correctly!')

# Save the model using pickle (use protocol 4 for efficiency)
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f, protocol=4)
