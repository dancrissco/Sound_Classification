import random
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to generate sound variations
def generate_variations(num_samples, base_frequency, base_duration, base_amplitude, noise_level, sample_rate=44100):
    X = []
    y = []
    for i in range(num_samples):
        frequency = base_frequency + random.uniform(-50, 50)
        amplitude = base_amplitude + random.uniform(-0.1, 0.1)
        clean_sound = np.sin(2 * np.pi * frequency * np.linspace(0, base_duration, int(sample_rate * base_duration)))
        clean_features = np.mean(clean_sound)
        X.append(clean_features)
        y.append(0)  # Label 0 for clean sound
        
        noisy_sound = clean_sound + noise_level * np.random.normal(0, 1, len(clean_sound))
        noisy_features = np.mean(noisy_sound)
        X.append(noisy_features)
        y.append(1)  # Label 1 for noisy sound
    return np.array(X).reshape(-1, 1), np.array(y)

# Function to train the model
def train_model():
    num_samples = 30  # Set your desired number of samples
    base_frequency = 440
    base_duration = 1.0
    base_amplitude = 0.5
    noise_level = 0.1

    # Generate sound variations
    X, y = generate_variations(num_samples, base_frequency, base_duration, base_amplitude, noise_level)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training completed. Test accuracy: {accuracy * 100:.2f}%")
    
    # Save the trained model
    with open('sound_classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("Model saved successfully.")

if __name__ == "__main__":
    train_model()
