import numpy as np
import pickle
import sounddevice as sd

# Function to load the saved model and predict based on a new sample
def predict_sample():
    base_frequency = 440
    base_duration = 1.0
    base_amplitude = 0.5
    noise_level = 0.1

    # Generate new sound and add noise
    new_clean_sound = np.sin(2 * np.pi * base_frequency * np.linspace(0, base_duration, int(44100 * base_duration)))
    new_noisy_sound = new_clean_sound + noise_level * np.random.normal(0, 1, len(new_clean_sound))
    
    # Extract features
    new_features = np.mean(new_noisy_sound).reshape(1, -1)

    # Load the trained model
    with open('sound_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    # Predict
    prediction = clf.predict(new_features)
    
    # Display the result
    result = "Clean" if prediction == 0 else "Noisy"
    print(f"Prediction: The sound is classified as '{result}'.")

    # Play the generated sound
    sd.play(new_noisy_sound, 44100)
    sd.wait()

if __name__ == "__main__":
    predict_sample()
