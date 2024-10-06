# Sound_Classification
User Interface to classify sound on a trained model

This project generates sound samples, trains a machine learning model to classify them as "Clean" or "Noisy", and allows for real-time predictions.

## Files
- `train_model.py`: Generates sound samples, trains a `RandomForestClassifier`, and saves the model.
- `predict_model.py`: Loads the saved model, generates a new sound sample, and predicts whether it's "Clean" or "Noisy."

## How to Use
1. **Training the Model**:
   Run the `train_model.py` script to train the model and save it as `sound_classifier.pkl`.
   ```bash
   python train_model.py
