# Create extract_model.py
import tensorflow as tf
from tensorflow import keras
import json

# Load trained model
model = keras.models.load_model('results_azure_forecast_1hr/models/gru_attention_best.h5')

# Save architecture as JSON
architecture_json = model.to_json()
with open('gru_attention_architecture.json', 'w') as f:
    json.dump(architecture_json, f, indent=2)

# Save weights separately
model.save_weights('gru_attention_weights.h5')
