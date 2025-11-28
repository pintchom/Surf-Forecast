from src.models.lstm_model import train_lstm_model

# Train LSTM for station 46221 only
station = '46221'
model, history, metrics = train_lstm_model(station)
print(f"\nStation {station} LSTM training completed!")

# Print key results
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
print(f"Training epochs: {len(history.history['loss'])}")