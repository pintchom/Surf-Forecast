from src.models.enhanced_lstm_model import train_enhanced_lstm

# Test attention-based LSTM on station 46221 (larger dataset)
print("Testing Enhanced LSTM with Attention Mechanism")
print("=" * 50)

try:
    model, history, metrics = train_enhanced_lstm(
        station_id='46221',
        architecture='attention',  # Try attention first
        lookback_hours=24,  # Keep same for fair comparison
        lstm_layers=[128, 64, 32],  # Deeper than original [64, 32]
        learning_rate=0.001,  # Higher than original 0.0001
        epochs=200,
        batch_size=64  # Larger batch size
    )
    
    print("\n" + "="*50)
    print("ATTENTION LSTM vs ORIGINAL LSTM COMPARISON")
    print("="*50)
    
    # Compare with original LSTM results
    print(f"Enhanced LSTM (Attention):")
    print(f"  Overall RMSE: {metrics['overall']['rmse']:.4f}")
    print(f"  WVHT_1h RMSE: {metrics['by_target']['WVHT_1h']['rmse']:.4f}")
    print(f"  WVHT_6h RMSE: {metrics['by_target']['WVHT_6h']['rmse']:.4f}")
    
    print(f"\nOriginal LSTM (for reference):")
    print(f"  Overall RMSE: 0.5965")
    print(f"  WVHT_1h RMSE: 0.3603") 
    print(f"  WVHT_6h RMSE: 0.5545")
    
    # Calculate improvement
    original_rmse = 0.5965
    enhanced_rmse = metrics['overall']['rmse']
    improvement = ((original_rmse - enhanced_rmse) / original_rmse) * 100
    
    print(f"\nPerformance Change: {improvement:+.1f}%")
    
    if improvement > 0:
        print("✅ Enhanced LSTM shows improvement!")
    else:
        print("❌ Enhanced LSTM did not improve performance")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()