import argparse
import os
import sys

# Ensure validate_private module from the code directory can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))
try:
    from validate_private import run_v17_inference
except ImportError as e:
    print(f"Import failed: {e}. Please ensure you run this script from the project root directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test Inference Script: Read weights, process features CSV, and generate predictions")
    
    # Required parameters
    parser.add_argument("--input_csv", type=str, required=True, 
                        help="Path to the input features CSV file for prediction (e.g., data/test_features.csv)")
    
    # Defaults pointing to trained models and output paths
    parser.add_argument("--output_csv", type=str, default="test_predictions.csv", 
                        help="Path to save the generated prediction results (default: test_predictions.csv)")
    parser.add_argument("--weights_dir", type=str, default="exp_v17_paper/models", 
                        help="Directory containing the .pt model weights (default: exp_v17_paper/models)")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"❌ Error: Input file '{args.input_csv}' not found.")
        sys.exit(1)
        
    if not os.path.exists(args.weights_dir):
        print(f"❌ Error: Model weights directory '{args.weights_dir}' not found. Please ensure training is completed!")
        sys.exit(1)

    print("====================================")
    print("🔧 Starting V17 Inference:")
    print(f"➡️ Input Data: {args.input_csv}")
    print(f"⬅️ Output Results: {args.output_csv}")
    print(f"🧠 Weights Directory: {args.weights_dir}")
    print("====================================")
    
    # Reuse the full TTA inference logic from validate_private.py
    run_v17_inference(args.input_csv, args.output_csv, args.weights_dir)
    
    print("\n✅ Inference execution completed!")

if __name__ == "__main__":
    main()
