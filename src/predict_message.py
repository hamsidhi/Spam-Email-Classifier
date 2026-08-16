"""
Grand Line Message Bounty Detector — Spam Message Classifier
Script: predict_message.py
Description: Interactive command-line interface for testing SMS messages
             against the trained scikit-learn spam detection pipeline.
"""

import sys
from pathlib import Path
import joblib


def get_model_path():
    """
    Resolves project root and locates saved model pipeline artifact.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    model_path = project_root / "models" / "spam_message_pipeline.joblib"
    return model_path


def load_trained_pipeline(model_path):
    """
    Loads serialized joblib pipeline. Exits gracefully if missing.
    """
    if not model_path.exists():
        print("\n" + "=" * 65)
        print(" [ERROR] TRAINED MODEL PIPELINE NOT FOUND ")
        print("=" * 65)
        print(f"Target Path: {model_path}")
        print("\n[INSTRUCTION] You must train the model first by running:")
        print("              python src/train_model.py\n")
        sys.exit(1)
        
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        print(f"\n[ERROR] Failed to load model file: {e}")
        sys.exit(1)


def predict_message(message: str, pipeline):
    """
    Accepts a single raw text string and returns predicted label ('ham' or 'spam')
    along with estimated model probability.
    """
    # Wrap text in a list for scikit-learn pipeline vectorizer
    text_data = [message]
    
    # Generate label prediction
    prediction = pipeline.predict(text_data)[0]
    
    # Calculate estimated probabilities if model supports predict_proba
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(text_data)[0]
        classes = pipeline.classes_
        
        # Extract estimated probability for predicted class
        prob_dict = dict(zip(classes, probabilities))
        estimated_prob = prob_dict[prediction]
    else:
        estimated_prob = None
        
    return prediction, estimated_prob


def display_welcome_banner():
    """
    Displays themed CLI introduction banner.
    """
    print("\n" + "=" * 65)
    print("      GRAND LINE MESSAGE BOUNTY DETECTOR — INSPECTOR CLI     ")
    print("=" * 65)
    print(" Welcome, Navigator! Enter any message to check for pirate spam.")
    print(" Type 'exit', 'quit', or 'q' to end session.")
    print("=" * 65 + "\n")


def main():
    model_path = get_model_path()
    pipeline = load_trained_pipeline(model_path)
    
    display_welcome_banner()
    
    while True:
        try:
            user_input = input("GrandLine-Inspector> ").strip()
            
            # Handle exit commands
            if user_input.lower() in {"exit", "quit", "q"}:
                print("\n[INFO] Closing inspector CLI session. May fair winds guide your journey!\n")
                break
                
            # Handle empty inputs gracefully
            if not user_input:
                print(" [NOTICE] Empty message received. Please enter text to evaluate.\n")
                continue
                
            # Predict label and estimated probability
            label, probability = predict_message(user_input, pipeline)
            
            # Format results
            print("-" * 65)
            if label == "spam":
                print(f" Result             : [MARINE ALERT] Classified as 'spam'")
            else:
                print(f" Result             : [CREW MESSAGE] Classified as 'ham'")
                
            if probability is not None:
                prob_pct = probability * 100
                print(f" Estimated Probability: {prob_pct:.2f}%")
            else:
                print(f" Estimated Probability: N/A")
                
            print("-" * 65 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n[INFO] Session interrupted by user. Exiting safely.")
            break
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}\n")


if __name__ == "__main__":
    main()
