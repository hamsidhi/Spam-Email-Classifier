"""
Grand Line Message Bounty Detector — One Piece Anime Interactive Inspector
Script: predict_message.py
Description: Interactive terminal inspector that loads the saved Marine Spam Pipeline
             and evaluates incoming text messages for pirate spam signals.
"""

import sys
from pathlib import Path
import joblib


def locate_marine_model_path():
    """
    Resolves project root and returns path to saved joblib model artifact.
    """
    script_dir = Path(__file__).resolve().parent
    grand_line_root = script_dir.parent
    model_artifact_path = grand_line_root / "models" / "spam_message_pipeline.joblib"
    return model_artifact_path


def load_marine_pipeline(model_artifact_path):
    """
    Loads saved Marine Spam Pipeline from disk.
    Exits safely if model artifact is missing.
    """
    if not model_artifact_path.exists():
        print("\n" + "=" * 65)
        print(" [ERROR] MARINE SPAM PIPELINE NOT FOUND ")
        print("=" * 65)
        print(f"Target Path: {model_artifact_path}")
        print("\n[INSTRUCTION] Train the Marine classifier first by running:")
        print("              python src/train_model.py\n")
        sys.exit(1)
        
    try:
        marine_pipeline = joblib.load(model_artifact_path)
        return marine_pipeline
    except Exception as error_msg:
        print(f"\n[ERROR] Could not load Marine Den Den Mushi model: {error_msg}")
        sys.exit(1)


def inspect_message(incoming_text: str, marine_pipeline):
    """
    Accepts one incoming text string and predicts whether it is 'ham' or 'spam'
    along with estimated model probability.
    """
    # Wrap incoming message text in a list for scikit-learn pipeline
    scroll_data = [incoming_text]
    
    # Predict label class
    bounty_label = marine_pipeline.predict(scroll_data)[0]
    
    # Extract probability if supported
    if hasattr(marine_pipeline, "predict_proba"):
        probability_scores = marine_pipeline.predict_proba(scroll_data)[0]
        class_labels = marine_pipeline.classes_
        
        prob_mapping = dict(zip(class_labels, probability_scores))
        estimated_probability = prob_mapping[bounty_label]
    else:
        estimated_probability = None
        
    return bounty_label, estimated_probability


def display_grand_line_banner():
    """
    Displays themed CLI introduction banner.
    """
    print("\n" + "=" * 65)
    print("      GRAND LINE MESSAGE BOUNTY DETECTOR — INSPECTOR CLI     ")
    print("=" * 65)
    print(" Welcome, Navigator! Enter any message scroll to test for pirate spam.")
    print(" Type 'exit', 'quit', or 'q' to return to your ship.")
    print("=" * 65 + "\n")


def main():
    model_artifact_path = locate_marine_model_path()
    marine_pipeline = load_marine_pipeline(model_artifact_path)
    
    display_grand_line_banner()
    
    while True:
        try:
            navigator_input = input("GrandLine-Inspector> ").strip()
            
            # Check for exit commands
            if navigator_input.lower() in {"exit", "quit", "q"}:
                print("\n[INFO] Closing inspector CLI session. May fair winds guide your journey!\n")
                break
                
            # Handle empty inputs gracefully
            if not navigator_input:
                print(" [NOTICE] Empty scroll received. Enter text to inspect.\n")
                continue
                
            # Inspect message
            bounty_label, estimated_prob = inspect_message(navigator_input, marine_pipeline)
            
            # Format themed output
            print("-" * 65)
            if bounty_label == "spam":
                print(f" Result             : [MARINE ALERT] Classified as 'spam'")
                print(f" Bounty Status      : [ALERT] Pirate Spam Notice Detected!")
            else:
                print(f" Result             : [CREW MESSAGE] Classified as 'ham'")
                print(f" Bounty Status      : [SAFE] Communication from Straw Hat Crew!")
                
            if estimated_prob is not None:
                prob_percentage = estimated_prob * 100
                print(f" Model Probability  : {prob_percentage:.2f}%")
            else:
                print(f" Model Probability  : N/A")
                
            print("-" * 65 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n[INFO] Inspector session interrupted by user. Safe voyage!")
            break
        except Exception as err:
            print(f"\n[ERROR] An unexpected error occurred: {err}\n")


if __name__ == "__main__":
    main()
