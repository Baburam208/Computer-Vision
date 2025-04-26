import argparse
from engine.inference import Predictor
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Predict snake name")
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('image_path', type=str, help='Path to image for prediction')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--show_prob', action='store_true', help='Show class probabilities')
    args = parser.parse_args()

    # Initialize predictor
    predictor = Predictor(args.model_path, args.config)

    # Make prediction
    if args.show_prob:
        pred_class, probabilities = predictor.predict(args.image_path, return_probabilities=True)

        print(f"Predicted class: {pred_class}")
        print("Class probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"Class {i}: {prob:.4f}")
    else:
        pred_class = predictor.predict(args.image_path)
        print(f"Predicted class: {pred_class}")

    # Display image
    try:
        img = Image.open(args.image_path)
        img.show(title=f"Predicted class: {pred_class}")
    except :
        pass


if __name__ == '__main__':
    main()
