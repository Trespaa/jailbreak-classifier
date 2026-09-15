"""
predict.py -- clasifica un prompt nuevo con el modelo ya entrenado.

Ejecutar:
    python src/predict.py "texto del prompt aqui"
"""

import sys
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "jailbreak_classifier.joblib"


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "prompt text"')
        sys.exit(1)

    prompt = sys.argv[1]
    model = joblib.load(MODEL_PATH)
    pred = model.predict([prompt])[0]
    prob = model.predict_proba([prompt])[0][1]

    label = "JAILBREAK" if pred == 1 else "regular"
    print(f"Prediction: {label}  (jailbreak probability: {prob:.3f})")


if __name__ == "__main__":
    main()