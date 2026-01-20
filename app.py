
import numpy as np
import librosa
import joblib
import gradio as gr

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")


def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


cry_remedies = {
    "belly_pain": {
        "reason": "Baby may have gas or stomach discomfort.",
        "remedy": "Gently massage the tummy, burp the baby, and keep the baby upright after feeding."
    },
    "burping": {
        "reason": "Baby needs to burp after feeding.",
        "remedy": "Hold the baby upright and gently pat the back until burping occurs."
    },
    "discomfort": {
        "reason": "Baby may feel uncomfortable due to wet diaper or temperature.",
        "remedy": "Change diaper and ensure comfortable room temperature and clothing."
    },
    "hungry": {
        "reason": "Baby is hungry and needs feeding.",
        "remedy": "Breastfeed or bottle-feed the baby with proper posture."
    },
    "tired": {
        "reason": "Baby is tired or overstimulated.",
        "remedy": "Reduce noise and light, gently rock the baby, and help them sleep."
    }
}


def predict_audio(file_path):
    if file_path is None:
        return "❌ Please upload a baby cry audio file (.wav)."

    features = extract_features(file_path)
    pred = model.predict([features])[0]
    label = encoder.inverse_transform([pred])[0]

    key = label.lower().strip()
    info = cry_remedies.get(
        key,
        {
            "reason": "General discomfort or unclear reason.",
            "remedy": "Comfort the baby and consult a pediatrician if crying persists."
        }
    )

    return (
        f"🍼 Predicted Cry Type: {label}\n\n"
        f"📌 Possible Reason:\n{info['reason']}\n\n"
        f"🏠 Home Remedy:\n{info['remedy']}\n\n"
        "⚠️ Note: This is not a medical diagnosis."
    )


ui = gr.Interface(
    fn=predict_audio,
    inputs=gr.Audio(type="filepath", label="Upload Baby Cry (.wav)"),
    outputs=gr.Textbox(lines=10, label="Prediction Result"),
    title="Newborn Cry Classification & Home Care Assistant",
    description="Upload a baby cry audio to identify the cry type and get safe home-care suggestions."
)

ui.launch()
