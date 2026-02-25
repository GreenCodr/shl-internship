🎧 Grammar Scoring Engine

Audio → Grammar Score (Regression) | SHL Internship Project

A lightweight, end-to-end ML pipeline that listens to a .wav audio response and predicts a grammar score using deep audio embeddings + regression + ensembling.

Goal: Given spoken audio, automatically estimate how grammatically correct the speech is — and output a numeric score.

⸻

✨ What’s inside?

✅ Audio loading + preprocessing
✅ Feature extraction using a pretrained speech model (embeddings)
✅ Regression models (Ridge / boosting models)
✅ Cross-validation (OOF predictions)
✅ Stacking + blending ensembles
✅ Calibration (optional) to squeeze extra performance
✅ Submission file generation (submission.csv)

⸻

🧠 How it works (big picture)

1) 🔊 Input: Speech Audio

User provides voice samples as .wav files.

2) 🧬 Feature Extraction (Audio → Embeddings)

Instead of handcrafted features only (pitch, MFCC, etc.), we use deep speech representations from pretrained models and convert each audio into a fixed-size numeric vector.

3) 📈 Regression Model Training

We train models to predict a continuous score (label).

4) 🧩 Ensembles (Stacking + Blending)

We combine multiple models to improve stability and accuracy using:
	•	OOF predictions
	•	weighted blending
	•	stacking meta-model

5) 🎯 Output: Grammar Score

For each test audio, we generate predicted scores and export a CSV.

⸻

🗂️ Repository Structure
.
├── shl-internship.ipynb     # Main notebook (pipeline + training + inference)
└── README.md                # You’re reading it 🙂
📊 Outputs You’ll Get

When you run the notebook, it generates:

✅ Model artifacts (often .npy)
	•	OOF predictions for training
	•	test predictions

✅ Submissions (.csv)

Examples:
	•	submission_stack_final.csv
	•	submission_blend.csv
	•	submission_calibrated.csv

⸻

🧪 Key ML Concepts Used
	•	K-Fold Cross Validation
	•	OOF predictions for safe ensembling
	•	Ridge regression baseline
	•	Boosting models for non-linear patterns
	•	Stacking meta-model
	•	Blending weights search
	•	Calibration (Isotonic / linear correction)

⸻

🌟 Why this approach?

Speech is messy: background noise, accent differences, speed, pauses…
Using pretrained speech embeddings helps the model focus on language patterns instead of raw waveforms.

Ensembles improve:
	•	performance consistency
	•	generalization
	•	robustness on unseen speakers

⸻

🔮 Future Improvements

If you want to evolve this into a production-ready system:
	•	Add VAD (voice activity detection) and silence trimming
	•	Add noise reduction + normalization
	•	Experiment with whisper transcription + grammar NLP scoring
	•	Train a hybrid model: Audio Embeddings + Text Embeddings
	•	Deploy with Streamlit / FastAPI as a scoring API
