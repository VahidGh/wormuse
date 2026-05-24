"""Chopin score learning demo, kept outside `pyannow`.

The notebook imports these helpers so the notebook cells stay thin and
demonstration-focused.
"""

from .data import ScoreDataset, build_score_dataset, fourier_time_features, roll_to_note_events
from .model import TimeScoreNet, build_model, weighted_bce_with_logits, score_logits
from .train import train_score_model, predict_probabilities, best_threshold
from .render import score_to_events, evaluate_reconstruction, render_audio
