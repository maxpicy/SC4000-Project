# Tests for src/feature_engineering.py
import numpy as np
import pytest
from src.data_loader import load_gnss, load_ground_truth
from src.feature_engineering import (
    build_feature_matrix,
    train_lgbm_residual_model,
    predict_satellite_weights,
)


def test_feature_matrix_shape(sample_device_dir):
    gnss = load_gnss(sample_device_dir)
    gt   = load_ground_truth(sample_device_dir)
    feat = build_feature_matrix(gnss, gt)
    assert "target_residual_m" in feat.columns
    assert "elevation_deg" in feat.columns
    assert "pr_minus_geometric_m" in feat.columns
    assert len(feat) > 0


def test_feature_matrix_no_gt(sample_device_dir):
    # build_feature_matrix works when ground truth is absent (test split).
    import pandas as pd
    gnss = load_gnss(sample_device_dir)
    feat = build_feature_matrix(gnss, pd.DataFrame())
    assert feat["target_residual_m"].isna().all()


def test_lgbm_model_trains(sample_device_dir):
    gnss  = load_gnss(sample_device_dir)
    gt    = load_ground_truth(sample_device_dir)
    feat  = build_feature_matrix(gnss, gt)
    model = train_lgbm_residual_model(feat)
    assert model is not None


def test_predict_weights_positive(sample_device_dir):
    gnss   = load_gnss(sample_device_dir)
    gt     = load_ground_truth(sample_device_dir)
    feat   = build_feature_matrix(gnss, gt)
    model  = train_lgbm_residual_model(feat)
    w      = predict_satellite_weights(feat, model)
    assert (w > 0).all()
    assert w.shape[0] == len(feat)
