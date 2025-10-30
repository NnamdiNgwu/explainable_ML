"""loads models and creates explainers"""
import json
import joblib
import torch
import pathlib
import shap
import logging
import sys
from sklearn.ensemble import RandomForestClassifier
from models import *  # noqa: F401,F403
from models.safe_smote import *  # noqa: F401,F403
from models.cybersecurity_transformer import build_cybersecurity_transformer_from_maps as bctm
from models.safe_smote import SafeSMOTE
from src.serving.utils.feature_mapping import TransformerExplanationMapper
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline


def _extract_rf_estimator(obj):
    """Return (bare_rf, rf_pipeline_or_None)."""
    if isinstance(obj, RandomForestClassifier):
        return obj, None
    if isinstance(obj, (ImbPipeline, SkPipeline)):
        last = obj.steps[-1][1] if obj.steps else None
        if isinstance(last, RandomForestClassifier):
            return last, obj
    return None, None


class ModelLoader:
    @staticmethod
    def _create_explainers(rf_model, rf_estimator,
                            transformer_model, embed_maps,
                            feature_lists, feature_names, 
                            preprocessor=None, rf_pipeline=None):
        explainers = {}
        try:
            logging.info("Creating SHAP TreeExplainer for RF estimator…")
            explainers['rf_explainer'] = shap.TreeExplainer(rf_estimator)
        except Exception as e:
            logging.error(f"Failed to create RF SHAP explainer: {e}")
            raise

        _, rf_pipeline = _extract_rf_estimator(rf_model)

        try:
            explainers['transformer_explainer'] = ModelLoader._create_explanation_mapper(
                rf_explainer=explainers['rf_explainer'],
                transformer_model=transformer_model,
                embed_maps=embed_maps,
                feature_lists=feature_lists,
                features_order=feature_names,
                preprocessor=preprocessor,
                rf_estimator=rf_estimator,
                rf_pipeline=rf_pipeline,
            )
        except Exception as e:
            logging.error(f"Failed to create Transformer explanation mapper: {e}")
            raise

        return explainers

    @staticmethod
    def _create_explanation_mapper(
        rf_explainer,
        transformer_model,
        embed_maps,
        feature_lists,
        features_order,
        preprocessor=None,
        rf_estimator=None,
        rf_pipeline=None,
    ):
        return TransformerExplanationMapper(
            rf_explainer=rf_explainer,
            transformer_model=transformer_model,
            feature_lists=feature_lists,
            features_order=features_order,
            embed_maps=embed_maps,
            preprocessor=preprocessor,
            rf_pipeline=rf_pipeline,
            rf_estimator=rf_estimator,
        )

    @staticmethod
    def load_all_models(model_dir):
        model_dir = pathlib.Path(model_dir)
        logging.info(f"Loading models from: {model_dir}")

        try:
            # Load config files
            cascade_config = json.loads((model_dir / "cascade_config.json").read_text())
            embed_maps     = json.loads((model_dir / "embedding_maps.json").read_text())
            feature_lists  = json.loads((model_dir / "feature_lists.json").read_text())

            CONT = feature_lists["CONTINUOUS_USED"]
            BOOL = feature_lists["BOOLEAN_USED"]
            HI   = feature_lists["HIGH_CAT_USED"]
            LO   = feature_lists["LOW_CAT_USED"]
            FEATURES_ORDER = CONT + BOOL + HI + LO

            # Build feature defaults
            feature_defaults = {}
            for f in CONT:
                feature_defaults[f] = 0.0
            for f in BOOL:
                feature_defaults[f] = False
            for f in (HI + LO):
                feature_defaults[f] = "unknown"

            # Load RF model (with error handling)
            logging.info(f"Loading RF model from {cascade_config['rf_model_path']}…")
            try:
                rf_model = joblib.load(cascade_config["rf_model_path"])
                logging.info(f"✓ Loaded RF model: {type(rf_model).__name__}")
            except ModuleNotFoundError as e:
                logging.error(f"❌ Pickle import error: {e}. Ensure models/ shim exists at repo root.")
                raise RuntimeError(
                    f"Failed to unpickle RF model. Missing module: {e}. "
                ) from e

            # Load preprocessors
            try:
                pre = None
                preprocessors_path = model_dir / "preprocessors.pkl"
                if preprocessors_path.exists():
                    try:
                        logging.info("Loading preprocessor..")
                        preprocessors = joblib.load(model_dir / "preprocessors.pkl")
                        pre = preprocessors['pre']
                        logging.info("Loaded preprocessors")
                    except Exception as e:
                       logging.error(f"Failed to load preprocessors: {e}")
            
                else:
                    logging.warning("No preprocessors.pkl found; proceeding without preprocessor.")
            except Exception as e:
                logging.error(f"Error loading preprocessors: {e}")
                raise

            # Extract RF estimator
            rf_estimator, _ = _extract_rf_estimator(rf_model)
            if rf_estimator is None:
                raise ValueError(
                    f"RandomForestClassifier not found in {type(rf_model).__name__}. "
                )
            logging.info(f"Extracted RF estimator with {rf_estimator.n_features_in_} features")

            # Build and load transformer
            logging.info("Building Transformer model…")
            cont_dim = len(CONT) + len(BOOL)
            transformer_model = bctm(embed_maps, continuous_dim=cont_dim, num_classes=3)
            
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logging.info(f"Loading Transformer weights to {device}…")
            
            try:
                state_dict = torch.load(
                    cascade_config["transformer_model_path"],
                    map_location=device  # Use detected device
                )
                # Strip _orig_mod prefix if present (from compiled models)
                if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                
                transformer_model.load_state_dict(state_dict)
                transformer_model.to(device)
                transformer_model.eval()
                logging.info("✓ Loaded and moved Transformer to device")
            except Exception as e:
                logging.error(f"Failed to load Transformer state: {e}")
                raise

            # Create explainers
            logging.info("Creating explainers…")
            explainers = ModelLoader._create_explainers(
                rf_model, rf_estimator,
                transformer_model, embed_maps,
                feature_lists, FEATURES_ORDER,
                preprocessor=pre,
                rf_pipeline=rf_model
            )
            logging.info("Explainers created")

            # Return fully loaded models dict
            models = {
                'rf': rf_model,
                'rf_estimator': rf_estimator,
                'transformer': transformer_model,
                'preprocessor': pre,
                'embed_maps': embed_maps,
                'feature_lists': feature_lists,
                'feature_names': FEATURES_ORDER,
                'feature_defaults': feature_defaults,
                'tau': float(cascade_config['tau']),
                'tau2': float(cascade_config['tau2']),
                'device': device,
                'cascade_config': cascade_config,
                **explainers,
            }

            logging.info(
                f"All models loaded successfully. "
                # f"RF: {rf_estimator.n_estimators} trees, "
                f"Transformer: {type(transformer_model).__name__}, "
                f"Device: {device}"
            )

            return models

        except Exception as e:
            logging.error(f"❌ Failed to load models: {e}", exc_info=True)
            raise