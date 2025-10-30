# shap_utils.py
import logging
import numpy as np


def _safe_get_shap_values(explainer, X, prediction_class):
    """Safely get SHAP values handling binary/multiclass differences."""
    shap_values_all = explainer.shap_values(X)

    logging.info(f"SHAP raw output type: {type(shap_values_all)}")

    if isinstance(shap_values_all, list):
        logging.info(f"SHAP list length: {len(shap_values_all)}")
        for i, arr in enumerate(shap_values_all):
            logging.info(f"  Class {i} shape: {getattr(arr, 'shape', 'no shape')}")
        # choose requested class (fallback to last if OOR), then first row
        class_array = shap_values_all[prediction_class if prediction_class < len(shap_values_all) else -1]
        result = class_array[0] if getattr(class_array, "ndim", 1) == 2 else np.asarray(class_array)
        return np.asarray(result)

    # ndarray path
    arr = np.asarray(shap_values_all)
    logging.info(f"SHAP single array shape: {getattr(arr, 'shape', 'no shape')}")

    # Common multiclass: (N, F, C). We always pass N==1.
    if arr.ndim == 3:
        # select first row and the requested class channel -> (F,)
        return arr[0, :, int(prediction_class)]

    # Binary/other: (N, F) or (F,)
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 1:
        return arr

    # Last resort
    return arr.flatten()



# def collapse_preprocessed_shap_to_raw(shap_vec, preprocessor, raw_features):
#     """
#     Collapse SHAP in preprocessed space (e.g., 57) back to raw features (e.g., 33).
#     - For passthrough numeric/boolean: keep as-is.
#     - For OneHotEncoder groups: sum signed contributions over the group.
#     Returns (collapsed_values: np.ndarray of len(raw_features), collapsed_names: list[str])
#     """
#     # Fallbacks
#     shap_vec = np.asarray(shap_vec).reshape(-1)
#     if preprocessor is None or not hasattr(preprocessor, "transformers_"):
#         # No info to collapse; best effort: truncate/pad to raw length
#         n = len(raw_features)
#         if shap_vec.size == n:
#             return shap_vec, list(raw_features)
#         if shap_vec.size > n:
#             return shap_vec[:n], list(raw_features)
#         out = np.zeros(n, dtype=shap_vec.dtype); out[:shap_vec.size] = shap_vec
#         return out, list(raw_features)

#     collapsed = np.zeros(len(raw_features), dtype=float)
#     # Build a map raw_feature -> indices in preprocessed
#     col_idx = 0
#     for name, trans, cols in preprocessor.transformers_:
#         if name == 'remainder' and trans == 'drop':
#             continue

#         # Normalize cols to a list of raw feature names/indices
#         if isinstance(cols, (list, tuple, np.ndarray)):
#             cols_list = list(cols)
#         elif hasattr(cols, 'tolist'):
#             cols_list = cols.tolist()
#         else:
#             cols_list = [cols]

#         if trans == 'drop':
#             continue

#         # Passthrough
#         if trans == 'passthrough':
#             for j, raw in enumerate(cols_list):
#                 raw_name = raw if isinstance(raw, str) else raw_features[raw]
#                 ridx = raw_features.index(raw_name)
#                 collapsed[ridx] += float(shap_vec[col_idx])
#                 col_idx += 1
#             continue

#         # OneHotEncoder or similar with get_feature_names_out
#         if hasattr(trans, 'get_feature_names_out'):
#             # number of generated columns
#             out_names = trans.get_feature_names_out(cols_list)
#             width = len(out_names)
#             group_slice = shap_vec[col_idx: col_idx + width]
#             # Sum signed contributions back to the original raw feature(s).
#             # If multiple input cols, split width by input feature
#             if hasattr(trans, 'categories_') and len(cols_list) == len(trans.categories_):
#                 offset = 0
#                 for raw, cats in zip(cols_list, trans.categories_):
#                     raw_name = raw if isinstance(raw, str) else raw_features[raw]
#                     w = len(cats)
#                     ridx = raw_features.index(raw_name)
#                     collapsed[ridx] += float(np.sum(group_slice[offset: offset + w]))
#                     offset += w
#             else:
#                 # Unknown structure: attribute all back to the first raw col
#                 raw_name = cols_list[0] if isinstance(cols_list[0], str) else raw_features[cols_list[0]]
#                 ridx = raw_features.index(raw_name)
#                 collapsed[ridx] += float(np.sum(group_slice))
#             col_idx += width
#             continue

#         # Transformers that output fixed width without names
#         # Try to attribute equally to inputs
#         width = getattr(trans, 'n_features_out_', None)
#         if width is None:
#             # Last resort: assume same width as len(cols_list)
#             width = len(cols_list)
#         group_slice = shap_vec[col_idx: col_idx + width]
#         share = float(np.sum(group_slice)) / max(1, len(cols_list))
#         for raw in cols_list:
#             raw_name = raw if isinstance(raw, str) else raw_features[raw]
#             ridx = raw_features.index(raw_name)
#             collapsed[ridx] += share
#         col_idx += width

#     return collapsed, list(raw_features)
