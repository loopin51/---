# ==============================================================================
# File: utils_calibration.py
# Description: Functions for creating master calibration frames.
# ==============================================================================
import numpy as np
import logging

logger_calib = logging.getLogger(__name__)

def create_master_bias_from_data(bias_images_stack):
    if bias_images_stack is None or bias_images_stack.size == 0:
        logger_calib.warning("Bias image stack is None or empty. Cannot create Master Bias.")
        return None
    logger_calib.info(f"Creating Master Bias from stack of shape {bias_images_stack.shape}...")
    try:
        master_bias = np.median(bias_images_stack, axis=0).astype(np.float32)
        logger_calib.info(f"Master Bias created with shape {master_bias.shape}.")
        return master_bias
    except Exception as e:
        logger_calib.error(f"Error creating Master Bias: {e}", exc_info=True)
        raise RuntimeError(f"Master Bias 생성 중 오류: {str(e)[:100]}")

def create_master_dark_from_data(dark_images_stack, master_bias_data):
    if dark_images_stack is None or dark_images_stack.size == 0:
        logger_calib.warning("Dark image stack is None or empty. Cannot create Master Dark.")
        return None
    logger_calib.info(f"Creating Master Dark from stack of shape {dark_images_stack.shape}...")
    
    subtracted_darks = dark_images_stack.copy() 

    if master_bias_data is not None:
        logger_calib.debug(f"Master Bias (shape {master_bias_data.shape}) provided for Dark calibration (shape {dark_images_stack.shape[1:]}).")
        if master_bias_data.shape != dark_images_stack.shape[1:]:
            err_msg = f"Master Bias shape {master_bias_data.shape} does not match Dark frame shape {dark_images_stack.shape[1:]}."
            logger_calib.error(err_msg)
            raise ValueError(err_msg)
        try:
            subtracted_darks = subtracted_darks - master_bias_data 
            logger_calib.info("Successfully subtracted Master Bias from Dark frames.")
        except Exception as e:
            logger_calib.error(f"Error subtracting Master Bias from Dark frames: {e}", exc_info=True)
            raise RuntimeError(f"Dark 프레임에서 Master Bias 차감 중 오류: {str(e)[:100]}")
    else:
        logger_calib.info("Master Bias not provided. Skipping bias subtraction for Master Dark.")
        
    try:
        master_dark = np.median(subtracted_darks, axis=0).astype(np.float32)
        logger_calib.info(f"Master Dark created with shape {master_dark.shape}.")
        return master_dark
    except Exception as e:
        logger_calib.error(f"Error creating Master Dark from subtracted darks: {e}", exc_info=True)
        raise RuntimeError(f"Master Dark 생성 중 오류: {str(e)[:100]}")

def create_master_flat_from_data(flat_images_stack, master_bias_data, master_dark_for_flat_data):
    if flat_images_stack is None or flat_images_stack.size == 0:
        logger_calib.warning("Flat image stack is None or empty. Cannot create Master Flat.")
        return None
    logger_calib.info(f"Creating Master Flat from stack of shape {flat_images_stack.shape}...")

    subtracted_flats = flat_images_stack.copy()

    if master_bias_data is not None:
        logger_calib.debug(f"Master Bias (shape {master_bias_data.shape}) provided for Flat calibration (shape {flat_images_stack.shape[1:]}).")
        if master_bias_data.shape != flat_images_stack.shape[1:]:
            err_msg = f"Master Bias shape {master_bias_data.shape} does not match Flat frame shape {flat_images_stack.shape[1:]}."
            logger_calib.error(err_msg)
            raise ValueError(err_msg)
        try:
            subtracted_flats = subtracted_flats - master_bias_data
            logger_calib.info("Successfully subtracted Master Bias from Flat frames.")
        except Exception as e:
            logger_calib.error(f"Error subtracting Master Bias from Flat frames: {e}", exc_info=True)
            raise RuntimeError(f"Flat 프레임에서 Master Bias 차감 중 오류: {str(e)[:100]}")
    else:
        logger_calib.info("Master Bias not provided. Skipping bias subtraction for Master Flat.")

    if master_dark_for_flat_data is not None:
        logger_calib.debug(f"Master Dark for Flat (shape {master_dark_for_flat_data.shape}) provided for Flat calibration (shape {flat_images_stack.shape[1:]}).")
        if master_dark_for_flat_data.shape != flat_images_stack.shape[1:]:
            err_msg = f"Master Dark for Flat shape {master_dark_for_flat_data.shape} does not match Flat frame shape {flat_images_stack.shape[1:]}."
            logger_calib.error(err_msg)
            raise ValueError(err_msg)
        try:
            subtracted_flats = subtracted_flats - master_dark_for_flat_data
            logger_calib.info("Successfully subtracted Master Dark (for Flat) from Flat frames.")
        except Exception as e:
            logger_calib.error(f"Error subtracting Master Dark (for Flat) from Flat frames: {e}", exc_info=True)
            raise RuntimeError(f"Flat 프레임에서 Master Dark (Flat용) 차감 중 오류: {str(e)[:100]}")
    else:
        logger_calib.info("Master Dark for Flat not provided. Skipping dark subtraction for Master Flat.")
        
    try:
        logger_calib.debug("Calculating combined flat using mean.")
        combined_flat = np.mean(subtracted_flats, axis=0).astype(np.float32)
        logger_calib.info(f"Combined Flat created with shape {combined_flat.shape}.")
    except Exception as e:
        logger_calib.error(f"Error creating combined Flat using mean: {e}", exc_info=True)
        raise RuntimeError(f"통합 Flat 생성 중 오류: {str(e)[:100]}")

    try:
        mean_val = np.mean(combined_flat)
        logger_calib.info(f"Mean value of combined_flat for normalization: {mean_val}")
        if not np.isfinite(mean_val) or mean_val < 1e-9: 
            logger_calib.warning(f"Mean of combined flat is {mean_val}. Skipping normalization as it might be problematic or cause division by zero/negative.")
            master_flat = combined_flat 
        else:
            master_flat = combined_flat / mean_val
            logger_calib.info("Successfully normalized Master Flat.")
        return master_flat
    except Exception as e:
        logger_calib.error(f"Error normalizing Master Flat: {e}", exc_info=True)
        raise RuntimeError(f"Master Flat 정규화 중 오류: {str(e)[:100]}")

