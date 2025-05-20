# ==============================================================================
# File: utils_calibration.py
# Description: Functions for creating master calibration frames using ccdproc.
# ==============================================================================
import numpy as np
import logging
from astropy.nddata import CCDData
from astropy import units as u
import ccdproc as ccdp 

logger_calib = logging.getLogger(__name__)

def create_master_bias_ccdproc(bias_file_paths, mem_limit=2e8): 
    if not bias_file_paths:
        logger_calib.warning("BIAS 파일 경로 리스트가 비어있습니다.")
        return None
    logger_calib.info(f"{len(bias_file_paths)}개의 BIAS 파일로 마스터 BIAS 생성 시작 (ccdproc)...")
    try:
        master_bias = ccdp.combine(
            bias_file_paths, 
            method='median',
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5, 
            mem_limit=mem_limit, 
            unit=u.adu 
        )
        logger_calib.info(f"Master BIAS 생성 완료 (ccdproc). Shape: {master_bias.shape}")
        return master_bias 
    except Exception as e:
        logger_calib.error(f"ccdproc로 Master BIAS 생성 중 오류: {e}", exc_info=True)
        raise RuntimeError(f"Master BIAS 생성 실패 (ccdproc): {str(e)[:100]}")

def create_master_dark_ccdproc(dark_file_paths, master_bias_ccd, mem_limit=2e8): 
    if not dark_file_paths:
        logger_calib.warning("DARK 파일 경로 리스트가 비어있습니다.")
        return None
    logger_calib.info(f"{len(dark_file_paths)}개의 DARK 파일로 마스터 DARK 생성 시작 (ccdproc)...")
    
    try:
        dark_ccds_corrected = []
        for df_path in dark_file_paths:
            try:
                dark_ccd_raw = CCDData.read(df_path, unit=u.adu, memmap=False)
                if master_bias_ccd is not None:
                    if dark_ccd_raw.shape == master_bias_ccd.shape:
                        dark_corrected_single = ccdp.subtract_bias(dark_ccd_raw, master_bias_ccd)
                        dark_ccds_corrected.append(dark_corrected_single)
                    else:
                        logger_calib.warning(f"DARK 파일 {df_path}와 Master BIAS의 크기가 불일치. BIAS 보정 건너<0xEB><0><0x8F><0xEB><0x82><0xB4>니다.")
                        dark_ccds_corrected.append(dark_ccd_raw) 
                else:
                    dark_ccds_corrected.append(dark_ccd_raw) 
            except Exception as e_read_dark:
                logger_calib.error(f"DARK 파일 {df_path} 로드 또는 BIAS 보정 중 오류: {e_read_dark}")
                continue 
        
        if not dark_ccds_corrected:
            logger_calib.warning("유효한 BIAS 보정된 DARK 프레임이 없습니다.")
            return None

        master_dark = ccdp.combine(
            dark_ccds_corrected,
            method='median',
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
            mem_limit=mem_limit,
            unit=u.adu 
        )
        logger_calib.info(f"Master DARK (Corrected) 생성 완료 (ccdproc). Shape: {master_dark.shape}")
        return master_dark
    except Exception as e:
        logger_calib.error(f"ccdproc로 Master DARK 생성 중 오류: {e}", exc_info=True)
        raise RuntimeError(f"Master DARK 생성 실패 (ccdproc): {str(e)[:100]}")

def create_master_flat_ccdproc(flat_file_paths, master_bias_ccd, master_dark_for_flat_ccd, mem_limit=2e8):
    if not flat_file_paths:
        logger_calib.warning("FLAT 파일 경로 리스트가 비어있습니다.")
        return None
    logger_calib.info(f"{len(flat_file_paths)}개의 FLAT 파일로 마스터 FLAT 생성 시작 (ccdproc)...")

    try:
        flat_ccds_processed = []
        for ff_path in flat_file_paths:
            try:
                flat_ccd_raw = CCDData.read(ff_path, unit=u.adu, memmap=False)
                # ccd_process 호출 시 master_dark 대신 dark_frame 사용
                processed_single_flat = ccdp.ccd_process(
                    flat_ccd_raw,
                    master_bias=master_bias_ccd if master_bias_ccd is not None and flat_ccd_raw.shape == master_bias_ccd.shape else None,
                    dark_frame=master_dark_for_flat_ccd if master_dark_for_flat_ccd is not None and flat_ccd_raw.shape == master_dark_for_flat_ccd.shape else None, # 변경됨
                    error=False,
                )
                flat_ccds_processed.append(processed_single_flat)
            except Exception as e_read_flat:
                logger_calib.error(f"FLAT 파일 {ff_path} 로드 또는 BIAS/DARK 보정 중 오류: {e_read_flat}")
                continue
        
        if not flat_ccds_processed:
            logger_calib.warning("유효한 BIAS/DARK 보정된 FLAT 프레임이 없습니다.")
            return None
        
        combined_flat_unnorm = ccdp.combine(
            flat_ccds_processed,
            method='average', 
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
            mem_limit=mem_limit,
            unit=u.adu
        )
        
        mean_flat_val = np.nanmean(combined_flat_unnorm.data)
        if mean_flat_val is not None and not np.isclose(mean_flat_val, 0) and np.isfinite(mean_flat_val):
            final_master_flat = combined_flat_unnorm.divide(mean_flat_val * combined_flat_unnorm.unit) 
            logger_calib.info(f"Master FLAT (Corrected, Normalized by mean {mean_flat_val:.2f}) 생성 완료 (ccdproc). Shape: {final_master_flat.shape}")
            return final_master_flat
        else:
            logger_calib.warning(f"결합된 Master FLAT의 평균값({mean_flat_val})이 유효하지 않아 정규화 실패. 정규화 안된 Flat 반환.")
            return combined_flat_unnorm

    except Exception as e:
        logger_calib.error(f"ccdproc로 Master FLAT 생성 중 오류: {e}", exc_info=True)
        raise RuntimeError(f"Master FLAT 생성 실패 (ccdproc): {str(e)[:100]}")
