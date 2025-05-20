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

def create_master_bias_ccdproc(bias_file_paths, mem_limit=2e9): 
    if not bias_file_paths:
        logger_calib.warning("BIAS 파일 경로 리스트가 비어있습니다.")
        return None
    logger_calib.info(f"{len(bias_file_paths)}개의 BIAS 파일로 마스터 BIAS 생성 시작 (ccdproc)...")
    try:
        master_bias = ccdp.combine(
            bias_file_paths, 
            method='average',
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5, 
            mem_limit=mem_limit, 
            unit=u.adu 
        )
        logger_calib.info(f"Master BIAS 생성 완료 (ccdproc). Shape: {master_bias.shape}")
        return master_bias 
    except Exception as e:
        logger_calib.error(f"ccdproc로 Master BIAS 생성 중 오류: {e}", exc_info=True)
        raise RuntimeError(f"Master BIAS 생성 실패 (ccdproc): {str(e)[:100]}")

def create_master_dark_ccdproc(dark_file_paths, master_bias_ccd, mem_limit=2e9): 
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
                        logger_calib.warning(f"DARK 파일 {df_path}와 Master BIAS의 크기가 불일치. BIAS 보정 건너뜁니다.")
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
            method='average',
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
            mem_limit=mem_limit,
            unit=u.adu 
        )
        logger_calib.info(f"Master DARK (Corrected) 생성 완료 (ccdproc). Shape: {master_dark.shape}")
        return master_dark
    except Exception as e:
        logger_calib.error(f"ccdproc로 Master DARK 생성 중 오류: {e}", exc_info=True)
        raise RuntimeError(f"Master DARK 생성 실패 (ccdproc): {str(e)[:100]}")

def create_preliminary_master_flat_ccdproc(flat_file_paths, mem_limit=2e9):
    """
    ccdproc를 사용하여 BIAS나 DARK를 빼지 않은 "예비 마스터 플랫"을 생성합니다.
    단순히 여러 Raw Flat들을 결합(평균 또는 중앙값)합니다. 정규화도 하지 않습니다.
    """
    if not flat_file_paths:
        logger_calib.warning("예비 마스터 플랫 생성을 위한 FLAT 파일 경로 리스트가 비어있습니다.")
        return None
    logger_calib.info(f"{len(flat_file_paths)}개의 FLAT 파일로 예비 마스터 플랫 생성 시작 (ccdproc)...")
    try:
        # Raw Flat들을 바로 결합합니다.
        prelim_master_flat = ccdp.combine(
            flat_file_paths,
            method='average', # 또는 'median', 플랫은 보통 평균 사용
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
            mem_limit=mem_limit,
            unit=u.adu # 입력 파일들의 단위를 가정
        )
        logger_calib.info(f"예비 Master FLAT 생성 완료 (ccdproc). Shape: {prelim_master_flat.shape}")
        return prelim_master_flat
    except Exception as e:
        logger_calib.error(f"ccdproc로 예비 Master FLAT 생성 중 오류: {e}", exc_info=True)
        raise RuntimeError(f"예비 Master FLAT 생성 실패 (ccdproc): {str(e)[:100]}")
