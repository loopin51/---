# ==============================================================================
# File: utils_calibration.py
# Description: NumPy + Astropy-FITS 만으로 Master Bias / Dark / Flat 생성과 
#              Light 프레임 수동 보정 기능을 제공합니다.
# ==============================================================================
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
from astropy.io import fits
from utils.fits import get_fits_keyword # utils.fits에서 함수 임포트 (경로 주의)
import astropy.units as u # 명시적 단위 사용을 위해
import os

logger_calib = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
# 기본 유틸                                                                    #
# -----------------------------------------------------------------------------#
def _load_fits(path):
    """
    지정된 경로의 FITS 파일을 로드하여 NumPy 배열 데이터와 헤더를 반환합니다.

    Args:
        path (str or Path): FITS 파일 경로.

    Returns:
        tuple: (np.ndarray, astropy.io.fits.Header) 또는 (None, None)
    """
    try:
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data
            if data is not None:
                return data.astype("float32"), hdul[0].header
            # 확장 HDU 확인 (일부 FITS 파일은 주 HDU에 데이터가 없을 수 있음)
            for hdu_ext in hdul[1:]:
                if hdu_ext.data is not None and hdu_ext.is_image:
                    logger_calib.debug(f"Data found in extension for {path}")
                    return hdu_ext.data.astype("float32"), hdu_ext.header
            logger_calib.warning(f"No image data found in {path}")
            return None, None
    except Exception as e:
        logger_calib.error(f"Error loading FITS file {path}: {e}")
        return None, None


def _median_combine(arrays):
    """
    NumPy 배열 리스트를 중앙값으로 결합합니다. NaN 값을 무시합니다.

    Args:
        arrays (list of np.ndarray): 결합할 NumPy 배열들의 리스트.

    Returns:
        np.ndarray: 중앙값 결합된 배열.
    """
    if not arrays:
        logger_calib.warning("결합할 배열이 없습니다.")
        return None
    try:
        stack = np.stack(arrays, axis=0)
        return np.nanmedian(stack, axis=0)
    except Exception as e:
        logger_calib.error(f"배열 결합 중 오류: {e}")
        return None

def _average_combine(arrays):
    """
    NumPy 배열 리스트를 평균값으로 결합합니다. NaN 값을 무시합니다.

    Args:
        arrays (list of np.ndarray): 결합할 NumPy 배열들의 리스트.

    Returns:
        np.ndarray: 평균값 결합된 배열.
    """
    if not arrays:
        logger_calib.warning("결합할 배열이 없습니다.")
        return None
    try:
        stack = np.stack(arrays, axis=0)
        return np.nanmean(stack, axis=0)
    except Exception as e:
        logger_calib.error(f"배열 결합 중 오류: {e}")
        return None

def _save_fits(path: Path, data, header, history_tag):
    """
    NumPy 데이터를 FITS 파일로 저장하고 HISTORY 키워드를 추가합니다.

    Args:
        path (Path): 저장할 파일 경로.
        data (np.ndarray): 저장할 데이터.
        header (astropy.io.fits.Header): 사용할 헤더.
        history_tag (str): HISTORY에 추가할 태그.
    """
    try:
        hdu = fits.PrimaryHDU(data.astype("float32"), header)
        hdu.header.add_history(f"{history_tag} {datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}")
        path.parent.mkdir(parents=True, exist_ok=True)
        hdu.writeto(path, overwrite=True)
        logger_calib.info(f"FITS 파일 저장 완료: {path}")
    except Exception as e:
        logger_calib.error(f"FITS 파일 저장 실패 {path}: {e}")


# -----------------------------------------------------------------------------#
# 1. Master Bias                                                               #
# -----------------------------------------------------------------------------#
def create_master_bias(bias_paths):
    """
    여러 Bias 프레임들을 중앙값으로 결합하여 Master Bias NumPy 배열을 생성합니다.

    Args:
        bias_paths (list): BIAS 파일 경로 리스트.

    Returns:
        np.ndarray or None: 생성된 마스터 BIAS 데이터, 실패 시 None.
    """
    if not bias_paths:
        logger_calib.warning("Bias 경로 리스트가 비어 있습니다.")
        return None

    logger_calib.info(f"{len(bias_paths)}개의 Bias 프레임으로 Master Bias 생성 시작...")
    datas = []
    for p in bias_paths:
        data, _ = _load_fits(p)
        if data is not None:
            datas.append(data)
    
    if not datas:
        logger_calib.warning("유효한 Bias 데이터를 로드하지 못했습니다.")
        return None

    mbias = _median_combine(datas)
    if mbias is not None:
        logger_calib.info(f"Master Bias 생성 완료 | shape={mbias.shape}")
    else:
        logger_calib.error("Master Bias 생성 실패.")
    return mbias


# -----------------------------------------------------------------------------#
# 2. Master Dark (Bias 보정 후 결합)                                           #
# -----------------------------------------------------------------------------#
def create_master_dark(dark_paths, master_bias_data_np=None):
    """
    동일 노출 시간의 Dark 프레임들을 (선택적으로) Master Bias로 보정한 후,
    중앙값으로 결합하여 Master Dark NumPy 배열을 생성합니다.

    Args:
        dark_paths (list): DARK 파일 경로 리스트.
        master_bias_data_np (np.ndarray, optional): 마스터 BIAS 데이터. Defaults to None.

    Returns:
        tuple: (master_dark_data_np, first_dark_header) 또는 (None, None)
               master_dark_data_np (np.ndarray or None): 생성된 마스터 DARK 데이터.
               first_dark_header (Header or None): 첫 번째 DARK 파일의 헤더 (노출시간 등 정보 참조용).
    """
    if not dark_paths:
        logger_calib.warning("Dark 경로 리스트가 비어 있습니다.")
        return None, None

    logger_calib.info(f"{len(dark_paths)}개의 Dark 프레임으로 Master Dark 생성 시작...")
    datas = []
    first_dark_header = None
    for i, p in enumerate(dark_paths):
        d, hdr = _load_fits(p)
        if d is not None:
            if i == 0: first_dark_header = hdr # 첫 번째 헤더 저장
            if master_bias_data_np is not None and d.shape == master_bias_data_np.shape:
                d = d - master_bias_data_np
            datas.append(d)
    
    if not datas:
        logger_calib.warning("유효한 Dark 데이터를 로드하거나 처리하지 못했습니다.")
        return None, None

    mdark = _median_combine(datas)
    if mdark is not None:
        logger_calib.info(f"Master Dark 생성 완료 | shape={mdark.shape}")
    else:
        logger_calib.error("Master Dark 생성 실패.")
    return mdark, first_dark_header


# -----------------------------------------------------------------------------#
# 3. Master Flat (정석적 방법: Bias, Dark 보정 후 정규화)                         #
# -----------------------------------------------------------------------------#
def create_master_flat(flat_paths, master_bias_data_np=None, master_dark_data_dict=None):
    """
    Raw Flat 목록(동일 필터)을 Bias 및 Dark 보정한 후,
    각 프레임을 중앙값으로 정규화하고, 이들을 중앙값으로 결합한 뒤,
    최종 결과를 다시 중앙값으로 정규화하여 Master Flat NumPy 배열을 생성합니다.

    Args:
        flat_paths (list): 동일 필터의 Raw FLAT 파일 경로 리스트.
        master_bias_data_np (np.ndarray, optional): 마스터 BIAS 데이터.
        master_dark_data_dict (dict, optional): {exptime(float): (master_dark_data, master_dark_header)}
                                                형태의 마스터 DARK 데이터 딕셔너리.

    Returns:
        tuple: (master_flat_data_np, first_flat_header) 또는 (None, None)
    """
    if not flat_paths:
        logger_calib.warning("Flat 경로 리스트가 비어 있습니다.")
        return None, None

    logger_calib.info(f"{len(flat_paths)}개의 Flat 프레임으로 Master Flat 생성 시작...")
    processed_flats = []
    first_flat_header = None

    for i, p in enumerate(flat_paths):
        flat_data, flat_header = _load_fits(p)
        if flat_data is None:
            continue
        if i == 0: first_flat_header = flat_header

        # Bias 보정
        if master_bias_data_np is not None and flat_data.shape == master_bias_data_np.shape:
            flat_data = flat_data - master_bias_data_np
        
        # Dark 보정 (Flat의 노출시간에 맞는 Dark 사용)
        flat_exptime = get_fits_keyword(flat_header, ['EXPTIME', 'EXPOSURE'], data_type=float)
        mdark_for_flat_tuple = master_dark_data_dict.get(flat_exptime) if (flat_exptime and master_dark_data_dict) else None
        
        if mdark_for_flat_tuple:
            mdark_for_flat_data, _ = mdark_for_flat_tuple
            if mdark_for_flat_data is not None and mdark_for_flat_data.shape == flat_data.shape:
                flat_data = flat_data - mdark_for_flat_data
            else:
                logger_calib.warning(f"Flat {p}에 대한 Dark 프레임 크기 불일치 또는 없음. Dark 보정 생략.")
        else:
            logger_calib.warning(f"Flat {p} (노출: {flat_exptime}s)에 맞는 Dark 프레임 없음. Dark 보정 생략.")

        # 각 Flat 프레임을 중앙값으로 정규화
        median_val = np.nanmedian(flat_data)
        if median_val is not None and not np.isclose(median_val, 0) and np.isfinite(median_val):
            processed_flats.append(flat_data / median_val)
        else:
            logger_calib.warning(f"Flat {p}의 중앙값이 유효하지 않아 정규화 없이 추가 (또는 제외). 값: {median_val}")
            processed_flats.append(flat_data) # 또는 제외하도록 수정 가능

    if not processed_flats:
        logger_calib.warning("처리할 유효한 Flat 프레임이 없습니다.")
        return None, None

    # 정규화된 플랫들을 중앙값 결합
    combined_flat = _median_combine(processed_flats)
    if combined_flat is None:
        logger_calib.error("결합된 Master Flat 생성 실패.")
        return None, first_flat_header # 헤더라도 반환 시도

    # 최종적으로 한 번 더 중앙값으로 정규화
    final_median_val = np.nanmedian(combined_flat)
    if final_median_val is not None and not np.isclose(final_median_val, 0) and np.isfinite(final_median_val):
        master_flat_data = combined_flat / final_median_val
    else:
        master_flat_data = combined_flat # 정규화 실패 시 결합된 플랫 그대로 사용
        logger_calib.warning(f"최종 Master Flat 정규화 실패 (중앙값: {final_median_val}). 정규화 안된 플랫 사용.")

    logger_calib.info(f"Master Flat 생성 완료 | shape={master_flat_data.shape}")
    return master_flat_data, first_flat_header


# -----------------------------------------------------------------------------#
# 4. Light 프레임 수동 보정                                                    #
# -----------------------------------------------------------------------------#
def calibrate_light_numpy(
    raw_light_data_np,
    raw_light_header,
    master_bias_data_np=None,
    master_dark_data_dict=None,  # {exptime(float): (master_dark_data_np, master_dark_header)}
    master_flat_data_dict=None,  # {filter_str.upper(): (master_flat_data_np, master_flat_header)}
    do_bias_correction=True,
    do_dark_correction=True,
    do_flat_correction=True
):
    """
    NumPy 배열을 사용하여 LIGHT 프레임을 수동으로 보정합니다.
    BIAS -> DARK -> FLAT 순서로 보정합니다.

    Args:
        raw_light_data_np (np.ndarray): 원본 LIGHT 데이터.
        raw_light_header (fits.Header): 원본 LIGHT 헤더.
        master_bias_data_np (np.ndarray, optional): 마스터 BIAS.
        master_dark_data_dict (dict, optional): {노출시간: (다크데이터, 다크헤더)} 딕셔너리.
        master_flat_data_dict (dict, optional): {필터: (플랫데이터, 플랫헤더)} 딕셔너리.
        do_bias_correction (bool, optional): BIAS 보정 여부.
        do_dark_correction (bool, optional): DARK 보정 여부.
        do_flat_correction (bool, optional): FLAT 보정 여부.

    Returns:
        tuple: (calibrated_data_np, processing_log_list)
    """
    processing_log = []
    calibrated_data = raw_light_data_np.copy()

    # BIAS 보정
    if do_bias_correction:
        if master_bias_data_np is not None and calibrated_data.shape == master_bias_data_np.shape:
            calibrated_data -= master_bias_data_np
            processing_log.append("수동 보정: BIAS 보정됨.")
        else:
            processing_log.append("경고 (수동 보정): BIAS 보정 생략 (BIAS 데이터 없거나 크기 불일치).")
    else:
        processing_log.append("수동 보정: BIAS 보정 건너뜀.")

    # DARK 보정
    md_to_use_data_light = None
    if do_dark_correction and master_dark_data_dict:
        light_exptime = get_fits_keyword(raw_light_header, ['EXPTIME', 'EXPOSURE'], data_type=float)
        if light_exptime is not None and light_exptime > 0:
            dark_tuple = master_dark_data_dict.get(light_exptime)
            if dark_tuple:
                md_to_use_data_light, _ = dark_tuple # 헤더는 여기선 불필요
                if md_to_use_data_light is not None and calibrated_data.shape == md_to_use_data_light.shape:
                    calibrated_data -= md_to_use_data_light
                    processing_log.append(f"수동 보정: DARK (Exp: {light_exptime}s) 보정됨.")
                else:
                    processing_log.append(f"경고 (수동 보정): LIGHT DARK 보정 생략 (Exp: {light_exptime}s DARK 데이터 없거나 크기 불일치).")
            else:
                processing_log.append(f"경고 (수동 보정): LIGHT DARK 보정 생략 (Exp: {light_exptime}s에 맞는 DARK 없음).")
        else:
            processing_log.append("경고 (수동 보정): LIGHT DARK 보정 생략 (LIGHT 노출시간 정보 없음).")
    elif do_dark_correction:
        processing_log.append("경고 (수동 보정): LIGHT DARK 보정 생략 (Master DARK 딕셔너리 없음).")
    else:
        processing_log.append("수동 보정: LIGHT DARK 보정 건너뜀.")
            
    # FLAT 보정
    final_mf_data_for_light = None
    if do_flat_correction and master_flat_data_dict:
        light_filter = get_fits_keyword(raw_light_header, ['FILTER'], default_value='Generic').upper()
        flat_tuple = master_flat_data_dict.get(light_filter)
        if not flat_tuple and light_filter != 'Generic': # 특정 필터 플랫 없고 Generic 시도
             flat_tuple = master_flat_data_dict.get('Generic')
             if flat_tuple: processing_log.append(f"  {light_filter} 필터 플랫 없어 Generic 플랫 사용 시도.")
        
        if flat_tuple:
            final_mf_data_for_light, _ = flat_tuple # 이 플랫은 이미 최종 보정 및 정규화된 상태로 가정
            if final_mf_data_for_light is not None and calibrated_data.shape == final_mf_data_for_light.shape:
                safe_flat = np.where(np.abs(final_mf_data_for_light) < 1e-5, 1.0, final_mf_data_for_light)
                calibrated_data /= safe_flat
                processing_log.append(f"수동 보정: FLAT ({light_filter}) 보정됨.")
            else:
                processing_log.append(f"경고 (수동 보정): FLAT 보정 생략 (최종 FLAT 없거나 크기 불일치 - 필터: {light_filter}).")
        else:
            processing_log.append(f"경고 (수동 보정): FLAT 보정 생략 (필터 {light_filter}에 맞는 Master FLAT 없음).")
    elif do_flat_correction:
         processing_log.append("경고 (수동 보정): FLAT 보정 생략 (Master FLAT 딕셔너리 없음).")
    else:
        processing_log.append("수동 보정: FLAT 보정 건너뜀.")
            
    return calibrated_data, processing_log


# -----------------------------------------------------------------------------#
# 5. 편의 함수 – 한 폴더에서 Master 프레임 일괄 생성                            #
#    (build_master_frames.py 로직 간단 포트)                                   #
# -----------------------------------------------------------------------------#
def build_masters_from_folders(
    bias_dir=None,
    dark_dirs=None,     # {"exp_str": folder_path} 형태 (예: {"5.0": "/path/to/darks_5sec"})
    flat_dirs=None      # {"filter_str": folder_path} 형태 (예: {"B": "/path/to/flats_B"})
):
    """
    지정된 디렉토리 구조로부터 Master Bias, Master Darks (노출시간별), 
    Master Flats (필터별, 최종 보정 및 정규화됨)를 생성합니다.

    Args:
        bias_dir (str, optional): BIAS 프레임들이 있는 디렉토리 경로.
        dark_dirs (dict, optional): 키는 노출 시간(문자열), 값은 해당 노출 시간의 
                                   DARK 프레임들이 있는 디렉토리 경로인 딕셔너리.
        flat_dirs (dict, optional): 키는 필터명(문자열), 값은 해당 필터의 
                                   FLAT 프레임들이 있는 디렉토리 경로인 딕셔너리.

    Returns:
        tuple: (master_bias_data, master_darks_data_dict, master_flats_data_dict)
               master_bias_data (np.ndarray): 생성된 마스터 BIAS.
               master_darks_data_dict (dict): {노출시간(float): (다크데이터_np, 다크헤더)} 형태.
               master_flats_data_dict (dict): {필터명(str.upper()): (플랫데이터_np, 플랫헤더)} 형태.
    """
    mbias_data = None
    if bias_dir and os.path.isdir(bias_dir):
        bias_paths = [str(p) for p in Path(bias_dir).glob("*.f*t*")] # .fit, .fits, .fts
        if bias_paths:
            mbias_data = create_master_bias(bias_paths)

    mdark_data_dict = {}
    if dark_dirs:
        for exp_str, ddir_str in dark_dirs.items():
            if os.path.isdir(ddir_str):
                dark_paths = [str(p) for p in Path(ddir_str).glob("*.f*t*")]
                if dark_paths:
                    try:
                        exp_time_float = float(exp_str)
                        mdark_data, mdark_header = create_master_dark(dark_paths, mbias_data)
                        if mdark_data is not None:
                            mdark_data_dict[exp_time_float] = (mdark_data, mdark_header)
                    except ValueError:
                        logger_calib.error(f"Dark 디렉토리 키 '{exp_str}'를 float으로 변환할 수 없습니다.")

    mflat_data_dict = {}
    if flat_dirs:
        for filt_str, fdir_str in flat_dirs.items():
            if os.path.isdir(fdir_str):
                flat_paths = [str(p) for p in Path(fdir_str).glob("*.f*t*")]
                if flat_paths:
                    mflat_data, mflat_header = create_master_flat(flat_paths, mbias_data, mdark_data_dict)
                    if mflat_data is not None:
                        mflat_data_dict[filt_str.upper()] = (mflat_data, mflat_header)

    return mbias_data, mdark_data_dict, mflat_data_dict

