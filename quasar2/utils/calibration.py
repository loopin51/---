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

def create_master_bias_ccdproc(bias_file_paths, mem_limit=2e8): # mem_limit 추가
    """ ccdproc를 사용하여 마스터 BIAS를 생성합니다. """
    if not bias_file_paths:
        logger_calib.warning("BIAS 파일 경로 리스트가 비어있습니다.")
        return None
    logger_calib.info(f"{len(bias_file_paths)}개의 BIAS 파일로 마스터 BIAS 생성 시작 (ccdproc)...")
    try:
        # 이전 NumPy 로직 주석 처리
        # bias_images_stack, _, _ = load_fits_from_gradio_files(bias_file_objs, "BIAS") # 이 함수는 이제 파일 객체 리스트를 받음
        # if bias_images_stack is None:
        #     raise ValueError("BIAS 이미지 스택 로드 실패")
        # master_bias_data = np.median(bias_images_stack.astype(np.float32), axis=0).astype(np.float32)
        # return CCDData(master_bias_data, unit=u.adu, meta=fits.Header()) # 예시로 CCDData 반환

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
    """ ccdproc를 사용하여 (BIAS 보정된) 마스터 DARK를 생성합니다. """
    if not dark_file_paths:
        logger_calib.warning("DARK 파일 경로 리스트가 비어있습니다.")
        return None
    logger_calib.info(f"{len(dark_file_paths)}개의 DARK 파일로 마스터 DARK 생성 시작 (ccdproc)...")
    
    try:
        # 이전 NumPy 로직 주석 처리
        # dark_images_stack, _, _ = load_fits_from_gradio_files(dark_file_objs, "DARK")
        # if dark_images_stack is None:
        #     raise ValueError("DARK 이미지 스택 로드 실패")
        # subtracted_darks_np = dark_images_stack.astype(np.float32)
        # if master_bias_data is not None:
        #     if master_bias_data.shape == subtracted_darks_np.shape[1:]:
        #         subtracted_darks_np = subtracted_darks_np - master_bias_data
        #     else: # 크기 불일치 경고
        #         pass
        # master_dark_data = np.median(subtracted_darks_np, axis=0).astype(np.float32)
        # return CCDData(master_dark_data, unit=u.adu, meta=fits.Header())

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

def scale_by_nanmedian(arr_data): # ccdproc combine의 scale 인자에 사용될 함수
    """ NumPy 배열의 중앙값으로 스케일링하기 위한 함수 (NaN 무시) """
    # arr_data는 NumPy 배열이어야 함
    if not isinstance(arr_data, np.ndarray):
        if hasattr(arr_data, 'data') and isinstance(arr_data.data, np.ndarray): # CCDData 객체인 경우
            arr_data = arr_data.data
        else:
            logger_calib.error(f"scale_by_nanmedian: 입력 데이터 타입 오류 ({type(arr_data)}). NumPy 배열 필요.")
            return 1.0 # 스케일링 안 함
            
    median_val = np.nanmedian(arr_data)
    if median_val is None or np.isclose(median_val, 0) or not np.isfinite(median_val):
        logger_calib.warning(f"스케일링을 위한 중앙값 계산 실패 또는 0에 가까움 ({median_val}). 스케일링 없이 1.0 반환.")
        return 1.0 
    return median_val


def create_master_flat_ccdproc(flat_file_paths, master_bias_ccd, master_dark_for_flat_ccd, mem_limit=2e8):
    """ ccdproc를 사용하여 (BIAS 및 DARK 보정, 정규화된) 마스터 FLAT을 생성합니다. """
    if not flat_file_paths:
        logger_calib.warning("FLAT 파일 경로 리스트가 비어있습니다.")
        return None
    logger_calib.info(f"{len(flat_file_paths)}개의 FLAT 파일로 마스터 FLAT 생성 시작 (ccdproc)...")

    try:
        # 이전 NumPy 로직 주석 처리
        # flat_images_stack, _, _ = load_fits_from_gradio_files(flat_file_objs, "FLAT")
        # if flat_images_stack is None: raise ValueError("FLAT 이미지 스택 로드 실패")
        # subtracted_flats_np = flat_images_stack.astype(np.float32)
        # if master_bias_data is not None: subtracted_flats_np -= master_bias_data
        # if master_dark_for_flat_data is not None: subtracted_flats_np -= master_dark_for_flat_data
        # combined_flat_np = np.mean(subtracted_flats_np, axis=0).astype(np.float32)
        # mean_val_np = np.mean(combined_flat_np)
        # master_flat_data = combined_flat_np / mean_val_np if mean_val_np > 1e-9 else combined_flat_np
        # return CCDData(master_flat_data, unit=u.adu, meta=fits.Header())

        flat_ccds_processed = []
        for ff_path in flat_file_paths:
            try:
                flat_ccd_raw = CCDData.read(ff_path, unit=u.adu, memmap=False)
                # ccd_process는 CCDData를 반환
                processed_single_flat = ccdp.ccd_process(
                    flat_ccd_raw,
                    master_bias=master_bias_ccd if master_bias_ccd is not None and flat_ccd_raw.shape == master_bias_ccd.shape else None,
                    master_dark=master_dark_for_flat_ccd if master_dark_for_flat_ccd is not None and flat_ccd_raw.shape == master_dark_for_flat_ccd.shape else None,
                    error=False,
                )
                flat_ccds_processed.append(processed_single_flat)
            except Exception as e_read_flat:
                logger_calib.error(f"FLAT 파일 {ff_path} 로드 또는 BIAS/DARK 보정 중 오류: {e_read_flat}")
                continue
        
        if not flat_ccds_processed:
            logger_calib.warning("유효한 BIAS/DARK 보정된 FLAT 프레임이 없습니다.")
            return None
        
        # ccdproc.combine은 scale 함수가 각 이미지에 곱해질 값을 반환하도록 기대함.
        # 즉, 중앙값으로 나누려면 1/중앙값을 반환해야 함.
        # 또는, ccdproc.Combiner를 직접 사용하여 더 세밀하게 제어 가능.
        # 여기서는 예제처럼 scale 함수를 사용하되, 각 이미지를 스케일링하는 대신,
        # 결합 후 전체를 평균으로 나누는 방식으로 정규화.
        
        # 방법 1: 각 이미지를 중앙값으로 스케일링 후 평균 결합 (예제와 유사)
        # def inv_median_scaling(data): # data는 CCDData 객체
        #     median = scale_by_nanmedian(data.data) # NumPy 배열 전달
        #     return 1.0 / median if median != 0 else 1.0

        # combined_flat_scaled = ccdp.combine(
        #     flat_ccds_processed,
        #     method='average',
        #     scale=inv_median_scaling, # 각 CCDData에 적용될 스케일링 함수
        #     sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
        #     mem_limit=mem_limit,
        #     unit=u.adu # 결과 단위 명시
        # )
        # # 이미 각 프레임이 스케일링되었으므로, 추가 정규화는 필요 없거나, 전체 평균이 1이 되도록 다시 정규화
        # final_master_flat = combined_flat_scaled 
        # final_mean = np.nanmean(final_master_flat.data)
        # if final_mean is not None and not np.isclose(final_mean, 0) and np.isfinite(final_mean):
        #      final_master_flat = final_master_flat.divide(final_mean * final_master_flat.unit / final_master_flat.unit)


        # 방법 2: 먼저 평균 결합 후, 전체를 평균값으로 정규화 (더 간단)
        combined_flat_unnorm = ccdp.combine(
            flat_ccds_processed,
            method='average', 
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
            mem_limit=mem_limit,
            unit=u.adu
        )
        mean_flat_val = np.nanmean(combined_flat_unnorm.data)
        if mean_flat_val is not None and not np.isclose(mean_flat_val, 0) and np.isfinite(mean_flat_val):
            final_master_flat = combined_flat_unnorm.divide(mean_flat_val * combined_flat_unnorm.unit / combined_flat_unnorm.unit) # 단위 유지
            logger_calib.info(f"Master FLAT (Corrected, Normalized by mean {mean_flat_val:.2f}) 생성 완료 (ccdproc). Shape: {final_master_flat.shape}")
            return final_master_flat
        else:
            logger_calib.warning(f"결합된 Master FLAT의 평균값({mean_flat_val})이 유효하지 않아 정규화 실패. 정규화 안된 Flat 반환.")
            return combined_flat_unnorm

    except Exception as e:
        logger_calib.error(f"ccdproc로 Master FLAT 생성 중 오류: {e}", exc_info=True)
        raise RuntimeError(f"Master FLAT 생성 실패 (ccdproc): {str(e)[:100]}")
