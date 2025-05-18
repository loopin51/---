# ==============================================================================
# File: utils_fits.py
# Description: Utility functions for FITS file handling and image preview.
# ==============================================================================
import os
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch, LogStretch, LinearStretch
from PIL import Image, ImageDraw 
import logging

logger_fits = logging.getLogger(__name__) 

def get_fits_keyword(header, keys, default_value=None, data_type=str):
    """ FITS 헤더에서 여러 가능한 키워드 중 하나를 찾아 값을 반환합니다. """
    if header is None:
        return default_value
    for key in keys:
        value = header.get(key)
        if value is not None:
            try:
                if data_type is float: return float(value)
                if data_type is int: return int(value)
                return str(value).strip() # 기본적으로 문자열로 반환하고 공백 제거
            except ValueError:
                logger_fits.warning(f"키워드 '{key}'의 값 '{value}'을(를) {data_type}으로 변환 불가. 다음 키워드 시도.")
                continue
    return default_value

def load_fits_from_gradio_files(file_objects, object_type="프레임"):
    logger_fits.debug(f"Attempting to load {len(file_objects) if file_objects else 0} FITS files for {object_type}.")
    if not file_objects:
        logger_fits.warning(f"No file objects provided to load_fits_from_gradio_files for {object_type}.")
        return None, None, [] 
    all_images_list = []
    first_valid_header = None # 첫 번째 '유효한' 헤더를 저장
    all_headers = [] 
    loaded_file_count = 0

    for i, file_obj in enumerate(file_objects):
        current_header_for_list = None # 현재 파일의 헤더 (오류 시 None)
        if file_obj is None or not hasattr(file_obj, 'name') or file_obj.name is None:
            logger_fits.warning(f"File object at index {i} for {object_type} is invalid or has no name.")
            all_headers.append(current_header_for_list)
            continue
        file_path = file_obj.name
        logger_fits.info(f"Loading FITS file for {object_type}: {file_path}")
        try:
            with fits.open(file_path) as hdul:
                data_hdu = None
                header_from_hdu = None
                if hdul[0].data is not None and hdul[0].data.size > 0 :
                     data_hdu = hdul[0]
                     header_from_hdu = hdul[0].header.copy()
                else: 
                    for hdu_ext_idx, hdu_ext in enumerate(hdul[1:], start=1):
                        if isinstance(hdu_ext, (fits.ImageHDU, fits.CompImageHDU)) and hdu_ext.data is not None and hdu_ext.data.size > 0:
                            data_hdu = hdu_ext
                            header_from_hdu = hdu_ext.header.copy()
                            logger_fits.debug(f"Data found in HDU extension {hdu_ext_idx} for {object_type}")
                            break
                
                if data_hdu is None:
                    logger_fits.warning(f"No valid image data found in FITS file for {object_type}: {file_path}")
                    all_headers.append(None) # 유효 데이터 없으면 헤더도 None
                    continue

                data = data_hdu.data.astype(np.float32)
                current_header_for_list = header_from_hdu # 유효한 헤더
                all_headers.append(current_header_for_list)

                if first_valid_header is None and current_header_for_list is not None:
                    first_valid_header = current_header_for_list
                
                if all_images_list and data.shape != all_images_list[0].shape: # 첫 번째 유효 이미지와 크기 비교
                    err_msg = f"{object_type}들 간 이미지 크기 불일치. 예상: {all_images_list[0].shape}, 실제: {data.shape} ({os.path.basename(file_path)})."
                    logger_fits.error(err_msg)
                    raise ValueError(err_msg) 

                all_images_list.append(data)
                loaded_file_count += 1
        except FileNotFoundError:
            logger_fits.error(f"FITS file not found for {object_type}: {file_path}")
            all_headers.append(None) 
            raise
        except ValueError as ve: # 크기 불일치 등
            logger_fits.error(f"ValueError during FITS loading from {file_path} for {object_type}: {ve}")
            # ValueError 발생 시 해당 파일은 스택에 추가되지 않음. all_headers에는 이미 None이 추가되었거나, 이전 유효 헤더가 있을 수 있음.
            # 일관성을 위해, 오류 발생 시 해당 파일의 헤더는 None으로 간주하고 넘어가는 것이 나을 수 있으나,
            # 여기서는 예외를 발생시켜 호출자가 처리하도록 함.
            raise
        except Exception as e:
            logger_fits.error(f"Failed to load FITS file {file_path} for {object_type}: {e}", exc_info=True)
            all_headers.append(None)
            raise RuntimeError(f"{object_type} FITS 파일 로딩 중 오류 ({os.path.basename(file_path)}): {str(e)[:100]}")

    if not all_images_list: 
        logger_fits.warning(f"No valid FITS images were loaded for {object_type}.")
        return None, None, all_headers 
    
    logger_fits.info(f"Successfully loaded {loaded_file_count} FITS files into a list for {object_type}.")
    try:
        stacked_images = np.stack(all_images_list, axis=0)
        logger_fits.info(f"Stacked {len(all_images_list)} images for {object_type}. Stack shape: {stacked_images.shape}")
        return stacked_images, first_valid_header, all_headers # first_header 대신 first_valid_header
    except Exception as e: 
        logger_fits.error(f"Failed to stack images for {object_type}: {e}", exc_info=True)
        raise RuntimeError(f"{object_type} 이미지 스택 중 오류: {e}")


def load_single_fits_from_path(file_path, description="파일"):
    if not file_path or not os.path.exists(file_path):
        logger_fits.warning(f"{description} 경로가 유효하지 않거나 파일이 없습니다: {file_path}")
        return None, None
    logger_fits.info(f"Loading single FITS {description} from path: {file_path}")
    try:
        with fits.open(file_path) as hdul:
            data_hdu = None
            if hdul[0].data is not None and hdul[0].data.size > 0 :
                 data_hdu = hdul[0]
            else: 
                for hdu_ext_idx, hdu_ext in enumerate(hdul[1:], start=1):
                    if isinstance(hdu_ext, (fits.ImageHDU, fits.CompImageHDU)) and hdu_ext.data is not None and hdu_ext.data.size > 0:
                        data_hdu = hdu_ext
                        logger_fits.debug(f"Data found in HDU extension {hdu_ext_idx} for {description}")
                        break
            
            if data_hdu is None:
                logger_fits.warning(f"No valid image data found in any HDU for FITS {description}: {file_path}")
                return None, None
                
            data = data_hdu.data.astype(np.float32)
            header = data_hdu.header.copy()
            logger_fits.info(f"Successfully loaded {description} from {file_path}, shape: {data.shape}")
            return data, header
    except Exception as e:
        logger_fits.error(f"Failed to load single FITS {description} from {file_path}: {e}", exc_info=True)
        raise RuntimeError(f"{description} 로딩 오류 ({os.path.basename(file_path)}): {str(e)[:100]}")

def save_fits_image(data, header, base_filename, output_dir, timestamp_str):
    if data is None:
        logger_fits.warning(f"No data provided to save for {base_filename}.")
        return None
    
    filename = f"{base_filename}_{timestamp_str}.fits"
    filepath = os.path.join(output_dir, filename)
    
    logger_fits.info(f"Attempting to save FITS file to: {filepath}")

    if header is None:
        logger_fits.warning("No header provided for saving FITS. Creating a minimal header.")
        header = fits.Header()
    
    header['HISTORY'] = f'Generated by Gradio Astro App on {timestamp_str}'
    header['FILENAME'] = filename 

    try:
        hdu = fits.PrimaryHDU(data.astype(np.float32), header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(filepath, overwrite=True, output_verify='fix')
        logger_fits.info(f"Successfully saved FITS file: {filepath}")
        return filepath
    except Exception as e:
        logger_fits.error(f"Error saving FITS file {filepath}: {e}", exc_info=True)
        raise RuntimeError(f"FITS 파일 저장 중 오류 ({filename}): {str(e)[:100]}")

def create_preview_image(fits_data_2d, stretch_type='asinh', a_param=0.1):
    logger_fits.info(f"미리보기 이미지 생성 시도 (데이터 형태: {fits_data_2d.shape if fits_data_2d is not None else 'None'}, 스트레칭: {stretch_type})")
    if fits_data_2d is None or fits_data_2d.ndim != 2:
        logger_fits.warning("미리보기를 위한 데이터가 유효하지 않음 (None 또는 2D가 아님).")
        return None
    try:
        if not np.all(np.isfinite(fits_data_2d)):
            logger_fits.debug("데이터에 NaN 또는 Inf 포함. 유효한 값으로 대체 시도.")
            min_valid = np.nanmin(fits_data_2d[np.isfinite(fits_data_2d)]) if np.any(np.isfinite(fits_data_2d)) else 0
            max_valid = np.nanmax(fits_data_2d[np.isfinite(fits_data_2d)]) if np.any(np.isfinite(fits_data_2d)) else 1
            fits_data_2d = np.nan_to_num(fits_data_2d, nan=min_valid, posinf=max_valid, neginf=min_valid)
        
        try:
            interval = ZScaleInterval(contrast=0.25)
            vmin, vmax = interval.get_limits(fits_data_2d)
        except IndexError: 
            logger_fits.warning("ZScaleInterval 실패. Min/Max로 대체.")
            vmin, vmax = np.min(fits_data_2d), np.max(fits_data_2d)

        if stretch_type == 'asinh': stretch = AsinhStretch(a=a_param)
        elif stretch_type == 'log': stretch = LogStretch()
        else: stretch = LinearStretch()

        if np.isclose(vmin, vmax) or vmax < vmin :
            logger_fits.warning(f"vmin ({vmin})과 vmax ({vmax})가 거의 같거나 순서가 잘못됨. 범위 조정 시도.")
            min_val_data, max_val_data = np.min(fits_data_2d), np.max(fits_data_2d)
            if np.all(np.isclose(fits_data_2d, fits_data_2d.flat[0])):
                scaled_data_arr = np.full_like(fits_data_2d, 127, dtype=np.uint8) 
                pil_image = Image.fromarray(scaled_data_arr, mode='L')
                logger_fits.debug("데이터가 완전히 균일. 중간 회색으로 미리보기 생성.")
                return pil_image
            else: 
                vmin = min_val_data - ( (max_val_data - min_val_data) * 0.01 + 1e-5 ) 
                vmax = max_val_data + ( (max_val_data - min_val_data) * 0.01 + 1e-5 )
                if vmax <= vmin: vmax = vmin + 1e-4 
                logger_fits.debug(f"조정된 vmin={vmin}, vmax={vmax}")
        
        norm = ImageNormalize(fits_data_2d, vmin=vmin, vmax=vmax, stretch=stretch, clip=True)
        scaled_data = norm(fits_data_2d) 

        image_8bit = (np.clip(scaled_data, 0, 1) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_8bit, mode='L')
        logger_fits.info("미리보기용 PIL 이미지 성공적으로 생성.")
        return pil_image
    except Exception as e:
        logger_fits.error(f"미리보기 이미지 생성 중 오류 발생: {e}", exc_info=True)
        return None

def draw_roi_on_pil_image(base_pil_image, roi_x_min, roi_x_max, roi_y_min, roi_y_max):
    if base_pil_image is None:
        logger_fits.warning("ROI를 그릴 기본 이미지가 없습니다.")
        return None 
    
    try:
        img_with_roi = base_pil_image.copy()
        if img_with_roi.mode == 'L':
            img_with_roi = img_with_roi.convert('RGB')
            
        draw = ImageDraw.Draw(img_with_roi)
        
        if roi_x_min is not None and roi_x_max is not None and \
           roi_y_min is not None and roi_y_max is not None:
            
            img_width, img_height = img_with_roi.size
            x0 = np.clip(int(roi_x_min), 0, img_width -1)
            x1 = np.clip(int(roi_x_max), 0, img_width -1)
            y0 = np.clip(int(roi_y_min), 0, img_height -1)
            y1 = np.clip(int(roi_y_max), 0, img_height -1)

            if x1 > x0 and y1 > y0: 
                draw.rectangle([(x0, y0), (x1, y1)], outline="lime", width=2) 
                logger_fits.debug(f"ROI 사각형 그림: ({x0},{y0})-({x1},{y1})")
            else:
                logger_fits.debug("ROI 좌표가 유효하지 않아 사각형을 그리지 않음 (예: x_max <= x_min).")
        
        return img_with_roi
    except Exception as e:
        logger_fits.error(f"ROI 시각화 이미지 생성 중 오류: {e}", exc_info=True)
        return base_pil_image 

