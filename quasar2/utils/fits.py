# ==============================================================================
# File: utils_fits.py
# Description: Utility functions for FITS file handling and image preview.
# ==============================================================================
import os
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch, LogStretch, LinearStretch
from PIL import Image, ImageDraw, ImageFont # ImageFont 추가
import logging

logger_fits = logging.getLogger(__name__) 

def get_fits_keyword(header, keys, default_value=None, data_type=str, quiet=False):
    if header is None:
        if not quiet: logger_fits.warning("헤더가 None이므로 키워드를 찾을 수 없습니다.")
        return default_value
    for key in keys:
        value = header.get(key)
        if value is not None:
            try:
                if data_type is float: return float(value)
                if data_type is int: return int(value)
                return str(value).strip() 
            except ValueError:
                if not quiet: logger_fits.warning(f"키워드 '{key}'의 값 '{value}'을(를) {data_type}으로 변환 불가. 다음 키워드 시도.")
                continue
    if not quiet and default_value is None: 
        logger_fits.debug(f"키워드 목록 {keys}에 해당하는 값을 헤더에서 찾지 못했습니다.")
    return default_value

def load_fits_data_and_headers_from_file_objs(file_objects, object_type="프레임"):
    logger_fits.debug(f"Attempting to load data and headers from {len(file_objects) if file_objects else 0} FITS files for {object_type}.")
    if not file_objects:
        logger_fits.warning(f"No file objects provided for {object_type}.")
        return [] 

    loaded_data_header_pairs = []
    
    for i, file_obj in enumerate(file_objects):
        if file_obj is None or not hasattr(file_obj, 'name') or file_obj.name is None:
            logger_fits.warning(f"File object at index {i} for {object_type} is invalid.")
            loaded_data_header_pairs.append((None, None)) 
            continue
        
        file_path = file_obj.name
        logger_fits.info(f"Loading FITS file for {object_type}: {file_path}")
        try:
            data, header = load_single_fits_from_path(file_path, description=f"{object_type} 파일 {i+1}")
            loaded_data_header_pairs.append((data, header))
        except Exception as e:
            logger_fits.error(f"Error loading {file_path} for {object_type}: {e}", exc_info=True)
            loaded_data_header_pairs.append((None, None)) 
    
    return loaded_data_header_pairs


def load_single_fits_from_path(file_path, description="파일"):
    if not file_path or not os.path.exists(file_path):
        logger_fits.warning(f"{description} 경로가 유효하지 않거나 파일이 없습니다: {file_path}")
        return None, None
    logger_fits.info(f"Loading single FITS {description} from path: {file_path}")
    try:
        with fits.open(file_path, memmap=False) as hdul:
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
                
            data = data_hdu.data 
            if data is not None: data = data.astype(np.float32) 
            header = data_hdu.header.copy()
            logger_fits.info(f"Successfully loaded {description} from {file_path}, shape: {data.shape if data is not None else 'N/A'}")
            return data, header
    except Exception as e:
        logger_fits.error(f"Failed to load single FITS {description} from {file_path}: {e}", exc_info=True)
        return None, None 

def save_fits_image(data_object, header_object, base_filename, output_dir, timestamp_str):
    if data_object is None:
        logger_fits.warning(f"No data provided to save for {base_filename}.")
        return None
    
    filename = f"{base_filename}_{timestamp_str}.fits"
    filepath = os.path.join(output_dir, filename)
    
    logger_fits.info(f"Attempting to save FITS file to: {filepath}")

    data_to_save = None
    header_to_save = None

    if hasattr(data_object, 'data') and hasattr(data_object, 'header'): # CCDData 객체인 경우
        data_to_save = data_object.data
        header_to_save = data_object.header.copy() 
        if header_object is not None and header_to_save is not header_object : 
            for card in header_object:
                if card not in header_to_save:
                    try:
                        header_to_save.set(card, header_object[card], header_object.comments[card])
                    except Exception as e_hdr:
                        logger_fits.warning(f"Could not set header card {card} from additional header: {e_hdr}")

    elif isinstance(data_object, np.ndarray): 
        data_to_save = data_object
        header_to_save = header_object.copy() if header_object is not None else fits.Header()
    else:
        logger_fits.error(f"Unsupported data type for saving: {type(data_object)}")
        return None

    if header_to_save is None:
        logger_fits.warning("No header provided for saving FITS. Creating a minimal header.")
        header_to_save = fits.Header()
    
    header_to_save['HISTORY'] = f'Generated by Gradio Astro App on {timestamp_str}'
    header_to_save['FILENAME'] = filename 

    try:
        fits.writeto(filepath, data_to_save.astype(np.float32), header=header_to_save, overwrite=True, output_verify='fix')
        logger_fits.info(f"Successfully saved FITS file: {filepath}")
        return filepath
    except Exception as e:
        logger_fits.error(f"Error saving FITS file {filepath}: {e}", exc_info=True)
        raise RuntimeError(f"FITS 파일 저장 중 오류 ({filename}): {str(e)[:100]}")

def create_preview_image(fits_data_obj, stretch_type='asinh', a_param=0.1):
    logger_fits.info(f"미리보기 이미지 생성 시도 (스트레칭: {stretch_type})")
    
    data_array = None
    if hasattr(fits_data_obj, 'data'): 
        data_array = fits_data_obj.data
    elif isinstance(fits_data_obj, np.ndarray): 
        data_array = fits_data_obj
    
    if data_array is None or data_array.ndim != 2:
        logger_fits.warning("미리보기를 위한 데이터가 유효하지 않음 (None 또는 2D가 아님).")
        return None
        
    data_array = data_array.astype(np.float32) 

    try:
        if not np.all(np.isfinite(data_array)):
            logger_fits.debug("데이터에 NaN 또는 Inf 포함. 유효한 값으로 대체 시도.")
            min_valid = np.nanmin(data_array[np.isfinite(data_array)]) if np.any(np.isfinite(data_array)) else 0
            max_valid = np.nanmax(data_array[np.isfinite(data_array)]) if np.any(np.isfinite(data_array)) else 1
            data_array = np.nan_to_num(data_array, nan=min_valid, posinf=max_valid, neginf=min_valid)
        
        try:
            interval = ZScaleInterval(contrast=0.25)
            vmin, vmax = interval.get_limits(data_array)
        except IndexError: 
            logger_fits.warning("ZScaleInterval 실패. Min/Max로 대체.")
            vmin, vmax = np.min(data_array), np.max(data_array)

        if stretch_type == 'asinh': stretch = AsinhStretch(a=a_param)
        elif stretch_type == 'log': stretch = LogStretch()
        else: stretch = LinearStretch()

        if np.isclose(vmin, vmax) or vmax < vmin :
            logger_fits.warning(f"vmin ({vmin})과 vmax ({vmax})가 거의 같거나 순서가 잘못됨. 범위 조정 시도.")
            min_val_data, max_val_data = np.min(data_array), np.max(data_array)
            if np.all(np.isclose(data_array, data_array.flat[0])):
                scaled_data_arr_pil = np.full_like(data_array, 127, dtype=np.uint8) 
                pil_image = Image.fromarray(scaled_data_arr_pil, mode='L')
                logger_fits.debug("데이터가 완전히 균일. 중간 회색으로 미리보기 생성.")
                return pil_image
            else: 
                vmin = min_val_data - ( (max_val_data - min_val_data) * 0.01 + 1e-5 ) 
                vmax = max_val_data + ( (max_val_data - min_val_data) * 0.01 + 1e-5 )
                if vmax <= vmin: vmax = vmin + 1e-4 
                logger_fits.debug(f"조정된 vmin={vmin}, vmax={vmax}")
        
        norm = ImageNormalize(data_array, vmin=vmin, vmax=vmax, stretch=stretch, clip=True)
        scaled_data_norm = norm(data_array) 

        image_8bit = (np.clip(scaled_data_norm, 0, 1) * 255).astype(np.uint8)
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

def draw_photometry_results_on_image(base_pil_image, stars_info_list, 
                                     roi_coords=None, 
                                     circle_radius=10, 
                                     circle_color="yellow", 
                                     text_color="yellow", 
                                     font_size=12):
    """
    PIL 이미지 위에 별 위치에 원을 그리고 등급/ID 텍스트를 표시합니다.
    stars_info_list: 각 별의 정보를 담은 딕셔너리 리스트. 
                     각 딕셔너리는 'x', 'y', 'mag_display', 'id_text' 키를 가져야 함.
    roi_coords: (x_min, x_max, y_min, y_max) 튜플. 이 범위 내의 별만 그림.
    """
    if base_pil_image is None:
        logger_fits.warning("별 정보를 그릴 기본 이미지가 없습니다.")
        return None
    
    try:
        img_with_stars = base_pil_image.copy()
        if img_with_stars.mode == 'L': # 흑백이면 RGB로 변환해야 컬러로 그릴 수 있음
            img_with_stars = img_with_stars.convert('RGB')
        draw = ImageDraw.Draw(img_with_stars)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size) # 기본 폰트 시도
        except IOError:
            font = ImageFont.load_default() # 실패 시 Gradio 기본 폰트
            logger_fits.warning("Arial 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

        img_width, img_height = img_with_stars.size
        
        roi_x0, roi_x1, roi_y0, roi_y1 = -1, img_width+1, -1, img_height+1 # 기본값: 전체 영역
        if roi_coords:
            roi_x0 = int(roi_coords[0])
            roi_x1 = int(roi_coords[1])
            roi_y0 = int(roi_coords[2])
            roi_y1 = int(roi_coords[3])
            logger_fits.debug(f"ROI 적용하여 별 마킹: X({roi_x0}-{roi_x1}), Y({roi_y0}-{roi_y1})")

        drawn_star_count = 0
        for star_info in stars_info_list:
            x, y = star_info.get('x'), star_info.get('y')
            mag_display = star_info.get('mag_display', np.nan) # 표시할 등급
            id_text = star_info.get('id_text', '') # 표시할 ID

            if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                continue

            # ROI 필터링
            if roi_coords and not (roi_x0 <= x <= roi_x1 and roi_y0 <= y <= roi_y1):
                continue
            
            # 이미지 경계 내에 있는지 확인
            if not (0 <= x < img_width and 0 <= y < img_height):
                continue

            # 원 그리기 (중심점 기준)
            bbox = [
                x - circle_radius, y - circle_radius,
                x + circle_radius, y + circle_radius
            ]
            draw.ellipse(bbox, outline=circle_color, width=1)
            
            # 텍스트 표시
            text_to_show = ""
            if np.isfinite(mag_display):
                text_to_show += f"{mag_display:.2f}"
            if id_text and id_text not in ["N/A", "WCS 없음", "SIMBAD 오류", "좌표 없음"]:
                if text_to_show: text_to_show += f" ({id_text})"
                else: text_to_show = id_text
            
            if text_to_show:
                # 텍스트 위치는 원의 약간 오른쪽 아래로 조정
                text_position = (x + circle_radius + 2, y + circle_radius + 2)
                draw.text(text_position, text_to_show, fill=text_color, font=font)
            drawn_star_count +=1
        
        logger_fits.info(f"{drawn_star_count}개의 별을 이미지에 마킹했습니다.")
        return img_with_stars

    except Exception as e:
        logger_fits.error(f"별 정보 시각화 중 오류: {e}", exc_info=True)
        return base_pil_image # 오류 시 원본 이미지 반환
