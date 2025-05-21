# ==============================================================================
# File: utils_fits.py
# Description: FITS 파일 처리 및 이미지 미리보기 관련 유틸리티 함수 모음.
# 이 모듈은 FITS 파일 로딩, 저장, 헤더 키워드 추출,
# 이미지 데이터로부터 미리보기용 PIL 이미지 생성,
# ROI(관심 영역) 및 측광 결과 시각화 기능을 제공합니다.
# ==============================================================================
import os
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch, LogStretch, LinearStretch
from PIL import Image, ImageDraw, ImageFont 
import logging
from astropy.nddata import CCDData  # 필요 시 import


# 로거 설정
logger_fits = logging.getLogger(__name__) 

def get_fits_keyword(header, keys, default_value=None, data_type=str, quiet=False):
    """
    FITS 헤더에서 여러 가능한 키워드 중 하나를 찾아 값을 반환합니다.

    Args:
        header (astropy.io.fits.Header): 검색할 FITS 헤더 객체.
        keys (list): 검색할 키워드 문자열 리스트. 리스트 순서대로 우선순위를 가집니다.
        default_value (any, optional): 키워드를 찾지 못했을 때 반환할 기본값. Defaults to None.
        data_type (type, optional): 반환할 값의 데이터 타입 (str, float, int). Defaults to str.
        quiet (bool, optional): 키워드를 찾지 못했을 때 경고 로그를 출력할지 여부. Defaults to False.

    Returns:
        any: 찾은 키워드의 값 또는 default_value. 타입은 data_type에 따라 변환됩니다.
    """
    if header is None:
        if not quiet: logger_fits.warning("헤더가 None이므로 키워드를 찾을 수 없습니다.")
        return default_value
    for key in keys:
        value = header.get(key)
        if value is not None:
            try:
                if data_type is float: return float(value)
                if data_type is int: return int(value)
                return str(value).strip() # 문자열의 경우 양쪽 공백 제거
            except ValueError:
                if not quiet: logger_fits.warning(f"키워드 '{key}'의 값 '{value}'을(를) {data_type}으로 변환 불가. 다음 키워드 시도.")
                continue # 다음 키워드로 넘어감
    if not quiet and default_value is None: 
        logger_fits.debug(f"키워드 목록 {keys}에 해당하는 값을 헤더에서 찾지 못했습니다.")
    return default_value

def load_fits_data_and_headers_from_file_objs(file_objects, object_type="프레임"):
    """
    Gradio 파일 객체 리스트에서 각 FITS 파일의 데이터와 헤더를 로드합니다.
    주로 파일 경로 리스트를 생성한 후 개별적으로 또는 작은 그룹으로 파일을 로드하는 데 사용됩니다.

    Args:
        file_objects (list): Gradio의 파일 업로드 컴포넌트에서 반환된 파일 객체 리스트.
        object_type (str, optional): 로깅 메시지에 사용될 파일의 종류 (예: "BIAS", "LIGHT"). Defaults to "프레임".

    Returns:
        list: 각 요소가 (data, header) 튜플인 리스트. 파일 로드 실패 시 해당 요소는 (None, None)이 됩니다.
    """
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
    """
    지정된 경로에서 단일 FITS 파일을 로드하여 데이터와 헤더를 반환합니다.

    Args:
        file_path (str): 로드할 FITS 파일의 경로.
        description (str, optional): 로깅 메시지에 사용될 파일 설명. Defaults to "파일".

    Returns:
        tuple: (data, header) 튜플. 로드 실패 시 (None, None) 반환.
               data는 NumPy 배열 (float32), header는 astropy.io.fits.Header 객체.
    """
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
    """
    NumPy 배열 또는 CCDData 객체를 FITS 파일로 저장합니다.

    Args:
        data_object (np.ndarray or CCDData): 저장할 이미지 데이터.
        header_object (astropy.io.fits.Header): 저장할 헤더. CCDData 객체인 경우 data_object.header가 우선 사용됨.
        base_filename (str): 기본 파일 이름 (타임스탬프 및 확장자 제외).
        output_dir (str): 파일이 저장될 디렉토리 경로.
        timestamp_str (str): 파일명에 추가될 타임스탬프 문자열.

    Returns:
        str or None: 저장된 파일의 전체 경로, 실패 시 None.
    """
    if data_object is None:
        logger_fits.warning(f"No data provided to save for {base_filename}.")
        return None
    
    filename = f"{base_filename}_{timestamp_str}.fits"
    filepath = os.path.join(output_dir, filename)
    
    logger_fits.info(f"Attempting to save FITS file to: {filepath}")

    data_to_save = None
    header_to_save = None

    if hasattr(data_object, 'data') and hasattr(data_object, 'header'): 
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
    """
    2D FITS 데이터 (NumPy 배열 또는 CCDData 객체)로부터 미리보기용 PIL 이미지를 생성합니다.
    memoryview 객체도 처리할 수 있도록 수정되었습니다.
    """
    logger_fits.info(f"미리보기 이미지 생성 시도 (스트레칭: {stretch_type})")
        # ---------- 입력 → NumPy 2D 배열로 변환 ----------
    if isinstance(fits_data_obj, CCDData):          # ① CCDData
        data_array = fits_data_obj.data
    elif isinstance(fits_data_obj, np.ndarray):     # ② 이미 ndarray
        data_array = fits_data_obj
    elif isinstance(fits_data_obj, memoryview):     # ③ memoryview
        data_array = np.asarray(fits_data_obj)
    elif hasattr(fits_data_obj, "data"):            # ④ 기타 객체의 .data
        tmp = fits_data_obj.data
        data_array = np.asarray(tmp)                # memoryview일 수도 있으니 확실히 변환
    else:
        logger_fits.warning("지원되지 않는 데이터 타입입니다.")
        return None
    
    if data_array is None or data_array.ndim != 2:
        logger_fits.warning("미리보기를 위한 데이터가 유효하지 않음 (None 또는 2D가 아님).")
        return None
        
    # data_array가 None이 아니고, NumPy 배열로 변환된 후 astype 호출
    if data_array is not None: # 추가된 None 체크
        data_array = data_array.astype(np.float32) # NumPy 배열로 변환 후 astype 호출
    else: # 만약 data_array가 여전히 None이면 (위의 조건들에서 할당되지 않았다면)
        logger_fits.warning("미리보기 데이터 배열을 가져올 수 없습니다.")
        return None

    # ... (이하 기존 로직 동일) ...
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
    """
    주어진 PIL 이미지 위에 ROI(관심 영역) 사각형을 그립니다.

    Args:
        base_pil_image (PIL.Image.Image): 바탕이 될 PIL 이미지.
        roi_x_min (int): ROI의 x 최소 좌표.
        roi_x_max (int): ROI의 x 최대 좌표.
        roi_y_min (int): ROI의 y 최소 좌표.
        roi_y_max (int): ROI의 y 최대 좌표.

    Returns:
        PIL.Image.Image or None: ROI가 그려진 PIL 이미지, 실패 시 원본 이미지 또는 None.
    """
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
                                     font_size=10):
    """
    PIL 이미지 위에 별 위치에 원을 그리고 등급/ID 텍스트를 표시합니다.

    Args:
        base_pil_image (PIL.Image.Image): 바탕이 될 PIL 이미지.
        stars_info_list (list): 각 별의 정보를 담은 딕셔너리 리스트. 
                                각 딕셔너리는 'x', 'y', 'mag_display', 'id_text' 키를 가져야 함.
        roi_coords (tuple, optional): (x_min, x_max, y_min, y_max) 튜플. 
                                     이 범위 내의 별들만 그림. Defaults to None (전체 영역).
        circle_radius (int, optional): 별 위치에 그릴 원의 반지름. Defaults to 10.
        circle_color (str, optional): 원의 색상. Defaults to "yellow".
        text_color (str, optional): 텍스트 색상. Defaults to "yellow".
        font_size (int, optional): 텍스트 폰트 크기. Defaults to 10.

    Returns:
        PIL.Image.Image or None: 별 정보가 그려진 PIL 이미지, 실패 시 원본 이미지 또는 None.
    """
    if base_pil_image is None:
        logger_fits.warning("별 정보를 그릴 기본 이미지가 없습니다.")
        return None
    
    try:
        img_with_stars = base_pil_image.copy()
        if img_with_stars.mode == 'L': 
            img_with_stars = img_with_stars.convert('RGB')
        draw = ImageDraw.Draw(img_with_stars)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size) 
        except IOError:
            font = ImageFont.load_default() 
            logger_fits.warning("Arial 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

        img_width, img_height = img_with_stars.size
        
        roi_x0, roi_x1, roi_y0, roi_y1 = 0, img_width, 0, img_height 
        if roi_coords and len(roi_coords) == 4:
            roi_x0 = int(roi_coords[0])
            roi_x1 = int(roi_coords[1])
            roi_y0 = int(roi_coords[2])
            roi_y1 = int(roi_coords[3])
            logger_fits.debug(f"ROI 적용하여 별 마킹: X({roi_x0}-{roi_x1}), Y({roi_y0}-{roi_y1})")

        drawn_star_count = 0
        for star_info in stars_info_list:
            x, y = star_info.get('x'), star_info.get('y')
            mag_display = star_info.get('mag_display', np.nan) 
            id_text = star_info.get('id_text', '') 

            if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                continue

            if roi_coords and not (roi_x0 <= x <= roi_x1 and roi_y0 <= y <= roi_y1):
                continue
            
            if not (0 <= x < img_width and 0 <= y < img_height):
                continue

            bbox = [x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius]
            draw.ellipse(bbox, outline=circle_color, width=1)
            
            text_to_show = ""
            if np.isfinite(mag_display): text_to_show += f"{mag_display:.2f}"
            if id_text and id_text not in ["N/A", "WCS 없음", "SIMBAD 오류", "좌표 없음"]:
                if text_to_show: text_to_show += f" ({id_text.split('(')[0].strip()})" 
                else: text_to_show = id_text.split('(')[0].strip()
            
            if text_to_show:
                text_position = (x + circle_radius + 2, y - font_size / 2) 
                draw.text(text_position, text_to_show, fill=text_color, font=font)
            drawn_star_count +=1
        
        logger_fits.info(f"{drawn_star_count}개의 별을 이미지에 마킹했습니다.")
        return img_with_stars

    except Exception as e:
        logger_fits.error(f"별 정보 시각화 중 오류: {e}", exc_info=True)
        return base_pil_image 

