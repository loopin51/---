# ==============================================================================
# File: ui_handlers.py
# Description: Gradio UI event handler functions.
# ==============================================================================
import os
import gradio as gr
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt 
import csv 

from utils.fits import load_fits_from_gradio_files, load_single_fits_from_path, save_fits_image, create_preview_image, draw_roi_on_pil_image
from utils.calibration import create_master_bias_from_data, create_master_dark_from_data, create_master_flat_from_data
from utils.photometry import detect_stars_extinction, find_brightest_star_extinction, calculate_flux_extinction, detect_stars_dao, perform_aperture_photometry_on_detections
from utils.astro import (
    calculate_altitude_extinction, calculate_airmass_extinction, 
    calculate_instrumental_magnitude, perform_linear_regression_extinction,
    convert_pixel_to_wcs, calculate_standard_magnitude, query_simbad_for_object,
    match_stars_by_coords
)
from astropy.io import fits 
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from PIL import Image 
from astropy.table import Table 


logger_ui = logging.getLogger(__name__)

def handle_tab1_master_frame_creation(bias_file_objs, dark_file_objs, flat_file_objs_all, temp_dir):
    status_messages = []
    ui_bias_path, ui_dark_path = None, None
    ui_flat_b_path, ui_flat_v_path, ui_flat_generic_path = None, None, None
    state_bias_path, state_dark_path_corrected = None, None
    state_flat_b_path_corrected, state_flat_v_path_corrected, state_flat_generic_path_corrected = None, None, None
    master_bias_data, bias_header = None, None
    master_dark_corrected_data, dark_header = None, None 
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if bias_file_objs:
        try:
            status_messages.append(f"BIAS: {len(bias_file_objs)}개 파일 처리 시작...")
            bias_images_stack, bias_header_loaded, _ = load_fits_from_gradio_files(bias_file_objs, "BIAS")
            if bias_header_loaded: bias_header = bias_header_loaded 
            master_bias_data = create_master_bias_from_data(bias_images_stack)
            saved_path = save_fits_image(master_bias_data, bias_header, "master_bias", temp_dir, current_timestamp_str)
            if saved_path: ui_bias_path = state_bias_path = saved_path; status_messages.append(f"BIAS: 생성 완료: {os.path.basename(ui_bias_path)}")
            else: status_messages.append("BIAS: 생성 실패 (저장 오류 또는 데이터 없음).")
        except Exception as e: logger_ui.error("BIAS 처리 중 오류", exc_info=True); status_messages.append(f"BIAS 처리 오류: {str(e)}"); master_bias_data = None
    else: status_messages.append("BIAS: 업로드된 파일 없음.")

    if dark_file_objs:
        try:
            status_messages.append(f"DARK: {len(dark_file_objs)}개 파일 처리 시작...")
            dark_images_stack, dark_header_loaded, _ = load_fits_from_gradio_files(dark_file_objs, "DARK")
            if dark_header_loaded: dark_header = dark_header_loaded
            master_dark_corrected_data = create_master_dark_from_data(dark_images_stack, master_bias_data) 
            saved_path = save_fits_image(master_dark_corrected_data, dark_header, "master_dark_corrected", temp_dir, current_timestamp_str)
            if saved_path: ui_dark_path = state_dark_path_corrected = saved_path; status_messages.append(f"DARK (Corrected): 생성 완료: {os.path.basename(ui_dark_path)}")
            else: status_messages.append("DARK (Corrected): 생성 실패.")
        except Exception as e: logger_ui.error("DARK 처리 중 오류", exc_info=True); status_messages.append(f"DARK 처리 오류: {str(e)}"); master_dark_corrected_data = None
    else: status_messages.append("DARK: 업로드된 파일 없음.")

    if flat_file_objs_all:
        status_messages.append(f"FLAT: 총 {len(flat_file_objs_all)}개 파일 처리 시작...")
        flats_b_objs, flats_v_objs, flats_generic_objs = [], [], []
        
        for idx, flat_obj in enumerate(flat_file_objs_all): 
            if flat_obj and flat_obj.name:
                try:
                    _, header_temp = load_single_fits_from_path(flat_obj.name, "FLAT (header check)")
                    if header_temp:
                        filter_val = header_temp.get('FILTER', 'UNKNOWN').strip().upper()
                        if filter_val == 'B': flats_b_objs.append(flat_obj)
                        elif filter_val == 'V': flats_v_objs.append(flat_obj)
                        else: flats_generic_objs.append(flat_obj); status_messages.append(f"FLAT 파일 '{os.path.basename(flat_obj.name)}' 필터 '{filter_val}' -> Generic 분류.")
                    else: status_messages.append(f"FLAT 파일 '{os.path.basename(flat_obj.name)}' 헤더 읽기 불가 -> Generic 분류."); flats_generic_objs.append(flat_obj)
                except Exception as e_flat_head:
                    status_messages.append(f"FLAT 파일 '{os.path.basename(flat_obj.name)}' 헤더 읽기 오류 ({e_flat_head}) -> Generic 분류."); flats_generic_objs.append(flat_obj)
            else: status_messages.append("유효하지 않은 FLAT 파일 객체 발견.")

        for filter_char, flat_objs_list_current in [('B', flats_b_objs), ('V', flats_v_objs), ('Generic', flats_generic_objs)]:
            if flat_objs_list_current: 
                try:
                    status_messages.append(f"Master FLAT ({filter_char}): {len(flat_objs_list_current)}개 파일로 생성 시작...")
                    flat_images_stack, first_flat_header, _ = load_fits_from_gradio_files(flat_objs_list_current, f"FLAT ({filter_char})")
                    if flat_images_stack is None: status_messages.append(f"Master FLAT ({filter_char}): 이미지 스택 생성 실패."); continue
                    master_flat_data = create_master_flat_from_data(flat_images_stack, master_bias_data, master_dark_corrected_data)
                    save_header_flat = first_flat_header if first_flat_header else (bias_header if bias_header else fits.Header())
                    saved_path = save_fits_image(master_flat_data, save_header_flat, f"master_flat_{filter_char.lower()}_corrected", temp_dir, current_timestamp_str)
                    if saved_path:
                        status_messages.append(f"Master FLAT ({filter_char}, Corrected): 생성 완료: {os.path.basename(saved_path)}")
                        if filter_char == 'B': ui_flat_b_path = state_flat_b_path_corrected = saved_path
                        elif filter_char == 'V': ui_flat_v_path = state_flat_v_path_corrected = saved_path
                        else: ui_flat_generic_path = state_flat_generic_path_corrected = saved_path
                    else: status_messages.append(f"Master FLAT ({filter_char}, Corrected): 생성 실패.")
                except Exception as e_mf: logger_ui.error(f"Master FLAT ({filter_char}) 처리 중 오류", exc_info=True); status_messages.append(f"Master FLAT ({filter_char}) 처리 오류: {str(e_mf)}")
            else: status_messages.append(f"Master FLAT ({filter_char}): 업로드된 해당 필터 파일 없음.")
    else: status_messages.append("FLAT: 업로드된 파일 없음.")
        
    final_status = "\n".join(status_messages)
    logger_ui.info("Tab 1: Master frame generation finished.")
    return ui_bias_path, ui_dark_path, ui_flat_b_path, ui_flat_v_path, ui_flat_generic_path, \
           state_bias_path, state_dark_path_corrected, \
           state_flat_b_path_corrected, state_flat_v_path_corrected, state_flat_generic_path_corrected, \
           final_status


def handle_tab2_light_frame_calibration(
    light_file_objs_list, 
    uploaded_mf_b_corr_obj, uploaded_mf_v_corr_obj, 
    state_mb_p, state_md_p_corrected, 
    state_mf_b_p_corr, state_mf_v_p_corr, state_mf_gen_p_corr, 
    preview_stretch_type, preview_asinh_a,
    temp_dir):
    status_messages = []
    calibrated_light_file_paths_for_ui = []
    output_preview_pil_image = None 
    mb_data, md_corr_data = None, None 
    used_masters_info = {"bias": "미사용", "dark": "미사용"} 
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    try: 
        if state_mb_p and os.path.exists(state_mb_p):
            mb_data, _ = load_single_fits_from_path(state_mb_p, "저장된 Master BIAS (탭1)")
            used_masters_info["bias"] = f"탭1 ({os.path.basename(state_mb_p)})" if mb_data is not None else "탭1 로드 실패"
        else: status_messages.append("경고: 탭1 Master BIAS 없음. BIAS 보정 생략 가능.")
        
        if state_md_p_corrected and os.path.exists(state_md_p_corrected): 
            md_corr_data, _ = load_single_fits_from_path(state_md_p_corrected, "저장된 Master DARK (탭1-Corrected)")
            used_masters_info["dark"] = f"탭1 ({os.path.basename(state_md_p_corrected)})" if md_corr_data is not None else "탭1 로드 실패"
        else: status_messages.append("경고: 탭1 Master DARK 없음. DARK 보정 생략 가능.")
        
        status_messages.extend([f"사용될 Master BIAS: {used_masters_info['bias']}", f"사용될 Master DARK: {used_masters_info['dark']}"])
    except Exception as e:
        logger_ui.error("탭2 BIAS/DARK 마스터 로드 중 오류", exc_info=True); status_messages.append(f"BIAS/DARK 마스터 로드 실패: {str(e)}"); return [], None, "\n".join(status_messages)

    if not light_file_objs_list: status_messages.append("보정할 LIGHT 프레임 없음."); return [], None, "\n".join(status_messages)

    status_messages.append(f"{len(light_file_objs_list)}개의 LIGHT 프레임 보정을 시작합니다...")
    first_calibrated_image_data_for_preview = None

    for i, light_file_obj in enumerate(light_file_objs_list):
        light_filename = "알 수 없는 파일"
        flat_to_use_for_this_light = None 
        flat_source_for_this_frame = "미사용"
        try:
            if light_file_obj is None or not hasattr(light_file_obj, 'name') or light_file_obj.name is None: status_messages.append(f"LIGHT 파일 {i+1} 유효X."); continue
            light_filename = os.path.basename(light_file_obj.name)
            status_messages.append(f"--- {light_filename} 보정 중 ({i+1}/{len(light_file_objs_list)}) ---")
            light_data, light_header = load_single_fits_from_path(light_file_obj.name, f"LIGHT ({light_filename})")
            if light_data is None or light_header is None: status_messages.append(f"{light_filename} 로드 실패."); continue
            
            calibrated_light = light_data.copy().astype(np.float32)
            current_light_filter = light_header.get('FILTER', 'UNKNOWN').strip().upper()

            if current_light_filter == 'B':
                if uploaded_mf_b_corr_obj and uploaded_mf_b_corr_obj.name:
                    flat_to_use_for_this_light, _ = load_single_fits_from_path(uploaded_mf_b_corr_obj.name, "탭2 업로드 Master FLAT B (Corrected)")
                    if flat_to_use_for_this_light is not None: flat_source_for_this_frame = f"탭2 업로드 B ({os.path.basename(uploaded_mf_b_corr_obj.name)})"
                elif state_mf_b_p_corr and os.path.exists(state_mf_b_p_corr):
                    flat_to_use_for_this_light, _ = load_single_fits_from_path(state_mf_b_p_corr, "탭1 Master FLAT B")
                    if flat_to_use_for_this_light is not None: flat_source_for_this_frame = f"탭1 B ({os.path.basename(state_mf_b_p_corr)})"
            elif current_light_filter == 'V':
                if uploaded_mf_v_corr_obj and uploaded_mf_v_corr_obj.name:
                    flat_to_use_for_this_light, _ = load_single_fits_from_path(uploaded_mf_v_corr_obj.name, "탭2 업로드 Master FLAT V (Corrected)")
                    if flat_to_use_for_this_light is not None: flat_source_for_this_frame = f"탭2 업로드 V ({os.path.basename(uploaded_mf_v_corr_obj.name)})"
                elif state_mf_v_p_corr and os.path.exists(state_mf_v_p_corr):
                    flat_to_use_for_this_light, _ = load_single_fits_from_path(state_mf_v_p_corr, "탭1 Master FLAT V")
                    if flat_to_use_for_this_light is not None: flat_source_for_this_frame = f"탭1 V ({os.path.basename(state_mf_v_p_corr)})"
            
            if flat_to_use_for_this_light is None and state_mf_gen_p_corr and os.path.exists(state_mf_gen_p_corr): 
                flat_to_use_for_this_light, _ = load_single_fits_from_path(state_mf_gen_p_corr, "탭1 Master FLAT Generic")
                if flat_to_use_for_this_light is not None: flat_source_for_this_frame = f"탭1 Generic ({os.path.basename(state_mf_gen_p_corr)})"
            
            status_messages.append(f"{light_filename}: 사용된 Master FLAT ({current_light_filter}용) -> {flat_source_for_this_frame}")

            if mb_data is not None and mb_data.shape == calibrated_light.shape: calibrated_light -= mb_data; status_messages.append(f"{light_filename}: BIAS 적용.")
            if md_corr_data is not None and md_corr_data.shape == calibrated_light.shape: calibrated_light -= md_corr_data; status_messages.append(f"{light_filename}: DARK 적용.")
            
            if flat_to_use_for_this_light is not None and flat_to_use_for_this_light.shape == calibrated_light.shape:
                safe_flat = np.where(flat_to_use_for_this_light < 0.01, 1.0, flat_to_use_for_this_light)
                if not np.all(np.isclose(safe_flat, 0)): calibrated_light /= safe_flat; status_messages.append(f"{light_filename}: FLAT ({flat_source_for_this_frame}) 적용.")
                else: status_messages.append(f"{light_filename}: FLAT 데이터 0, 보정 불가.")
            else: status_messages.append(f"{light_filename}: 사용 가능한 FLAT 없음 ({current_light_filter}). FLAT 보정 생략.")

            if first_calibrated_image_data_for_preview is None and calibrated_light is not None:
                first_calibrated_image_data_for_preview = calibrated_light.copy(); status_messages.append(f"{light_filename}: 미리보기용 선택.")
            
            light_header['HISTORY'] = f'Calibrated App v0.10 (B:{used_masters_info["bias"]!="미사용"},D:{used_masters_info["dark"]!="미사용"},F:{flat_source_for_this_frame!="미사용"})'
            saved_path = save_fits_image(calibrated_light, light_header, f"calibrated_{os.path.splitext(light_filename)[0]}", temp_dir, current_timestamp_str)
            if saved_path: calibrated_light_file_paths_for_ui.append(saved_path); status_messages.append(f"{light_filename}: 보정 완료 및 저장: {os.path.basename(saved_path)}")
            else: status_messages.append(f"{light_filename}: 저장 실패.")
        except Exception as e: logger_ui.error(f"LIGHT ({light_filename}) 보정 오류", exc_info=True); status_messages.append(f"{light_filename} 보정 오류: {str(e)}")

    if first_calibrated_image_data_for_preview is not None:
        try:
            output_preview_pil_image = create_preview_image(first_calibrated_image_data_for_preview, stretch_type=preview_stretch_type, a_param=preview_asinh_a)
            status_messages.append("미리보기 생성 완료." if output_preview_pil_image else "미리보기 생성 실패.")
        except Exception as e: logger_ui.error("미리보기 생성 함수 호출 오류", exc_info=True); status_messages.append(f"미리보기 생성 오류: {str(e)}")
            
    if not calibrated_light_file_paths_for_ui: status_messages.append("성공적으로 보정된 LIGHT 프레임 없음.")
    final_status = "\n".join(status_messages)
    logger_ui.info("Tab 2: Light frame calibration finished."); return calibrated_light_file_paths_for_ui, output_preview_pil_image, final_status


def handle_tab3_extinction_analysis(
    light_file_objs, 
    uploaded_mb_path_obj, uploaded_md_raw_path_obj,
    uploaded_mf_b_raw_path_obj, uploaded_mf_v_raw_path_obj,
    state_mb_p, state_md_p_corrected, 
    state_mf_b_p_corr, state_mf_v_p_corr, state_mf_gen_p_corr, 
    star_detection_thresh_factor,
    temp_dir):
    status_log = []
    all_frame_results_for_df = [] 
    plot_image_fig = None 
    summary_text = "분석 결과가 없습니다."
    
    mb_data, md_corrected_data_final = None, None 
    mf_b_corrected_data_final, mf_v_corrected_data_final = None, None
    try:
        status_log.append("--- Master Bias 준비 중 (탭3) ---")
        if uploaded_mb_path_obj and uploaded_mb_path_obj.name:
            mb_data, _ = load_single_fits_from_path(uploaded_mb_path_obj.name, "업로드된 Master BIAS (탭3)")
        elif state_mb_p and os.path.exists(state_mb_p): mb_data, _ = load_single_fits_from_path(state_mb_p, "탭1 Master BIAS")
        status_log.append(f"Master BIAS: {'사용' if mb_data is not None else '사용 안함/로드 실패'}")

        status_log.append("--- Master Dark (Corrected) 준비 중 (탭3) ---")
        if uploaded_md_raw_path_obj and uploaded_md_raw_path_obj.name:
            raw_md_data, _ = load_single_fits_from_path(uploaded_md_raw_path_obj.name, "업로드된 Raw Master DARK (탭3)")
            if raw_md_data is not None:
                if mb_data is not None and raw_md_data.shape == mb_data.shape: md_corrected_data_final = raw_md_data - mb_data; status_log.append("Raw DARK에서 BIAS 차감.")
                else: md_corrected_data_final = raw_md_data; status_log.append("경고: Raw DARK에 BIAS 차감 못함.")
            else: status_log.append("업로드된 Raw Master DARK 로드 실패.")
        elif state_md_p_corrected and os.path.exists(state_md_p_corrected): 
            md_corrected_data_final, _ = load_single_fits_from_path(state_md_p_corrected, "탭1 Master DARK (Corrected)")
            status_log.append("탭1 Master DARK (Corrected) 사용." if md_corrected_data_final is not None else "탭1 Master DARK 로드 실패.")
        else: status_log.append("사용 가능 Master DARK 없음.")

        status_log.append("--- Master Flat B (Corrected) 준비 중 (탭3) ---")
        if uploaded_mf_b_raw_path_obj and uploaded_mf_b_raw_path_obj.name:
            raw_mf_b, _ = load_single_fits_from_path(uploaded_mf_b_raw_path_obj.name, "업로드된 Raw Master FLAT B (탭3)")
            if raw_mf_b is not None:
                temp_mf_b = raw_mf_b.copy()
                if mb_data is not None and temp_mf_b.shape == mb_data.shape: temp_mf_b -= mb_data
                if md_corrected_data_final is not None and temp_mf_b.shape == md_corrected_data_final.shape: temp_mf_b -= md_corrected_data_final
                mean_b = np.mean(temp_mf_b); mf_b_corrected_data_final = temp_mf_b / mean_b if mean_b > 1e-9 else temp_mf_b
                status_log.append("업로드된 Raw Master FLAT B 처리 완료 (정규화 시도).")
            else: status_log.append("업로드된 Raw Master FLAT B 로드 실패.")
        elif state_mf_b_p_corr and os.path.exists(state_mf_b_p_corr): 
            mf_b_corrected_data_final, _ = load_single_fits_from_path(state_mf_b_p_corr, "탭1 Master FLAT B (Corrected)")
            status_log.append("탭1 Master FLAT B 사용." if mf_b_corrected_data_final is not None else "탭1 Master FLAT B 로드 실패.")
        elif state_mf_gen_p_corr and os.path.exists(state_mf_gen_p_corr): 
            mf_b_corrected_data_final, _ = load_single_fits_from_path(state_mf_gen_p_corr, "탭1 Master FLAT Generic (B용)")
            status_log.append("탭1 Generic Master FLAT을 B필터용으로 사용 (주의)." if mf_b_corrected_data_final is not None else "탭1 Generic Master FLAT 로드 실패 (B용).")
        else: status_log.append("사용 가능 Master FLAT B 없음.")
        
        status_log.append("--- Master Flat V (Corrected) 준비 중 (탭3) ---")
        if uploaded_mf_v_raw_path_obj and uploaded_mf_v_raw_path_obj.name:
            raw_mf_v, _ = load_single_fits_from_path(uploaded_mf_v_raw_path_obj.name, "업로드된 Raw Master FLAT V (탭3)")
            if raw_mf_v is not None:
                temp_mf_v = raw_mf_v.copy()
                if mb_data is not None and temp_mf_v.shape == mb_data.shape: temp_mf_v -= mb_data
                if md_corrected_data_final is not None and temp_mf_v.shape == md_corrected_data_final.shape: temp_mf_v -= md_corrected_data_final
                mean_v = np.mean(temp_mf_v); mf_v_corrected_data_final = temp_mf_v / mean_v if mean_v > 1e-9 else temp_mf_v
                status_log.append("업로드된 Raw Master FLAT V 처리 완료 (정규화 시도).")
            else: status_log.append("업로드된 Raw Master FLAT V 로드 실패.")
        elif state_mf_v_p_corr and os.path.exists(state_mf_v_p_corr): 
            mf_v_corrected_data_final, _ = load_single_fits_from_path(state_mf_v_p_corr, "탭1 Master FLAT V (Corrected)")
            status_log.append("탭1 Master FLAT V 사용." if mf_v_corrected_data_final is not None else "탭1 Master FLAT V 로드 실패.")
        elif state_mf_gen_p_corr and os.path.exists(state_mf_gen_p_corr): 
            mf_v_corrected_data_final, _ = load_single_fits_from_path(state_mf_gen_p_corr, "탭1 Master FLAT Generic (V용)")
            status_log.append("탭1 Generic Master FLAT을 V필터용으로 사용 (주의)." if mf_v_corrected_data_final is not None else "탭1 Generic Master FLAT 로드 실패 (V용).")
        else: status_log.append("사용 가능 Master FLAT V 없음.")

    except Exception as e:
        logger_ui.error("탭3 마스터 프레임 준비 중 치명적 오류", exc_info=True); status_log.append(f"마스터 프레임 준비 오류: {e}")
        df_headers_err = ["File", "Filter", "Airmass", "Altitude", "Inst. Mag.", "Flux", "Star X", "Star Y", "Ap. Radius", "Error"]
        return None, "오류 발생", (df_headers_err, [["오류"]*len(df_headers_err)]), "\n".join(status_log)

    if not light_file_objs:
        status_log.append("분석할 LIGHT 프레임 없음."); df_headers_no_light = ["File", "Filter", "Airmass", "Altitude", "Inst. Mag.", "Flux", "Star X", "Star Y", "Ap. Radius", "Error"]
        return None, "LIGHT 파일 없음", (df_headers_no_light, [["LIGHT 파일 없음"]*len(df_headers_no_light)]), "\n".join(status_log)

    status_log.append(f"--- {len(light_file_objs)}개 LIGHT 프레임 분석 시작 ---")
    light_files_with_headers = []
    for light_obj in light_file_objs:
        if light_obj and light_obj.name and os.path.exists(light_obj.name):
            try: _, header = load_single_fits_from_path(light_obj.name, "LIGHT header check"); light_files_with_headers.append({'path': light_obj.name, 'header': header})
            except Exception as e_head: status_log.append(f"경고: {os.path.basename(light_obj.name)} 헤더 읽기 오류 ({e_head}).")
        else: status_log.append(f"경고: 유효하지 않은 LIGHT 파일 객체.")

    processed_results_for_analysis = [] 
    for file_info in light_files_with_headers:
        file_path, header = file_info['path'], file_info['header']
        if not header: status_log.append(f"경고: {os.path.basename(file_path)} 헤더 정보 없음."); continue
        
        current_filter = header.get('FILTER', 'UNKNOWN').strip().upper()
        mf_to_use = mf_b_corrected_data_final if current_filter == 'B' else (mf_v_corrected_data_final if current_filter == 'V' else None)
        if mf_to_use is None: status_log.append(f"경고: {os.path.basename(file_path)} 필터 '{current_filter}'에 대한 Master Flat 없음. Flat 보정 생략.")

        current_result = {'file': os.path.basename(file_path), 'filter': current_filter, 'error_message': None}
        try:
            image_data, img_header = load_single_fits_from_path(file_path, f"LIGHT ({current_result['file']})")
            if image_data is None: raise ValueError("LIGHT 데이터 로드 실패")
            header_to_use = img_header if img_header else header 

            cal_img = image_data.astype(np.float32)
            if mb_data is not None and mb_data.shape == cal_img.shape: cal_img -= mb_data
            if md_corrected_data_final is not None and md_corrected_data_final.shape == cal_img.shape: cal_img -= md_corrected_data_final
            if mf_to_use is not None and mf_to_use.shape == cal_img.shape:
                safe_mf = np.where(mf_to_use < 0.01, 1.0, mf_to_use)
                if not np.all(np.isclose(safe_mf, 0)): cal_img /= safe_mf
            
            stars = detect_stars_extinction(cal_img, star_detection_thresh_factor) 
            brightest = find_brightest_star_extinction(stars)
            if brightest is None: raise ValueError("가장 밝은 별 탐지 실패 (Tab3)")
            
            flux, ap_rad, bg_tot = calculate_flux_extinction(cal_img, brightest) 
            if flux is None: raise ValueError("Flux 계산 실패 (Tab3)")
            current_result.update({'flux': flux, 'star_x': brightest['xcentroid'], 'star_y': brightest['ycentroid'], 'aperture_radius': ap_rad})
            
            inst_mag = calculate_instrumental_magnitude(flux)
            if inst_mag is None: raise ValueError("기기 등급 계산 실패")
            current_result['instrumental_magnitude'] = inst_mag
            
            alt = calculate_altitude_extinction(header_to_use) 
            airmass = calculate_airmass_extinction(header_to_use) 
            if airmass is None: raise ValueError("대기질량 계산 실패")
            current_result.update({'altitude': alt, 'airmass': airmass})
            
            status_log.append(f"처리 완료 ({current_result['file']}): F={current_filter}, AM={airmass:.3f}, Mag={inst_mag:.3f}")
            processed_results_for_analysis.append(current_result)

        except Exception as e_proc:
            logger_ui.error(f"파일 {current_result['file']} 처리 중 오류", exc_info=True)
            current_result['error_message'] = str(e_proc); status_log.append(f"오류 ({current_result['file']}): {str(e_proc)}")
            processed_results_for_analysis.append(current_result)

    results_b = [r for r in processed_results_for_analysis if r.get('filter') == 'B' and r.get('airmass') is not None and r.get('instrumental_magnitude') is not None and r.get('error_message') is None]
    results_v = [r for r in processed_results_for_analysis if r.get('filter') == 'V' and r.get('airmass') is not None and r.get('instrumental_magnitude') is not None and r.get('error_message') is None]
    summary_lines = []
    slope_b, intercept_b, r_sq_b, model_b = None, None, None, None
    slope_v, intercept_v, r_sq_v, model_v = None, None, None, None

    if len(results_b) >= 2:
        slope_b, intercept_b, r_sq_b, model_b = perform_linear_regression_extinction([r['airmass'] for r in results_b], [r['instrumental_magnitude'] for r in results_b])
        if slope_b is not None: summary_lines.append(f"B 필터: k_B={slope_b:.4f}, m0_B={intercept_b:.4f}, R²={r_sq_b:.4f} ({len(results_b)}개)")
        else: summary_lines.append(f"B 필터: 선형 회귀 실패 ({len(results_b)}개).")
    elif results_b: summary_lines.append(f"B 필터: 데이터 부족 ({len(results_b)}개)으로 회귀 불가.")
    else: summary_lines.append("B 필터: 유효 데이터 없음.")

    if len(results_v) >= 2:
        slope_v, intercept_v, r_sq_v, model_v = perform_linear_regression_extinction([r['airmass'] for r in results_v], [r['instrumental_magnitude'] for r in results_v])
        if slope_v is not None: summary_lines.append(f"V 필터: k_V={slope_v:.4f}, m0_V={intercept_v:.4f}, R²={r_sq_v:.4f} ({len(results_v)}개)")
        else: summary_lines.append(f"V 필터: 선형 회귀 실패 ({len(results_v)}개).")
    elif results_v: summary_lines.append(f"V 필터: 데이터 부족 ({len(results_v)}개)으로 회귀 불가.")
    else: summary_lines.append("V 필터: 유효 데이터 없음.")
    summary_text = "\n".join(summary_lines) if summary_lines else "분석 결과가 없습니다."

    try:
        plt.style.use('seaborn-v0_8-whitegrid') 
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_created = False
        if results_b and model_b:
            x_b_arr = np.array([r['airmass'] for r in results_b]); y_b_arr = np.array([r['instrumental_magnitude'] for r in results_b])
            ax.scatter(x_b_arr, y_b_arr, color='blue', label=f'B Data ({len(x_b_arr)})', alpha=0.7, edgecolor='k')
            if len(x_b_arr) > 0: 
                x_fit_b = np.array([np.min(x_b_arr), np.max(x_b_arr)]); y_fit_b = model_b.predict(x_fit_b.reshape(-1,1))
                ax.plot(x_fit_b, y_fit_b, color='dodgerblue', ls='--', label=f'B Fit (k={slope_b:.3f}, R²={r_sq_b:.3f})')
            plot_created = True
        if results_v and model_v:
            x_v_arr = np.array([r['airmass'] for r in results_v]); y_v_arr = np.array([r['instrumental_magnitude'] for r in results_v])
            ax.scatter(x_v_arr, y_v_arr, color='green', label=f'V Data ({len(x_v_arr)})', alpha=0.7, edgecolor='k')
            if len(x_v_arr) > 0:
                x_fit_v = np.array([np.min(x_v_arr), np.max(x_v_arr)]); y_fit_v = model_v.predict(x_fit_v.reshape(-1,1))
                ax.plot(x_fit_v, y_fit_v, color='forestgreen', ls='--', label=f'V Fit (k={slope_v:.3f}, R²={r_sq_v:.3f})')
            plot_created = True
        if plot_created:
            ax.set_xlabel('Airmass (X)'); ax.set_ylabel('Instrumental Magnitude (m_inst)'); ax.set_title('Atmospheric Extinction (m_inst = m0 + kX)')
            ax.invert_yaxis(); ax.legend(loc='best'); ax.grid(True, which='both', ls=':', lw=0.5); fig.tight_layout()
            plot_image_fig = fig; status_log.append("그래프 생성 완료.")
        else: status_log.append("그래프 생성 데이터 부족."); fig_empty, ax_empty = plt.subplots(); ax_empty.text(0.5,0.5,"No data to plot",ha='center',va='center'); plot_image_fig = fig_empty
        plt.close(plot_image_fig) 
    except Exception as e_plot:
        logger_ui.error("그래프 생성 중 오류", exc_info=True); status_log.append(f"그래프 생성 오류: {e_plot}")
        fig_err, ax_err = plt.subplots(); ax_err.text(0.5,0.5,f"Plotting error: {e_plot}",ha='center',va='center',color='red'); plot_image_fig = fig_err
        plt.close(plot_image_fig)

    df_headers = ["File", "Filter", "Airmass", "Altitude", "Inst. Mag.", "Flux", "Star X", "Star Y", "Ap. Radius", "Error"]
    for r_item in processed_results_for_analysis:
        all_frame_results_for_df.append([
            r_item.get('file', 'N/A'), r_item.get('filter', 'N/A'),
            f"{r_item.get('airmass'):.3f}" if r_item.get('airmass') is not None else 'N/A',
            f"{r_item.get('altitude'):.2f}" if r_item.get('altitude') is not None else 'N/A',
            f"{r_item.get('instrumental_magnitude'):.3f}" if r_item.get('instrumental_magnitude') is not None else 'N/A',
            f"{r_item.get('flux'):.2e}" if r_item.get('flux') is not None else 'N/A',
            f"{r_item.get('star_x'):.1f}" if r_item.get('star_x') is not None else 'N/A',
            f"{r_item.get('star_y'):.1f}" if r_item.get('star_y') is not None else 'N/A',
            f"{r_item.get('aperture_radius'):.1f}" if r_item.get('aperture_radius') is not None else 'N/A',
            r_item.get('error_message', '')
        ])
    
    final_log = "\n".join(status_log)
    logger_ui.info("대기소광계수 분석 완료.")
    return plot_image_fig, summary_text, (df_headers, all_frame_results_for_df) if all_frame_results_for_df else (df_headers, [["결과 없음"]*len(df_headers)]), final_log


def handle_tab4_detailed_photometry(
    light_b_file_objs, light_v_file_objs,
    std_star_b_file_obj, std_star_v_file_obj,
    std_b_mag_known_input, std_v_mag_known_input,
    tab4_uploaded_mb_obj, tab4_uploaded_md_raw_obj, 
    tab4_uploaded_mf_b_raw_obj, tab4_uploaded_mf_v_raw_obj,
    state_mb_p, state_md_p_corrected, 
    state_mf_b_p_corr, state_mf_v_p_corr, state_mf_gen_p_corr,
    k_b_input, m0_b_input_user, k_v_input, m0_v_input_user, 
    dao_fwhm_input, dao_thresh_nsigma_input, phot_aperture_radius_input, 
    roi_x_min, roi_x_max, roi_y_min, roi_y_max,
    simbad_query_radius_arcsec, 
    temp_dir):
    status_log = []
    all_stars_final_data_for_df = [] 
    csv_output_path = None
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger_ui.info("탭 4 상세 측광 분석 시작...")

    if not light_b_file_objs and not light_v_file_objs:
        status_log.append("오류: B 또는 V 필터 LIGHT 프레임을 하나 이상 업로드해야 합니다.")
        return (["Error Message"], [["LIGHT 프레임 없음"]]), None, "\n".join(status_log)
    try:
        k_b, m0_b_user_val = float(k_b_input), float(m0_b_input_user)
        k_v, m0_v_user_val = float(k_v_input), float(m0_v_input_user)
        fwhm, thresh_nsigma = float(dao_fwhm_input), float(dao_thresh_nsigma_input)
        ap_radius_phot = float(phot_aperture_radius_input)
        simbad_radius = float(simbad_query_radius_arcsec)
        roi_x0, roi_x1 = int(roi_x_min), int(roi_x_max)
        roi_y0, roi_y1 = int(roi_y_min), int(roi_y_max)
        use_roi = True
        if not (roi_x1 > roi_x0 and roi_y1 > roi_y0):
            status_log.append("경고: ROI 범위가 유효하지 않습니다. 전체 이미지를 사용합니다.")
            use_roi = False
    except ValueError:
        status_log.append("오류: 입력 파라미터(소광계수, 영점 등)는 숫자여야 합니다.")
        return (["Error Message"], [["입력 파라미터 오류"]]), None, "\n".join(status_log)

    final_mb_data, final_md_corr_data, final_mf_b_corr_data, final_mf_v_corr_data = None, None, None, None
    if tab4_uploaded_mb_obj and tab4_uploaded_mb_obj.name: final_mb_data, _ = load_single_fits_from_path(tab4_uploaded_mb_obj.name, "탭4 업로드 BIAS")
    elif state_mb_p and os.path.exists(state_mb_p): final_mb_data, _ = load_single_fits_from_path(state_mb_p, "탭1 BIAS")
    status_log.append(f"Master BIAS: {'사용' if final_mb_data is not None else '미사용/로드실패'}")

    if tab4_uploaded_md_raw_obj and tab4_uploaded_md_raw_obj.name:
        raw_md, _ = load_single_fits_from_path(tab4_uploaded_md_raw_obj.name, "탭4 업로드 Raw DARK")
        if raw_md is not None and final_mb_data is not None and raw_md.shape == final_mb_data.shape: final_md_corr_data = raw_md - final_mb_data
        elif raw_md is not None: final_md_corr_data = raw_md; status_log.append("경고: 탭4 Raw DARK에 BIAS 차감 못함.")
    elif state_md_p_corrected and os.path.exists(state_md_p_corrected): final_md_corr_data, _ = load_single_fits_from_path(state_md_p_corrected, "탭1 DARK(Corrected)")
    status_log.append(f"Master DARK (Corrected): {'사용' if final_md_corr_data is not None else '미사용/로드실패'}")

    for filt_char, uploaded_mf_raw_obj, state_mf_filt_p, state_mf_gen_p, mf_final_var_name_str in [
        ('B', tab4_uploaded_mf_b_raw_obj, state_mf_b_p_corr, state_mf_gen_p_corr, "final_mf_b_corr_data"),
        ('V', tab4_uploaded_mf_v_raw_obj, state_mf_v_p_corr, state_mf_gen_p_corr, "final_mf_v_corr_data")
    ]:
        temp_mf_data = None
        if uploaded_mf_raw_obj and uploaded_mf_raw_obj.name:
            raw_mf, _ = load_single_fits_from_path(uploaded_mf_raw_obj.name, f"탭4 업로드 Raw FLAT {filt_char}")
            if raw_mf is not None:
                temp_mf_data = raw_mf.copy()
                if final_mb_data is not None and temp_mf_data.shape == final_mb_data.shape: temp_mf_data -= final_mb_data
                if final_md_corr_data is not None and temp_mf_data.shape == final_md_corr_data.shape: temp_mf_data -= final_md_corr_data
                mean_val = np.mean(temp_mf_data)
                if mean_val > 1e-9: temp_mf_data /= mean_val
                else: status_log.append(f"경고: 탭4 업로드 Raw FLAT {filt_char} 정규화 실패.")
        elif state_mf_filt_p and os.path.exists(state_mf_filt_p):
            temp_mf_data, _ = load_single_fits_from_path(state_mf_filt_p, f"탭1 Master FLAT {filt_char}")
        elif state_mf_gen_p and os.path.exists(state_mf_gen_p):
            temp_mf_data, _ = load_single_fits_from_path(state_mf_gen_p, f"탭1 Master FLAT Generic ({filt_char}용)")
            if temp_mf_data is not None: status_log.append(f"탭1 Generic FLAT을 {filt_char}필터용으로 사용 (주의).")
        
        if mf_final_var_name_str == "final_mf_b_corr_data": final_mf_b_corr_data = temp_mf_data
        elif mf_final_var_name_str == "final_mf_v_corr_data": final_mf_v_corr_data = temp_mf_data
        status_log.append(f"Master FLAT {filt_char} (Corrected): {'사용' if temp_mf_data is not None else '미사용/로드실패'}")

    if final_mb_data is None or final_md_corr_data is None:
        status_log.append("오류: BIAS 또는 DARK 마스터 프레임이 준비되지 않아 처리를 중단합니다.")
        return (["Error Message"], [["필수 마스터 프레임(BIAS/DARK) 없음"]]), None, "\n".join(status_log)

    m0_eff_b, m0_eff_v = m0_b_user_val, m0_v_user_val
    status_log.append(f"초기 영점: m0_B={m0_eff_b:.3f}, m0_V={m0_eff_v:.3f} (사용자 입력 또는 기본값)")

    for std_filt_char, std_file_obj, std_mag_known_in, k_coeff_std_val, mf_corr_std_use, m0_eff_var_name in [
        ('B', std_star_b_file_obj, std_b_mag_known_input, k_b, final_mf_b_corr_data, 'm0_eff_b'),
        ('V', std_star_v_file_obj, std_v_mag_known_input, k_v, final_mf_v_corr_data, 'm0_eff_v')
    ]:
        if std_file_obj and hasattr(std_file_obj, 'name') and std_file_obj.name:
            status_log.append(f"--- {std_filt_char}필터 표준별 처리: {os.path.basename(std_file_obj.name)} ---")
            std_data, std_header = load_single_fits_from_path(std_file_obj.name, f"{std_filt_char} 표준별")
            if std_data is not None and std_header is not None:
                cal_std_img = std_data.astype(np.float32)
                if final_mb_data is not None: cal_std_img -= final_mb_data
                if final_md_corr_data is not None: cal_std_img -= final_md_corr_data
                if mf_corr_std_use is not None:
                    safe_mf_std = np.where(mf_corr_std_use < 0.01, 1.0, mf_corr_std_use)
                    if not np.all(np.isclose(safe_mf_std, 0)): cal_std_img /= safe_mf_std
                
                std_stars_table = detect_stars_dao(cal_std_img, fwhm, thresh_nsigma)
                if std_stars_table and len(std_stars_table) > 0:
                    if 'flux' in std_stars_table.colnames : std_stars_table.sort('flux', reverse=True)
                    brightest_std_star_photutils = std_stars_table[0]
                    std_phot_table = perform_aperture_photometry_on_detections(cal_std_img, Table([brightest_std_star_photutils]), ap_radius_phot)
                    if std_phot_table and 'net_flux' in std_phot_table.colnames and len(std_phot_table) > 0:
                        m_inst_std = calculate_instrumental_magnitude(std_phot_table['net_flux'][0])
                        x_std = calculate_airmass_extinction(std_header)
                        m_std_known_val = np.nan
                        if std_mag_known_in is not None:
                            try: m_std_known_val = float(std_mag_known_in)
                            except: status_log.append(f"{std_filt_char} 표준 등급 입력값 유효X.")
                        
                        if not np.isfinite(m_std_known_val):
                            std_ra, std_dec = convert_pixel_to_wcs(std_phot_table['xcentroid'][0], std_phot_table['ycentroid'][0], std_header)
                            if np.isfinite(std_ra):
                                simbad_id_std = query_simbad_for_object(std_ra, std_dec, 2.0)
                                status_log.append(f"{std_filt_char} 표준별 SIMBAD: {simbad_id_std} (등급 자동 추출 미구현)")
                        
                        if np.isfinite(m_std_known_val) and m_inst_std is not None and x_std is not None and k_coeff_std_val is not None:
                            calc_m0 = m_inst_std - k_coeff_std_val * x_std - m_std_known_val
                            if m0_eff_var_name == 'm0_eff_b': m0_eff_b = calc_m0
                            elif m0_eff_var_name == 'm0_eff_v': m0_eff_v = calc_m0
                            status_log.append(f"{std_filt_char}필터 영점(m0_eff) 계산됨: {calc_m0:.3f} (표준별 사용)")
                        else: status_log.append(f"{std_filt_char}필터 표준별 정보 부족으로 영점 자동 계산 불가. 사용자 입력 m0 사용.")
                    else: status_log.append(f"{std_filt_char}필터 표준별 측광 실패.")
                else: status_log.append(f"{std_filt_char}필터 표준별 이미지에서 별 탐지 실패.")
            else: status_log.append(f"{std_filt_char}필터 표준별 파일 로드 실패.")
        else: status_log.append(f"{std_filt_char}필터 표준별 파일 미업로드. 사용자 입력 m0 사용.")

    filter_processed_stars_data = {'B': [], 'V': []} 
    for filter_char_loop, light_objs_loop, k_coeff_loop, m0_eff_loop, mf_corr_to_use in [
        ('B', light_b_file_objs, k_b, m0_eff_b, final_mf_b_corr_data), 
        ('V', light_v_file_objs, k_v, m0_eff_v, final_mf_v_corr_data)
    ]:
        if not light_objs_loop: continue
        status_log.append(f"--- {filter_char_loop} 필터 대상 프레임 처리 시작 ({len(light_objs_loop)}개) ---")
        for light_obj_item in light_objs_loop:
            if not (light_obj_item and light_obj_item.name and os.path.exists(light_obj_item.name)): continue
            filename_loop = os.path.basename(light_obj_item.name)
            status_log.append(f"처리 중: {filename_loop} ({filter_char_loop})")
            try:
                image_data, header = load_single_fits_from_path(light_obj_item.name, f"{filter_char_loop} LIGHT")
                if image_data is None or header is None: status_log.append(f"오류: {filename_loop} 로드 실패."); continue
                cal_img = image_data.astype(np.float32)
                if final_mb_data is not None: cal_img -= final_mb_data
                if final_md_corr_data is not None: cal_img -= final_md_corr_data
                if mf_corr_to_use is not None: 
                    safe_mf = np.where(mf_corr_to_use < 0.01, 1.0, mf_corr_to_use)
                    if not np.all(np.isclose(safe_mf,0)): cal_img /= safe_mf
                
                detected_stars_table = detect_stars_dao(cal_img, fwhm, thresh_nsigma)
                if detected_stars_table is None or len(detected_stars_table) == 0: status_log.append(f"{filename_loop}: 별 탐지 실패."); continue

                phot_input_table = detected_stars_table
                if use_roi: 
                    x_dao, y_dao = detected_stars_table['xcentroid'], detected_stars_table['ycentroid']
                    roi_m = (x_dao >= roi_x0) & (x_dao <= roi_x1) & (y_dao >= roi_y0) & (y_dao <= roi_y1)
                    stars_in_roi = detected_stars_table[roi_m]
                    if not stars_in_roi: status_log.append(f"{filename_loop}: ROI 내 별 없음."); continue
                    status_log.append(f"{filename_loop}: {len(stars_in_roi)}개 별 ROI 내에 있음.")
                    phot_input_table = stars_in_roi
                
                phot_results_table = perform_aperture_photometry_on_detections(cal_img, phot_input_table, ap_radius_phot)
                if phot_results_table is None or 'net_flux' not in phot_results_table.colnames: status_log.append(f"{filename_loop}: 측광 실패."); continue
                
                ras, decs = convert_pixel_to_wcs(phot_results_table['xcentroid'], phot_results_table['ycentroid'], header)
                airmass_val = calculate_airmass_extinction(header)
                
                for star_idx, star_phot_info in enumerate(phot_results_table):
                    inst_flux = star_phot_info['net_flux']
                    inst_mag = calculate_instrumental_magnitude(inst_flux)
                    std_mag_val = calculate_standard_magnitude(inst_mag, airmass_val, k_coeff_loop, m0_eff_loop) if inst_mag is not None and airmass_val is not None else np.nan
                    
                    filter_processed_stars_data[filter_char_loop].append({
                        'file': filename_loop, 'filter': filter_char_loop,
                        'x': star_phot_info['xcentroid'], 'y': star_phot_info['ycentroid'],
                        'ra_deg': ras[star_idx] if ras is not None and star_idx < len(ras) else np.nan, 
                        'dec_deg': decs[star_idx] if decs is not None and star_idx < len(decs) else np.nan,
                        'flux': inst_flux, 'inst_mag': inst_mag, 'std_mag': std_mag_val, 
                        'airmass': airmass_val, 'header': header 
                    })
                status_log.append(f"{filename_loop}: {len(phot_results_table)}개 별 처리 완료 (ROI 적용됨).")
            except Exception as e_frame_tab4_proc:
                logger_ui.error(f"{filename_loop} 처리 중 오류 (탭4)", exc_info=True)
                status_log.append(f"오류 ({filename_loop}): {str(e_frame_tab4_proc)}")

    final_display_list = []
    processed_b_stars = filter_processed_stars_data['B']
    processed_v_stars = filter_processed_stars_data['V']
    v_coords_for_matching, v_data_for_matching = [], []
    if processed_v_stars:
        for star_v in processed_v_stars:
            if np.isfinite(star_v['ra_deg']) and np.isfinite(star_v['dec_deg']):
                v_coords_for_matching.append(SkyCoord(star_v['ra_deg'], star_v['dec_deg'], unit='deg', frame='icrs'))
                v_data_for_matching.append(star_v)
    v_catalog_sc = SkyCoord(v_coords_for_matching) if v_coords_for_matching else None
    v_matched_in_b_loop = [False] * len(v_data_for_matching)

    for b_star in processed_b_stars:
        entry = b_star.copy(); entry['mag_std_v'] = np.nan; entry['b_minus_v'] = np.nan; entry['simbad_id'] = "N/A"
        if v_catalog_sc and np.isfinite(b_star['ra_deg']) and np.isfinite(b_star['dec_deg']):
            b_star_sc = SkyCoord(b_star['ra_deg'], b_star['dec_deg'], unit='deg', frame='icrs')
            idx_v, sep2d_v, _ = b_star_sc.match_to_catalog_sky(v_catalog_sc) 
            if sep2d_v.arcsec < simbad_radius: 
                matched_v_data = v_data_for_matching[idx_v]; v_matched_in_b_loop[idx_v] = True 
                entry['mag_std_v'] = matched_v_data['std_mag']
                if np.isfinite(entry['std_mag']) and np.isfinite(entry['mag_std_v']): entry['b_minus_v'] = entry['std_mag'] - entry['mag_std_v']
        final_display_list.append(entry)
    if v_catalog_sc:
        for idx, v_star_data in enumerate(v_data_for_matching):
            if not v_matched_in_b_loop[idx]:
                final_display_list.append({'file': v_star_data['file'], 'filter': 'V_only', 'x': v_star_data['x'], 'y': v_star_data['y'],
                                           'ra_deg': v_star_data['ra_deg'], 'dec_deg': v_star_data['dec_deg'], 'flux': v_star_data['flux'], 
                                           'inst_mag': v_star_data['inst_mag'], 'std_mag': np.nan, 'mag_std_v': v_star_data['std_mag'], 
                                           'b_minus_v': np.nan, 'airmass': v_star_data['airmass'], 'simbad_id': 'N/A'})
    status_log.append(f"별 정보 통합 및 B-V 계산 완료. 총 {len(final_display_list)}개 별 항목 생성.")
    if final_display_list:
        status_log.append("SIMBAD 정보 조회 중...")
        for star_entry in final_display_list:
            ra_q, dec_q = star_entry.get('ra_deg', np.nan), star_entry.get('dec_deg', np.nan)
            if np.isfinite(ra_q) and np.isfinite(dec_q): star_entry['simbad_id'] = query_simbad_for_object(ra_q, dec_q, simbad_radius)
            else: star_entry['simbad_id'] = "WCS 없음"
        status_log.append("SIMBAD 정보 조회 완료.")
        final_display_list.sort(key=lambda s: (s.get('std_mag', np.inf) if np.isfinite(s.get('std_mag', np.inf)) else np.inf, 
                                               s.get('mag_std_v', np.inf) if np.isfinite(s.get('mag_std_v', np.inf)) else np.inf, 
                                               -(s.get('flux', -np.inf) if np.isfinite(s.get('flux', -np.inf)) else -np.inf)))
        for rank, star_entry in enumerate(final_display_list): star_entry['rank'] = rank + 1
        status_log.append("밝기 순 정렬 완료.")

    df_headers = ["Rank", "RA(deg)", "Dec(deg)", "StdMag B", "StdMag V", "B-V", 
                  "Flux", "InstMag", "Airmass", "Filter", "File", "X", "Y", "SIMBAD ID"]
    for s_data in final_display_list:
        all_stars_final_data_for_df.append([
            s_data.get('rank', ''),
            f"{s_data.get('ra_deg', np.nan):.5f}" if np.isfinite(s_data.get('ra_deg', np.nan)) else "N/A",
            f"{s_data.get('dec_deg', np.nan):.5f}" if np.isfinite(s_data.get('dec_deg', np.nan)) else "N/A",
            f"{s_data.get('std_mag', np.nan):.3f}" if np.isfinite(s_data.get('std_mag', np.nan)) else "N/A", 
            f"{s_data.get('mag_std_v', np.nan):.3f}" if np.isfinite(s_data.get('mag_std_v', np.nan)) else "N/A", 
            f"{s_data.get('b_minus_v', np.nan):.3f}" if np.isfinite(s_data.get('b_minus_v', np.nan)) else "N/A",
            f"{s_data.get('flux', np.nan):.2e}" if np.isfinite(s_data.get('flux', np.nan)) else "N/A", 
            f"{s_data.get('inst_mag', np.nan):.3f}" if np.isfinite(s_data.get('inst_mag', np.nan)) else "N/A", 
            f"{s_data.get('airmass', np.nan):.3f}" if np.isfinite(s_data.get('airmass', np.nan)) else "N/A",
            s_data.get('filter', "N/A"), s_data.get('file', "N/A"),
            f"{s_data.get('x', np.nan):.1f}" if np.isfinite(s_data.get('x', np.nan)) else "N/A",
            f"{s_data.get('y', np.nan):.1f}" if np.isfinite(s_data.get('y', np.nan)) else "N/A",
            s_data.get('simbad_id', "N/A")
        ])

    if all_stars_final_data_for_df:
        csv_filename = f"detailed_photometry_results_{current_timestamp_str}.csv"
        csv_output_path = os.path.join(temp_dir, csv_filename)
        try:
            with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile); writer.writerow(df_headers); writer.writerows(all_stars_final_data_for_df)
            status_log.append(f"결과 CSV 저장: {csv_filename}")
        except Exception as e_csv:
            logger_ui.error("CSV 파일 저장 오류", exc_info=True); status_log.append(f"CSV 저장 오류: {str(e_csv)}"); csv_output_path = None
            
    final_log = "\n".join(status_log)
    logger_ui.info("탭4: 상세 측광 분석 완료.")
    return (df_headers, all_stars_final_data_for_df) if all_stars_final_data_for_df else (df_headers, [["결과 없음"]*len(df_headers)]), \
           csv_output_path, \
           final_log

def handle_tab4_roi_preview_update(
    b_light_file_for_roi_obj, v_light_file_for_roi_obj, 
    current_roi_image_data_b_state, current_roi_image_data_v_state, 
    roi_x_min_val, roi_x_max_val, roi_y_min_val, roi_y_max_val 
    ):
    status_log = []
    output_pil_image_with_roi = None
    new_image_data_b_state = current_roi_image_data_b_state 
    new_image_data_v_state = current_roi_image_data_v_state
    
    slider_x_min_update = gr.update() 
    slider_x_max_update = gr.update()
    slider_y_min_update = gr.update()
    slider_y_max_update = gr.update()

    image_data_to_draw_on = None
    filename_for_log = "알 수 없음"
    newly_loaded_data_tuple = None 

    b_file_path = b_light_file_for_roi_obj.name if b_light_file_for_roi_obj and hasattr(b_light_file_for_roi_obj, 'name') else None
    v_file_path = v_light_file_for_roi_obj.name if v_light_file_for_roi_obj and hasattr(v_light_file_for_roi_obj, 'name') else None

    if b_file_path and os.path.exists(b_file_path):
        filename_for_log = os.path.basename(b_file_path)
        data, _ = load_single_fits_from_path(b_file_path, f"ROI용 B 이미지 ({filename_for_log})")
        if data is not None: 
            newly_loaded_data_tuple = (data, True) 
            status_log.append(f"{filename_for_log} (B)를 ROI 미리보기용으로 로드.")
    elif v_file_path and os.path.exists(v_file_path) and newly_loaded_data_tuple is None: 
        filename_for_log = os.path.basename(v_file_path)
        data, _ = load_single_fits_from_path(v_file_path, f"ROI용 V 이미지 ({filename_for_log})")
        if data is not None: 
            newly_loaded_data_tuple = (data, False) 
            status_log.append(f"{filename_for_log} (V)를 ROI 미리보기용으로 로드.")
    
    if newly_loaded_data_tuple:
        image_data_to_draw_on, is_b = newly_loaded_data_tuple
        if is_b:
            new_image_data_b_state = image_data_to_draw_on
            new_image_data_v_state = None 
        else:
            new_image_data_v_state = image_data_to_draw_on
            new_image_data_b_state = None
        
        h, w = image_data_to_draw_on.shape
        slider_x_min_update = gr.update(minimum=0, maximum=w - 1 if w > 0 else 0, value=0)
        slider_x_max_update = gr.update(minimum=0, maximum=w - 1 if w > 0 else 0, value=w - 1 if w > 0 else 0)
        slider_y_min_update = gr.update(minimum=0, maximum=h - 1 if h > 0 else 0, value=0)
        slider_y_max_update = gr.update(minimum=0, maximum=h - 1 if h > 0 else 0, value=h - 1 if h > 0 else 0)
        status_log.append(f"ROI 슬라이더 범위 업데이트: W={w}, H={h}")
    elif current_roi_image_data_b_state is not None: 
        image_data_to_draw_on = current_roi_image_data_b_state
        filename_for_log = "이전 B 이미지"
        status_log.append("이전에 로드된 B 이미지를 ROI 미리보기용으로 사용.")
    elif current_roi_image_data_v_state is not None: 
        image_data_to_draw_on = current_roi_image_data_v_state
        filename_for_log = "이전 V 이미지"
        status_log.append("이전에 로드된 V 이미지를 ROI 미리보기용으로 사용.")
    
    if image_data_to_draw_on is not None:
        base_pil_preview = create_preview_image(image_data_to_draw_on) 
        if base_pil_preview:
            output_pil_image_with_roi = draw_roi_on_pil_image(
                base_pil_preview, 
                roi_x_min_val, roi_x_max_val, 
                roi_y_min_val, roi_y_max_val
            )
            status_log.append(f"ROI 업데이트됨: X({roi_x_min_val}-{roi_x_max_val}), Y({roi_y_min_val}-{roi_y_max_val}) on {filename_for_log}")
        else:
            status_log.append(f"{filename_for_log}의 기본 미리보기 생성 실패.")
    else:
        status_log.append("ROI를 표시할 이미지가 없습니다.")

    return output_pil_image_with_roi, \
           slider_x_min_update, slider_x_max_update, slider_y_min_update, slider_y_max_update, \
           new_image_data_b_state, new_image_data_v_state, \
           "\n".join(status_log)

