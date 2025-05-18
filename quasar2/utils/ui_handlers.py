# ==============================================================================
# File: ui_handlers.py
# Description: Gradio UI event handler functions.
# ==============================================================================

import os
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt # Tab 3에서 그래프 생성에 사용
import csv # Tab 4에서 CSV 파일 생성에 사용
# import pandas as pd # 만약 CSV 생성 또는 데이터 처리에 pandas를 선호한다면 사용

# Astropy 관련 모듈 (Tab 4 등에서 직접 사용될 수 있는 부분)
from astropy.coordinates import SkyCoord # Tab 4 별 좌표 매칭에 사용
import astropy.units as u # Tab 4 별 좌표 매칭에 사용
from astropy.table import Table # DAOStarFinder 결과가 Table 객체일 경우를 대비 (직접 사용은 최소화)
from astropy.io import fits # FITS 파일 읽기 및 쓰기

# 프로젝트 유틸리티 모듈에서 함수 임포트
from utils.fits import (
    load_fits_from_gradio_files,
    load_single_fits_from_path,
    save_fits_image,
    create_preview_image
)
from utils.calibration import (
    create_master_bias_from_data,
    create_master_dark_from_data,
    create_master_flat_from_data
)
from utils.photometry import (
    detect_stars_extinction, # Tab 3용
    find_brightest_star_extinction, # Tab 3용
    calculate_flux_extinction, # Tab 3용
    detect_stars_dao, # Tab 4용
    perform_aperture_photometry_on_detections # Tab 4용
)
from utils.astro import (
    calculate_altitude_extinction,
    calculate_airmass_extinction,
    calculate_instrumental_magnitude,
    perform_linear_regression_extinction,
    convert_pixel_to_wcs, # Tab 4용
    calculate_standard_magnitude, # Tab 4용
    query_simbad_for_object, # Tab 4용
    match_stars_by_coords # Tab 4용
)


logger_ui = logging.getLogger(__name__)

def handle_tab1_master_frame_creation(bias_file_objs, dark_file_objs, flat_file_objs_all, temp_dir):
    status_messages = []
    # UI 출력용 경로
    ui_bias_path, ui_dark_path = None, None
    ui_flat_b_path, ui_flat_v_path, ui_flat_generic_path = None, None, None
    # 상태 저장용 경로
    state_bias_path, state_dark_path_corrected = None, None
    state_flat_b_path_corrected, state_flat_v_path_corrected, state_flat_generic_path_corrected = None, None, None
    
    master_bias_data, bias_header = None, None
    master_dark_corrected_data, dark_header = None, None # Bias가 빠진 Dark
    
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # 1. Master BIAS 생성
    if bias_file_objs:
        try:
            status_messages.append(f"BIAS: {len(bias_file_objs)}개 파일 처리 시작...")
            # load_fits_from_gradio_files는 이제 (stacked_images, first_header, all_headers) 반환
            bias_images_stack, bias_header_loaded, _ = load_fits_from_gradio_files(bias_file_objs, "BIAS")
            if bias_header_loaded: bias_header = bias_header_loaded 
            master_bias_data = create_master_bias_from_data(bias_images_stack)
            saved_path = save_fits_image(master_bias_data, bias_header, "master_bias", temp_dir, current_timestamp_str)
            if saved_path: ui_bias_path = state_bias_path = saved_path; status_messages.append(f"BIAS: 생성 완료: {os.path.basename(ui_bias_path)}")
            else: status_messages.append("BIAS: 생성 실패 (저장 오류 또는 데이터 없음).")
        except Exception as e: logger_ui.error("BIAS 처리 중 오류", exc_info=True); status_messages.append(f"BIAS 처리 오류: {str(e)}"); master_bias_data = None
    else: status_messages.append("BIAS: 업로드된 파일 없음.")

    # 2. Master DARK (Corrected) 생성
    if dark_file_objs:
        try:
            status_messages.append(f"DARK: {len(dark_file_objs)}개 파일 처리 시작...")
            dark_images_stack, dark_header_loaded, _ = load_fits_from_gradio_files(dark_file_objs, "DARK")
            if dark_header_loaded: dark_header = dark_header_loaded
            # create_master_dark_from_data는 bias_corrected dark를 반환
            master_dark_corrected_data = create_master_dark_from_data(dark_images_stack, master_bias_data) 
            saved_path = save_fits_image(master_dark_corrected_data, dark_header, "master_dark_corrected", temp_dir, current_timestamp_str)
            if saved_path: ui_dark_path = state_dark_path_corrected = saved_path; status_messages.append(f"DARK (Corrected): 생성 완료: {os.path.basename(ui_dark_path)}")
            else: status_messages.append("DARK (Corrected): 생성 실패.")
        except Exception as e: logger_ui.error("DARK 처리 중 오류", exc_info=True); status_messages.append(f"DARK 처리 오류: {str(e)}"); master_dark_corrected_data = None
    else: status_messages.append("DARK: 업로드된 파일 없음.")

    # 3. Master FLAT (필터별 및 Generic) 생성
    if flat_file_objs_all:
        status_messages.append(f"FLAT: 총 {len(flat_file_objs_all)}개 파일 처리 시작...")
        # 필터별로 flat 파일 객체와 헤더 분리
        flats_b_objs, flats_v_objs, flats_generic_objs = [], [], []
        
        # load_fits_from_gradio_files는 이제 all_headers도 반환하므로, 이를 활용
        # 하지만 여기서는 각 파일 객체를 직접 순회하며 헤더를 읽어 필터링
        for idx, flat_obj in enumerate(flat_file_objs_all): # 각 파일 객체 순회
            if flat_obj and flat_obj.name:
                try:
                    # 헤더만 읽기 위해 임시 로드 (데이터는 나중에 그룹별로 로드)
                    _, header_temp = load_single_fits_from_path(flat_obj.name, "FLAT (header check)")
                    if header_temp:
                        filter_val = header_temp.get('FILTER', 'UNKNOWN').strip().upper()
                        if filter_val == 'B':
                            flats_b_objs.append(flat_obj)
                        elif filter_val == 'V':
                            flats_v_objs.append(flat_obj)
                        else:
                            flats_generic_objs.append(flat_obj)
                            status_messages.append(f"FLAT 파일 '{os.path.basename(flat_obj.name)}'의 필터 '{filter_val}'는 Generic으로 분류됩니다.")
                    else: # 헤더 로드 실패 시 Generic으로
                        status_messages.append(f"FLAT 파일 '{os.path.basename(flat_obj.name)}'의 헤더를 읽을 수 없어 Generic으로 분류됩니다.")
                        flats_generic_objs.append(flat_obj)
                except Exception as e_flat_head:
                    status_messages.append(f"FLAT 파일 '{os.path.basename(flat_obj.name)}' 헤더 읽기 오류 ({e_flat_head}), Generic으로 분류.")
                    flats_generic_objs.append(flat_obj)
            else:
                status_messages.append("유효하지 않은 FLAT 파일 객체 발견.")

        # 필터별 마스터 플랫 생성
        for filter_char, flat_objs_list in [('B', flats_b_objs), ('V', flats_v_objs), ('Generic', flats_generic_objs)]:
            if flat_objs_list: # 해당 필터/타입의 파일이 있을 때만 처리
                try:
                    status_messages.append(f"Master FLAT ({filter_char}): {len(flat_objs_list)}개 파일로 생성 시작...")
                    # 여기서 flat_objs_list를 load_fits_from_gradio_files에 전달하여 스택 생성
                    flat_images_stack, first_flat_header, _ = load_fits_from_gradio_files(flat_objs_list, f"FLAT ({filter_char})")
                    
                    if flat_images_stack is None: # 스택 생성 실패 시
                        status_messages.append(f"Master FLAT ({filter_char}): 이미지 스택 생성 실패. 건너<0xEB><0><0x8F><0xEB><0x82><0xB4>니다.")
                        continue

                    master_flat_data = create_master_flat_from_data(flat_images_stack, master_bias_data, master_dark_corrected_data)
                    save_header_flat = first_flat_header if first_flat_header else (bias_header if bias_header else fits.Header())
                    saved_path = save_fits_image(master_flat_data, save_header_flat, f"master_flat_{filter_char.lower()}_corrected", temp_dir, current_timestamp_str)
                    
                    if saved_path:
                        status_messages.append(f"Master FLAT ({filter_char}, Corrected): 생성 완료: {os.path.basename(saved_path)}")
                        if filter_char == 'B': ui_flat_b_path = state_flat_b_path_corrected = saved_path
                        elif filter_char == 'V': ui_flat_v_path = state_flat_v_path_corrected = saved_path
                        else: ui_flat_generic_path = state_flat_generic_path_corrected = saved_path
                    else:
                        status_messages.append(f"Master FLAT ({filter_char}, Corrected): 생성 실패.")
                except Exception as e_mf:
                    logger_ui.error(f"Master FLAT ({filter_char}) 처리 중 오류", exc_info=True)
                    status_messages.append(f"Master FLAT ({filter_char}) 처리 오류: {str(e_mf)}")
            else:
                status_messages.append(f"Master FLAT ({filter_char}): 업로드된 해당 필터 파일 없음.")
    else:
        status_messages.append("FLAT: 업로드된 파일 없음.")
        
    final_status = "\n".join(status_messages)
    logger_ui.info("Tab 1: Master frame generation finished.")
    return ui_bias_path, ui_dark_path, ui_flat_b_path, ui_flat_v_path, ui_flat_generic_path, \
           state_bias_path, state_dark_path_corrected, \
           state_flat_b_path_corrected, state_flat_v_path_corrected, state_flat_generic_path_corrected, \
           final_status


def handle_tab2_light_frame_calibration(
    light_file_objs_list, 
    # 탭2에서 직접 업로드 (Corrected 상태로 업로드 가정)
    uploaded_mf_b_corr_obj, uploaded_mf_v_corr_obj, 
    # 탭1 상태 변수
    state_mb_p, state_md_p_corrected, 
    state_mf_b_p_corr, state_mf_v_p_corr, state_mf_gen_p_corr, 
    preview_stretch_type, preview_asinh_a,
    temp_dir):
    status_messages = []
    calibrated_light_file_paths_for_ui = []
    output_preview_pil_image = None 
    mb_data, md_corr_data = None, None 
    used_masters_info = {"bias": "미사용", "dark": "미사용"} # Flat은 프레임별로 기록
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    try: 
        # BIAS 결정 (탭2 업로드 우선순위 없음, 탭1 상태만 사용)
        if state_mb_p and os.path.exists(state_mb_p):
            mb_data, _ = load_single_fits_from_path(state_mb_p, "저장된 Master BIAS (탭1)")
            used_masters_info["bias"] = f"탭1 ({os.path.basename(state_mb_p)})" if mb_data is not None else "탭1 로드 실패"
        else:
            status_messages.append("경고: 탭1에서 생성된 Master BIAS를 찾을 수 없습니다. BIAS 보정이 생략될 수 있습니다.")
        
        # DARK (Corrected) 결정 (탭2 업로드 우선순위 없음, 탭1 상태만 사용)
        if state_md_p_corrected and os.path.exists(state_md_p_corrected): 
            md_corr_data, _ = load_single_fits_from_path(state_md_p_corrected, "저장된 Master DARK (탭1-Corrected)")
            used_masters_info["dark"] = f"탭1 ({os.path.basename(state_md_p_corrected)})" if md_corr_data is not None else "탭1 로드 실패"
        else:
            status_messages.append("경고: 탭1에서 생성된 Master DARK를 찾을 수 없습니다. DARK 보정이 생략될 수 있습니다.")
        
        status_messages.extend([f"사용될 Master BIAS: {used_masters_info['bias']}", f"사용될 Master DARK: {used_masters_info['dark']}"])
    except Exception as e:
        logger_ui.error("탭2 BIAS/DARK 마스터 프레임 로드 중 오류", exc_info=True); status_messages.append(f"BIAS/DARK 마스터 로드 실패: {str(e)}"); return [], None, "\n".join(status_messages)

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

            # 이 LIGHT 프레임에 사용할 FLAT 결정
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
            
            if flat_to_use_for_this_light is None and state_mf_gen_p_corr and os.path.exists(state_mf_gen_p_corr): # 필터별 플랫 없으면 Generic 시도
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
            
            light_header['HISTORY'] = f'Calibrated App v0.9 (B:{used_masters_info["bias"]!="미사용"},D:{used_masters_info["dark"]!="미사용"},F:{flat_source_for_this_frame!="미사용"})'
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
    # 탭4에서 직접 업로드된 마스터 프레임 (Raw 상태 가정)
    tab4_uploaded_mb_obj, tab4_uploaded_md_raw_obj, 
    tab4_uploaded_mf_b_raw_obj, tab4_uploaded_mf_v_raw_obj,
    # 탭1에서 생성된 마스터 프레임 경로 (Corrected 상태)
    state_mb_p, state_md_p_corrected, 
    state_mf_b_p_corr, state_mf_v_p_corr, state_mf_gen_p_corr,
    # 사용자 입력 파라미터
    k_b_input, m0_b_input, k_v_input, m0_v_input, 
    dao_fwhm_input, dao_thresh_nsigma_input, phot_aperture_radius_input, 
    simbad_query_radius_arcsec, 
    temp_dir):
    status_log = []
    all_stars_final_data_for_df = [] 
    csv_output_path = None
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger_ui.info("탭 4 상세 측광 분석 시작...")

    # --- 1. 입력값 유효성 검사 ---
    if not light_b_file_objs and not light_v_file_objs:
        status_log.append("오류: B 또는 V 필터 LIGHT 프레임을 하나 이상 업로드해야 합니다.")
        return (["Error Message"], [["LIGHT 프레임 없음"]]), None, "\n".join(status_log)
    try:
        k_b, m0_b = float(k_b_input), float(m0_b_input)
        k_v, m0_v = float(k_v_input), float(m0_v_input)
        fwhm, thresh_nsigma = float(dao_fwhm_input), float(dao_thresh_nsigma_input)
        ap_radius_phot = float(phot_aperture_radius_input)
        simbad_radius = float(simbad_query_radius_arcsec)
    except ValueError:
        status_log.append("오류: 입력 파라미터(소광계수, 영점 등)는 숫자여야 합니다.")
        return (["Error Message"], [["입력 파라미터 오류"]]), None, "\n".join(status_log)

    # --- 2. 최종 사용할 마스터 프레임 결정 및 준비 ---
    # BIAS
    final_mb_data = None
    if tab4_uploaded_mb_obj and tab4_uploaded_mb_obj.name:
        final_mb_data, _ = load_single_fits_from_path(tab4_uploaded_mb_obj.name, "탭4 업로드 Master BIAS")
        status_log.append("탭4 업로드 Master BIAS 사용." if final_mb_data is not None else "탭4 업로드 Master BIAS 로드 실패.")
    elif state_mb_p and os.path.exists(state_mb_p):
        final_mb_data, _ = load_single_fits_from_path(state_mb_p, "탭1 Master BIAS")
        status_log.append("탭1 Master BIAS 사용." if final_mb_data is not None else "탭1 Master BIAS 로드 실패.")
    if final_mb_data is None: status_log.append("경고: 사용 가능한 Master BIAS 없음. BIAS 보정 생략."); # 처리를 중단할 수도 있음

    # DARK (Corrected)
    final_md_corr_data = None
    if tab4_uploaded_md_raw_obj and tab4_uploaded_md_raw_obj.name:
        raw_md, _ = load_single_fits_from_path(tab4_uploaded_md_raw_obj.name, "탭4 업로드 Raw Master DARK")
        if raw_md is not None:
            if final_mb_data is not None and raw_md.shape == final_mb_data.shape:
                final_md_corr_data = raw_md - final_mb_data; status_log.append("탭4 업로드 Raw DARK에서 BIAS 차감하여 사용.")
            else: final_md_corr_data = raw_md; status_log.append("경고: 탭4 업로드 Raw DARK에 BIAS 차감 못함 (BIAS 없거나 크기 불일치). 그대로 사용.")
        else: status_log.append("탭4 업로드 Raw DARK 로드 실패.")
    elif state_md_p_corrected and os.path.exists(state_md_p_corrected):
        final_md_corr_data, _ = load_single_fits_from_path(state_md_p_corrected, "탭1 Master DARK (Corrected)")
        status_log.append("탭1 Master DARK (Corrected) 사용." if final_md_corr_data is not None else "탭1 Master DARK (Corrected) 로드 실패.")
    if final_md_corr_data is None: status_log.append("경고: 사용 가능한 Master DARK 없음. DARK 보정 생략.")

    # FLAT B (Corrected)
    final_mf_b_corr_data = None
    if tab4_uploaded_mf_b_raw_obj and tab4_uploaded_mf_b_raw_obj.name:
        raw_mf_b, _ = load_single_fits_from_path(tab4_uploaded_mf_b_raw_obj.name, "탭4 업로드 Raw Master FLAT B")
        if raw_mf_b is not None:
            temp_mf_b = raw_mf_b.copy()
            if final_mb_data is not None and temp_mf_b.shape == final_mb_data.shape: temp_mf_b -= final_mb_data
            if final_md_corr_data is not None and temp_mf_b.shape == final_md_corr_data.shape: temp_mf_b -= final_md_corr_data
            mean_b = np.mean(temp_mf_b); final_mf_b_corr_data = temp_mf_b / mean_b if mean_b > 1e-9 else temp_mf_b
            status_log.append("탭4 업로드 Raw Master FLAT B 처리하여 사용 (정규화 시도).")
        else: status_log.append("탭4 업로드 Raw Master FLAT B 로드 실패.")
    elif state_mf_b_p_corr and os.path.exists(state_mf_b_p_corr):
        final_mf_b_corr_data, _ = load_single_fits_from_path(state_mf_b_p_corr, "탭1 Master FLAT B (Corrected)")
        status_log.append("탭1 Master FLAT B 사용." if final_mf_b_corr_data is not None else "탭1 Master FLAT B 로드 실패.")
    elif state_mf_gen_p_corr and os.path.exists(state_mf_gen_p_corr):
        final_mf_b_corr_data, _ = load_single_fits_from_path(state_mf_gen_p_corr, "탭1 Master FLAT Generic (B용)")
        status_log.append("탭1 Generic Master FLAT을 B필터용으로 사용 (주의)." if final_mf_b_corr_data is not None else "탭1 Generic Master FLAT 로드 실패 (B용).")
    if final_mf_b_corr_data is None: status_log.append("경고: 사용 가능한 Master FLAT B 없음. B필터 FLAT 보정 생략.")

    # FLAT V (Corrected) - B와 유사 로직
    final_mf_v_corr_data = None
    if tab4_uploaded_mf_v_raw_obj and tab4_uploaded_mf_v_raw_obj.name:
        raw_mf_v, _ = load_single_fits_from_path(tab4_uploaded_mf_v_raw_obj.name, "탭4 업로드 Raw Master FLAT V")
        if raw_mf_v is not None:
            temp_mf_v = raw_mf_v.copy()
            if final_mb_data is not None and temp_mf_v.shape == final_mb_data.shape: temp_mf_v -= final_mb_data
            if final_md_corr_data is not None and temp_mf_v.shape == final_md_corr_data.shape: temp_mf_v -= final_md_corr_data
            mean_v = np.mean(temp_mf_v); final_mf_v_corr_data = temp_mf_v / mean_v if mean_v > 1e-9 else temp_mf_v
            status_log.append("탭4 업로드 Raw Master FLAT V 처리하여 사용 (정규화 시도).")
        else: status_log.append("탭4 업로드 Raw Master FLAT V 로드 실패.")
    elif state_mf_v_p_corr and os.path.exists(state_mf_v_p_corr):
        final_mf_v_corr_data, _ = load_single_fits_from_path(state_mf_v_p_corr, "탭1 Master FLAT V (Corrected)")
        status_log.append("탭1 Master FLAT V 사용." if final_mf_v_corr_data is not None else "탭1 Master FLAT V 로드 실패.")
    elif state_mf_gen_p_corr and os.path.exists(state_mf_gen_p_corr):
        final_mf_v_corr_data, _ = load_single_fits_from_path(state_mf_gen_p_corr, "탭1 Master FLAT Generic (V용)")
        status_log.append("탭1 Generic Master FLAT을 V필터용으로 사용 (주의)." if final_mf_v_corr_data is not None else "탭1 Generic Master FLAT 로드 실패 (V용).")
    if final_mf_v_corr_data is None: status_log.append("경고: 사용 가능한 Master FLAT V 없음. V필터 FLAT 보정 생략.")

    # 필수 마스터 프레임 확인 (예: BIAS는 거의 필수)
    if final_mb_data is None: 
        status_log.append("치명적 오류: Master BIAS를 사용할 수 없습니다. 처리를 중단합니다.")
        df_headers_fatal = ["Error Message"]; return (df_headers_fatal, [["Master BIAS 없음"]]), None, "\n".join(status_log)


    # --- 3. 필터별 프레임 처리 및 별 정보 추출 ---
    filter_processed_stars_data = {'B': [], 'V': []} 

    for filter_char_loop, light_objs_loop, k_coeff_loop, m0_val_loop, mf_corr_to_use in [
        ('B', light_b_file_objs, k_b, m0_b, final_mf_b_corr_data), 
        ('V', light_v_file_objs, k_v, m0_v, final_mf_v_corr_data)
    ]:
        if not light_objs_loop: continue
        status_log.append(f"--- {filter_char_loop} 필터 프레임 처리 시작 ({len(light_objs_loop)}개) ---")
        
        for light_obj_item in light_objs_loop:
            if not (light_obj_item and light_obj_item.name and os.path.exists(light_obj_item.name)):
                status_log.append(f"경고: 유효하지 않은 {filter_char_loop} 필터 파일 객체."); continue
            
            filename_loop = os.path.basename(light_obj_item.name)
            status_log.append(f"처리 중: {filename_loop} ({filter_char_loop})")
            try:
                image_data, header = load_single_fits_from_path(light_obj_item.name, f"{filter_char_loop} LIGHT")
                if image_data is None or header is None: status_log.append(f"오류: {filename_loop} 로드 실패."); continue

                cal_img = image_data.astype(np.float32)
                if final_mb_data is not None and final_mb_data.shape == cal_img.shape: cal_img -= final_mb_data
                if final_md_corr_data is not None and final_md_corr_data.shape == cal_img.shape: cal_img -= final_md_corr_data
                if mf_corr_to_use is not None and mf_corr_to_use.shape == cal_img.shape:
                    safe_mf = np.where(mf_corr_to_use < 0.01, 1.0, mf_corr_to_use)
                    if not np.all(np.isclose(safe_mf, 0)): cal_img /= safe_mf
                else: status_log.append(f"경고: {filename_loop} ({filter_char_loop})에 대한 Flat 보정 생략 (Master Flat 없음).")
                
                detected_stars_table = detect_stars_dao(cal_img, fwhm, thresh_nsigma)
                if detected_stars_table is None or len(detected_stars_table) == 0:
                    status_log.append(f"{filename_loop}: 별 탐지 실패 또는 별 없음."); continue
                
                phot_results_table = perform_aperture_photometry_on_detections(cal_img, detected_stars_table, ap_radius_phot)
                if phot_results_table is None or 'net_flux' not in phot_results_table.colnames:
                    status_log.append(f"{filename_loop}: 측광 실패."); continue

                ras, decs = convert_pixel_to_wcs(phot_results_table['xcentroid'], phot_results_table['ycentroid'], header)
                airmass_val = calculate_airmass_extinction(header)
                
                for star_row in phot_results_table:
                    inst_flux = star_row['net_flux']
                    inst_mag = calculate_instrumental_magnitude(inst_flux)
                    std_mag = None
                    if inst_mag is not None and airmass_val is not None:
                        std_mag = calculate_standard_magnitude(inst_mag, airmass_val, k_coeff_loop, m0_val_loop)
                    
                    star_idx_in_table = phot_results_table.index_of_row(star_row) 
                    
                    filter_processed_stars_data[filter_char_loop].append({
                        'file': filename_loop, 'filter': filter_char_loop,
                        'x': star_row['xcentroid'], 'y': star_row['ycentroid'],
                        'ra_deg': ras[star_idx_in_table] if ras is not None and star_idx_in_table < len(ras) else np.nan, 
                        'dec_deg': decs[star_idx_in_table] if decs is not None and star_idx_in_table < len(decs) else np.nan,
                        'flux': inst_flux, 'inst_mag': inst_mag, 'std_mag': std_mag,
                        'airmass': airmass_val, 'header': header 
                    })
                status_log.append(f"{filename_loop}: {len(phot_results_table)}개 별 처리 완료.")
            except Exception as e_frame_tab4:
                logger_ui.error(f"{filename_loop} 처리 중 오류 (탭4)", exc_info=True)
                status_log.append(f"오류 ({filename_loop}): {str(e_frame_tab4)}")

    # --- 4. 별 정보 통합, B-V 계산, SIMBAD 질의, 정렬 ---
    final_display_list = []
    processed_b_stars = filter_processed_stars_data['B']
    processed_v_stars = filter_processed_stars_data['V']

    v_coords_for_matching = []
    v_data_for_matching = []
    if processed_v_stars:
        for star_v in processed_v_stars:
            if np.isfinite(star_v['ra_deg']) and np.isfinite(star_v['dec_deg']):
                v_coords_for_matching.append(SkyCoord(star_v['ra_deg'], star_v['dec_deg'], unit='deg', frame='icrs'))
                v_data_for_matching.append(star_v)
    
    v_catalog_sc = SkyCoord(v_coords_for_matching) if v_coords_for_matching else None
    v_matched_in_b_loop = [False] * len(v_data_for_matching)

    for b_star in processed_b_stars:
        entry = b_star.copy() 
        entry['mag_std_v'] = np.nan 
        entry['b_minus_v'] = np.nan
        entry['simbad_id'] = "N/A"
        
        if v_catalog_sc and np.isfinite(b_star['ra_deg']) and np.isfinite(b_star['dec_deg']):
            b_star_sc = SkyCoord(b_star['ra_deg'], b_star['dec_deg'], unit='deg', frame='icrs')
            idx_v, sep2d_v, _ = b_star_sc.match_to_catalog_sky(v_catalog_sc) 
            
            if sep2d_v.arcsec < simbad_radius: 
                matched_v_data = v_data_for_matching[idx_v]
                v_matched_in_b_loop[idx_v] = True 
                entry['mag_std_v'] = matched_v_data['std_mag']
                if np.isfinite(entry['std_mag']) and np.isfinite(entry['mag_std_v']):
                    entry['b_minus_v'] = entry['std_mag'] - entry['mag_std_v']
        final_display_list.append(entry)

    if v_catalog_sc:
        for idx, v_star_data in enumerate(v_data_for_matching):
            if not v_matched_in_b_loop[idx]:
                final_display_list.append({
                    'file': v_star_data['file'], 'filter': 'V_only', 
                    'x': v_star_data['x'], 'y': v_star_data['y'],
                    'ra_deg': v_star_data['ra_deg'], 'dec_deg': v_star_data['dec_deg'],
                    'flux': v_star_data['flux'], 'inst_mag': v_star_data['inst_mag'], 
                    'std_mag': np.nan, 
                    'mag_std_v': v_star_data['std_mag'], 
                    'b_minus_v': np.nan,
                    'airmass': v_star_data['airmass'], 'simbad_id': 'N/A'
                })
    
    status_log.append(f"별 정보 통합 및 B-V 계산 완료. 총 {len(final_display_list)}개 별 항목 생성.")

    if final_display_list:
        status_log.append("SIMBAD 정보 조회 중...")
        for star_entry in final_display_list:
            ra_q, dec_q = star_entry.get('ra_deg', np.nan), star_entry.get('dec_deg', np.nan)
            if np.isfinite(ra_q) and np.isfinite(dec_q):
                star_entry['simbad_id'] = query_simbad_for_object(ra_q, dec_q, simbad_radius)
            else: star_entry['simbad_id'] = "WCS 없음"
        status_log.append("SIMBAD 정보 조회 완료.")

        final_display_list.sort(key=lambda s: (
            s.get('std_mag', np.inf) if np.isfinite(s.get('std_mag', np.inf)) else np.inf, 
            s.get('mag_std_v', np.inf) if np.isfinite(s.get('mag_std_v', np.inf)) else np.inf, 
            -(s.get('flux', -np.inf) if np.isfinite(s.get('flux', -np.inf)) else -np.inf) 
        ))
        for rank, star_entry in enumerate(final_display_list): star_entry['rank'] = rank + 1
        status_log.append("밝기 순 정렬 완료.")


    # --- 5. DataFrame 및 CSV용 데이터 준비 ---
    df_headers = ["Rank", "RA(deg)", "Dec(deg)", "StdMag B", "StdMag V", "B-V", 
                  "Flux", "InstMag", "Airmass", "Filter", "File", "X", "Y", "SIMBAD ID"]
    
    for s_data in final_display_list:
        ra_disp = s_data.get('ra_deg', np.nan)
        dec_disp = s_data.get('dec_deg', np.nan)
        file_disp = s_data.get('file', "N/A")
        filter_disp = s_data.get('filter', "N/A")
        x_disp = s_data.get('x', np.nan)
        y_disp = s_data.get('y', np.nan)
        flux_disp = s_data.get('flux', np.nan)
        imag_disp = s_data.get('inst_mag', np.nan)
        smag_b_disp = s_data.get('std_mag', np.nan) 
        smag_v_disp = s_data.get('mag_std_v', np.nan) 
        bv_disp = s_data.get('b_minus_v', np.nan)
        airmass_disp = s_data.get('airmass', np.nan)

        all_stars_final_data_for_df.append([
            s_data.get('rank', ''),
            f"{ra_disp:.5f}" if np.isfinite(ra_disp) else "N/A",
            f"{dec_disp:.5f}" if np.isfinite(dec_disp) else "N/A",
            f"{smag_b_disp:.3f}" if np.isfinite(smag_b_disp) else "N/A",
            f"{smag_v_disp:.3f}" if np.isfinite(smag_v_disp) else "N/A",
            f"{bv_disp:.3f}" if np.isfinite(bv_disp) else "N/A",
            f"{flux_disp:.2e}" if np.isfinite(flux_disp) else "N/A",
            f"{imag_disp:.3f}" if np.isfinite(imag_disp) else "N/A",
            f"{airmass_disp:.3f}" if np.isfinite(airmass_disp) else "N/A",
            filter_disp, file_disp,
            f"{x_disp:.1f}" if np.isfinite(x_disp) else "N/A",
            f"{y_disp:.1f}" if np.isfinite(y_disp) else "N/A",
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
