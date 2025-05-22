# ==============================================================================
# File: ui_handlers.py
# Description: Gradio UI event handler functions.
# ==============================================================================
import os
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt 
import csv 
import pandas as pd 
import gradio as gr

from utils.fits import ( 
    load_single_fits_from_path, save_fits_image, 
    create_preview_image, draw_roi_on_pil_image, get_fits_keyword,
    draw_photometry_results_on_image 
)
from utils.calibration import (
    create_master_bias_ccdproc, create_master_dark_ccdproc, 
    create_preliminary_master_flat_ccdproc, # 예비 플랫 생성 함수
    manual_ccd_calibration # 수동 보정 함수 추가
)
from utils.photometry import (
    detect_stars_extinction, find_brightest_star_extinction, 
    calculate_flux_extinction, detect_stars_dao, 
    perform_aperture_photometry_on_detections
)
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
from astropy.nddata import CCDData 
# import ccdproc as ccdp # ccd_process는 더 이상 직접 사용하지 않음


logger_ui = logging.getLogger(__name__)

def handle_tab1_master_frame_creation(bias_file_objs, dark_file_objs, flat_file_objs_all, temp_dir):
    """
    탭 1: 마스터 프레임 생성 핸들러.
    BIAS, DARK (노출시간별), 예비 FLAT (필터별)을 생성하고 저장합니다.
    생성된 DARK 및 예비 FLAT 파일 경로 리스트를 UI에 전달합니다.
    """
    status_messages = []
    ui_bias_path_out = None 
    ui_darks_paths_out = [] 
    ui_flats_paths_out = [] 
    
    state_bias_path_out = None
    state_darks_corrected_dict_out = {} 
    state_prelim_flats_dict_out = {} 
    
    master_bias_ccd = None # CCDData 객체로 BIAS를 다룸
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # 1. Master BIAS 생성
    if bias_file_objs:
        try:
            status_messages.append(f"BIAS: {len(bias_file_objs)}개 파일 처리 시작 (ccdproc)...")
            bias_file_paths = [f.name for f in bias_file_objs if f and f.name]
            if bias_file_paths:
                master_bias_ccd = create_master_bias_ccdproc(bias_file_paths) 
                if master_bias_ccd:
                    bias_header = master_bias_ccd.header if master_bias_ccd.header else fits.Header()
                    saved_path = save_fits_image(master_bias_ccd, bias_header, "master_bias_ccdproc", temp_dir, current_timestamp_str)
                    if saved_path: 
                        ui_bias_path_out = saved_path 
                        state_bias_path_out = saved_path 
                        status_messages.append(f"BIAS: 생성 완료: {os.path.basename(ui_bias_path_out)}")
                    else: status_messages.append("BIAS: 생성 실패 (저장 오류).")
                else: status_messages.append("BIAS: ccdproc으로 마스터 BIAS 생성 실패.")
            else: status_messages.append("BIAS: 유효한 파일 경로 없음.")
        except Exception as e: logger_ui.error("BIAS 처리 중 오류", exc_info=True); status_messages.append(f"BIAS 처리 오류: {str(e)}"); master_bias_ccd = None
    else: status_messages.append("BIAS: 업로드된 파일 없음.")

    # 2. Master DARK (노출 시간별 Corrected) 생성
    if dark_file_objs:
        status_messages.append(f"DARK: 총 {len(dark_file_objs)}개 파일 처리 시작 (ccdproc)...")
        grouped_dark_paths_by_exp = {} 
        temp_dark_headers_for_saving = {} 
        for df_obj in dark_file_objs:
            if df_obj and df_obj.name and os.path.exists(df_obj.name):
                try:
                    _, header = load_single_fits_from_path(df_obj.name, "DARK (header check for grouping)")
                    if header:
                        exp_time = get_fits_keyword(header, ['EXPTIME', 'EXPOSURE'], default_value=-1.0, data_type=float)
                        if exp_time > 0:
                            if exp_time not in grouped_dark_paths_by_exp:
                                grouped_dark_paths_by_exp[exp_time] = []
                                temp_dark_headers_for_saving[exp_time] = header 
                            grouped_dark_paths_by_exp[exp_time].append(df_obj.name)
                        else: status_messages.append(f"경고: DARK 파일 '{os.path.basename(df_obj.name)}' 노출시간 정보 부족/유효X.")
                    else: status_messages.append(f"경고: DARK 파일 '{os.path.basename(df_obj.name)}' 헤더 읽기 실패.")
                except Exception as e_head_dark: status_messages.append(f"경고: DARK 파일 '{os.path.basename(df_obj.name)}' 헤더 읽기 오류: {e_head_dark}.")
            else: status_messages.append("유효하지 않은 DARK 파일 객체 발견.")
            
        for exp_time, dark_paths_list in grouped_dark_paths_by_exp.items():
            if not dark_paths_list: continue
            try:
                status_messages.append(f"Master DARK (Exp: {exp_time}s): {len(dark_paths_list)}개 파일로 생성 시작...")
                master_dark_corrected_ccd = create_master_dark_ccdproc(dark_paths_list, master_bias_ccd) # BIAS 보정된 DARK
                if master_dark_corrected_ccd:
                    current_dark_header = master_dark_corrected_ccd.header if master_dark_corrected_ccd.header else temp_dark_headers_for_saving.get(exp_time, fits.Header())
                    base_fn = f"master_dark_exp{exp_time:.2f}s_ccdproc".replace('.', '_')
                    saved_path = save_fits_image(master_dark_corrected_ccd, current_dark_header, base_fn, temp_dir, current_timestamp_str)
                    if saved_path:
                        state_darks_corrected_dict_out[exp_time] = saved_path
                        ui_darks_paths_out.append(saved_path) 
                        status_messages.append(f"Master DARK (Exp {exp_time}s): {os.path.basename(saved_path)} 생성 완료 (경로: {saved_path})")
                    else: status_messages.append(f"Master DARK (Exp: {exp_time}s): 생성 실패 (저장 오류).")
                else: status_messages.append(f"Master DARK (Exp: {exp_time}s): ccdproc 생성 실패.")
            except Exception as e_md: logger_ui.error(f"Master DARK (Exp: {exp_time}s) 처리 중 오류", exc_info=True); status_messages.append(f"Master DARK (Exp: {exp_time}s) 오류: {str(e_md)}")
        if not ui_darks_paths_out: status_messages.append("유효한 Master Dark 생성 실패 또는 처리할 파일 없음.")
    else: status_messages.append("DARK: 업로드된 파일 없음.")

    # 3. 예비 Master FLAT (필터별) 생성 - RAW FLAT 결합
    if flat_file_objs_all:
        status_messages.append(f"예비 FLAT: 총 {len(flat_file_objs_all)}개 파일 처리 시작 (ccdproc)...")
        flat_files_info_grouped_by_filter = {} 
        for ff_obj in flat_file_objs_all:
            if ff_obj and ff_obj.name and os.path.exists(ff_obj.name):
                try:
                    _, header = load_single_fits_from_path(ff_obj.name, "FLAT (header check for grouping)")
                    if header:
                        filter_val = get_fits_keyword(header, ['FILTER'], 'Generic').upper()
                        if filter_val not in flat_files_info_grouped_by_filter:
                            flat_files_info_grouped_by_filter[filter_val] = {'paths': [], 'header': header}
                        flat_files_info_grouped_by_filter[filter_val]['paths'].append(ff_obj.name)
                    else: status_messages.append(f"경고: FLAT 파일 '{os.path.basename(ff_obj.name)}' 헤더 읽기 실패.")
                except Exception as e_head_flat: status_messages.append(f"경고: FLAT 파일 '{os.path.basename(ff_obj.name)}' 헤더 읽기 오류: {e_head_flat}.")
            else: status_messages.append("유효하지 않은 FLAT 파일 객체 발견.")

        for filter_name, info_dict in flat_files_info_grouped_by_filter.items():
            flat_paths_list = info_dict['paths']
            first_header_in_group = info_dict['header']
            if not flat_paths_list: continue
            try:
                status_messages.append(f"예비 Master FLAT ({filter_name}): {len(flat_paths_list)}개 파일로 생성 시작...")
                prelim_master_flat_ccd = create_preliminary_master_flat_ccdproc(flat_paths_list) 
                if prelim_master_flat_ccd:
                    current_flat_header = prelim_master_flat_ccd.header if prelim_master_flat_ccd.header else first_header_in_group
                    base_fn = f"prelim_master_flat_{filter_name}_raw_combined" 
                    saved_path = save_fits_image(prelim_master_flat_ccd, current_flat_header, base_fn, temp_dir, current_timestamp_str)
                    if saved_path:
                        state_prelim_flats_dict_out[filter_name] = saved_path 
                        ui_flats_paths_out.append(saved_path)
                        status_messages.append(f"예비 Master FLAT ({filter_name}): {os.path.basename(saved_path)} 생성 완료 (경로: {saved_path})")
                    else: status_messages.append(f"예비 Master FLAT ({filter_name}): 생성 실패 (저장 오류).")
                else: status_messages.append(f"예비 Master FLAT ({filter_name}): ccdproc 생성 실패.")
            except Exception as e_mf_grp: logger_ui.error(f"예비 Master FLAT ({filter_name}) 처리 중 오류", exc_info=True); status_messages.append(f"예비 Master FLAT ({filter_name}) 오류: {str(e_mf_grp)}")
        if not ui_flats_paths_out: status_messages.append("유효한 예비 Master Flat 생성 실패 또는 처리할 파일 없음.")
    else: status_messages.append("FLAT: 업로드된 파일 없음.")
        
    all_masters_created_successfully = True 
    if not state_bias_path_out and bias_file_objs: all_masters_created_successfully = False
    if not state_darks_corrected_dict_out and dark_file_objs: all_masters_created_successfully = False
    if not state_prelim_flats_dict_out and flat_file_objs_all: all_masters_created_successfully = False
    
    if all_masters_created_successfully and (bias_file_objs or dark_file_objs or flat_file_objs_all):
        status_messages.append("모든 마스터 프레임 생성 성공. 업로드된 원본 임시 파일 삭제 시도...")
        files_to_delete_paths = []
        if bias_file_objs: files_to_delete_paths.extend([f.name for f in bias_file_objs if f and f.name])
        if dark_file_objs: files_to_delete_paths.extend([f.name for f in dark_file_objs if f and f.name])
        if flat_file_objs_all: files_to_delete_paths.extend([f.name for f in flat_file_objs_all if f and f.name])
        
        deleted_count = 0
        failed_to_delete_count = 0
        for f_path in set(files_to_delete_paths): 
            try:
                if os.path.exists(f_path):
                    os.remove(f_path)
                    logger_ui.info(f"임시 업로드 파일 삭제 성공: {f_path}")
                    deleted_count += 1
                else:
                    logger_ui.warning(f"임시 업로드 파일 경로를 찾을 수 없음: {f_path}")
            except Exception as e_del:
                logger_ui.error(f"임시 업로드 파일 삭제 실패 {f_path}: {e_del}")
                failed_to_delete_count += 1
        status_messages.append(f"임시 업로드 파일 삭제 완료: 성공 {deleted_count}개, 실패 {failed_to_delete_count}개.")
    elif (bias_file_objs or dark_file_objs or flat_file_objs_all): 
        status_messages.append("일부 마스터 프레임 생성 실패로 업로드된 원본 임시 파일을 삭제하지 않습니다.")

    final_status = "\n".join(status_messages)
    logger_ui.info("Tab 1: Master frame generation finished.")
    
    return ui_bias_path_out, ui_darks_paths_out, ui_flats_paths_out, \
           state_bias_path_out, state_darks_corrected_dict_out, state_prelim_flats_dict_out, \
           final_status


def handle_tab2_light_frame_calibration(
    light_file_objs_list, 
    tab2_uploaded_bias_obj, tab2_uploaded_dark_raw_files, 
    tab2_uploaded_flat_b_obj, tab2_uploaded_flat_v_obj, 
    state_mb_p, state_md_dict_corr, state_prelim_mf_dict, 
    preview_stretch_type, preview_asinh_a,
    temp_dir):
    """
    탭 2: LIGHT 프레임 보정 핸들러.
    NumPy 기반 수동 보정 로직을 사용합니다.
    FLAT은 예비 플랫을 로드 후, LIGHT 프레임의 노출시간에 맞는 DARK로 실시간 보정하여 사용합니다.
    """
    status_messages = []
    calibrated_light_file_paths_for_ui = []
    output_preview_pil_image = None 
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # 1. 최종 사용할 Master BIAS 결정 (NumPy 배열)
    final_mb_data = None 
    if tab2_uploaded_bias_obj and tab2_uploaded_bias_obj.name:
        mb_data_temp, _ = load_single_fits_from_path(tab2_uploaded_bias_obj.name, "탭2 업로드 Master BIAS")
        if mb_data_temp is not None: final_mb_data = mb_data_temp
    elif state_mb_p and os.path.exists(state_mb_p):
        mb_data_temp, _ = load_single_fits_from_path(state_mb_p, "탭1 Master BIAS")
        if mb_data_temp is not None: final_mb_data = mb_data_temp
    if final_mb_data is None: status_messages.append("경고: 사용 가능한 Master BIAS 없음. BIAS 보정 생략됨.")
    else: status_messages.append(f"Master BIAS 사용 준비 완료 (소스: {'탭2 업로드' if tab2_uploaded_bias_obj and tab2_uploaded_bias_obj.name and final_mb_data is not None else '탭1 상태' if final_mb_data is not None else '사용 불가'}).")

    # 2. 탭2에서 업로드된 Raw Dark들을 BIAS 보정하여 NumPy 딕셔너리 생성
    tab2_processed_darks_data_dict = {} 
    if tab2_uploaded_dark_raw_files: 
        status_messages.append(f"탭2 업로드된 DARK 파일 {len(tab2_uploaded_dark_raw_files)}개 처리 시작...")
        for dark_file_obj in tab2_uploaded_dark_raw_files:
            if dark_file_obj and dark_file_obj.name:
                raw_md_data, raw_md_header = load_single_fits_from_path(dark_file_obj.name, f"탭2 업로드 Raw DARK ({os.path.basename(dark_file_obj.name)})")
                if raw_md_data is not None and raw_md_header is not None:
                    exp_time = get_fits_keyword(raw_md_header, ['EXPTIME', 'EXPOSURE'], default_value=-1.0, data_type=float)
                    if exp_time > 0:
                        corrected_dark_data = raw_md_data.copy() 
                        if final_mb_data is not None and raw_md_data.shape == final_mb_data.shape:
                            corrected_dark_data = raw_md_data - final_mb_data
                        if exp_time not in tab2_processed_darks_data_dict: 
                            tab2_processed_darks_data_dict[exp_time] = (corrected_dark_data, raw_md_header)
                            status_messages.append(f"탭2 업로드 DARK (Exp: {exp_time}s) 사용 준비 완료 (BIAS 보정됨).")
                        else:
                            status_messages.append(f"경고: 탭2에 동일 노출시간({exp_time}s)의 DARK가 여러 개 업로드됨. 첫 번째 파일만 사용.")
                    else: status_messages.append(f"경고: 탭2 DARK ({os.path.basename(dark_file_obj.name)}) 노출시간 정보 없음.")
                else: status_messages.append(f"탭2 DARK ({os.path.basename(dark_file_obj.name)}) 로드 실패.")
    
    # 3. 탭2에서 업로드된 필터별 "예비" Flat (Raw) - NumPy 딕셔너리
    tab2_uploaded_prelim_flats_dict = {} 
    for filt_char, uploaded_mf_obj in [('B', tab2_uploaded_flat_b_obj), ('V', tab2_uploaded_flat_v_obj)]:
        if uploaded_mf_obj and uploaded_mf_obj.name:
            mf_data_raw, mf_header = load_single_fits_from_path(uploaded_mf_obj.name, f"탭2 업로드 (예비) FLAT {filt_char}")
            if mf_data_raw is not None:
                tab2_uploaded_prelim_flats_dict[filt_char] = (mf_data_raw, mf_header)
                status_messages.append(f"탭2 업로드 예비 Master FLAT {filt_char} 사용 준비 완료 (Raw).")
            else: status_messages.append(f"탭2 업로드 Master FLAT {filt_char} 로드 실패.")


    if not light_file_objs_list: status_messages.append("보정할 LIGHT 프레임 없음."); return [], None, "\n".join(status_messages)
    status_messages.append(f"{len(light_file_objs_list)}개의 LIGHT 프레임 보정을 시작합니다...")
    first_calibrated_image_data_for_preview = None

    for i, light_file_obj in enumerate(light_file_objs_list):
        light_filename = "알 수 없는 파일"; 
        md_to_use_data_light = None; md_to_use_header_light = None
        final_mf_data_for_light = None
        dark_source_msg, flat_source_msg = "미사용", "미사용"
        
        try:
            if light_file_obj is None or not hasattr(light_file_obj, 'name') or light_file_obj.name is None: continue
            light_filename = os.path.basename(light_file_obj.name)
            status_messages.append(f"--- {light_filename} 보정 중 ---")
            light_data_raw_np, light_header = load_single_fits_from_path(light_file_obj.name, f"LIGHT ({light_filename})")
            if light_data_raw_np is None or light_header is None: status_messages.append(f"{light_filename} 로드 실패."); continue
            
            current_light_filter = get_fits_keyword(light_header, ['FILTER'], 'Generic').upper()
            current_light_exptime = get_fits_keyword(light_header, ['EXPTIME', 'EXPOSURE'], -1.0, float)

            # DARK 결정 (LIGHT 프레임용 - NumPy 배열과 헤더)
            if current_light_exptime > 0:
                if current_light_exptime in tab2_processed_darks_data_dict: 
                    md_to_use_data_light, md_to_use_header_light = tab2_processed_darks_data_dict[current_light_exptime]
                    dark_source_msg = f"탭2 업로드 Dark (Exp {current_light_exptime}s, BIAS 보정됨)"
                elif state_md_dict_corr and current_light_exptime in state_md_dict_corr: 
                    dark_path = state_md_dict_corr[current_light_exptime]
                    if dark_path and os.path.exists(dark_path): 
                        d_data, d_hdr = load_single_fits_from_path(dark_path, f"탭1 Dark {current_light_exptime}s (BIAS 보정됨)")
                        if d_data is not None: 
                            md_to_use_data_light = d_data
                            md_to_use_header_light = d_hdr
                        dark_source_msg = f"탭1 Dark ({os.path.basename(dark_path)}, BIAS 보정됨)"
            if md_to_use_data_light is None: status_messages.append(f"경고: {light_filename} (Exp: {current_light_exptime}s)에 맞는 Master DARK 없음. DARK 보정 생략.")
            
            # 예비 FLAT 결정 (NumPy 배열과 헤더)
            prelim_flat_data_to_process = None
            prelim_flat_header_to_process = None
            if current_light_filter in tab2_uploaded_prelim_flats_dict: 
                prelim_flat_data_to_process, prelim_flat_header_to_process = tab2_uploaded_prelim_flats_dict[current_light_filter]
                flat_source_msg = f"탭2 업로드 예비 Flat ({current_light_filter})"
            elif state_prelim_mf_dict: 
                prelim_flat_path = state_prelim_mf_dict.get(current_light_filter)
                if not prelim_flat_path: prelim_flat_path = state_prelim_mf_dict.get('Generic')
                if prelim_flat_path and os.path.exists(prelim_flat_path):
                    pf_data, pf_hdr = load_single_fits_from_path(prelim_flat_path, f"탭1 예비 Flat {current_light_filter or 'Generic'}")
                    if pf_data is not None: 
                        prelim_flat_data_to_process = pf_data
                        prelim_flat_header_to_process = pf_hdr
                    flat_source_msg = f"탭1 예비 Flat ({os.path.basename(prelim_flat_path)})"
            
            # 수동 보정 함수 호출
            calibrated_light_data_np, calib_log_list = manual_ccd_calibration(
                raw_light_data_np=light_data_raw_np,
                raw_light_header=light_header, # LIGHT 헤더 전달
                master_bias_data_np=final_mb_data,
                master_dark_data_np=md_to_use_data_light, # LIGHT 프레임용 DARK
                master_dark_header=md_to_use_header_light, # LIGHT 프레임용 DARK의 헤더
                prelim_flat_data_np=prelim_flat_data_to_process,
                prelim_flat_header=prelim_flat_header_to_process,
                do_bias_correction=True, 
                do_dark_correction=True, 
                do_flat_correction=True  
            )
            status_messages.extend(calib_log_list)
            
            if first_calibrated_image_data_for_preview is None:
                first_calibrated_image_data_for_preview = calibrated_light_data_np
            
            light_header['HISTORY'] = f'Calibrated App v0.17.3 (ManualNumPy) (B:{final_mb_data is not None},D:{md_to_use_data_light is not None},F:{prelim_flat_data_to_process is not None})'
            saved_path = save_fits_image(calibrated_light_data_np, light_header, f"calibrated_manual_{os.path.splitext(light_filename)[0]}", temp_dir, current_timestamp_str)
            if saved_path: 
                calibrated_light_file_paths_for_ui.append(saved_path)
                status_messages.append(f"{light_filename}: 수동 보정 완료: {os.path.basename(saved_path)}")
            else: 
                status_messages.append(f"{light_filename}: 수동 보정 후 저장 실패.")
        except Exception as e: 
            logger_ui.error(f"LIGHT ({light_filename}) 수동 보정 오류", exc_info=True)
            status_messages.append(f"{light_filename} 수동 보정 오류: {str(e)}")

    if first_calibrated_image_data_for_preview is not None:
        try: output_preview_pil_image = create_preview_image(first_calibrated_image_data_for_preview, stretch_type=preview_stretch_type, a_param=preview_asinh_a)
        except Exception as e: logger_ui.error("미리보기 생성 오류", exc_info=True); status_messages.append(f"미리보기 생성 오류: {str(e)}")
    if not calibrated_light_file_paths_for_ui: status_messages.append("성공적으로 보정된 LIGHT 프레임 없음.")
    final_status = "\n".join(status_messages)
    logger_ui.info("Tab 2: Light frame calibration finished."); return calibrated_light_file_paths_for_ui, output_preview_pil_image, final_status


def handle_tab3_extinction_analysis(
    light_file_objs, 
    uploaded_mb_path_obj, uploaded_md_raw_files_objs,
    uploaded_mf_b_raw_path_obj, uploaded_mf_v_raw_path_obj,
    state_mb_p, state_md_dict_corr, state_prelim_mf_dict, 
    star_detection_thresh_factor, # 이 값은 이제 DAOStarFinder의 threshold_nsigma로 사용됨
    temp_dir):
    """
    탭 3: 대기소광계수 분석 핸들러.
    LIGHT 프레임들을 수동 보정하고, 가장 밝은 별의 기기등급과 대기질량을 계산하여
    필터별로 대기소광계수(k)와 영점(m0)을 추정합니다.
    """
    status_log = []
    all_frame_results_for_df_list = [] 
    plot_image_fig = None 
    summary_text = "분석 결과가 없습니다."
    fwhm_for_dao_tab3 = 3.0 # 탭3에서 사용할 DAOStarFinder의 FWHM 값 (UI 입력으로 변경 가능)
    
    # 1. Master BIAS 결정 (NumPy 배열)
    final_mb_data = None
    if uploaded_mb_path_obj and uploaded_mb_path_obj.name:
        mb_data_temp, _ = load_single_fits_from_path(uploaded_mb_path_obj.name, "탭3 업로드 Master BIAS")
        if mb_data_temp is not None: final_mb_data = mb_data_temp
    elif state_mb_p and os.path.exists(state_mb_p):
        mb_data_temp, _ = load_single_fits_from_path(state_mb_p, "탭1 Master BIAS")
        if mb_data_temp is not None: final_mb_data = mb_data_temp
    status_log.append(f"Master BIAS: {'사용' if final_mb_data is not None else '미사용/로드실패'}")

    # 2. 탭3에서 업로드된 Raw Dark들을 BIAS 보정하여 NumPy 딕셔너리 생성
    tab3_processed_darks_data_dict = {} 
    if uploaded_md_raw_files_objs: 
        status_log.append(f"탭3 업로드된 DARK 파일 {len(uploaded_md_raw_files_objs)}개 처리 시작...")
        for dark_file_obj in uploaded_md_raw_files_objs:
            if dark_file_obj and dark_file_obj.name:
                raw_md_data, raw_md_header = load_single_fits_from_path(dark_file_obj.name, f"탭3 업로드 Raw DARK ({os.path.basename(dark_file_obj.name)})")
                if raw_md_data is not None and raw_md_header is not None:
                    exp_time = get_fits_keyword(raw_md_header, ['EXPTIME', 'EXPOSURE'], default_value=-1.0, data_type=float)
                    if exp_time > 0:
                        corrected_dark_data = raw_md_data.copy()
                        if final_mb_data is not None and raw_md_data.shape == final_mb_data.shape:
                            corrected_dark_data = raw_md_data - final_mb_data
                        if exp_time not in tab3_processed_darks_data_dict: 
                            tab3_processed_darks_data_dict[exp_time] = (corrected_dark_data, raw_md_header)
                            status_log.append(f"탭3 업로드 DARK (Exp: {exp_time}s) 사용 준비 완료 (BIAS 보정됨).")
                        else:
                            status_log.append(f"경고: 탭3에 동일 노출시간({exp_time}s)의 DARK가 여러 개 업로드됨. 첫 번째 파일만 사용.")
                    else: status_log.append(f"경고: 탭3 DARK ({os.path.basename(dark_file_obj.name)}) 노출시간 정보 없음.")
                else: status_log.append(f"탭3 DARK ({os.path.basename(dark_file_obj.name)}) 로드 실패.")
    
    # 3. 탭3에서 업로드된 필터별 "예비" Flat (Raw) - NumPy 딕셔너리
    tab3_uploaded_prelim_flats_data_dict = {} 
    for filt_char, uploaded_mf_obj in [('B', uploaded_mf_b_raw_path_obj), ('V', uploaded_mf_v_raw_path_obj)]:
        if uploaded_mf_obj and uploaded_mf_obj.name:
            mf_data_raw, mf_header = load_single_fits_from_path(uploaded_mf_obj.name, f"탭3 업로드 (예비) FLAT {filt_char}")
            if mf_data_raw is not None:
                tab3_uploaded_prelim_flats_data_dict[filt_char] = (mf_data_raw, mf_header)
                status_log.append(f"탭3 업로드 예비 Master FLAT {filt_char} 사용 준비 완료 (Raw).")
            else: status_log.append(f"탭3 업로드 Master FLAT {filt_char} 로드 실패.")

    if not light_file_objs:
        status_log.append("분석할 LIGHT 프레임 없음.")
        df_headers_no_light = ["File", "Filter", "Airmass", "Altitude", "Inst. Mag.", "Flux", "Star X", "Star Y", "Ap. Radius", "Error"]
        df_empty = pd.DataFrame([["LIGHT 파일 없음"]*len(df_headers_no_light)], columns=df_headers_no_light)
        return None, "LIGHT 파일 없음", df_empty, "\n".join(status_log)

    status_log.append(f"--- {len(light_file_objs)}개 LIGHT 프레임 분석 시작 ---")
    
    processed_results_for_analysis = [] 
    for light_file_obj in light_file_objs: 
        if not (light_file_obj and light_file_obj.name and os.path.exists(light_file_obj.name)): continue
        
        light_filename = os.path.basename(light_file_obj.name)
        status_log.append(f"--- {light_filename} 처리 중 ---")
        current_result = {'file': light_filename, 'error_message': None}
        md_to_use_data_light = None; md_to_use_header_light = None
        prelim_flat_data_to_process = None; prelim_flat_header_to_process = None

        try:
            light_data_raw_np, light_header = load_single_fits_from_path(light_file_obj.name, f"LIGHT ({light_filename})")
            if light_data_raw_np is None or light_header is None: raise ValueError("LIGHT 데이터 로드 실패")
            
            current_filter = get_fits_keyword(light_header, ['FILTER'], 'UNKNOWN').upper()
            current_exptime = get_fits_keyword(light_header, ['EXPTIME', 'EXPOSURE'], -1.0, float)
            current_result['filter'] = current_filter

            # DARK 결정 (LIGHT 프레임용 - NumPy)
            if current_exptime > 0:
                if current_exptime in tab3_processed_darks_data_dict: 
                    md_to_use_data_light, md_to_use_header_light = tab3_processed_darks_data_dict[current_exptime]
                elif state_md_dict_corr and current_exptime in state_md_dict_corr:
                    dark_path = state_md_dict_corr[current_exptime]
                    if dark_path and os.path.exists(dark_path): 
                        d_data, d_hdr = load_single_fits_from_path(dark_path, f"탭1 Dark {current_exptime}s (BIAS 보정됨)")
                        if d_data is not None: md_to_use_data_light, md_to_use_header_light = d_data, d_hdr
            if md_to_use_data_light is None: status_log.append(f"경고: {light_filename} (Exp: {current_exptime}s)에 맞는 Master DARK 없음.")
            
            # 예비 FLAT 결정 (NumPy)
            if current_filter in tab3_uploaded_prelim_flats_data_dict:
                prelim_flat_data_to_process, prelim_flat_header_to_process = tab3_uploaded_prelim_flats_data_dict[current_filter]
            elif state_prelim_mf_dict:
                prelim_flat_path = state_prelim_mf_dict.get(current_filter)
                if not prelim_flat_path: prelim_flat_path = state_prelim_mf_dict.get('Generic')
                if prelim_flat_path and os.path.exists(prelim_flat_path):
                    pf_data, pf_hdr = load_single_fits_from_path(prelim_flat_path, f"탭1 예비 Flat {current_filter or 'Generic'}")
                    if pf_data is not None: 
                        prelim_flat_data_to_process = pf_data
                        prelim_flat_header_to_process = pf_hdr
            
            # 수동 보정 함수 호출
            calibrated_data_np, calib_log_list = manual_ccd_calibration(
                raw_light_data_np=light_data_raw_np,
                raw_light_header=light_header,
                master_bias_data_np=final_mb_data,
                master_dark_data_np=md_to_use_data_light,
                master_dark_header=md_to_use_header_light,
                prelim_flat_data_np=prelim_flat_data_to_process,
                prelim_flat_header=prelim_flat_header_to_process,
                do_bias_correction=True, 
                do_dark_correction=True, 
                do_flat_correction=True
            )
            status_log.extend(calib_log_list)
            
            # 별 탐지 및 측광 (DAOStarFinder 사용)
            sources_table = detect_stars_extinction(
                calibrated_data_np, 
                fwhm_dao=fwhm_for_dao_tab3, 
                threshold_nsigma_dao=float(star_detection_thresh_factor)
            )
            brightest = find_brightest_star_extinction(sources_table, fwhm_for_radius_approx=fwhm_for_dao_tab3)
            if brightest is None: raise ValueError("가장 밝은 별 탐지 실패 (DAO)")
            
            flux, ap_rad, _ = calculate_flux_extinction(calibrated_data_np, brightest)
            if flux is None: raise ValueError("Flux 계산 실패")
            current_result.update({'flux': flux, 'star_x': brightest['xcentroid'], 'star_y': brightest['ycentroid'], 'aperture_radius': ap_rad})
            inst_mag = calculate_instrumental_magnitude(flux)
            if inst_mag is None: raise ValueError("기기 등급 계산 실패")
            current_result['instrumental_magnitude'] = inst_mag
            alt = calculate_altitude_extinction(light_header) 
            airmass = calculate_airmass_extinction(light_header) 
            if airmass is None: raise ValueError("대기질량 계산 실패")
            current_result.update({'altitude': alt, 'airmass': airmass})
            status_log.append(f"처리 완료 ({current_result['file']}): F={current_filter}, AM={airmass:.3f}, Mag={inst_mag:.3f}")
            processed_results_for_analysis.append(current_result)
        except Exception as e_proc_tab3:
            logger_ui.error(f"파일 {light_filename} 처리 중 오류 (탭3)", exc_info=True)
            current_result['error_message'] = str(e_proc_tab3); status_log.append(f"오류 ({light_filename}): {str(e_proc_tab3)}")
            processed_results_for_analysis.append(current_result)

    results_b = [r for r in processed_results_for_analysis if r.get('filter') == 'B' and r.get('airmass') is not None and r.get('instrumental_magnitude') is not None and r.get('error_message') is None]
    results_v = [r for r in processed_results_for_analysis if r.get('filter') == 'V' and r.get('airmass') is not None and r.get('instrumental_magnitude') is not None and r.get('error_message') is None]
    summary_lines = []
    slope_b, intercept_b, r_sq_b, model_b = None, None, None, None
    slope_v, intercept_v, r_sq_v, model_v = None, None, None, None

    if len(results_b) >= 2:
        slope_b, intercept_b, r_sq_b, model_b = perform_linear_regression_extinction([r['airmass'] for r in results_b], [r['instrumental_magnitude'] for r in results_b])
        if slope_b is not None: summary_lines.append(f"B 필터: k_B={slope_b:.4f}, m0_B={intercept_b:.4f}, R²={r_sq_b:.4f} ({len(results_b)}개)")
    elif results_b: summary_lines.append(f"B 필터: 데이터 부족 ({len(results_b)}개)으로 회귀 불가.")
    else: summary_lines.append("B 필터: 유효 데이터 없음.")

    if len(results_v) >= 2:
        slope_v, intercept_v, r_sq_v, model_v = perform_linear_regression_extinction([r['airmass'] for r in results_v], [r['instrumental_magnitude'] for r in results_v])
        if slope_v is not None: summary_lines.append(f"V 필터: k_V={slope_v:.4f}, m0_V={intercept_v:.4f}, R²={r_sq_v:.4f} ({len(results_v)}개)")
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
                x_fit_b_plot = np.array([np.min(x_b_arr), np.max(x_b_arr)])
                y_fit_b_plot = model_b.predict(x_fit_b_plot.reshape(-1,1))
                ax.plot(x_fit_b_plot, y_fit_b_plot, color='dodgerblue', ls='--', label=f'B Fit (k={slope_b:.3f}, R²={r_sq_b:.3f})')
            plot_created = True
        if results_v and model_v:
            x_v_arr = np.array([r['airmass'] for r in results_v]); y_v_arr = np.array([r['instrumental_magnitude'] for r in results_v])
            ax.scatter(x_v_arr, y_v_arr, color='green', label=f'V Data ({len(x_v_arr)})', alpha=0.7, edgecolor='k')
            if len(x_v_arr) > 0:
                x_fit_v_plot = np.array([np.min(x_v_arr), np.max(x_v_arr)])
                y_fit_v_plot = model_v.predict(x_fit_v_plot.reshape(-1,1))
                ax.plot(x_fit_v_plot, y_fit_v_plot, color='forestgreen', ls='--', label=f'V Fit (k={slope_v:.3f}, R²={r_sq_v:.3f})')
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
        all_frame_results_for_df_list.append([
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
    
    df_results = pd.DataFrame(
        all_frame_results_for_df_list or [["결과 없음"]*len(df_headers)],
        columns=df_headers
    )
    final_log = "\n".join(status_log)
    logger_ui.info("대기소광계수 분석 완료.")
    return plot_image_fig, summary_text, df_results, final_log


def handle_tab4_detailed_photometry(
    light_b_file_objs, light_v_file_objs,
    std_star_b_file_obj, std_star_v_file_obj,
    std_b_mag_known_input, std_v_mag_known_input,
    # 아래 마스터 프레임 관련 파라미터들 제거됨
    # tab4_uploaded_mb_obj, tab4_uploaded_md_raw_obj, 
    # tab4_uploaded_mf_b_raw_obj, tab4_uploaded_mf_v_raw_obj,
    # state_mb_p, state_md_dict_corr, 
    # state_prelim_mf_dict, 
    k_b_input, m0_b_input_user, k_v_input, m0_v_input_user, 
    dao_fwhm_input, dao_thresh_nsigma_input, phot_aperture_radius_input, 
    roi_x_min, roi_x_max, roi_y_min, roi_y_max,
    simbad_query_radius_arcsec, 
    temp_dir):
    """
    탭 4: 상세 측광 및 카탈로그 매칭 핸들러.
    사용자가 이미 보정한 LIGHT 프레임과 표준별 프레임을 업로드하면,
    이를 사용하여 측광, 표준 등급 계산, 카탈로그 매칭을 수행합니다.
    내부적인 BIAS, DARK, FLAT 보정 과정은 생략됩니다.
    """
    status_log = []
    all_stars_final_data_for_df = [] 
    csv_output_path = None
    photometry_preview_image_pil = None 
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger_ui.info("탭 4 상세 측광 분석 시작 (보정된 파일 사용)...")

    # --- 1. 입력값 유효성 검사 (동일) ---
    if not light_b_file_objs and not light_v_file_objs:
        status_log.append("오류: B 또는 V 필터 LIGHT 프레임을 하나 이상 업로드해야 합니다.")
        df_error_headers = ["Error Message"] # DataFrame 반환을 위해 헤더 정의
        df_error_data = [["LIGHT 프레임 없음"]]
        return pd.DataFrame(df_error_data, columns=df_error_headers), None, None, "\n".join(status_log)
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
        df_error_headers = ["Error Message"]
        df_error_data = [["입력 파라미터 오류"]]
        return pd.DataFrame(df_error_data, columns=df_error_headers), None, None, "\n".join(status_log)

    # --- 2. 마스터 프레임 준비 로직 제거 ---
    # 사용자가 보정된 파일을 업로드하므로, 이 부분은 필요 없음.
    status_log.append("탭 4: 사용자가 보정한 파일을 직접 사용합니다. 내부 보정 과정은 생략됩니다.")

    # --- 3. 표준별 처리로 유효 영점(m0_eff) 계산 ---
    m0_eff_b, m0_eff_v = m0_b_user_val, m0_v_user_val 
    status_log.append(f"초기 영점: m0_B={m0_eff_b:.3f}, m0_V={m0_eff_v:.3f} (사용자 입력 또는 기본값)")

    for std_filt_char, std_file_obj, std_mag_known_in, k_coeff_std_val, m0_eff_var_name_str in [
        ('B', std_star_b_file_obj, std_b_mag_known_input, k_b, 'm0_eff_b'),
        ('V', std_star_v_file_obj, std_v_mag_known_input, k_v, 'm0_eff_v')
    ]:
        if std_file_obj and hasattr(std_file_obj, 'name') and std_file_obj.name:
            status_log.append(f"--- {std_filt_char}필터 표준별 처리 (보정된 파일 사용): {os.path.basename(std_file_obj.name)} ---")
            # 업로드된 표준별 파일을 이미 보정된 것으로 간주하고 바로 로드
            cal_std_data, std_header = load_single_fits_from_path(std_file_obj.name, f"보정된 {std_filt_char} 표준별")
            
            if cal_std_data is not None and std_header is not None:
                std_stars_table = detect_stars_dao(cal_std_data, fwhm, thresh_nsigma)
                if std_stars_table and len(std_stars_table) > 0:
                    if 'flux' in std_stars_table.colnames : std_stars_table.sort('flux', reverse=True)
                    brightest_std_star_photutils = std_stars_table[0]
                    std_phot_table = perform_aperture_photometry_on_detections(cal_std_data, Table([brightest_std_star_photutils]), ap_radius_phot)
                    if std_phot_table and 'net_flux' in std_phot_table.colnames and len(std_phot_table) > 0:
                        m_inst_std = calculate_instrumental_magnitude(std_phot_table['net_flux'][0])
                        x_std = calculate_airmass_extinction(std_header)
                        m_std_known_val = np.nan
                        if std_mag_known_in is not None:
                            try: m_std_known_val = float(std_mag_known_in)
                            except: status_log.append(f"{std_filt_char} 표준 등급 입력값 유효X.")
                        
                        if not np.isfinite(m_std_known_val):
                            std_ra, std_dec = convert_pixel_to_wcs(std_phot_table['xcentroid'][0], std_phot_table['ycentroid'][0], std_header)
                            if np.isfinite(std_ra): # WCS 변환 성공 시
                                simbad_id_std = query_simbad_for_object(std_ra, std_dec, 2.0)
                                status_log.append(f"{std_filt_char} 표준별 SIMBAD: {simbad_id_std} (참고용, 등급 자동 추출 미지원)")
                        
                        if np.isfinite(m_std_known_val) and m_inst_std is not None and x_std is not None and k_coeff_std_val is not None:
                            calc_m0 = m_std_known_val - (m_inst_std - k_coeff_std_val * x_std) 
                            if m0_eff_var_name_str == 'm0_eff_b': m0_eff_b = calc_m0
                            elif m0_eff_var_name_str == 'm0_eff_v': m0_eff_v = calc_m0
                            status_log.append(f"{std_filt_char}필터 영점(m0_eff) 계산됨: {calc_m0:.3f} (표준별 사용)")
                        else: 
                            status_log.append(f"{std_filt_char}필터 표준별 정보 부족으로 영점 자동 계산 불가. 사용자 입력 m0 사용.")
                            if m0_eff_var_name_str == 'm0_eff_b': m0_eff_b = m0_b_user_val
                            elif m0_eff_var_name_str == 'm0_eff_v': m0_eff_v = m0_v_user_val
                    else: status_log.append(f"{std_filt_char}필터 표준별 측광 실패.")
                else: status_log.append(f"{std_filt_char}필터 표준별 이미지에서 별 탐지 실패.")
            else: status_log.append(f"{std_filt_char}필터 표준별 파일 로드 실패.")
        else: 
            status_log.append(f"{std_filt_char}필터 표준별 파일 미업로드. 사용자 입력 m0 사용.")
            if m0_eff_var_name_str == 'm0_eff_b': m0_eff_b = m0_b_user_val
            elif m0_eff_var_name_str == 'm0_eff_v': m0_eff_v = m0_v_user_val

    # --- 4. 대상 LIGHT 프레임 처리 ---
    filter_processed_stars_data = {'B': [], 'V': []} 
    first_light_b_calibrated_data_for_preview = None 
    first_light_v_calibrated_data_for_preview = None

    for filter_char_loop, light_objs_loop, k_coeff_loop, m0_eff_loop_val in [ 
        ('B', light_b_file_objs, k_b, m0_eff_b), 
        ('V', light_v_file_objs, k_v, m0_eff_v)
    ]:
        if not light_objs_loop: continue
        status_log.append(f"--- {filter_char_loop} 필터 대상 프레임 처리 시작 ({len(light_objs_loop)}개) ---")
        for i_light, light_obj_item in enumerate(light_objs_loop): 
            if not (light_obj_item and light_obj_item.name and os.path.exists(light_obj_item.name)): continue
            filename_loop = os.path.basename(light_obj_item.name)
            status_log.append(f"처리 중 (보정된 파일 사용): {filename_loop} ({filter_char_loop})")
            try:
                # 업로드된 파일을 이미 보정된 것으로 간주하고 바로 로드
                calibrated_data_target, header = load_single_fits_from_path(light_obj_item.name, f"보정된 {filter_char_loop} LIGHT")
                if calibrated_data_target is None or header is None: 
                    status_log.append(f"오류: {filename_loop} 로드 실패."); continue
                
                if i_light == 0: 
                    if filter_char_loop == 'B': first_light_b_calibrated_data_for_preview = calibrated_data_target
                    elif filter_char_loop == 'V': first_light_v_calibrated_data_for_preview = calibrated_data_target

                detected_stars_table = detect_stars_dao(calibrated_data_target, fwhm, thresh_nsigma)
                if detected_stars_table is None or len(detected_stars_table) == 0: status_log.append(f"{filename_loop}: 별 탐지 실패."); continue
                
                phot_input_table = detected_stars_table
                if use_roi: 
                    x_dao, y_dao = detected_stars_table['xcentroid'], detected_stars_table['ycentroid']
                    roi_m = (x_dao >= roi_x0) & (x_dao <= roi_x1) & (y_dao >= roi_y0) & (y_dao <= roi_y1)
                    stars_in_roi_table = detected_stars_table[roi_m]
                    if not len(stars_in_roi_table)>0: status_log.append(f"{filename_loop}: ROI 내 별 없음."); continue 
                    status_log.append(f"{filename_loop}: {len(stars_in_roi_table)}개 별이 ROI 내에 있음.")
                    phot_input_table = stars_in_roi_table
                
                phot_results_table = perform_aperture_photometry_on_detections(calibrated_data_target, phot_input_table, ap_radius_phot)
                if phot_results_table is None or 'net_flux' not in phot_results_table.colnames: status_log.append(f"{filename_loop}: 측광 실패."); continue
                
                ras, decs = convert_pixel_to_wcs(phot_results_table['xcentroid'], phot_results_table['ycentroid'], header)
                airmass_val = calculate_airmass_extinction(header)
                
                for star_idx_loop, star_phot_info in enumerate(phot_results_table):
                    inst_flux = star_phot_info['net_flux']
                    inst_mag = calculate_instrumental_magnitude(inst_flux)
                    std_mag_val = calculate_standard_magnitude(inst_mag, airmass_val, k_coeff_loop, m0_eff_loop_val) if inst_mag is not None and airmass_val is not None else np.nan
                    
                    filter_processed_stars_data[filter_char_loop].append({
                        'file': filename_loop, 'filter': filter_char_loop,
                        'x': star_phot_info['xcentroid'], 'y': star_phot_info['ycentroid'],
                        'ra_deg': ras[star_idx_loop] if ras is not None and star_idx_loop < len(ras) else np.nan, 
                        'dec_deg': decs[star_idx_loop] if decs is not None and star_idx_loop < len(decs) else np.nan,
                        'flux': inst_flux, 'inst_mag': inst_mag, 'std_mag': std_mag_val, 
                        'airmass': airmass_val, 'header': header 
                    })
                status_log.append(f"{filename_loop}: {len(phot_results_table)}개 별 처리 완료 (ROI 적용됨).")
            except Exception as e_frame_tab4_proc:
                logger_ui.error(f"{filename_loop} 처리 중 오류 (탭4)", exc_info=True)
                status_log.append(f"오류 ({filename_loop}): {str(e_frame_tab4_proc)}")

    # --- 5. 별 정보 통합, B-V 계산, SIMBAD 질의, 정렬 (이전과 동일) ---
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

    preview_base_data = first_light_b_calibrated_data_for_preview if first_light_b_calibrated_data_for_preview is not None else first_light_v_calibrated_data_for_preview
    if preview_base_data is not None:
        base_pil = create_preview_image(preview_base_data)
        if base_pil:
            stars_for_preview_drawing = []
            preview_filename = None 
            if first_light_b_calibrated_data_for_preview is not None and light_b_file_objs: preview_filename = os.path.basename(light_b_file_objs[0].name)
            elif first_light_v_calibrated_data_for_preview is not None and light_v_file_objs: preview_filename = os.path.basename(light_v_file_objs[0].name)

            if preview_filename:
                for star_info_item in final_display_list: 
                    if star_info_item.get('file') == preview_filename:
                        stars_for_preview_drawing.append({
                            'x': star_info_item.get('x'), 
                            'y': star_info_item.get('y'),
                            'mag_display': star_info_item.get('std_mag') if star_info_item.get('filter') != 'V_only' else star_info_item.get('mag_std_v'),
                            'id_text': star_info_item.get('simbad_id', '')
                        })
            
            roi_params_for_draw = (roi_x0, roi_x1, roi_y0, roi_y1) if use_roi else None
            photometry_preview_image_pil = draw_photometry_results_on_image(base_pil, stars_for_preview_drawing, roi_coords=roi_params_for_draw)
            if photometry_preview_image_pil: status_log.append("측광 결과 미리보기 이미지 생성 완료.")
            else: status_log.append("측광 결과 미리보기 이미지 생성 실패.")
    else:
        status_log.append("측광 결과 미리보기용 기본 이미지 없음.")


    # --- 6. DataFrame 및 CSV용 데이터 준비 ---
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

    df_results_tab4 = pd.DataFrame(
        all_stars_final_data_for_df or [["결과 없음"]*len(df_headers)], 
        columns=df_headers
    )

    if all_stars_final_data_for_df:
        csv_filename = f"detailed_photometry_results_{current_timestamp_str}.csv"
        csv_output_path = os.path.join(temp_dir, csv_filename)
        try:
            df_results_tab4.to_csv(csv_output_path, index=False, encoding='utf-8-sig') 
            status_log.append(f"결과 CSV 저장: {csv_filename}")
        except Exception as e_csv:
            logger_ui.error("CSV 파일 저장 오류", exc_info=True); status_log.append(f"CSV 저장 오류: {str(e_csv)}"); csv_output_path = None
            
    final_log = "\n".join(status_log)
    logger_ui.info("탭4: 상세 측광 분석 완료.")
    return df_results_tab4, csv_output_path, photometry_preview_image_pil, final_log


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

def handle_tab5_plot_hr_diagram(csv_file_obj, temp_dir):
    status_log = []
    hr_plot_image = None
    
    if csv_file_obj is None or not hasattr(csv_file_obj, 'name') or csv_file_obj.name is None:
        status_log.append("오류: CSV 파일을 업로드하세요.")
        return None, "\n".join(status_log)

    csv_file_path = csv_file_obj.name
    status_log.append(f"CSV 파일 로드 시도: {os.path.basename(csv_file_path)}")

    try:
        df = pd.read_csv(csv_file_path)
        status_log.append(f"{len(df)}개의 행을 CSV에서 읽었습니다.")

        required_cols = ['StdMag V', 'B-V', 'SIMBAD ID'] 
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            status_log.append(f"오류: CSV 파일에 필요한 컬럼이 없습니다 - {', '.join(missing_cols)}")
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "필요한 데이터 컬럼 없음", ha='center', va='center'); hr_plot_image = fig
            plt.close(fig)
            return hr_plot_image, "\n".join(status_log)

        for col in ['StdMag V', 'B-V']:
            if df[col].dtype == 'object': 
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        valid_data = df[df['StdMag V'].notna() & df['B-V'].notna()].copy() 

        if valid_data.empty:
            status_log.append("경고: H-R도를 그릴 유효한 데이터(V등급, B-V 색지수)가 없습니다.")
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "유효 데이터 없음", ha='center', va='center'); hr_plot_image = fig
            plt.close(fig)
            return hr_plot_image, "\n".join(status_log)

        status_log.append(f"{len(valid_data)}개의 유효한 별 데이터로 H-R도 생성 중...")

        v_mag = valid_data['StdMag V'].astype(float)
        bv_color = valid_data['B-V'].astype(float)
        simbad_ids = valid_data['SIMBAD ID']

        plt.style.use('seaborn-v0_8-darkgrid') 
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.get_cmap('RdYlBu_r') 
        
        bv_min, bv_max = bv_color.min(), bv_color.max()
        if bv_min == bv_max: 
            normalized_bv = np.full_like(bv_color, 0.5) 
        else:
            normalized_bv = (bv_color - bv_min) / (bv_max - bv_min)
        
        scatter = ax.scatter(bv_color, v_mag, c=normalized_bv, cmap=cmap, s=50, alpha=0.8, edgecolors='k', linewidths=0.5)
        
        for i in range(len(valid_data)):
            sid = simbad_ids.iloc[i]
            if pd.notna(sid) and sid not in ["N/A", "WCS 없음", "SIMBAD 오류", "좌표 없음"]:
                ax.text(bv_color.iloc[i] + 0.01, v_mag.iloc[i], sid.split('(')[0].strip(), fontsize=7, alpha=0.7) 

        ax.set_xlabel('B-V Color Index')
        ax.set_ylabel('V Standard Magnitude')
        ax.set_title('H-R Diagram (Color-Magnitude Diagram)')
        ax.invert_yaxis() 
        
        cbar = fig.colorbar(scatter, ax=ax, label='Normalized B-V Color Index (Blue to Red)')
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        hr_plot_image = fig
        status_log.append("H-R도(색-등급도) 생성 완료.")
        # plt.close(fig) # Gradio가 Figure 객체를 직접 처리

    except pd.errors.EmptyDataError:
        status_log.append("오류: CSV 파일이 비어있거나 읽을 수 없습니다.")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "CSV 파일 비어있음", ha='center', va='center'); hr_plot_image = fig
        plt.close(fig)
    except KeyError as ke:
        status_log.append(f"오류: CSV 파일에 필요한 컬럼이 없습니다 - {ke}")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"필요 컬럼 없음: {ke}", ha='center', va='center'); hr_plot_image = fig
        plt.close(fig)
    except Exception as e:
        logger_ui.error("H-R도 생성 중 오류", exc_info=True)
        status_log.append(f"H-R도 생성 중 오류: {str(e)}")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"오류: {e}", ha='center', va='center', color='red'); hr_plot_image = fig
        plt.close(fig)
        
    return hr_plot_image, "\n".join(status_log)

def handle_tab5_plot_hr_diagram(csv_file_obj, temp_dir):
    status_log = []
    hr_plot_image = None
    
    if csv_file_obj is None or not hasattr(csv_file_obj, 'name') or csv_file_obj.name is None:
        status_log.append("오류: CSV 파일을 업로드하세요.")
        return None, "\n".join(status_log)

    csv_file_path = csv_file_obj.name
    status_log.append(f"CSV 파일 로드 시도: {os.path.basename(csv_file_path)}")

    try:
        df = pd.read_csv(csv_file_path)
        status_log.append(f"{len(df)}개의 행을 CSV에서 읽었습니다.")

        required_cols = ['StdMag V', 'B-V', 'SIMBAD ID'] 
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            status_log.append(f"오류: CSV 파일에 필요한 컬럼이 없습니다 - {', '.join(missing_cols)}")
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "필요한 데이터 컬럼 없음", ha='center', va='center'); hr_plot_image = fig
            plt.close(fig)
            return hr_plot_image, "\n".join(status_log)

        for col in ['StdMag V', 'B-V']:
            if df[col].dtype == 'object': 
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        valid_data = df[df['StdMag V'].notna() & df['B-V'].notna()].copy() 

        if valid_data.empty:
            status_log.append("경고: H-R도를 그릴 유효한 데이터(V등급, B-V 색지수)가 없습니다.")
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "유효 데이터 없음", ha='center', va='center'); hr_plot_image = fig
            plt.close(fig)
            return hr_plot_image, "\n".join(status_log)

        status_log.append(f"{len(valid_data)}개의 유효한 별 데이터로 H-R도 생성 중...")

        v_mag = valid_data['StdMag V'].astype(float)
        bv_color = valid_data['B-V'].astype(float)
        simbad_ids = valid_data['SIMBAD ID']

        plt.style.use('seaborn-v0_8-darkgrid') 
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.get_cmap('RdYlBu_r') 
        
        bv_min, bv_max = bv_color.min(), bv_color.max()
        if bv_min == bv_max: 
            normalized_bv = np.full_like(bv_color, 0.5) 
        else:
            normalized_bv = (bv_color - bv_min) / (bv_max - bv_min)
        
        scatter = ax.scatter(bv_color, v_mag, c=normalized_bv, cmap=cmap, s=50, alpha=0.8, edgecolors='k', linewidths=0.5)
        
        for i in range(len(valid_data)):
            sid = simbad_ids.iloc[i]
            if pd.notna(sid) and sid not in ["N/A", "WCS 없음", "SIMBAD 오류", "좌표 없음"]:
                ax.text(bv_color.iloc[i] + 0.01, v_mag.iloc[i], sid.split('(')[0].strip(), fontsize=7, alpha=0.7) 

        ax.set_xlabel('B-V Color Index')
        ax.set_ylabel('V Standard Magnitude')
        ax.set_title('H-R Diagram (Color-Magnitude Diagram)')
        ax.invert_yaxis() 
        
        cbar = fig.colorbar(scatter, ax=ax, label='Normalized B-V Color Index (Blue to Red)')
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        hr_plot_image = fig
        status_log.append("H-R도(색-등급도) 생성 완료.")
        # plt.close(fig) # Gradio가 Figure 객체를 직접 처리

    except pd.errors.EmptyDataError:
        status_log.append("오류: CSV 파일이 비어있거나 읽을 수 없습니다.")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "CSV 파일 비어있음", ha='center', va='center'); hr_plot_image = fig
        plt.close(fig)
    except KeyError as ke:
        status_log.append(f"오류: CSV 파일에 필요한 컬럼이 없습니다 - {ke}")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"필요 컬럼 없음: {ke}", ha='center', va='center'); hr_plot_image = fig
        plt.close(fig)
    except Exception as e:
        logger_ui.error("H-R도 생성 중 오류", exc_info=True)
        status_log.append(f"H-R도 생성 중 오류: {str(e)}")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"오류: {e}", ha='center', va='center', color='red'); hr_plot_image = fig
        plt.close(fig)
        
    return hr_plot_image, "\n".join(status_log)