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


from astropy.io import fits 
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from PIL import Image 
from astropy.table import Table 
from astropy.nddata import CCDData # CCDData 사용
import ccdproc as ccdp # ccdproc 사용

from utils.fits import load_single_fits_from_path, save_fits_image, create_preview_image, draw_roi_on_pil_image, get_fits_keyword
from utils.calibration import create_master_bias_ccdproc, create_master_dark_ccdproc, create_master_flat_ccdproc # ccdproc 함수로 변경
from utils.photometry import detect_stars_extinction, find_brightest_star_extinction, calculate_flux_extinction, detect_stars_dao, perform_aperture_photometry_on_detections
from utils.astro import (
    calculate_altitude_extinction, calculate_airmass_extinction, 
    calculate_instrumental_magnitude, perform_linear_regression_extinction,
    convert_pixel_to_wcs, calculate_standard_magnitude, query_simbad_for_object,
    match_stars_by_coords
)


logger_ui = logging.getLogger(__name__)
def handle_tab1_master_frame_creation(bias_file_objs, dark_file_objs, flat_file_objs_all, temp_dir):
    status_messages = []
    # ... (UI 변수 및 상태 변수 초기화는 이전과 유사) ...
    ui_bias_path = None
    ui_dark_output_msg = "생성된 Master Dark 없음"
    ui_flat_b_output_msg = "생성된 Master Flat B 없음"
    ui_flat_v_output_msg = "생성된 Master Flat V 없음"
    ui_flat_generic_output_msg = "생성된 Master Flat Generic 없음"
    
    state_bias_path_out = None
    state_darks_corrected_dict_out = {} 
    state_flats_corrected_dict_out = {} 
    
    master_bias_ccd = None # CCDData 객체로 저장
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # 1. Master BIAS 생성 (ccdproc 사용)
    if bias_file_objs:
        try:
            status_messages.append(f"BIAS: {len(bias_file_objs)}개 파일 처리 시작 (ccdproc)...")
            bias_file_paths = [f.name for f in bias_file_objs if f and f.name]
            if bias_file_paths:
                master_bias_ccd = create_master_bias_ccdproc(bias_file_paths) # CCDData 객체 반환
                if master_bias_ccd:
                    bias_header = master_bias_ccd.header if master_bias_ccd.header else fits.Header()
                    saved_path = save_fits_image(master_bias_ccd, bias_header, "master_bias_ccdproc", temp_dir, current_timestamp_str)
                    if saved_path: ui_bias_path = state_bias_path_out = saved_path; status_messages.append(f"BIAS: 생성 완료: {os.path.basename(ui_bias_path)}")
                    else: status_messages.append("BIAS: 생성 실패 (저장 오류).")
                else: status_messages.append("BIAS: ccdproc으로 마스터 BIAS 생성 실패.")
            else: status_messages.append("BIAS: 유효한 파일 경로 없음.")
        except Exception as e: logger_ui.error("BIAS 처리 중 오류", exc_info=True); status_messages.append(f"BIAS 처리 오류: {str(e)}"); master_bias_ccd = None
    else: status_messages.append("BIAS: 업로드된 파일 없음.")

    # 2. Master DARK (노출 시간별 Corrected) 생성 (ccdproc 사용)
    if dark_file_objs:
        status_messages.append(f"DARK: 총 {len(dark_file_objs)}개 파일 처리 시작 (ccdproc)...")
        
        # 헤더를 먼저 읽어 노출시간별로 파일 경로 그룹화
        grouped_dark_paths_by_exp = {} # key: exp_time, value: list of file_paths
        temp_dark_headers_for_saving = {} # key: exp_time, value: 첫번째 헤더 (저장용)

        for df_obj in dark_file_objs:
            if df_obj and df_obj.name and os.path.exists(df_obj.name):
                try:
                    _, header = load_single_fits_from_path(df_obj.name, "DARK (header check for grouping)")
                    if header:
                        exp_time = get_fits_keyword(header, ['EXPTIME', 'EXPOSURE'], default_value=-1.0, data_type=float)
                        if exp_time > 0:
                            if exp_time not in grouped_dark_paths_by_exp:
                                grouped_dark_paths_by_exp[exp_time] = []
                                temp_dark_headers_for_saving[exp_time] = header # 대표 헤더로 저장
                            grouped_dark_paths_by_exp[exp_time].append(df_obj.name)
                        else: status_messages.append(f"경고: DARK 파일 '{os.path.basename(df_obj.name)}' 노출시간 정보 부족/유효X. 건너<0xEB><0x8F><0xEB><0x82><0xB4>니다.")
                    else: status_messages.append(f"경고: DARK 파일 '{os.path.basename(df_obj.name)}' 헤더 읽기 실패. 건너<0xEB><0x8F><0xEB><0x82><0xB4>니다.")
                except Exception as e_head_dark: status_messages.append(f"경고: DARK 파일 '{os.path.basename(df_obj.name)}' 헤더 읽기 오류: {e_head_dark}. 건너<0xEB><0x8F><0xEB><0x82><0xB4>니다.")
            else: status_messages.append("유효하지 않은 DARK 파일 객체 발견. 건너<0xEB><0x8F><0xEB><0x82><0xB4>니다.")
            
        created_darks_info = []
        for exp_time, dark_paths_list in grouped_dark_paths_by_exp.items():
            if not dark_paths_list: continue
            try:
                status_messages.append(f"Master DARK (Exp: {exp_time}s): {len(dark_paths_list)}개 파일로 생성 시작...")
                # ccdproc.combine은 파일 경로 리스트를 직접 받음
                master_dark_corrected_ccd = create_master_dark_ccdproc(dark_paths_list, master_bias_ccd) 
                if master_dark_corrected_ccd:
                    current_dark_header = master_dark_corrected_ccd.header if master_dark_corrected_ccd.header else temp_dark_headers_for_saving.get(exp_time, fits.Header())
                    base_fn = f"master_dark_exp{exp_time:.2f}s_ccdproc".replace('.', '_')
                    saved_path = save_fits_image(master_dark_corrected_ccd, current_dark_header, base_fn, temp_dir, current_timestamp_str)
                    if saved_path:
                        state_darks_corrected_dict_out[exp_time] = saved_path
                        created_darks_info.append(f"Exp {exp_time}s: {os.path.basename(saved_path)}")
                    else: status_messages.append(f"Master DARK (Exp: {exp_time}s): 생성 실패 (저장 오류).")
                else: status_messages.append(f"Master DARK (Exp: {exp_time}s): ccdproc 생성 실패.")
            except Exception as e_md: logger_ui.error(f"Master DARK (Exp: {exp_time}s) 처리 중 오류", exc_info=True); status_messages.append(f"Master DARK (Exp: {exp_time}s) 오류: {str(e_md)}")
        
        if created_darks_info: ui_dark_output_msg = "생성된 Master Darks:\n" + "\n".join(created_darks_info)
        else: ui_dark_output_msg = "유효한 Master Dark 생성 실패 또는 처리할 파일 없음."
    else: status_messages.append("DARK: 업로드된 파일 없음.")

    # 3. Master FLAT (필터별, 노출 시간별 Corrected) 생성 (ccdproc 사용)
    if flat_file_objs_all:
        status_messages.append(f"FLAT: 총 {len(flat_file_objs_all)}개 파일 처리 시작 (ccdproc)...")
        
        flat_files_info_grouped = {} # Key: (filter, exp_time), Value: {'paths': [], 'header': first_header}
        for ff_obj in flat_file_objs_all:
            if ff_obj and ff_obj.name and os.path.exists(ff_obj.name):
                try:
                    _, header = load_single_fits_from_path(ff_obj.name, "FLAT (header check for grouping)")
                    if header:
                        filter_val = get_fits_keyword(header, ['FILTER'], 'Generic').upper()
                        exp_time = get_fits_keyword(header, ['EXPTIME', 'EXPOSURE'], -1.0, float)
                        if exp_time > 0:
                            group_key = (filter_val, exp_time)
                            if group_key not in flat_files_info_grouped:
                                flat_files_info_grouped[group_key] = {'paths': [], 'header': header}
                            flat_files_info_grouped[group_key]['paths'].append(ff_obj.name)
                        else: status_messages.append(f"경고: Flat 파일 '{os.path.basename(ff_obj.name)}' 노출시간 정보 부족/유효X.")
                    else: status_messages.append(f"경고: Flat 파일 '{os.path.basename(ff_obj.name)}' 헤더 읽기 실패.")
                except Exception as e_head_flat: status_messages.append(f"경고: Flat 파일 '{os.path.basename(ff_obj.name)}' 헤더 읽기 오류: {e_head_flat}.")
            else: status_messages.append("유효하지 않은 FLAT 파일 객체 발견.")

        created_flats_b_info, created_flats_v_info, created_flats_g_info = [], [], []
        for group_key, info_dict in flat_files_info_grouped.items():
            filter_name, exp_time = group_key
            flat_paths_list = info_dict['paths']
            first_header_in_group = info_dict['header']

            if not flat_paths_list: continue
            try:
                status_messages.append(f"Master FLAT ({filter_name}, Exp: {exp_time}s): {len(flat_paths_list)}개 파일로 생성 시작...")
                # 해당 노출 시간의 마스터 다크 찾기
                dark_for_flat_ccd = None
                master_dark_path = state_darks_corrected_dict_out.get(exp_time) # Tab1에서 방금 생성된 다크 사용
                if master_dark_path and os.path.exists(master_dark_path):
                    dark_data_temp, dark_header_temp = load_single_fits_from_path(master_dark_path, f"Dark for Flat {group_key}")
                    if dark_data_temp is not None: dark_for_flat_ccd = CCDData(dark_data_temp, header=dark_header_temp, unit=u.adu) 
                
                master_flat_ccd = create_master_flat_ccdproc(flat_paths_list, master_bias_ccd, dark_for_flat_ccd)
                if master_flat_ccd:
                    current_flat_header = master_flat_ccd.header if master_flat_ccd.header else first_header_in_group
                    base_fn = f"master_flat_{filter_name}_exp{exp_time:.2f}s_ccdproc".replace('.', '_')
                    saved_path = save_fits_image(master_flat_ccd, current_flat_header, base_fn, temp_dir, current_timestamp_str)
                    if saved_path:
                        state_flats_corrected_dict_out[group_key] = saved_path
                        info_str = f"({filter_name}, Exp {exp_time}s): {os.path.basename(saved_path)}"
                        if filter_name == 'B': created_flats_b_info.append(info_str)
                        elif filter_name == 'V': created_flats_v_info.append(info_str)
                        else: created_flats_g_info.append(info_str)
                    else: status_messages.append(f"Master FLAT ({filter_name}, Exp: {exp_time}s): 생성 실패.")
                else: status_messages.append(f"Master FLAT ({filter_name}, Exp: {exp_time}s): ccdproc 생성 실패.")
            except Exception as e_mf_grp: logger_ui.error(f"Master FLAT ({filter_name}, Exp: {exp_time}s) 처리 오류", exc_info=True); status_messages.append(f"Master FLAT ({filter_name}, Exp: {exp_time}s) 오류: {str(e_mf_grp)}")
        
        ui_flat_b_output_msg = "생성된 Master Flat B:\n" + "\n".join(created_flats_b_info) if created_flats_b_info else "생성된 Master Flat B 없음"
        ui_flat_v_output_msg = "생성된 Master Flat V:\n" + "\n".join(created_flats_v_info) if created_flats_v_info else "생성된 Master Flat V 없음"
        ui_flat_generic_output_msg = "생성된 Master Flat Generic:\n" + "\n".join(created_flats_g_info) if created_flats_g_info else "생성된 Master Flat Generic 없음"
    else: status_messages.append("FLAT: 업로드된 파일 없음.")
        
    final_status = "\n".join(status_messages)
    logger_ui.info("Tab 1: Master frame generation finished.")
    return ui_bias_path, ui_dark_output_msg, ui_flat_b_output_msg, ui_flat_v_output_msg, ui_flat_generic_output_msg, \
           state_bias_path_out, state_darks_corrected_dict_out, state_flats_corrected_dict_out, \
           final_status


def handle_tab2_light_frame_calibration(
    light_file_objs_list, 
    tab2_uploaded_bias_obj, tab2_uploaded_dark_raw_files, 
    tab2_uploaded_flat_b_obj, tab2_uploaded_flat_v_obj, 
    state_mb_p, state_md_dict_corr, state_mf_dict_corr, # 딕셔너리 형태로 받음
    preview_stretch_type, preview_asinh_a,
    temp_dir):
    status_messages = []
    calibrated_light_file_paths_for_ui = []
    output_preview_pil_image = None 
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # 1. 최종 사용할 Master BIAS 결정 (CCDData 객체로)
    final_mb_ccd = None 
    if tab2_uploaded_bias_obj and tab2_uploaded_bias_obj.name:
        mb_data_temp, mb_header_temp = load_single_fits_from_path(tab2_uploaded_bias_obj.name, "탭2 업로드 Master BIAS")
        if mb_data_temp is not None: final_mb_ccd = CCDData(mb_data_temp, header=mb_header_temp, unit=u.adu)
        status_messages.append("탭2 업로드 Master BIAS 사용." if final_mb_ccd is not None else "탭2 업로드 Master BIAS 로드 실패.")
    elif state_mb_p and os.path.exists(state_mb_p):
        mb_data_temp, mb_header_temp = load_single_fits_from_path(state_mb_p, "탭1 Master BIAS")
        if mb_data_temp is not None: final_mb_ccd = CCDData(mb_data_temp, header=mb_header_temp, unit=u.adu)
        status_messages.append("탭1 Master BIAS 사용." if final_mb_ccd is not None else "탭1 Master BIAS 로드 실패.")
    if final_mb_ccd is None: status_messages.append("경고: 사용 가능한 Master BIAS 없음. BIAS 보정 생략.")

    # 2. 탭2에서 업로드된 Raw Dark들을 처리하여 딕셔너리 생성 (CCDData 객체로)
    tab2_processed_darks_ccd_dict = {} # key: exp_time (float), value: dark_ccd_corrected (CCDData)
    if tab2_uploaded_dark_raw_files: 
        status_messages.append(f"탭2 업로드된 DARK 파일 {len(tab2_uploaded_dark_raw_files)}개 처리 시작...")
        for dark_file_obj in tab2_uploaded_dark_raw_files:
            if dark_file_obj and dark_file_obj.name:
                raw_md_data, raw_md_header = load_single_fits_from_path(dark_file_obj.name, f"탭2 업로드 Raw DARK ({os.path.basename(dark_file_obj.name)})")
                if raw_md_data is not None and raw_md_header is not None:
                    exp_time = get_fits_keyword(raw_md_header, ['EXPTIME', 'EXPOSURE'], default_value=-1.0, data_type=float)
                    if exp_time > 0:
                        raw_dark_ccd = CCDData(raw_md_data, header=raw_md_header, unit=u.adu)
                        corrected_dark_ccd = raw_dark_ccd 
                        if final_mb_ccd is not None and raw_dark_ccd.shape == final_mb_ccd.shape:
                            corrected_dark_ccd = ccdp.subtract_bias(raw_dark_ccd, final_mb_ccd)
                            status_messages.append(f"탭2 업로드 DARK ({os.path.basename(dark_file_obj.name)}, Exp: {exp_time}s) BIAS 차감.")
                        else:
                            status_messages.append(f"경고: 탭2 업로드 DARK ({os.path.basename(dark_file_obj.name)}, Exp: {exp_time}s) BIAS 차감 못함.")
                        
                        if exp_time not in tab2_processed_darks_ccd_dict: 
                            tab2_processed_darks_ccd_dict[exp_time] = corrected_dark_ccd
                            status_messages.append(f"탭2 업로드 DARK (Exp: {exp_time}s) 사용 준비 완료.")
                        else:
                            status_messages.append(f"경고: 탭2에 동일 노출시간({exp_time}s)의 DARK가 여러 개 업로드됨. 첫 번째 파일만 사용.")
                    else:
                        status_messages.append(f"경고: 탭2 업로드 DARK ({os.path.basename(dark_file_obj.name)}) 노출시간 정보 없음. 무시됨.")
                else: status_messages.append(f"탭2 업로드 DARK ({os.path.basename(dark_file_obj.name)}) 로드 실패.")
    
    # 3. 탭2에서 업로드된 필터별 Flat 처리 (CCDData 객체로)
    tab2_processed_flats_ccd_dict = {} # Key: (filter, exp_time), Value: flat_ccd_corrected
    for filt_char, uploaded_mf_obj in [('B', tab2_uploaded_flat_b_obj), ('V', tab2_uploaded_flat_v_obj)]:
        if uploaded_mf_obj and uploaded_mf_obj.name:
            mf_data_raw, mf_header = load_single_fits_from_path(uploaded_mf_obj.name, f"탭2 업로드 Master FLAT {filt_char}")
            if mf_data_raw is not None and mf_header is not None:
                exp_time_mf = get_fits_keyword(mf_header, ['EXPTIME', 'EXPOSURE'], -1.0, float)
                if exp_time_mf <= 0: exp_time_mf = -1.0 
                
                raw_flat_ccd = CCDData(mf_data_raw, header=mf_header, unit=u.adu)
                
                # 이 Flat에 맞는 Dark 찾기 (탭2 업로드 Dark 또는 탭1 Dark)
                dark_for_this_flat_ccd = tab2_processed_darks_ccd_dict.get(exp_time_mf) 
                if dark_for_this_flat_ccd is None and state_md_dict_corr: 
                    dark_path = state_md_dict_corr.get(exp_time_mf) # 정확 일치
                    if not dark_path and state_md_dict_corr: # 가장 가까운 것
                        available_exp = sorted([k for k in state_md_dict_corr.keys() if isinstance(k, (int, float)) and k > 0])
                        if available_exp: 
                            closest_exp = min(available_exp, key=lambda e_val: abs(e_val - exp_time_mf))
                            dark_path = state_md_dict_corr[closest_exp]
                    if dark_path and os.path.exists(dark_path): 
                        d_data, d_hdr = load_single_fits_from_path(dark_path, f"Dark for Flat {filt_char} {exp_time_mf}s")
                        if d_data is not None: dark_for_this_flat_ccd = CCDData(d_data, header=d_hdr, unit=u.adu)
                
                processed_flat_ccd = ccdp.ccd_process(raw_flat_ccd, master_bias=final_mb_ccd, dark_frame=dark_for_this_flat_ccd, error=False)
                mean_val = np.nanmean(processed_flat_ccd.data)
                if mean_val is not None and not np.isclose(mean_val, 0) and np.isfinite(mean_val):
                     processed_flat_ccd = processed_flat_ccd.divide(mean_val * processed_flat_ccd.unit)
                
                flat_key = (filt_char, exp_time_mf if exp_time_mf > 0 else -1.0) 
                tab2_processed_flats_ccd_dict[flat_key] = processed_flat_ccd 
                status_messages.append(f"탭2 업로드 Master FLAT {filt_char} (Exp: {exp_time_mf if exp_time_mf > 0 else '모름'}) 처리하여 사용 준비 완료.")
            else: status_messages.append(f"탭2 업로드 Master FLAT {filt_char} 로드 실패.")

    if not light_file_objs_list: status_messages.append("보정할 LIGHT 프레임 없음."); return [], None, "\n".join(status_messages)
    status_messages.append(f"{len(light_file_objs_list)}개의 LIGHT 프레임 보정을 시작합니다...")
    first_calibrated_image_data_for_preview = None

    for i, light_file_obj in enumerate(light_file_objs_list):
        light_filename = "알 수 없는 파일"
        md_to_use_ccd, mf_to_use_ccd = None, None
        dark_source_msg, flat_source_msg = "미사용", "미사용"
        try:
            if light_file_obj is None or not hasattr(light_file_obj, 'name') or light_file_obj.name is None: status_messages.append(f"LIGHT 파일 {i+1} 유효X."); continue
            light_filename = os.path.basename(light_file_obj.name)
            status_messages.append(f"--- {light_filename} 보정 중 ({i+1}/{len(light_file_objs_list)}) ---")
            
            light_data, light_header = load_single_fits_from_path(light_file_obj.name, f"LIGHT ({light_filename})")
            if light_data is None or light_header is None: status_messages.append(f"{light_filename} 로드 실패."); continue
            
            light_ccd_raw = CCDData(light_data, header=light_header, unit=u.adu) 
            current_light_filter = get_fits_keyword(light_header, ['FILTER'], 'Generic').upper()
            current_light_exptime = get_fits_keyword(light_header, ['EXPTIME', 'EXPOSURE'], -1.0, float)

            # DARK 결정
            if current_light_exptime > 0:
                if current_light_exptime in tab2_processed_darks_ccd_dict: 
                    md_to_use_ccd = tab2_processed_darks_ccd_dict[current_light_exptime]
                    dark_source_msg = f"탭2 업로드 Dark (Exp {current_light_exptime}s)"
                elif state_md_dict_corr and current_light_exptime in state_md_dict_corr: 
                    dark_path = state_md_dict_corr[current_light_exptime]
                    if dark_path and os.path.exists(dark_path): 
                        d_data, d_hdr = load_single_fits_from_path(dark_path, f"탭1 Dark {current_light_exptime}s")
                        if d_data is not None: md_to_use_ccd = CCDData(d_data, header=d_hdr, unit=u.adu)
                        dark_source_msg = f"탭1 Dark ({os.path.basename(dark_path)})"
            if md_to_use_ccd is None: status_messages.append(f"경고: {light_filename} (Exp: {current_light_exptime}s)에 맞는 Master DARK 없음. DARK 보정 생략.")
            else: status_messages.append(f"{light_filename}: 사용된 Master DARK -> {dark_source_msg}")

            # FLAT 결정
            flat_key_exact_tab2 = (current_light_filter, current_light_exptime if current_light_exptime > 0 else -1.0)
            flat_key_filter_any_exp_tab2 = (current_light_filter, -1.0) # 노출시간 모르는 경우의 키
            
            if flat_key_exact_tab2 in tab2_processed_flats_ccd_dict: 
                mf_to_use_ccd = tab2_processed_flats_ccd_dict[flat_key_exact_tab2]; flat_source_msg = f"탭2 업로드 Flat {flat_key_exact_tab2}"
            elif flat_key_filter_any_exp_tab2 in tab2_processed_flats_ccd_dict:
                mf_to_use_ccd = tab2_processed_flats_ccd_dict[flat_key_filter_any_exp_tab2]; flat_source_msg = f"탭2 업로드 Flat {flat_key_filter_any_exp_tab2}"
            elif state_mf_dict_corr: 
                flat_path_tab1 = state_mf_dict_corr.get(flat_key_exact_tab2)
                if not flat_path_tab1: 
                    paths_for_filter = [p for (f,e),p in state_mf_dict_corr.items() if f == current_light_filter]
                    if paths_for_filter: flat_path_tab1 = paths_for_filter[0] 
                    else: 
                        gen_keys = [gk for gk in state_mf_dict_corr.keys() if gk[0] == 'Generic']
                        if gen_keys: 
                            gen_path_exp_match = state_mf_dict_corr.get(('Generic', current_light_exptime if current_light_exptime > 0 else -1.0))
                            if gen_path_exp_match and os.path.exists(gen_path_exp_match): flat_path_tab1 = gen_path_exp_match
                            else: flat_path_tab1 = state_mf_dict_corr[gen_keys[0]] 
                if flat_path_tab1 and os.path.exists(flat_path_tab1):
                    mf_data, mf_hdr = load_single_fits_from_path(flat_path_tab1, f"탭1 Flat from {flat_path_tab1}")
                    if mf_data is not None: mf_to_use_ccd = CCDData(mf_data, header=mf_hdr, unit=u.adu)
                    flat_source_msg = f"탭1 Flat ({os.path.basename(flat_path_tab1)})"
            status_messages.append(f"{light_filename}: 사용된 Master FLAT -> {flat_source_msg}")

            calibrated_light_ccd = ccdp.ccd_process(
                light_ccd_raw,
                master_bias=final_mb_ccd if final_mb_ccd is not None and light_ccd_raw.shape == final_mb_ccd.shape else None,
                dark_frame=md_to_use_ccd if md_to_use_ccd is not None and light_ccd_raw.shape == md_to_use_ccd.shape else None,
                master_flat=mf_to_use_ccd if mf_to_use_ccd is not None and light_ccd_raw.shape == mf_to_use_ccd.shape else None,
                dark_exposure=light_header.get('EXPTIME')*u.s if light_header.get('EXPTIME') else None, 
                data_exposure=light_header.get('EXPTIME')*u.s if light_header.get('EXPTIME') else None,
                dark_scale=True, 
                error=False 
            )
            
            if first_calibrated_image_data_for_preview is None:
                first_calibrated_image_data_for_preview = calibrated_light_ccd.data 
            
            calibrated_light_ccd.header['HISTORY'] = f'Calibrated App v0.12 (B:{final_mb_ccd is not None},D:{dark_source_msg!="미사용"},F:{flat_source_msg!="미사용"})'
            saved_path = save_fits_image(calibrated_light_ccd, calibrated_light_ccd.header, f"calibrated_{os.path.splitext(light_filename)[0]}", temp_dir, current_timestamp_str)
            if saved_path: calibrated_light_file_paths_for_ui.append(saved_path); status_messages.append(f"{light_filename}: 보정 완료: {os.path.basename(saved_path)}")
            else: status_messages.append(f"{light_filename}: 저장 실패.")
        except Exception as e: logger_ui.error(f"LIGHT ({light_filename}) 보정 오류", exc_info=True); status_messages.append(f"{light_filename} 보정 오류: {str(e)}")

    if first_calibrated_image_data_for_preview is not None:
        try: output_preview_pil_image = create_preview_image(first_calibrated_image_data_for_preview, stretch_type=preview_stretch_type, a_param=preview_asinh_a)
        except Exception as e: logger_ui.error("미리보기 생성 오류", exc_info=True); status_messages.append(f"미리보기 생성 오류: {str(e)}")
            
    if not calibrated_light_file_paths_for_ui: status_messages.append("성공적으로 보정된 LIGHT 프레임 없음.")
    final_status = "\n".join(status_messages)
    logger_ui.info("Tab 2: Light frame calibration finished."); return calibrated_light_file_paths_for_ui, output_preview_pil_image, final_status


def handle_tab3_extinction_analysis(
    light_file_objs, 
    uploaded_mb_path_obj, uploaded_md_raw_path_obj,
    uploaded_mf_b_raw_path_obj, uploaded_mf_v_raw_path_obj,
    state_mb_p, state_md_dict_corr, state_mf_dict_corr, 
    star_detection_thresh_factor,
    temp_dir):
    # 이 함수도 handle_tab2_light_frame_calibration과 유사하게
    # Master Dark 및 Master Flat을 state_md_dict_corr와 state_mf_dict_corr (딕셔너리)에서
    # 각 LIGHT 프레임의 노출시간과 필터에 맞춰 가져와 사용하도록 수정해야 합니다.
    # 업로드된 마스터 프레임 처리 로직도 ccdproc 기반으로 변경 및 유사하게 적용됩니다.
    status_log = ["탭 3 대기소광계수 기능은 현재 ccdproc 및 노출/필터별 마스터 로직 업데이트 필요."]
    logger_ui.warning("handle_tab3_extinction_analysis needs refactoring for ccdproc and exp/filter specific masters from dict.")
    df_headers_stub = ["Message"]
    return None, "기능 업데이트 필요", (df_headers_stub, [["기능 업데이트 필요"]]), "\n".join(status_log)


def handle_tab4_detailed_photometry(
    light_b_file_objs, light_v_file_objs,
    std_star_b_file_obj, std_star_v_file_obj,
    std_b_mag_known_input, std_v_mag_known_input,
    tab4_uploaded_mb_obj, tab4_uploaded_md_raw_obj, 
    tab4_uploaded_mf_b_raw_obj, tab4_uploaded_mf_v_raw_obj,
    state_mb_p, state_md_dict_corr, 
    state_mf_dict_corr,
    k_b_input, m0_b_input_user, k_v_input, m0_v_input_user, 
    dao_fwhm_input, dao_thresh_nsigma_input, phot_aperture_radius_input, 
    roi_x_min, roi_x_max, roi_y_min, roi_y_max,
    simbad_query_radius_arcsec, 
    temp_dir):
    # 이 함수도 handle_tab2_light_frame_calibration과 유사하게
    # Master Dark 및 Master Flat을 state_md_dict_corr와 state_mf_dict_corr (딕셔너리)에서
    # 각 LIGHT 프레임의 노출시간과 필터에 맞춰 가져와 사용하고, ccd_process로 보정하도록 수정해야 합니다.
    # 업로드된 마스터 프레임 처리 로직도 ccdproc 기반으로 변경 및 유사하게 적용됩니다.
    status_log = ["탭 4 상세 측광 기능은 현재 ccdproc 및 노출/필터별 마스터 프레임 로직 업데이트 필요."]
    logger_ui.warning("handle_tab4_detailed_photometry needs refactoring for ccdproc and exp/filter specific masters from dict.")
    df_headers_stub = ["Message"]
    return (df_headers_stub, [["기능 업데이트 필요"]]), None, "\n".join(status_log)


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
