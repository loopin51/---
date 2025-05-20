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

from utils.fits import load_fits_from_gradio_files, load_single_fits_from_path, save_fits_image, create_preview_image, draw_roi_on_pil_image, get_fits_keyword, draw_photometry_results_on_image
from utils.calibration import create_master_bias_ccdproc, create_master_dark_ccdproc, create_master_flat_ccdproc 
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
from astropy.nddata import CCDData 
import ccdproc as ccdp 


logger_ui = logging.getLogger(__name__)

def handle_tab1_master_frame_creation(bias_file_objs, dark_file_objs, flat_file_objs_all, temp_dir):
    status_messages = []
    ui_bias_path = None
    ui_dark_output_msg = "생성된 Master Dark 없음"
    ui_flat_b_output_msg = "생성된 Master Flat B 없음"
    ui_flat_v_output_msg = "생성된 Master Flat V 없음"
    ui_flat_generic_output_msg = "생성된 Master Flat Generic 없음"
    
    state_bias_path_out = None
    state_darks_corrected_dict_out = {} 
    state_flats_corrected_dict_out = {} 
    
    master_bias_ccd = None 
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if bias_file_objs:
        try:
            status_messages.append(f"BIAS: {len(bias_file_objs)}개 파일 처리 시작 (ccdproc)...")
            bias_file_paths = [f.name for f in bias_file_objs if f and f.name]
            if bias_file_paths:
                master_bias_ccd = create_master_bias_ccdproc(bias_file_paths) 
                if master_bias_ccd:
                    bias_header = master_bias_ccd.header if master_bias_ccd.header else fits.Header()
                    saved_path = save_fits_image(master_bias_ccd, bias_header, "master_bias_ccdproc", temp_dir, current_timestamp_str)
                    if saved_path: ui_bias_path = state_bias_path_out = saved_path; status_messages.append(f"BIAS: 생성 완료: {os.path.basename(ui_bias_path)}")
                    else: status_messages.append("BIAS: 생성 실패 (저장 오류).")
                else: status_messages.append("BIAS: ccdproc으로 마스터 BIAS 생성 실패.")
            else: status_messages.append("BIAS: 유효한 파일 경로 없음.")
        except Exception as e: logger_ui.error("BIAS 처리 중 오류", exc_info=True); status_messages.append(f"BIAS 처리 오류: {str(e)}"); master_bias_ccd = None
    else: status_messages.append("BIAS: 업로드된 파일 없음.")

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
                        else: status_messages.append(f"경고: DARK 파일 '{os.path.basename(df_obj.name)}' 노출시간 정보 부족/유효X. 건너<0xEB><0><0x8F><0xEB><0x82><0xB4>니다.")
                    else: status_messages.append(f"경고: DARK 파일 '{os.path.basename(df_obj.name)}' 헤더 읽기 실패. 건너<0xEB><0><0x8F><0xEB><0x82><0xB4>니다.")
                except Exception as e_head_dark: status_messages.append(f"경고: DARK 파일 '{os.path.basename(df_obj.name)}' 헤더 읽기 오류: {e_head_dark}. 건너<0xEB><0><0x8F><0xEB><0x82><0xB4>니다.")
            else: status_messages.append("유효하지 않은 DARK 파일 객체 발견. 건너<0xEB><0><0x8F><0xEB><0x82><0xB4>니다.")
            
        created_darks_info = []
        for exp_time, dark_paths_list in grouped_dark_paths_by_exp.items():
            if not dark_paths_list: continue
            try:
                status_messages.append(f"Master DARK (Exp: {exp_time}s): {len(dark_paths_list)}개 파일로 생성 시작...")
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

    if flat_file_objs_all:
        status_messages.append(f"FLAT: 총 {len(flat_file_objs_all)}개 파일 처리 시작 (ccdproc)...")
        
        flat_files_info_grouped = {} 
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
                dark_for_flat_ccd = None
                if state_darks_corrected_dict_out:
                    master_dark_path = state_darks_corrected_dict_out.get(exp_time) 
                    if not master_dark_path and state_darks_corrected_dict_out:
                        available_exp_df = sorted([k for k in state_darks_corrected_dict_out.keys() if isinstance(k, (int, float))])
                        if available_exp_df: closest_exp_df = min(available_exp_df, key=lambda e: abs(e-exp_time)); master_dark_path = state_darks_corrected_dict_out[closest_exp_df]
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
            except Exception as e_mf_grp: logger_ui.error(f"Master FLAT ({filter_name}, Exp: {exp_time}s) 처리 중 오류", exc_info=True); status_messages.append(f"Master FLAT ({filter_name}, Exp: {exp_time}s) 오류: {str(e_mf_grp)}")
        
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
    state_mb_p, state_md_dict_corr, state_mf_dict_corr, 
    preview_stretch_type, preview_asinh_a,
    temp_dir):
    status_messages = []
    calibrated_light_file_paths_for_ui = []
    output_preview_pil_image = None 
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

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

    tab2_processed_darks_ccd_dict = {} 
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
                        if exp_time not in tab2_processed_darks_ccd_dict: 
                            tab2_processed_darks_ccd_dict[exp_time] = corrected_dark_ccd
                    else: status_messages.append(f"경고: 탭2 DARK ({os.path.basename(dark_file_obj.name)}) 노출시간 정보 없음.")
                else: status_messages.append(f"탭2 DARK ({os.path.basename(dark_file_obj.name)}) 로드 실패.")
    
    tab2_processed_flats_ccd_dict = {} 
    for filt_char, uploaded_mf_obj in [('B', tab2_uploaded_flat_b_obj), ('V', tab2_uploaded_flat_v_obj)]:
        if uploaded_mf_obj and uploaded_mf_obj.name:
            mf_data_raw, mf_header = load_single_fits_from_path(uploaded_mf_obj.name, f"탭2 업로드 Master FLAT {filt_char}")
            if mf_data_raw is not None and mf_header is not None:
                exp_time_mf = get_fits_keyword(mf_header, ['EXPTIME', 'EXPOSURE'], -1.0, float)
                if exp_time_mf <= 0: exp_time_mf = -1.0 
                raw_flat_ccd = CCDData(mf_data_raw, header=mf_header, unit=u.adu)
                dark_for_this_flat_ccd = tab2_processed_darks_ccd_dict.get(exp_time_mf) 
                if dark_for_this_flat_ccd is None and state_md_dict_corr: 
                    dark_path = state_md_dict_corr.get(exp_time_mf)
                    if not dark_path and state_md_dict_corr:
                        available_exp = sorted([k for k in state_md_dict_corr.keys() if isinstance(k, (int, float)) and k > 0])
                        if available_exp: closest_exp = min(available_exp, key=lambda e_val: abs(e_val - exp_time_mf)); dark_path = state_md_dict_corr[closest_exp]
                    if dark_path and os.path.exists(dark_path): 
                        d_data, d_hdr = load_single_fits_from_path(dark_path, f"Dark for Flat {filt_char} {exp_time_mf}s")
                        if d_data is not None: dark_for_this_flat_ccd = CCDData(d_data, header=d_hdr, unit=u.adu)
                
                processed_flat_ccd = ccdp.ccd_process(raw_flat_ccd, master_bias=final_mb_ccd, dark_frame=dark_for_this_flat_ccd, error=False) # dark_frame 사용
                mean_val = np.nanmean(processed_flat_ccd.data)
                if mean_val is not None and not np.isclose(mean_val, 0) and np.isfinite(mean_val): processed_flat_ccd = processed_flat_ccd.divide(mean_val * processed_flat_ccd.unit) 
                flat_key = (filt_char, exp_time_mf if exp_time_mf > 0 else -1.0) 
                tab2_processed_flats_ccd_dict[flat_key] = processed_flat_ccd 
                status_messages.append(f"탭2 업로드 Master FLAT {filt_char} (Exp: {exp_time_mf if exp_time_mf > 0 else '모름'}) 처리 완료.")
            else: status_messages.append(f"탭2 업로드 Master FLAT {filt_char} 로드 실패.")

    if not light_file_objs_list: status_messages.append("보정할 LIGHT 프레임 없음."); return [], None, "\n".join(status_messages)
    status_messages.append(f"{len(light_file_objs_list)}개의 LIGHT 프레임 보정을 시작합니다...")
    first_calibrated_image_data_for_preview = None

    for i, light_file_obj in enumerate(light_file_objs_list):
        light_filename = "알 수 없는 파일"; md_to_use_ccd, mf_to_use_ccd = None, None
        dark_source_msg, flat_source_msg = "미사용", "미사용"
        try:
            if light_file_obj is None or not hasattr(light_file_obj, 'name') or light_file_obj.name is None: status_messages.append(f"LIGHT 파일 {i+1} 유효X."); continue
            light_filename = os.path.basename(light_file_obj.name)
            status_messages.append(f"--- {light_filename} 보정 중 ---")
            light_data, light_header = load_single_fits_from_path(light_file_obj.name, f"LIGHT ({light_filename})")
            if light_data is None or light_header is None: status_messages.append(f"{light_filename} 로드 실패."); continue
            light_ccd_raw = CCDData(light_data, header=light_header, unit=u.adu) 
            current_light_filter = get_fits_keyword(light_header, ['FILTER'], 'Generic').upper()
            current_light_exptime = get_fits_keyword(light_header, ['EXPTIME', 'EXPOSURE'], -1.0, float)

            if current_light_exptime > 0:
                if current_light_exptime in tab2_processed_darks_ccd_dict: 
                    md_to_use_ccd = tab2_processed_darks_ccd_dict[current_light_exptime]; dark_source_msg = f"탭2 업로드 Dark (Exp {current_light_exptime}s)"
                elif state_md_dict_corr and current_light_exptime in state_md_dict_corr: 
                    dark_path = state_md_dict_corr[current_light_exptime]
                    if dark_path and os.path.exists(dark_path): 
                        d_data, d_hdr = load_single_fits_from_path(dark_path, f"탭1 Dark {current_light_exptime}s")
                        if d_data is not None: md_to_use_ccd = CCDData(d_data, header=d_hdr, unit=u.adu)
                        dark_source_msg = f"탭1 Dark ({os.path.basename(dark_path)})"
            if md_to_use_ccd is None: status_messages.append(f"경고: {light_filename} (Exp: {current_light_exptime}s)에 맞는 Master DARK 없음. DARK 보정 생략.")
            
            flat_key_exact_tab2 = (current_light_filter, current_light_exptime if current_light_exptime > 0 else -1.0)
            flat_key_filter_any_exp_tab2 = (current_light_filter, -1.0) 
            if flat_key_exact_tab2 in tab2_processed_flats_ccd_dict: mf_to_use_ccd = tab2_processed_flats_ccd_dict[flat_key_exact_tab2]; flat_source_msg = f"탭2 업로드 Flat {flat_key_exact_tab2}"
            elif flat_key_filter_any_exp_tab2 in tab2_processed_flats_ccd_dict: mf_to_use_ccd = tab2_processed_flats_ccd_dict[flat_key_filter_any_exp_tab2]; flat_source_msg = f"탭2 업로드 Flat {flat_key_filter_any_exp_tab2}"
            elif state_mf_dict_corr: 
                flat_path_tab1 = state_mf_dict_corr.get(flat_key_exact_tab2)
                if not flat_path_tab1: 
                    paths_for_filter = [p for (f,e),p in state_mf_dict_corr.items() if f == current_light_filter]
                    if paths_for_filter: 
                        closest_flat_key = min([(f,e) for (f,e) in state_mf_dict_corr.keys() if f == current_light_filter], key=lambda k_exp: abs(k_exp[1] - (current_light_exptime if current_light_exptime > 0 else float('inf'))) if k_exp[1]>0 else float('inf'), default=None)
                        if closest_flat_key: flat_path_tab1 = state_mf_dict_corr[closest_flat_key]
                    else: 
                        gen_keys = [gk for gk in state_mf_dict_corr.keys() if gk[0] == 'Generic']
                        if gen_keys: 
                            gen_path_exp_match = state_mf_dict_corr.get(('Generic', current_light_exptime if current_light_exptime > 0 else -1.0))
                            if gen_path_exp_match and os.path.exists(gen_path_exp_match): flat_path_tab1 = gen_path_exp_match
                            elif gen_keys: flat_path_tab1 = state_mf_dict_corr[gen_keys[0]]
                if flat_path_tab1 and os.path.exists(flat_path_tab1):
                    mf_data, mf_hdr = load_single_fits_from_path(flat_path_tab1, f"탭1 Flat from {flat_path_tab1}"); mf_to_use_ccd = CCDData(mf_data, header=mf_hdr, unit=u.adu) if mf_data is not None else None; flat_source_msg = f"탭1 Flat ({os.path.basename(flat_path_tab1)})"
            if mf_to_use_ccd is None: status_messages.append(f"경고: {light_filename} ({current_light_filter})에 맞는 Master FLAT 없음.")
            
            calibrated_light_ccd = ccdp.ccd_process(light_ccd_raw, master_bias=final_mb_ccd, dark_frame=md_to_use_ccd, master_flat=mf_to_use_ccd, dark_scale=True, error=False)
            if first_calibrated_image_data_for_preview is None: first_calibrated_image_data_for_preview = calibrated_light_ccd.data 
            calibrated_light_ccd.header['HISTORY'] = f'Calibrated App v0.13 (B:{final_mb_ccd is not None},D:{dark_source_msg!="미사용"},F:{flat_source_msg!="미사용"})'
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
    status_log = []
    all_frame_results_for_df = [] 
    plot_image_fig = None 
    summary_text = "분석 결과가 없습니다."
    
    # 1. Master BIAS 결정 (CCDData 객체로)
    final_mb_ccd = None
    if uploaded_mb_path_obj and uploaded_mb_path_obj.name:
        mb_data_temp, mb_hdr_temp = load_single_fits_from_path(uploaded_mb_path_obj.name, "탭3 업로드 Master BIAS")
        if mb_data_temp is not None: final_mb_ccd = CCDData(mb_data_temp, header=mb_hdr_temp, unit=u.adu)
    elif state_mb_p and os.path.exists(state_mb_p):
        mb_data_temp, mb_hdr_temp = load_single_fits_from_path(state_mb_p, "탭1 Master BIAS")
        if mb_data_temp is not None: final_mb_ccd = CCDData(mb_data_temp, header=mb_hdr_temp, unit=u.adu)
    status_log.append(f"Master BIAS: {'사용' if final_mb_ccd is not None else '사용 안함/로드 실패'}")
    if final_mb_ccd is None: status_log.append("경고: BIAS 보정 생략됨.")

    # 2. 업로드된 Raw Dark 처리 (단일 파일, 해당 노출시간으로 간주)
    tab3_uploaded_dark_ccd_corrected = None
    tab3_uploaded_dark_exp_time = -1.0 
    if uploaded_md_raw_path_obj and uploaded_md_raw_path_obj.name:
        raw_md_data, raw_md_header = load_single_fits_from_path(uploaded_md_raw_path_obj.name, "탭3 업로드 Raw Master DARK")
        if raw_md_data is not None and raw_md_header is not None:
            tab3_uploaded_dark_exp_time = get_fits_keyword(raw_md_header, ['EXPTIME', 'EXPOSURE'], -1.0, float)
            raw_dark_ccd = CCDData(raw_md_data, header=raw_md_header, unit=u.adu)
            if final_mb_ccd is not None and raw_dark_ccd.shape == final_mb_ccd.shape:
                tab3_uploaded_dark_ccd_corrected = ccdp.subtract_bias(raw_dark_ccd, final_mb_ccd)
            else: tab3_uploaded_dark_ccd_corrected = raw_dark_ccd 
            status_log.append(f"탭3 업로드 Raw DARK (Exp: {tab3_uploaded_dark_exp_time if tab3_uploaded_dark_exp_time > 0 else '모름'}) 처리 완료.")
        else: status_log.append("탭3 업로드 Raw Master DARK 로드 실패.")
    
    # 3. 업로드된 필터별 Raw Flat 처리
    tab3_uploaded_flats_ccd_dict = {} 
    for filt_char, uploaded_mf_raw_obj in [('B', uploaded_mf_b_raw_path_obj), ('V', uploaded_mf_v_raw_path_obj)]:
        if uploaded_mf_raw_obj and uploaded_mf_raw_obj.name:
            mf_data_raw, mf_header = load_single_fits_from_path(uploaded_mf_raw_obj.name, f"탭3 업로드 Raw Master FLAT {filt_char}")
            if mf_data_raw is not None and mf_header is not None:
                exp_time_mf = get_fits_keyword(mf_header, ['EXPTIME', 'EXPOSURE'], -1.0, float)
                raw_flat_ccd = CCDData(mf_data_raw, header=mf_header, unit=u.adu)
                dark_for_this_flat_ccd = None
                if tab3_uploaded_dark_ccd_corrected is not None and (tab3_uploaded_dark_exp_time == exp_time_mf or tab3_uploaded_dark_exp_time == -1.0):
                    dark_for_this_flat_ccd = tab3_uploaded_dark_ccd_corrected
                elif state_md_dict_corr: 
                    dark_path = state_md_dict_corr.get(exp_time_mf)
                    if not dark_path and exp_time_mf > 0:
                        available_exp_d = sorted([k for k in state_md_dict_corr.keys() if isinstance(k, (int, float)) and k > 0])
                        if available_exp_d: closest_exp_d = min(available_exp_d, key=lambda e: abs(e-exp_time_mf)); dark_path = state_md_dict_corr[closest_exp_d]
                    if dark_path and os.path.exists(dark_path): 
                        d_data, d_hdr = load_single_fits_from_path(dark_path, f"Dark for Flat {filt_char} {exp_time_mf}s")
                        if d_data is not None: dark_for_this_flat_ccd = CCDData(d_data, header=d_hdr, unit=u.adu)
                
                processed_flat_ccd = ccdp.ccd_process(raw_flat_ccd, master_bias=final_mb_ccd, dark_frame=dark_for_this_flat_ccd, error=False)
                mean_val = np.nanmean(processed_flat_ccd.data)
                if mean_val is not None and not np.isclose(mean_val, 0) and np.isfinite(mean_val): processed_flat_ccd = processed_flat_ccd.divide(mean_val * processed_flat_ccd.unit)
                flat_key = (filt_char, exp_time_mf if exp_time_mf > 0 else -1.0)
                tab3_uploaded_flats_ccd_dict[flat_key] = processed_flat_ccd
                status_log.append(f"탭3 업로드 Master FLAT {filt_char} (Exp: {exp_time_mf if exp_time_mf > 0 else '모름'}) 처리 완료.")
            else: status_log.append(f"탭3 업로드 Master FLAT {filt_char} 로드 실패.")

    if not light_file_objs:
        status_log.append("분석할 LIGHT 프레임 없음."); df_headers_no_light = ["File", "Filter", "Airmass", "Altitude", "Inst. Mag.", "Flux", "Star X", "Star Y", "Ap. Radius", "Error"]
        return None, "LIGHT 파일 없음", (df_headers_no_light, [["LIGHT 파일 없음"]*len(df_headers_no_light)]), "\n".join(status_log)

    status_log.append(f"--- {len(light_file_objs)}개 LIGHT 프레임 분석 시작 ---")
    
    processed_results_for_analysis = [] 
    for light_file_obj in light_file_objs: 
        if not (light_file_obj and light_file_obj.name and os.path.exists(light_file_obj.name)): continue
        
        light_filename = os.path.basename(light_file_obj.name)
        status_log.append(f"--- {light_filename} 처리 중 ---")
        current_result = {'file': light_filename, 'error_message': None}
        try:
            light_data, light_header = load_single_fits_from_path(light_file_obj.name, f"LIGHT ({light_filename})")
            if light_data is None or light_header is None: raise ValueError("LIGHT 데이터 로드 실패")
            
            light_ccd_raw = CCDData(light_data, header=light_header, unit=u.adu)
            current_filter = get_fits_keyword(light_header, ['FILTER'], 'UNKNOWN').upper()
            current_exptime = get_fits_keyword(light_header, ['EXPTIME', 'EXPOSURE'], -1.0, float)
            current_result['filter'] = current_filter

            md_to_use_ccd = None; dark_source_msg = "미사용"
            if current_exptime > 0:
                if tab3_uploaded_dark_ccd_corrected is not None and (tab3_uploaded_dark_exp_time == current_exptime or tab3_uploaded_dark_exp_time == -1.0):
                    md_to_use_ccd = tab3_uploaded_dark_ccd_corrected; dark_source_msg = f"탭3 업로드 Dark (Exp {tab3_uploaded_dark_exp_time if tab3_uploaded_dark_exp_time > 0 else '모름'})"
                elif state_md_dict_corr and current_exptime in state_md_dict_corr:
                    dark_path = state_md_dict_corr[current_exptime]
                    if dark_path and os.path.exists(dark_path): d_data, d_hdr = load_single_fits_from_path(dark_path, f"탭1 Dark {current_exptime}s"); md_to_use_ccd = CCDData(d_data, header=d_hdr, unit=u.adu) if d_data is not None else None; dark_source_msg = f"탭1 Dark ({os.path.basename(dark_path)})"
            if md_to_use_ccd is None: status_log.append(f"경고: {light_filename} (Exp: {current_exptime}s)에 맞는 Master DARK 없음.")
            
            mf_to_use_ccd = None; flat_source_msg = "미사용"
            flat_key_exact_tab3 = (current_filter, current_exptime if current_exptime > 0 else -1.0)
            flat_key_filter_any_exp_tab3 = (current_filter, -1.0)
            if flat_key_exact_tab3 in tab3_uploaded_flats_ccd_dict: mf_to_use_ccd = tab3_uploaded_flats_ccd_dict[flat_key_exact_tab3]; flat_source_msg = f"탭3 업로드 Flat {flat_key_exact_tab3}"
            elif flat_key_filter_any_exp_tab3 in tab3_uploaded_flats_ccd_dict: mf_to_use_ccd = tab3_uploaded_flats_ccd_dict[flat_key_filter_any_exp_tab3]; flat_source_msg = f"탭3 업로드 Flat {flat_key_filter_any_exp_tab3}"
            elif state_mf_dict_corr: 
                flat_path_tab1 = state_mf_dict_corr.get(flat_key_exact_tab3)
                if not flat_path_tab1:
                    paths_for_filter = [p for (f,e),p in state_mf_dict_corr.items() if f == current_filter]
                    if paths_for_filter: 
                        closest_flat_key = min([(f,e) for (f,e) in state_mf_dict_corr.keys() if f == current_filter], key=lambda k_exp: abs(k_exp[1] - current_exptime) if current_exptime > 0 and k_exp[1]>0 else float('inf'), default=None)
                        if closest_flat_key: flat_path_tab1 = state_mf_dict_corr[closest_flat_key]
                    else: 
                        gen_keys = [gk for gk in state_mf_dict_corr.keys() if gk[0] == 'Generic']
                        if gen_keys: 
                            gen_path_exp = state_mf_dict_corr.get(('Generic', current_exptime if current_exptime > 0 else -1.0))
                            if gen_path_exp and os.path.exists(gen_path_exp): flat_path_tab1 = gen_path_exp
                            else: flat_path_tab1 = state_mf_dict_corr[gen_keys[0]]
                if flat_path_tab1 and os.path.exists(flat_path_tab1):
                    mf_data, mf_hdr = load_single_fits_from_path(flat_path_tab1, f"탭1 Flat from {flat_path_tab1}"); mf_to_use_ccd = CCDData(mf_data, header=mf_hdr, unit=u.adu) if mf_data is not None else None; flat_source_msg = f"탭1 Flat ({os.path.basename(flat_path_tab1)})"
            if mf_to_use_ccd is None: status_log.append(f"경고: {light_filename} ({current_filter})에 맞는 Master FLAT 없음.")

            calibrated_ccd = ccdp.ccd_process(light_ccd_raw, master_bias=final_mb_ccd, dark_frame=md_to_use_ccd, master_flat=mf_to_use_ccd, dark_scale=True, error=False)
            
            stars = detect_stars_extinction(calibrated_ccd.data, star_detection_thresh_factor)
            brightest = find_brightest_star_extinction(stars)
            if brightest is None: raise ValueError("가장 밝은 별 탐지 실패")
            flux, ap_rad, _ = calculate_flux_extinction(calibrated_ccd.data, brightest)
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
    state_mb_p, state_md_dict_corr, 
    state_mf_dict_corr, # 탭1의 (filter, exp_time): path 딕셔너리
    k_b_input, m0_b_input_user, k_v_input, m0_v_input_user, 
    dao_fwhm_input, dao_thresh_nsigma_input, phot_aperture_radius_input, 
    roi_x_min, roi_x_max, roi_y_min, roi_y_max,
    simbad_query_radius_arcsec, 
    temp_dir):
    status_log = []
    all_stars_final_data_for_df = [] 
    csv_output_path = None
    photometry_preview_image_pil = None 
    current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger_ui.info("탭 4 상세 측광 분석 시작 (ccdproc 적용)...")

    # --- 1. 입력값 유효성 검사 ---
    if not light_b_file_objs and not light_v_file_objs:
        status_log.append("오류: B 또는 V 필터 LIGHT 프레임을 하나 이상 업로드해야 합니다.")
        return (["Error Message"], [["LIGHT 프레임 없음"]]), None, None, "\n".join(status_log)
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
        return (["Error Message"], [["입력 파라미터 오류"]]), None, None, "\n".join(status_log)

    # --- 2. 최종 사용할 마스터 프레임 결정 및 준비 (CCDData 객체로) ---
    final_mb_ccd = None
    if tab4_uploaded_mb_obj and tab4_uploaded_mb_obj.name:
        mb_data_temp, mb_hdr_temp = load_single_fits_from_path(tab4_uploaded_mb_obj.name, "탭4 업로드 BIAS")
        if mb_data_temp is not None: final_mb_ccd = CCDData(mb_data_temp, header=mb_hdr_temp, unit=u.adu)
    elif state_mb_p and os.path.exists(state_mb_p):
        mb_data_temp, mb_hdr_temp = load_single_fits_from_path(state_mb_p, "탭1 BIAS")
        if mb_data_temp is not None: final_mb_ccd = CCDData(mb_data_temp, header=mb_hdr_temp, unit=u.adu)
    status_log.append(f"Master BIAS: {'사용' if final_mb_ccd is not None else '미사용/로드실패'}")
    if final_mb_ccd is None: status_log.append("경고: BIAS 보정 생략됨.");

    # 탭4에서 업로드된 Raw Dark 처리 (단일 파일, 해당 노출시간으로 간주)
    tab4_uploaded_dark_ccd_corrected_dict = {} 
    if tab4_uploaded_md_raw_obj and tab4_uploaded_md_raw_obj.name:
        raw_md_data, raw_md_header = load_single_fits_from_path(tab4_uploaded_md_raw_obj.name, "탭4 업로드 Raw Master DARK")
        if raw_md_data is not None and raw_md_header is not None:
            exp_time_md = get_fits_keyword(raw_md_header, ['EXPTIME', 'EXPOSURE'], -1.0, float)
            raw_dark_ccd = CCDData(raw_md_data, header=raw_md_header, unit=u.adu)
            corrected_dark_ccd = raw_dark_ccd
            if final_mb_ccd is not None and raw_dark_ccd.shape == final_mb_ccd.shape:
                corrected_dark_ccd = ccdp.subtract_bias(raw_dark_ccd, final_mb_ccd)
            # 노출시간을 모르면 -1.0 키로 저장하여, 나중에 노출시간 모르는 LIGHT에 우선 적용 가능
            tab4_uploaded_dark_ccd_corrected_dict[exp_time_md if exp_time_md > 0 else -1.0] = corrected_dark_ccd
            status_log.append(f"탭4 업로드 Raw DARK (Exp: {exp_time_md if exp_time_md > 0 else '모름'}) 처리 완료.")
        else: status_log.append("탭4 업로드 Raw Master DARK 로드 실패.")
    
    # 탭4에서 업로드된 필터별 Raw Flat 처리
    tab4_uploaded_flats_ccd_dict = {} 
    for filt_char_up, uploaded_mf_raw_obj_tab4 in [('B', tab4_uploaded_mf_b_raw_obj), ('V', tab4_uploaded_mf_v_raw_obj)]:
        if uploaded_mf_raw_obj_tab4 and uploaded_mf_raw_obj_tab4.name:
            mf_data_raw, mf_header = load_single_fits_from_path(uploaded_mf_raw_obj_tab4.name, f"탭4 업로드 Raw FLAT {filt_char_up}")
            if mf_data_raw is not None and mf_header is not None:
                exp_time_mf = get_fits_keyword(mf_header, ['EXPTIME', 'EXPOSURE'], -1.0, float)
                raw_flat_ccd = CCDData(mf_data_raw, header=mf_header, unit=u.adu)
                
                dark_for_this_flat_ccd = tab4_uploaded_dark_ccd_corrected_dict.get(exp_time_mf if exp_time_mf > 0 else -1.0)
                if dark_for_this_flat_ccd is None and state_md_dict_corr: 
                    dark_path = state_md_dict_corr.get(exp_time_mf)
                    if not dark_path and exp_time_mf > 0: # 가장 가까운 노출시간
                        available_exp_d = sorted([k for k in state_md_dict_corr.keys() if isinstance(k, (int, float)) and k > 0])
                        if available_exp_d: closest_exp_d = min(available_exp_d, key=lambda e: abs(e-exp_time_mf)); dark_path = state_md_dict_corr[closest_exp_d]
                    if dark_path and os.path.exists(dark_path): 
                        d_data, d_hdr = load_single_fits_from_path(dark_path, f"Dark for Flat {filt_char_up} {exp_time_mf}s")
                        if d_data is not None: dark_for_this_flat_ccd = CCDData(d_data, header=d_hdr, unit=u.adu)
                
                processed_flat_ccd = ccdp.ccd_process(raw_flat_ccd, master_bias=final_mb_ccd, dark_frame=dark_for_this_flat_ccd, error=False)
                mean_val = np.nanmean(processed_flat_ccd.data)
                if mean_val is not None and not np.isclose(mean_val, 0) and np.isfinite(mean_val): processed_flat_ccd = processed_flat_ccd.divide(mean_val * processed_flat_ccd.unit)
                flat_key = (filt_char_up, exp_time_mf if exp_time_mf > 0 else -1.0)
                tab4_uploaded_flats_ccd_dict[flat_key] = processed_flat_ccd
                status_log.append(f"탭4 업로드 Master FLAT {filt_char_up} (Exp: {exp_time_mf if exp_time_mf > 0 else '모름'}) 처리 완료.")
            else: status_log.append(f"탭4 업로드 Master FLAT {filt_char_up} 로드 실패.")

    if final_mb_ccd is None: # Bias는 필수
        status_log.append("오류: Master BIAS를 사용할 수 없습니다. 처리를 중단합니다.")
        return (["Error Message"], [["Master BIAS 없음"]]), None, None, "\n".join(status_log)

    # --- 3. 표준별 처리로 유효 영점(m0_eff) 계산 ---
    m0_eff_b, m0_eff_v = m0_b_user_val, m0_v_user_val 
    status_log.append(f"초기 영점 설정: m0_B={m0_eff_b:.3f}, m0_V={m0_eff_v:.3f} (사용자 입력 또는 기본값)")

    for std_filt_char, std_file_obj, std_mag_known_in, k_coeff_std_val, m0_eff_var_name_str in [
        ('B', std_star_b_file_obj, std_b_mag_known_input, k_b, 'm0_eff_b'),
        ('V', std_star_v_file_obj, std_v_mag_known_input, k_v, 'm0_eff_v')
    ]:
        if std_file_obj and hasattr(std_file_obj, 'name') and std_file_obj.name:
            status_log.append(f"--- {std_filt_char}필터 표준별 처리: {os.path.basename(std_file_obj.name)} ---")
            std_data, std_header = load_single_fits_from_path(std_file_obj.name, f"{std_filt_char} 표준별")
            if std_data is not None and std_header is not None:
                std_ccd_raw = CCDData(std_data, header=std_header, unit=u.adu)
                std_exp_time = get_fits_keyword(std_header, ['EXPTIME', 'EXPOSURE'], -1.0, float)
                
                dark_for_std_ccd = tab4_uploaded_dark_ccd_corrected_dict.get(std_exp_time if std_exp_time > 0 else -1.0)
                if dark_for_std_ccd is None and state_md_dict_corr:
                    dark_path_std = state_md_dict_corr.get(std_exp_time)
                    if not dark_path_std and std_exp_time > 0:
                        available_exp_std_d = sorted([k for k in state_md_dict_corr.keys() if isinstance(k, (int, float)) and k > 0])
                        if available_exp_std_d: closest_exp_std_d = min(available_exp_std_d, key=lambda e: abs(e-std_exp_time)); dark_path_std = state_md_dict_corr[closest_exp_std_d]
                    if dark_path_std and os.path.exists(dark_path_std):
                        d_std_data, d_std_hdr = load_single_fits_from_path(dark_path_std, f"Dark for Std Star {std_filt_char}")
                        if d_std_data is not None: dark_for_std_ccd = CCDData(d_std_data, header=d_std_hdr, unit=u.adu)

                flat_for_std_ccd = None
                flat_key_std_exact = (std_filt_char, std_exp_time if std_exp_time > 0 else -1.0)
                flat_key_std_any_exp = (std_filt_char, -1.0)
                if flat_key_std_exact in tab4_uploaded_flats_ccd_dict: flat_for_std_ccd = tab4_uploaded_flats_ccd_dict[flat_key_std_exact]
                elif flat_key_std_any_exp in tab4_uploaded_flats_ccd_dict: flat_for_std_ccd = tab4_uploaded_flats_ccd_dict[flat_key_std_any_exp]
                elif state_mf_dict_corr:
                    flat_path_std_state = state_mf_dict_corr.get(flat_key_std_exact)
                    # ... (탭2와 유사한 폴백 로직) ...
                    if not flat_path_std_state:
                        paths_for_filter_std = [p for (f,e),p in state_mf_dict_corr.items() if f == std_filt_char]
                        if paths_for_filter_std: 
                            closest_flat_key_std = min([(f,e) for (f,e) in state_mf_dict_corr.keys() if f == std_filt_char], key=lambda k_exp: abs(k_exp[1] - std_exp_time) if std_exp_time > 0 and k_exp[1]>0 else float('inf'), default=None)
                            if closest_flat_key_std: flat_path_std_state = state_mf_dict_corr[closest_flat_key_std]
                        else: 
                            gen_keys_std = [gk for gk in state_mf_dict_corr.keys() if gk[0] == 'Generic']
                            if gen_keys_std: 
                                gen_path_exp_std = state_mf_dict_corr.get(('Generic', std_exp_time if std_exp_time > 0 else -1.0))
                                if gen_path_exp_std and os.path.exists(gen_path_exp_std): flat_path_std_state = gen_path_exp_std
                                elif gen_keys_std: flat_path_std_state = state_mf_dict_corr[gen_keys_std[0]]
                    if flat_path_std_state and os.path.exists(flat_path_std_state):
                        f_std_data, f_std_hdr = load_single_fits_from_path(flat_path_std_state, f"Flat for Std Star {std_filt_char}")
                        if f_std_data is not None: flat_for_std_ccd = CCDData(f_std_data, header=f_std_hdr, unit=u.adu)
                
                cal_std_ccd = ccdp.ccd_process(std_ccd_raw, master_bias=final_mb_ccd, dark_frame=dark_for_std_ccd, master_flat=flat_for_std_ccd, dark_scale=True, error=False)
                
                std_stars_table = detect_stars_dao(cal_std_ccd.data, fwhm, thresh_nsigma)
                if std_stars_table and len(std_stars_table) > 0:
                    if 'flux' in std_stars_table.colnames : std_stars_table.sort('flux', reverse=True)
                    brightest_std_star_photutils = std_stars_table[0]
                    std_phot_table = perform_aperture_photometry_on_detections(cal_std_ccd.data, Table([brightest_std_star_photutils]), ap_radius_phot)
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
                            if m0_eff_var_name_str == 'm0_eff_b': m0_eff_b = calc_m0
                            elif m0_eff_var_name_str == 'm0_eff_v': m0_eff_v = calc_m0
                            status_log.append(f"{std_filt_char}필터 영점(m0_eff) 계산됨: {calc_m0:.3f} (표준별 사용)")
                        else: status_log.append(f"{std_filt_char}필터 표준별 정보 부족으로 영점 자동 계산 불가. 사용자 입력 m0 사용.")
                    else: status_log.append(f"{std_filt_char}필터 표준별 측광 실패.")
                else: status_log.append(f"{std_filt_char}필터 표준별 이미지에서 별 탐지 실패.")
            else: status_log.append(f"{std_filt_char}필터 표준별 파일 로드 실패.")
        else: status_log.append(f"{std_filt_char}필터 표준별 파일 미업로드. 사용자 입력 m0 사용.")

    # --- 4. 대상 LIGHT 프레임 처리 ---
    filter_processed_stars_data = {'B': [], 'V': []} 
    first_light_b_calibrated_ccd_data_for_preview = None 
    first_light_v_calibrated_ccd_data_for_preview = None

    for filter_char_loop, light_objs_loop, k_coeff_loop, m0_eff_loop in [
        ('B', light_b_file_objs, k_b, m0_eff_b), 
        ('V', light_v_file_objs, k_v, m0_eff_v)
    ]:
        if not light_objs_loop: continue
        status_log.append(f"--- {filter_char_loop} 필터 대상 프레임 처리 시작 ({len(light_objs_loop)}개) ---")
        for i_light, light_obj_item in enumerate(light_objs_loop): # 첫번째 이미지 미리보기 위해 인덱스 사용
            if not (light_obj_item and light_obj_item.name and os.path.exists(light_obj_item.name)): continue
            filename_loop = os.path.basename(light_obj_item.name)
            status_log.append(f"처리 중: {filename_loop} ({filter_char_loop})")
            try:
                light_data, header = load_single_fits_from_path(light_obj_item.name, f"{filter_char_loop} LIGHT")
                if light_data is None or header is None: status_log.append(f"오류: {filename_loop} 로드 실패."); continue
                light_ccd_raw = CCDData(light_data, header=header, unit=u.adu)
                current_light_exptime = get_fits_keyword(header, ['EXPTIME', 'EXPOSURE'], -1.0, float)

                # DARK 결정 (탭4 업로드 > 탭1 상태)
                md_to_use_ccd_light = tab4_uploaded_dark_ccd_corrected_dict.get(current_light_exptime if current_light_exptime > 0 else -1.0)
                if md_to_use_ccd_light is None and state_md_dict_corr:
                    dark_path_light = state_md_dict_corr.get(current_light_exptime)
                    if not dark_path_light and current_light_exptime > 0:
                        available_exp_l_d = sorted([k for k in state_md_dict_corr.keys() if isinstance(k, (int, float)) and k > 0])
                        if available_exp_l_d: closest_exp_l_d = min(available_exp_l_d, key=lambda e: abs(e-current_light_exptime)); dark_path_light = state_md_dict_corr[closest_exp_l_d]
                    if dark_path_light and os.path.exists(dark_path_light):
                        d_l_data, d_l_hdr = load_single_fits_from_path(dark_path_light, f"Dark for LIGHT {filename_loop}")
                        if d_l_data is not None: md_to_use_ccd_light = CCDData(d_l_data, header=d_l_hdr, unit=u.adu)
                
                # FLAT 결정 (탭4 업로드 > 탭1 상태)
                mf_to_use_ccd_light = None
                flat_key_light_exact = (filter_char_loop, current_light_exptime if current_light_exptime > 0 else -1.0)
                flat_key_light_any_exp = (filter_char_loop, -1.0)
                if flat_key_light_exact in tab4_uploaded_flats_ccd_dict: mf_to_use_ccd_light = tab4_uploaded_flats_ccd_dict[flat_key_light_exact]
                elif flat_key_light_any_exp in tab4_uploaded_flats_ccd_dict: mf_to_use_ccd_light = tab4_uploaded_flats_ccd_dict[flat_key_light_any_exp]
                elif state_mf_dict_corr:
                    # (탭2와 유사한 폴백 로직으로 state_mf_dict_corr에서 탐색)
                    flat_path_light_state = state_mf_dict_corr.get(flat_key_light_exact)
                    if not flat_path_light_state:
                        paths_for_filter_light = [p for (f,e),p in state_mf_dict_corr.items() if f == filter_char_loop]
                        if paths_for_filter_light: 
                            closest_flat_key_light = min([(f,e) for (f,e) in state_mf_dict_corr.keys() if f == filter_char_loop], key=lambda k_exp: abs(k_exp[1] - current_light_exptime) if current_light_exptime > 0 and k_exp[1]>0 else float('inf'), default=None)
                            if closest_flat_key_light: flat_path_light_state = state_mf_dict_corr[closest_flat_key_light]
                        else: 
                            gen_keys_light = [gk for gk in state_mf_dict_corr.keys() if gk[0] == 'Generic']
                            if gen_keys_light: 
                                gen_path_exp_light = state_mf_dict_corr.get(('Generic', current_light_exptime if current_light_exptime > 0 else -1.0))
                                if gen_path_exp_light and os.path.exists(gen_path_exp_light): flat_path_light_state = gen_path_exp_light
                                elif gen_keys_light: flat_path_light_state = state_mf_dict_corr[gen_keys_light[0]]
                    if flat_path_light_state and os.path.exists(flat_path_light_state):
                        mf_l_data, mf_l_hdr = load_single_fits_from_path(flat_path_light_state, f"Flat for LIGHT {filename_loop}")
                        if mf_l_data is not None: mf_to_use_ccd_light = CCDData(mf_l_data, header=mf_l_hdr, unit=u.adu)

                calibrated_ccd = ccdp.ccd_process(light_ccd_raw, master_bias=final_mb_ccd, dark_frame=md_to_use_ccd_light, master_flat=mf_to_use_ccd_light, dark_scale=True, error=False)
                
                if i_light == 0: # 첫번째 LIGHT 프레임의 보정된 데이터 저장 (미리보기용)
                    if filter_char_loop == 'B': first_light_b_calibrated_ccd_data_for_preview = calibrated_ccd.data
                    elif filter_char_loop == 'V': first_light_v_calibrated_ccd_data_for_preview = calibrated_ccd.data

                detected_stars_table = detect_stars_dao(calibrated_ccd.data, fwhm, thresh_nsigma)
                if detected_stars_table is None or len(detected_stars_table) == 0: status_log.append(f"{filename_loop}: 별 탐지 실패."); continue
                
                phot_input_table = detected_stars_table
                if use_roi: 
                    x_dao, y_dao = detected_stars_table['xcentroid'], detected_stars_table['ycentroid']
                    roi_m = (x_dao >= roi_x0) & (x_dao <= roi_x1) & (y_dao >= roi_y0) & (y_dao <= roi_y1)
                    stars_in_roi_table = detected_stars_table[roi_m]
                    if not len(stars_in_roi_table)>0: status_log.append(f"{filename_loop}: ROI 내 별 없음."); continue # 수정: not stars_in_roi_table -> not len(stars_in_roi_table)>0
                    status_log.append(f"{filename_loop}: {len(stars_in_roi_table)}개 별이 ROI 내에 있음.")
                    phot_input_table = stars_in_roi_table
                
                phot_results_table = perform_aperture_photometry_on_detections(calibrated_ccd.data, phot_input_table, ap_radius_phot)
                if phot_results_table is None or 'net_flux' not in phot_results_table.colnames: status_log.append(f"{filename_loop}: 측광 실패."); continue
                
                ras, decs = convert_pixel_to_wcs(phot_results_table['xcentroid'], phot_results_table['ycentroid'], header)
                airmass_val = calculate_airmass_extinction(header)
                
                for star_idx_loop, star_phot_info in enumerate(phot_results_table):
                    inst_flux = star_phot_info['net_flux']
                    inst_mag = calculate_instrumental_magnitude(inst_flux)
                    std_mag_val = calculate_standard_magnitude(inst_mag, airmass_val, k_coeff_loop, m0_eff_loop) if inst_mag is not None and airmass_val is not None else np.nan
                    
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

    # --- 5. 별 정보 통합, B-V 계산, SIMBAD 질의, 정렬 ---
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

    # 측광 결과 미리보기 이미지 생성
    preview_base_data = first_light_b_calibrated_ccd_data_for_preview if first_light_b_calibrated_ccd_data_for_preview is not None else first_light_v_calibrated_ccd_data_for_preview
    if preview_base_data is not None:
        base_pil = create_preview_image(preview_base_data)
        if base_pil:
            stars_for_preview_drawing = []
            preview_filename = None 
            if first_light_b_calibrated_ccd_data_for_preview is not None and light_b_file_objs: preview_filename = os.path.basename(light_b_file_objs[0].name)
            elif first_light_v_calibrated_ccd_data_for_preview is not None and light_v_file_objs: preview_filename = os.path.basename(light_v_file_objs[0].name)

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
           csv_output_path, photometry_preview_image_pil, \
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

        plt.style.use('seaborn-v0_8-v0_8-darkgrid') 
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


# ==============================================================================
# File: astro_app_main.py
# Description: Main Gradio application script.
# ==============================================================================
import gradio as gr
import numpy as np 
from astropy.utils.exceptions import AstropyWarning 
import os
import tempfile
from datetime import datetime
import logging
import traceback
import warnings
import shutil
import matplotlib 
matplotlib.use('Agg') 

from utils.ui_handlers import (
    handle_tab1_master_frame_creation,
    handle_tab2_light_frame_calibration,
    handle_tab3_extinction_analysis,
    handle_tab4_detailed_photometry,
    handle_tab4_roi_preview_update,
    handle_tab5_plot_hr_diagram 
)

warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger_main = logging.getLogger(__name__)

try:
    script_directory = os.path.dirname(os.path.abspath(__file__))
    temp_subdir_name = "gradio_temp_outputs" 
    APP_TEMP_OUTPUT_DIR = os.path.join(script_directory, temp_subdir_name)
    os.makedirs(APP_TEMP_OUTPUT_DIR, exist_ok=True)
    logger_main.info(f"애플리케이션 임시 출력 디렉토리: {APP_TEMP_OUTPUT_DIR}")
except Exception as e:
    logger_main.critical(f"스크립트 경로에 애플리케이션 임시 출력 디렉토리 생성 실패: {e}", exc_info=True)
    try:
        APP_TEMP_OUTPUT_DIR = tempfile.mkdtemp(prefix="gradio_astro_app_fallback_")
        logger_main.warning(f"대체 경로로 시스템 임시 디렉토리 사용: {APP_TEMP_OUTPUT_DIR}")
    except Exception as e2:
        logger_main.critical(f"대체 임시 디렉토리 생성 실패: {e2}", exc_info=True)
        APP_TEMP_OUTPUT_DIR = "." 
        logger_main.error(f"최후의 수단으로 현재 작업 디렉토리를 임시 파일용으로 사용: {os.path.abspath(APP_TEMP_OUTPUT_DIR)}")


with gr.Blocks(title="천체사진 처리 도구 v0.13 (ccdproc 적용 및 로직 개선)", theme=gr.themes.Soft()) as app: 
    logger_main.info("Gradio Blocks UI 정의 시작.")
    gr.Markdown("# 천체사진 처리 도구 v0.13 (ccdproc 적용 및 H-R도 탭 추가)") 
    gr.Markdown("탭을 선택하여 원하는 작업을 수행하세요. 로그는 콘솔 및 각 탭의 로그 창에 출력됩니다.")

    # 상태 변수
    state_master_bias_path = gr.State(None)
    state_master_darks_corrected_dict = gr.State({}) 
    state_master_flats_corrected_dict = gr.State({}) 
    
    # 탭4 ROI용 상태 변수
    state_tab4_roi_image_data_b = gr.State(None) 
    state_tab4_roi_image_data_v = gr.State(None)


    with gr.Tabs():
        # --- 탭 1 정의 (UI 출력 부분 수정) ---
        with gr.TabItem("1. 마스터 프레임 생성 (ccdproc)"): 
            gr.Markdown("## 마스터 프레임 (BIAS, DARK, FLAT) 생성 (ccdproc 사용)")
            gr.Markdown("각 타입의 FITS 파일들을 업로드하여 마스터 보정 프레임을 생성합니다.\n"
                        "DARK는 노출시간별로, FLAT 프레임은 FITS 헤더의 'FILTER'와 'EXPTIME' 키워드를 읽어 필터/노출시간별로 자동 분류되어 처리됩니다.")
            with gr.Row():
                tab1_bias_input = gr.File(label="BIAS 프레임 업로드 (.fits)", file_count="multiple", file_types=[".fits", ".fit"], type="filepath")
                tab1_dark_input = gr.File(label="DARK 프레임 업로드 (다양한 노출시간 혼합 가능)", file_count="multiple", file_types=[".fits", ".fit"], type="filepath")
                tab1_flat_input_all = gr.File(label="FLAT 프레임 업로드 (다양한 필터/노출시간 혼합 가능)", file_count="multiple", file_types=[".fits", ".fit"], type="filepath")
            
            tab1_process_button = gr.Button("마스터 프레임 생성 시작 (ccdproc)", variant="primary") 
            
            with gr.Accordion("생성된 마스터 프레임 정보 및 로그", open=False):
                gr.Markdown("생성된 마스터 BIAS, DARK(노출시간별), FLAT(필터/노출시간별)은 내부적으로 저장되어 다른 탭에서 사용됩니다.")
                with gr.Row(): 
                    tab1_bias_output_ui = gr.File(label="다운로드: Master BIAS", interactive=False)
                    tab1_dark_output_ui_msg = gr.Textbox(label="Master DARKs 정보", value="노출시간별 Master Dark 생성됨 (로그 확인)", interactive=False, lines=3)
                with gr.Row(): 
                    tab1_flat_b_output_ui_msg = gr.Textbox(label="Master FLAT B 정보", value="생성된 Master Flat B 없음", interactive=False, lines=2)
                    tab1_flat_v_output_ui_msg = gr.Textbox(label="Master FLAT V 정보", value="생성된 Master Flat V 없음", interactive=False, lines=2)
                    tab1_flat_generic_output_ui_msg = gr.Textbox(label="Master FLAT Generic 정보", value="생성된 Master Flat Generic 없음", interactive=False, lines=2)

                tab1_status_output = gr.Textbox(label="처리 상태 및 요약 로그", lines=10, interactive=False, show_copy_button=True)

            tab1_process_button.click(
                fn=handle_tab1_master_frame_creation, 
                inputs=[tab1_bias_input, tab1_dark_input, tab1_flat_input_all, gr.State(APP_TEMP_OUTPUT_DIR)], 
                outputs=[
                    tab1_bias_output_ui, tab1_dark_output_ui_msg, 
                    tab1_flat_b_output_ui_msg, tab1_flat_v_output_ui_msg, tab1_flat_generic_output_ui_msg, 
                    state_master_bias_path, 
                    state_master_darks_corrected_dict, 
                    state_master_flats_corrected_dict, 
                    tab1_status_output
                ]
            )

        # --- 탭 2 정의 (입력 및 핸들러 호출 수정) ---
        with gr.TabItem("2. LIGHT 프레임 보정 (ccdproc)"): 
            gr.Markdown("## LIGHT 프레임 보정 및 미리보기 (ccdproc 사용)")
            gr.Markdown(
                "LIGHT 프레임(들)을 업로드하여 보정하고, 첫 번째 보정 결과를 미리봅니다.\n"
                "탭 1에서 생성된 마스터 프레임이 LIGHT 프레임의 필터와 노출시간에 맞춰 자동으로 사용됩니다.\n"
                "필요시 아래에서 마스터 BIAS, DARK(Raw, 다수 가능), 필터별 FLAT(Raw 또는 Corrected)을 직접 업로드하여 탭1의 결과보다 우선 사용할 수 있습니다."
            )
            with gr.Row():
                with gr.Column(scale=3): 
                    tab2_light_input = gr.File(label="LIGHT 프레임(들) 업로드 (.fits)", file_count="multiple", file_types=[".fits", ".fit"], type="filepath")
                    with gr.Accordion("(선택) 마스터 프레임 직접 업로드", open=False):
                        tab2_uploaded_bias_obj = gr.File(label="Master BIAS 직접 업로드", file_count="single", type="filepath")
                        tab2_uploaded_dark_raw_files = gr.File(label="Master DARK (Raw, 다수, 노출시간별)", file_count="multiple", type="filepath") 
                        tab2_uploaded_flat_b_obj = gr.File(label="Master FLAT B (Raw 또는 Corrected)", file_count="single", type="filepath")
                        tab2_uploaded_flat_v_obj = gr.File(label="Master FLAT V (Raw 또는 Corrected)", file_count="single", type="filepath")
                    with gr.Row():
                         tab2_preview_stretch = gr.Radio(['asinh', 'log', 'linear'], label="미리보기 스트레칭 방식", value='asinh', interactive=True)
                         tab2_asinh_a_param = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Asinh 'a' 값 (Asinh 선택 시)", interactive=True)
                    tab2_calibrate_button = gr.Button("LIGHT 프레임 보정 시작 (ccdproc)", variant="primary") 
                with gr.Column(scale=2): 
                    tab2_preview_image_output_ui = gr.Image(label="첫 번째 보정 이미지 미리보기", type="pil", interactive=False, show_share_button=False, show_download_button=False, height=400)
            with gr.Accordion("보정된 LIGHT 프레임 다운로드 및 로그", open=False):
                tab2_calibrated_lights_output_ui = gr.Files(label="다운로드: 보정된 LIGHT 프레임(들)", interactive=False)
                tab2_status_output = gr.Textbox(label="보정 상태 및 요약 로그", lines=15, interactive=False, show_copy_button=True)
            
            tab2_calibrate_button.click(
                fn=handle_tab2_light_frame_calibration, 
                inputs=[
                    tab2_light_input, 
                    tab2_uploaded_bias_obj, tab2_uploaded_dark_raw_files,
                    tab2_uploaded_flat_b_obj, tab2_uploaded_flat_v_obj,   
                    state_master_bias_path, 
                    state_master_darks_corrected_dict, 
                    state_master_flats_corrected_dict, 
                    tab2_preview_stretch, 
                    tab2_asinh_a_param,
                    gr.State(APP_TEMP_OUTPUT_DIR) 
                ],
                outputs=[tab2_calibrated_lights_output_ui, tab2_preview_image_output_ui, tab2_status_output]
            )


        # --- 탭 3 정의 (입력 상태 변수명 변경) ---
        with gr.TabItem("3. 대기소광계수 계산"):
            gr.Markdown("## 대기소광계수 계산")
            with gr.Row():
                with gr.Column(scale=2): 
                    tab3_light_files_input = gr.File(label="LIGHT 프레임 (B, V 필터 혼합, 다수 업로드)", file_count="multiple", file_types=[".fits", ".fit"], type="filepath")
                    with gr.Accordion("마스터 프레임 업로드 (선택 사항, Raw 권장)", open=False):
                        tab3_mb_input = gr.File(label="Master BIAS (Raw 또는 Corrected)", file_count="single", type="filepath")
                        tab3_md_raw_input = gr.File(label="Master DARK (Raw, Bias 미차감, 노출시간은 헤더 참조)", file_count="single", type="filepath") 
                        tab3_mf_b_raw_input = gr.File(label="Master FLAT B (Raw, Bias/Dark 미차감)", file_count="single", type="filepath")
                        tab3_mf_v_raw_input = gr.File(label="Master FLAT V (Raw, Bias/Dark 미차감)", file_count="single", type="filepath")
                    tab3_star_detect_thresh_slider = gr.Slider(minimum=3, maximum=100, value=30, step=1, label="별 탐지 임계 계수 (Sigma + 2, 높을수록 엄격)", interactive=True)
                    tab3_process_button = gr.Button("대기소광계수 계산 시작", variant="primary")
                with gr.Column(scale=3): 
                    tab3_plot_output = gr.Plot(label="등급 vs. 대기질량 그래프")
            with gr.Accordion("결과 요약 및 상세 데이터", open=True):
                tab3_summary_output = gr.Textbox(label="계산 결과 요약 (소광계수, 영점 등급)", lines=5, interactive=False, show_copy_button=True)
                tab3_dataframe_output = gr.DataFrame(label="개별 프레임 처리 결과", interactive=False, wrap=True, height=400) 
            tab3_log_output = gr.Textbox(label="처리 로그 및 상태", lines=15, interactive=False, show_copy_button=True)
            tab3_process_button.click(
                fn=handle_tab3_extinction_analysis, 
                inputs=[
                    tab3_light_files_input,
                    tab3_mb_input, tab3_md_raw_input,
                    tab3_mf_b_raw_input, tab3_mf_v_raw_input,
                    state_master_bias_path, 
                    state_master_darks_corrected_dict, 
                    state_master_flats_corrected_dict, 
                    tab3_star_detect_thresh_slider,
                    gr.State(APP_TEMP_OUTPUT_DIR) 
                ],
                outputs=[
                    tab3_plot_output,
                    tab3_summary_output,
                    tab3_dataframe_output,
                    tab3_log_output
                ]
            )
        
        # --- 탭 4 정의 (입력 상태 변수명 변경) ---
        with gr.TabItem("4. 상세 측광 및 카탈로그 분석"):
            gr.Markdown("## 상세 측광, 표준등급 계산 및 카탈로그 매칭")
            with gr.Row():
                with gr.Column(scale=2): 
                    gr.Markdown("#### 1. LIGHT 프레임 업로드 (필터별)")
                    tab4_light_b_input = gr.File(label="B 필터 LIGHT 프레임(들)", file_count="multiple", type="filepath", file_types=[".fits", ".fit"])
                    tab4_light_v_input = gr.File(label="V 필터 LIGHT 프레임(들)", file_count="multiple", type="filepath", file_types=[".fits", ".fit"])
                    
                    gr.Markdown("#### 1a. 표준별 FITS 파일 업로드 (선택 사항)")
                    tab4_std_star_b_file_input = gr.File(label="B필터 표준별 FITS 파일 (1개)", file_count="single", type="filepath")
                    tab4_std_star_v_file_input = gr.File(label="V필터 표준별 FITS 파일 (1개)", file_count="single", type="filepath")

                    gr.Markdown("#### 1b. 표준별 정보 입력 (선택 사항, 미입력 시 SIMBAD 조회 시도)")
                    with gr.Row():
                        tab4_std_b_mag_known_input = gr.Number(label="B필터 표준별의 알려진 B등급", interactive=True, info="예: 12.34")
                        tab4_std_v_mag_known_input = gr.Number(label="V필터 표준별의 알려진 V등급", interactive=True, info="예: 11.87")

                    with gr.Accordion("마스터 프레임 직접 업로드 (선택 사항, Raw 권장)", open=False):
                        tab4_uploaded_mb_obj = gr.File(label="Master BIAS (Raw 또는 Corrected)", file_count="single", type="filepath")
                        tab4_uploaded_md_raw_obj = gr.File(label="Master DARK (Raw, Bias 미차감, 노출시간은 헤더 참조)", file_count="single", type="filepath")
                        tab4_uploaded_mf_b_raw_obj = gr.File(label="Master FLAT B (Raw, Bias/Dark 미차감)", file_count="single", type="filepath")
                        tab4_uploaded_mf_v_raw_obj = gr.File(label="Master FLAT V (Raw, Bias/Dark 미차감)", file_count="single", type="filepath")

                    gr.Markdown("#### 2. 대기소광 및 기본 영점 파라미터 입력 (필수)")
                    with gr.Row():
                        tab4_kb_input = gr.Number(label="B필터 소광계수 (k_B)", value=0.25, interactive=True)
                        tab4_m0b_input_user = gr.Number(label="B필터 기본 영점 (m0_B, 표준별 정보 없을 시 사용)", value=20.0, interactive=True) 
                    with gr.Row():
                        tab4_kv_input = gr.Number(label="V필터 소광계수 (k_V)", value=0.15, interactive=True)
                        tab4_m0v_input_user = gr.Number(label="V필터 기본 영점 (m0_V, 표준별 정보 없을 시 사용)", value=20.0, interactive=True) 

                    gr.Markdown("#### 3. 별 탐지 및 측광 파라미터") 
                    tab4_dao_fwhm_input = gr.Slider(minimum=1.0, maximum=20.0, value=3.0, step=0.1, label="DAOStarFinder FWHM (pixels)", interactive=True)
                    tab4_dao_thresh_nsigma_input = gr.Slider(minimum=1.0, maximum=10.0, value=5.0, step=0.1, label="DAOStarFinder 탐지 임계값 (N x Sigma_Background)", interactive=True)
                    tab4_phot_ap_radius_input = gr.Slider(minimum=1.0, maximum=30.0, value=5.0, step=0.5, label="측광 조리개 반경 (pixels)", interactive=True)
                    
                    gr.Markdown("#### 4. ROI (Region of Interest) 설정")
                    tab4_roi_x_min_slider = gr.Slider(label="ROI X 시작", minimum=0, maximum=100, value=0, step=1, interactive=True) 
                    tab4_roi_x_max_slider = gr.Slider(label="ROI X 끝", minimum=0, maximum=100, value=100, step=1, interactive=True) 
                    tab4_roi_y_min_slider = gr.Slider(label="ROI Y 시작", minimum=0, maximum=100, value=0, step=1, interactive=True)
                    tab4_roi_y_max_slider = gr.Slider(label="ROI Y 끝", minimum=0, maximum=100, value=100, step=1, interactive=True)

                    gr.Markdown("#### 5. 카탈로그 검색 파라미터")
                    tab4_simbad_radius_input = gr.Slider(minimum=1.0, maximum=30.0, value=5.0, step=1.0, label="SIMBAD 검색 반경 (arcsec)", interactive=True)
                    tab4_process_button = gr.Button("상세 측광 분석 시작", variant="primary")

                with gr.Column(scale=3): 
                    gr.Markdown("#### ROI 시각화 및 분석 결과 테이블")
                    tab4_roi_preview_image_output = gr.Image(label="ROI 시각화 (첫 번째 B 또는 V 이미지)", type="pil", height=400, interactive=False, show_download_button=False, show_share_button=False) 
                    tab4_photometry_preview_output = gr.Image(label="상세 측광 결과 미리보기 (별 마킹)", type="pil", height=400, interactive=False, show_download_button=True, show_share_button=False) 
                    tab4_dataframe_output = gr.DataFrame(label="측광 결과", interactive=False, wrap=True, height=400) 
                    tab4_csv_download_button = gr.File(label="결과 CSV 파일 다운로드", interactive=False)
            
            tab4_log_output = gr.Textbox(label="처리 로그 및 상태", lines=15, interactive=False, show_copy_button=True)

            roi_preview_inputs = [tab4_light_b_input, tab4_light_v_input, state_tab4_roi_image_data_b, state_tab4_roi_image_data_v, 
                                  tab4_roi_x_min_slider, tab4_roi_x_max_slider, tab4_roi_y_min_slider, tab4_roi_y_max_slider]
            roi_preview_outputs = [tab4_roi_preview_image_output, tab4_roi_x_min_slider, tab4_roi_x_max_slider,
                                   tab4_roi_y_min_slider, tab4_roi_y_max_slider, state_tab4_roi_image_data_b, state_tab4_roi_image_data_v, tab4_log_output]
            tab4_light_b_input.upload(fn=handle_tab4_roi_preview_update, inputs=roi_preview_inputs, outputs=roi_preview_outputs)
            tab4_light_v_input.upload(fn=handle_tab4_roi_preview_update, inputs=roi_preview_inputs, outputs=roi_preview_outputs)
            for roi_slider_comp in [tab4_roi_x_min_slider, tab4_roi_x_max_slider, tab4_roi_y_min_slider, tab4_roi_y_max_slider]:
                roi_slider_comp.release(fn=handle_tab4_roi_preview_update, inputs=roi_preview_inputs, outputs=roi_preview_outputs)

            tab4_process_button.click(
                fn=handle_tab4_detailed_photometry,
                inputs=[
                    tab4_light_b_input, tab4_light_v_input,
                    tab4_std_star_b_file_input, tab4_std_star_v_file_input, 
                    tab4_std_b_mag_known_input, tab4_std_v_mag_known_input,
                    tab4_uploaded_mb_obj, tab4_uploaded_md_raw_obj, 
                    tab4_uploaded_mf_b_raw_obj, tab4_uploaded_mf_v_raw_obj,
                    state_master_bias_path, state_master_darks_corrected_dict, 
                    state_master_flats_corrected_dict, 
                    tab4_kb_input, tab4_m0b_input_user, tab4_kv_input, tab4_m0v_input_user,
                    tab4_dao_fwhm_input, tab4_dao_thresh_nsigma_input, tab4_phot_ap_radius_input,
                    tab4_roi_x_min_slider, tab4_roi_x_max_slider, 
                    tab4_roi_y_min_slider, tab4_roi_y_max_slider,
                    tab4_simbad_radius_input,
                    gr.State(APP_TEMP_OUTPUT_DIR)
                ],
                outputs=[
                    tab4_dataframe_output,
                    tab4_csv_download_button,
                    tab4_photometry_preview_output, 
                    tab4_log_output
                ]
            )
        
        # --- 탭 5 정의 (신규) ---
        with gr.TabItem("5. H-R도 (색-등급도) 그리기"):
            gr.Markdown("## H-R도 (색-등급도) 그리기")
            gr.Markdown("탭 4에서 생성된 CSV 파일을 업로드하여 H-R도(V등급 vs B-V 색지수)를 그립니다.")
            with gr.Row():
                with gr.Column(scale=1):
                    tab5_csv_input = gr.File(label="측광 결과 CSV 파일 업로드 (탭 4에서 생성)", file_types=[".csv"], type="filepath")
                    tab5_plot_hr_button = gr.Button("H-R도 그리기 시작", variant="primary")
                with gr.Column(scale=2):
                    tab5_hr_plot_output = gr.Plot(label="H-R도 (색-등급도)")
            tab5_log_output = gr.Textbox(label="처리 로그", lines=5, interactive=False, show_copy_button=True)

            tab5_plot_hr_button.click(
                fn=handle_tab5_plot_hr_diagram,
                inputs=[tab5_csv_input, gr.State(APP_TEMP_OUTPUT_DIR)], 
                outputs=[tab5_hr_plot_output, tab5_log_output]
            )

    
    gr.Markdown(f"모든 임시 생성 파일은 `{APP_TEMP_OUTPUT_DIR}` 디렉토리에 저장됩니다. 앱 종료 시 자동으로 삭제됩니다.")
    logger_main.info("Gradio UI 정의 완료.")

if __name__ == "__main__":
    logger_main.info("천체사진 처리 도구 Gradio 앱 시작 중...")
    import atexit
    def cleanup_temp_dir():
        logger_main.info(f"애플리케이션 종료. 임시 디렉토리 정리 시도: {APP_TEMP_OUTPUT_DIR}")
        try:
            if os.path.exists(APP_TEMP_OUTPUT_DIR) and APP_TEMP_OUTPUT_DIR != "." and os.path.isdir(APP_TEMP_OUTPUT_DIR): 
                shutil.rmtree(APP_TEMP_OUTPUT_DIR)
                logger_main.info(f"임시 디렉토리 성공적으로 삭제: {APP_TEMP_OUTPUT_DIR}")
            else: logger_main.info(f"임시 디렉토리가 없거나 유효하지 않아 정리 건너<0xEB><0><0x8F><0xEB><0x82><0xB4>니다: {APP_TEMP_OUTPUT_DIR}")
        except Exception as e: logger_main.error(f"임시 디렉토리 정리 중 오류 {APP_TEMP_OUTPUT_DIR}: {e}", exc_info=True)
    atexit.register(cleanup_temp_dir)
    logger_main.info(f"임시 디렉토리({APP_TEMP_OUTPUT_DIR}) 자동 정리 기능 등록됨 (atexit).")
    app.launch() 
    logger_main.info("Gradio 앱이 종료되었습니다.")

