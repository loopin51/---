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

# --- Gradio 업로드용 임시 디렉토리 설정 ---
GRADIO_CUSTOM_TEMP_DIR = None # 전역 변수로 선언
try:
    script_directory = os.path.dirname(os.path.abspath(__file__))
    gradio_uploads_dir_name = "gradio_temp_files"  
    GRADIO_CUSTOM_TEMP_DIR = os.path.join(script_directory, gradio_uploads_dir_name)
    os.makedirs(GRADIO_CUSTOM_TEMP_DIR, exist_ok=True)
    os.environ['GRADIO_TEMP_DIR'] = GRADIO_CUSTOM_TEMP_DIR 
    logger_main.info(f"Gradio 업로드용 임시 디렉토리 설정됨: {GRADIO_CUSTOM_TEMP_DIR}")
except Exception as e:
    logger_main.error(f"Gradio 업로드용 임시 디렉토리 설정 실패: {e}. 기본 시스템 임시 디렉토리가 사용됩니다.", exc_info=True)
    GRADIO_CUSTOM_TEMP_DIR = None # 설정 실패 시 None으로 유지

# --- 애플리케이션 자체 생성 파일용 임시 디렉토리 설정 ---
APP_TEMP_OUTPUT_DIR = None # 전역 변수로 선언
try:
    script_directory = os.path.dirname(os.path.abspath(__file__)) 
    app_outputs_dir_name = "gradio_temp_outputs" 
    APP_TEMP_OUTPUT_DIR = os.path.join(script_directory, app_outputs_dir_name)
    os.makedirs(APP_TEMP_OUTPUT_DIR, exist_ok=True)
    logger_main.info(f"애플리케이션 생성 파일용 임시 디렉토리: {APP_TEMP_OUTPUT_DIR}")
except Exception as e:
    logger_main.critical(f"스크립트 경로에 애플리케이션 생성 파일용 임시 디렉토리 생성 실패: {e}", exc_info=True)
    try:
        APP_TEMP_OUTPUT_DIR = tempfile.mkdtemp(prefix="gradio_astro_app_fallback_")
        logger_main.warning(f"대체 경로로 시스템 임시 디렉토리 사용 (애플리케이션 출력용): {APP_TEMP_OUTPUT_DIR}")
    except Exception as e2:
        logger_main.critical(f"대체 임시 디렉토리 생성 실패 (애플리케이션 출력용): {e2}", exc_info=True)
        APP_TEMP_OUTPUT_DIR = "." 
        logger_main.error(f"최후의 수단으로 현재 작업 디렉토리를 임시 파일용으로 사용 (애플리케이션 출력용): {os.path.abspath(APP_TEMP_OUTPUT_DIR)}")


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
                tab3_dataframe_output = gr.DataFrame(label="개별 프레임 처리 결과", interactive=False, wrap=True, max_height=400) 
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
                    tab4_dataframe_output = gr.DataFrame(label="측광 결과", interactive=False, wrap=True, max_height=400) 
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
        # APP_TEMP_OUTPUT_DIR (우리 앱이 생성한 파일용) 정리
        logger_main.info(f"애플리케이션 종료. 앱 출력 임시 디렉토리 정리 시도: {APP_TEMP_OUTPUT_DIR}")
        try:
            if APP_TEMP_OUTPUT_DIR and os.path.exists(APP_TEMP_OUTPUT_DIR) and APP_TEMP_OUTPUT_DIR != "." and os.path.isdir(APP_TEMP_OUTPUT_DIR): 
                shutil.rmtree(APP_TEMP_OUTPUT_DIR)
                logger_main.info(f"앱 출력 임시 디렉토리 성공적으로 삭제: {APP_TEMP_OUTPUT_DIR}")
            else:
                logger_main.info(f"앱 출력 임시 디렉토리가 없거나 유효하지 않아 정리 건너<0xEB><0><0x8F><0xEB><0x82><0xB4>니다: {APP_TEMP_OUTPUT_DIR}")
        except Exception as e:
            logger_main.error(f"앱 출력 임시 디렉토리 정리 중 오류 {APP_TEMP_OUTPUT_DIR}: {e}", exc_info=True)

        # GRADIO_CUSTOM_TEMP_DIR (Gradio 업로드 파일용) 정리
        # 이 변수는 astro_app_main.py 상단에서 설정되므로, 여기서도 접근 가능해야 함
        # 단, GRADIO_CUSTOM_TEMP_DIR이 None이 아니고, APP_TEMP_OUTPUT_DIR과 다른 경우에만 삭제 시도
        if 'GRADIO_CUSTOM_TEMP_DIR' in globals() and GRADIO_CUSTOM_TEMP_DIR and GRADIO_CUSTOM_TEMP_DIR != APP_TEMP_OUTPUT_DIR:
            logger_main.info(f"Gradio 업로드 임시 디렉토리 정리 시도: {GRADIO_CUSTOM_TEMP_DIR}")
            try:
                if os.path.exists(GRADIO_CUSTOM_TEMP_DIR) and os.path.isdir(GRADIO_CUSTOM_TEMP_DIR):
                    shutil.rmtree(GRADIO_CUSTOM_TEMP_DIR)
                    logger_main.info(f"Gradio 업로드 임시 디렉토리 성공적으로 삭제: {GRADIO_CUSTOM_TEMP_DIR}")
                else:
                    logger_main.info(f"Gradio 업로드 임시 디렉토리가 없거나 유효하지 않아 정리 건너<0xEB><0><0x8F><0xEB><0x82><0xB4>니다: {GRADIO_CUSTOM_TEMP_DIR}")
            except Exception as e:
                logger_main.error(f"Gradio 업로드 임시 디렉토리 정리 중 오류 {GRADIO_CUSTOM_TEMP_DIR}: {e}", exc_info=True)


    atexit.register(cleanup_temp_dir)
    logger_main.info(f"임시 디렉토리 자동 정리 기능 등록됨 (atexit).")

    app.launch(share=True, debug=True) 
    logger_main.info("Gradio 앱이 종료되었습니다.")