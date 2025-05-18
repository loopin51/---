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
    handle_tab4_detailed_photometry 
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
    APP_TEMP_OUTPUT_DIR = tempfile.mkdtemp(prefix="gradio_astro_app_")
    logger_main.info(f"Application temporary output directory: {APP_TEMP_OUTPUT_DIR}")
except Exception as e:
    logger_main.critical(f"Failed to create application temporary output directory: {e}", exc_info=True)
    APP_TEMP_OUTPUT_DIR = "." 


with gr.Blocks(title="천체사진 처리 도구 v0.9 (필터별 플랫 및 탭4 마스터 로직 개선)", theme=gr.themes.Soft()) as app:
    logger_main.info("Gradio Blocks UI 정의 시작.")
    gr.Markdown("# 천체사진 처리 도구 v0.9 (필터별 플랫 및 탭4 마스터 로직 개선)")
    gr.Markdown("탭을 선택하여 원하는 작업을 수행하세요. 로그는 콘솔 및 각 탭의 로그 창에 출력됩니다.")

    # 상태 변수: 탭1에서 생성된 마스터 프레임 경로 저장용
    state_master_bias_path = gr.State(None)
    state_master_dark_corrected_path = gr.State(None) 
    # 필터별 마스터 플랫 경로
    state_master_flat_b_corrected_path = gr.State(None)
    state_master_flat_v_corrected_path = gr.State(None)
    state_master_flat_generic_corrected_path = gr.State(None) # B, V 외 필터 또는 필터 정보 없는 경우

    with gr.Tabs():
        # --- 탭 1 정의 (수정) ---
        with gr.TabItem("1. 마스터 프레임 생성"):
            gr.Markdown("## 마스터 프레임 (BIAS, DARK, FLAT) 생성")
            gr.Markdown("각 타입의 FITS 파일들을 업로드하여 마스터 보정 프레임을 생성합니다.\n"
                        "FLAT 프레임은 FITS 헤더의 'FILTER' 키워드를 읽어 B, V, 기타(Generic)로 자동 분류되어 처리됩니다.")
            with gr.Row():
                tab1_bias_input = gr.File(label="BIAS 프레임 업로드 (.fits)", file_count="multiple", file_types=[".fits", ".fit"], type="filepath")
                tab1_dark_input = gr.File(label="DARK 프레임 업로드 (.fits)", file_count="multiple", file_types=[".fits", ".fit"], type="filepath")
                tab1_flat_input_all = gr.File(label="FLAT 프레임 업로드 (B, V, 기타 필터 혼합 가능)", file_count="multiple", file_types=[".fits", ".fit"], type="filepath")
            
            tab1_process_button = gr.Button("마스터 프레임 생성 시작", variant="primary")
            
            with gr.Accordion("생성된 마스터 프레임 다운로드 및 로그", open=False):
                with gr.Row():
                    tab1_bias_output_ui = gr.File(label="다운로드: Master BIAS", interactive=False)
                    tab1_dark_output_ui = gr.File(label="다운로드: Master DARK (Corrected)", interactive=False)
                with gr.Row():
                    tab1_flat_b_output_ui = gr.File(label="다운로드: Master FLAT B (Corrected)", interactive=False)
                    tab1_flat_v_output_ui = gr.File(label="다운로드: Master FLAT V (Corrected)", interactive=False)
                    tab1_flat_generic_output_ui = gr.File(label="다운로드: Master FLAT Generic (Corrected)", interactive=False)
                tab1_status_output = gr.Textbox(label="처리 상태 및 요약 로그", lines=10, interactive=False, show_copy_button=True)

            tab1_process_button.click(
                fn=handle_tab1_master_frame_creation, 
                inputs=[tab1_bias_input, tab1_dark_input, tab1_flat_input_all, gr.State(APP_TEMP_OUTPUT_DIR)], 
                outputs=[
                    tab1_bias_output_ui, tab1_dark_output_ui, 
                    tab1_flat_b_output_ui, tab1_flat_v_output_ui, tab1_flat_generic_output_ui,
                    state_master_bias_path, state_master_dark_corrected_path, 
                    state_master_flat_b_corrected_path, state_master_flat_v_corrected_path, state_master_flat_generic_corrected_path,
                    tab1_status_output
                ]
            )

        # --- 탭 2 정의 (수정: Master Flat 업로드 UI 변경 및 입력 상태 변수명 변경) ---
        with gr.TabItem("2. LIGHT 프레임 보정"):
            gr.Markdown("## LIGHT 프레임 보정 및 미리보기")
            gr.Markdown(
                "LIGHT 프레임(들)과 필요한 마스터 프레임들을 업로드하여 보정하고, 첫 번째 보정 결과를 미리봅니다.\n"
                "아래에서 필터별 Master Flat(Corrected)을 직접 업로드하거나, 미업로드 시 탭 1에서 생성된 프레임이 사용됩니다 (필터별 Flat 우선, 없으면 Generic Flat 사용)."
            )
            with gr.Row():
                with gr.Column(scale=3): 
                    tab2_light_input = gr.File(label="LIGHT 프레임(들) 업로드 (.fits)", file_count="multiple", file_types=[".fits", ".fit"], type="filepath")
                    with gr.Accordion("(선택) 마스터 프레임 직접 업로드", open=False):
                        # tab2_mb_input: BIAS는 탭1의 상태를 우선 사용하거나, 필요시 여기에 추가 가능
                        # tab2_md_input: DARK도 탭1의 상태를 우선 사용하거나, 필요시 여기에 추가 가능
                        tab2_mf_b_corr_input = gr.File(label="Master FLAT B (Corrected) 직접 업로드", file_count="single", file_types=[".fits", ".fit"], type="filepath")
                        tab2_mf_v_corr_input = gr.File(label="Master FLAT V (Corrected) 직접 업로드", file_count="single", file_types=[".fits", ".fit"], type="filepath")
                    with gr.Row():
                         tab2_preview_stretch = gr.Radio(['asinh', 'log', 'linear'], label="미리보기 스트레칭 방식", value='asinh', interactive=True)
                         tab2_asinh_a_param = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Asinh 'a' 값 (Asinh 선택 시)", interactive=True)
                    tab2_calibrate_button = gr.Button("LIGHT 프레임 보정 시작", variant="primary")
                with gr.Column(scale=2): 
                    tab2_preview_image_output_ui = gr.Image(label="첫 번째 보정 이미지 미리보기", type="pil", interactive=False, show_share_button=False, show_download_button=False, height=400)
            with gr.Accordion("보정된 LIGHT 프레임 다운로드 및 로그", open=False):
                tab2_calibrated_lights_output_ui = gr.Files(label="다운로드: 보정된 LIGHT 프레임(들)", interactive=False)
                tab2_status_output = gr.Textbox(label="보정 상태 및 요약 로그", lines=15, interactive=False, show_copy_button=True)
            tab2_calibrate_button.click(
                fn=handle_tab2_light_frame_calibration, 
                inputs=[
                    tab2_light_input, 
                    None, None, # 탭2에서 Bias, Dark 직접 업로드는 일단 생략 (탭1 상태 의존) -> 필요시 여기에 gr.File 추가
                    tab2_mf_b_corr_input, tab2_mf_v_corr_input, # 탭2에서 업로드한 필터별 Corrected Flat
                    state_master_bias_path, state_master_dark_corrected_path, 
                    state_master_flat_b_corrected_path, state_master_flat_v_corrected_path, state_master_flat_generic_corrected_path, 
                    tab2_preview_stretch, tab2_asinh_a_param,
                    gr.State(APP_TEMP_OUTPUT_DIR) 
                ],
                outputs=[tab2_calibrated_lights_output_ui, tab2_preview_image_output_ui, tab2_status_output]
            )

        # --- 탭 3 정의 (수정: 입력 상태 변수명 변경) ---
        with gr.TabItem("3. 대기소광계수 계산"):
            gr.Markdown("## 대기소광계수 계산")
            gr.Markdown(
                "여러 LIGHT 프레임(B, V 필터)과 보정 프레임들을 사용하여 대기소광계수를 계산합니다.\n"
                "LIGHT 프레임 FITS 헤더에 'FILTER' (B 또는 V), 'DATE-OBS', 관측지 정보, 목표 좌표 (또는 'CENTALT'/'AIRMASS') 정보가 필요합니다.\n"
                "마스터 프레임 미업로드 시 탭1에서 생성된 프레임을 사용합니다. 필터별 Flat(Raw) 직접 업로드를 권장합니다."
            )
            with gr.Row():
                with gr.Column(scale=2): 
                    tab3_light_files_input = gr.File(label="LIGHT 프레임 (B, V 필터 혼합, 다수 업로드)", file_count="multiple", file_types=[".fits", ".fit"], type="filepath")
                    with gr.Accordion("마스터 프레임 업로드 (선택 사항, Raw 권장)", open=False):
                        tab3_mb_input = gr.File(label="Master BIAS (Raw 또는 Corrected)", file_count="single", type="filepath")
                        tab3_md_raw_input = gr.File(label="Master DARK (Raw, Bias 미차감)", file_count="single", type="filepath")
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
                    state_master_bias_path, state_master_dark_corrected_path, 
                    state_master_flat_b_corrected_path, state_master_flat_v_corrected_path, state_master_flat_generic_corrected_path,
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
        
        # --- 탭 4 정의 (수정: 마스터 프레임 업로드 UI 추가 및 입력 상태 변수명 변경) ---
        with gr.TabItem("4. 상세 측광 및 카탈로그 분석"):
            gr.Markdown("## 상세 측광, 표준등급 계산 및 카탈로그 매칭")
            gr.Markdown(
                "B 및 V 필터 LIGHT 프레임을 업로드하고, 보정 및 측광 후 표준 등급을 계산합니다.\n"
                "FITS 헤더에 WCS 정보가 필요하며, 이를 바탕으로 SIMBAD 카탈로그 정보를 조회합니다.\n"
                "아래에서 마스터 프레임(Raw 권장)을 직접 업로드하거나, 미업로드 시 탭 1에서 생성된 프레임이 사용됩니다."
            )
            with gr.Row():
                with gr.Column(scale=2): 
                    gr.Markdown("#### 1. LIGHT 프레임 업로드 (필터별)")
                    tab4_light_b_input = gr.File(label="B 필터 LIGHT 프레임(들)", file_count="multiple", type="filepath", file_types=[".fits", ".fit"])
                    tab4_light_v_input = gr.File(label="V 필터 LIGHT 프레임(들)", file_count="multiple", type="filepath", file_types=[".fits", ".fit"])
                    
                    with gr.Accordion("마스터 프레임 직접 업로드 (선택 사항, Raw 권장)", open=False):
                        tab4_uploaded_mb_obj = gr.File(label="Master BIAS (Raw 또는 Corrected)", file_count="single", type="filepath")
                        tab4_uploaded_md_raw_obj = gr.File(label="Master DARK (Raw, Bias 미차감)", file_count="single", type="filepath")
                        tab4_uploaded_mf_b_raw_obj = gr.File(label="Master FLAT B (Raw, Bias/Dark 미차감)", file_count="single", type="filepath")
                        tab4_uploaded_mf_v_raw_obj = gr.File(label="Master FLAT V (Raw, Bias/Dark 미차감)", file_count="single", type="filepath")

                    gr.Markdown("#### 2. 대기소광 및 영점 파라미터 입력 (필수)")
                    with gr.Row():
                        tab4_kb_input = gr.Number(label="B필터 소광계수 (k_B)", value=0.25, interactive=True)
                        tab4_m0b_input = gr.Number(label="B필터 영점 (m0_B)", value=20.0, interactive=True)
                    with gr.Row():
                        tab4_kv_input = gr.Number(label="V필터 소광계수 (k_V)", value=0.15, interactive=True)
                        tab4_m0v_input = gr.Number(label="V필터 영점 (m0_V)", value=20.0, interactive=True)

                    gr.Markdown("#### 3. 별 탐지 및 측광 파라미터")
                    tab4_dao_fwhm_input = gr.Slider(minimum=1.0, maximum=20.0, value=3.0, step=0.1, label="DAOStarFinder FWHM (pixels)", interactive=True)
                    tab4_dao_thresh_nsigma_input = gr.Slider(minimum=1.0, maximum=10.0, value=5.0, step=0.1, label="DAOStarFinder 탐지 임계값 (N x Sigma_Background)", interactive=True)
                    tab4_phot_ap_radius_input = gr.Slider(minimum=1.0, maximum=30.0, value=5.0, step=0.5, label="측광 조리개 반경 (pixels)", interactive=True)
                    
                    gr.Markdown("#### 4. 카탈로그 검색 파라미터")
                    tab4_simbad_radius_input = gr.Slider(minimum=1.0, maximum=30.0, value=5.0, step=1.0, label="SIMBAD 검색 반경 (arcsec)", interactive=True)

                    tab4_process_button = gr.Button("상세 측광 분석 시작", variant="primary")

                with gr.Column(scale=3): 
                    gr.Markdown("#### 분석 결과 테이블")
                    tab4_dataframe_output = gr.DataFrame(label="측광 결과", interactive=False, wrap=True, height=600) 
                    tab4_csv_download_button = gr.File(label="결과 CSV 파일 다운로드", interactive=False)
            
            tab4_log_output = gr.Textbox(label="처리 로그 및 상태", lines=15, interactive=False, show_copy_button=True)

            tab4_process_button.click(
                fn=handle_tab4_detailed_photometry,
                inputs=[
                    tab4_light_b_input, tab4_light_v_input,
                    tab4_uploaded_mb_obj, tab4_uploaded_md_raw_obj, 
                    tab4_uploaded_mf_b_raw_obj, tab4_uploaded_mf_v_raw_obj,
                    state_master_bias_path, state_master_dark_corrected_path, 
                    state_master_flat_b_corrected_path, state_master_flat_v_corrected_path, state_master_flat_generic_corrected_path,
                    tab4_kb_input, tab4_m0b_input, tab4_kv_input, tab4_m0v_input,
                    tab4_dao_fwhm_input, tab4_dao_thresh_nsigma_input, tab4_phot_ap_radius_input,
                    tab4_simbad_radius_input,
                    gr.State(APP_TEMP_OUTPUT_DIR)
                ],
                outputs=[
                    tab4_dataframe_output,
                    tab4_csv_download_button,
                    tab4_log_output
                ]
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
            else:
                logger_main.info(f"임시 디렉토리가 없거나 유효하지 않아 정리 건너<0xEB><0><0x8F><0xEB><0x82><0xB4>니다: {APP_TEMP_OUTPUT_DIR}")
        except Exception as e:
            logger_main.error(f"임시 디렉토리 정리 중 오류 {APP_TEMP_OUTPUT_DIR}: {e}", exc_info=True)

    atexit.register(cleanup_temp_dir)
    logger_main.info(f"임시 디렉토리({APP_TEMP_OUTPUT_DIR}) 자동 정리 기능 등록됨 (atexit).")

    app.launch(share=True, debug=True, show_error=True)
    logger_main.info("Gradio 앱이 종료되었습니다.")
