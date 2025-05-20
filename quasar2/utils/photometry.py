# ==============================================================================
# File: utils_photometry.py
# Description: Functions for star detection and aperture photometry.
# (이 파일은 이전 버전과 동일하게 유지됩니다. 변경 사항 없음)
# ==============================================================================
import numpy as np
from astropy.stats import mad_std, SigmaClip
from scipy.ndimage import label, find_objects
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.detection import DAOStarFinder 
import logging

logger_phot = logging.getLogger(__name__)

def detect_stars_extinction(image_data, threshold_factor_input=31):
    logger_phot.debug(f"별 탐지 시작. 임계 계수 입력값: {threshold_factor_input}")
    if image_data is None:
        logger_phot.warning("별 탐지를 위한 이미지 데이터가 없습니다.")
        return []
    try:
        if not np.all(np.isfinite(image_data)):
            min_valid = np.nanmin(image_data[np.isfinite(image_data)]) if np.any(np.isfinite(image_data)) else 0
            image_data = np.nan_to_num(image_data, nan=min_valid)

        median_val = np.median(image_data)
        std_val = mad_std(image_data)
        threshold = median_val + (threshold_factor_input + 2) * std_val 
        logger_phot.debug(f"별 탐지 임계값: median={median_val:.2f}, std={std_val:.2f}, threshold={threshold:.2f}")

        binary_map = image_data > threshold
        labeled_map, num_features = label(binary_map)
        if num_features == 0:
            logger_phot.info("임계값 조건에 맞는 별이 탐지되지 않았습니다.")
            return []
        
        star_slices = find_objects(labeled_map)
        stars = []
        for i, star_slice in enumerate(star_slices):
            if star_slice is None: continue
            region = labeled_map[star_slice]
            image_region = image_data[star_slice]
            y_rel, x_rel = np.where(region == (i + 1))
            if y_rel.size == 0: continue

            star_pixels_image_values = image_region[y_rel, x_rel]
            flux = np.sum(star_pixels_image_values)
            
            y_abs = y_rel + star_slice[0].start
            x_abs = x_rel + star_slice[1].start
            
            if np.sum(star_pixels_image_values) > 1e-9: 
                x_mean = np.average(x_abs, weights=star_pixels_image_values)
                y_mean = np.average(y_abs, weights=star_pixels_image_values)
            else: 
                x_mean = np.mean(x_abs)
                y_mean = np.mean(y_abs)

            mean_brightness_in_segment = np.mean(star_pixels_image_values)
            threshold_brightness_for_radius = 0.7 * np.max(image_data) 
            if mean_brightness_in_segment > threshold_brightness_for_radius and y_rel.size > 0:
                radius = np.sqrt(y_rel.size / np.pi)
            else:
                dx = star_slice[1].stop - star_slice[1].start
                dy = star_slice[0].stop - star_slice[0].start
                radius = max(dx, dy) / 2.0
            radius = np.clip(radius, 2.0, 50.0) 

            stars.append({'flux': flux, 'xcentroid': x_mean, 'ycentroid': y_mean, 'radius': radius, 'label': i + 1})
        
        logger_phot.info(f"{len(stars)}개의 별 후보 탐지 완료.")
        return stars
    except Exception as e:
        logger_phot.error(f"별 탐지 중 오류: {e}", exc_info=True)
        raise RuntimeError(f"별 탐지 실패: {e}")

def find_brightest_star_extinction(stars):
    if not stars:
        logger_phot.warning("별 목록이 비어있어 가장 밝은 별을 찾을 수 없습니다.")
        return None
    try:
        brightest_star = max(stars, key=lambda star: star['flux'])
        logger_phot.debug(f"가장 밝은 별 탐지: Label {brightest_star.get('label', 'N/A')}, Flux {brightest_star['flux']:.2e}")
        return brightest_star
    except Exception as e:
        logger_phot.error(f"가장 밝은 별 탐색 중 오류: {e}", exc_info=True)
        raise RuntimeError(f"가장 밝은 별 탐색 실패: {e}")

def calculate_flux_extinction(image_data, star_position):
    logger_phot.debug(f"Flux 계산 시작: Star Pos ({star_position['xcentroid']:.1f}, {star_position['ycentroid']:.1f}), Radius {star_position['radius']:.1f}")
    if image_data is None or star_position is None:
        logger_phot.warning("Flux 계산을 위한 이미지 데이터 또는 별 위치 정보가 없습니다.")
        return None, None, None 
    try:
        aperture_radius = max(star_position['radius'], 2.0) 
        r_in = aperture_radius * 1.5
        r_out = aperture_radius * 2.5
        if r_out <= r_in: r_out = r_in + max(1.0, aperture_radius * 0.5)
        logger_phot.debug(f"Aperture radius: {aperture_radius:.2f}, Annulus r_in: {r_in:.2f}, r_out: {r_out:.2f}")

        aperture = CircularAperture((star_position['xcentroid'], star_position['ycentroid']), r=aperture_radius)
        annulus = CircularAnnulus((star_position['xcentroid'], star_position['ycentroid']), r_in=r_in, r_out=r_out)
        
        h, w = image_data.shape
        if not (0 <= star_position['xcentroid'] < w and 0 <= star_position['ycentroid'] < h):
            logger_phot.warning(f"별의 중심이 이미지 경계 밖에 있습니다.")
            return None, None, None

        sigma_clip_bg = SigmaClip(sigma=3.0, maxiters=5)
        bkg_estimator = np.median 
        
        phot_table = aperture_photometry(image_data, [aperture, annulus]) 
        
        annulus_mask = annulus.to_mask(method='center')
        if annulus_mask is None:
             logger_phot.warning("Annulus 마스크 생성 실패. 배경 0으로 가정.")
             background_mean_per_pix = 0.0
        else:
            annulus_data_raw = annulus_mask.multiply(image_data)
            if annulus_data_raw is None:
                logger_phot.warning("Annulus 데이터가 이미지와 곱해지지 않음. 배경 0으로 가정.")
                background_mean_per_pix = 0.0
            else:
                annulus_pixels = annulus_data_raw[annulus_mask.data > 0]
                annulus_pixels_finite = annulus_pixels[np.isfinite(annulus_pixels)]
                if annulus_pixels_finite.size == 0:
                    logger_phot.warning("Annulus 영역 내 유효한 픽셀 없어 배경 추정 불가. 배경 0으로 가정.")
                    background_mean_per_pix = 0.0
                else:
                    clipped_bg_pixels = sigma_clip_bg(annulus_pixels_finite, masked=False, return_bounds=False)
                    background_mean_per_pix = bkg_estimator(clipped_bg_pixels) if clipped_bg_pixels.size > 0 else 0.0
        
        aperture_area = aperture.area if hasattr(aperture, 'area') else np.pi * aperture_radius**2
        background_total_in_aperture = background_mean_per_pix * aperture_area
        
        source_flux_total = phot_table['aperture_sum_0'][0]
        final_flux = source_flux_total - background_total_in_aperture
        
        logger_phot.info(f"Flux 계산 완료: Total={source_flux_total:.2e}, Bg_in_Aperture={background_total_in_aperture:.2e}, Final={final_flux:.2e}")
        return final_flux, aperture_radius, background_total_in_aperture
    except Exception as e:
        logger_phot.error(f"Flux 계산 중 오류: {e}", exc_info=True)
        return None, None, None

def detect_stars_dao(image_data, fwhm, threshold_nsigma, background_sigma=None):
    logger_phot.debug(f"DAOStarFinder 별 탐지 시작. FWHM={fwhm}, Threshold Nsigma={threshold_nsigma}")
    if image_data is None:
        logger_phot.warning("DAOStarFinder를 위한 이미지 데이터가 없습니다.")
        return None
    
    if not np.all(np.isfinite(image_data)):
        logger_phot.warning("이미지 데이터에 NaN/Inf가 포함되어 있습니다. 0으로 대체합니다.")
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        if background_sigma is None:
            bkg_sigma = mad_std(image_data) 
            logger_phot.debug(f"배경 sigma (mad_std) 추정값: {bkg_sigma:.2f}")
        else:
            bkg_sigma = background_sigma

        if bkg_sigma < 1e-6: 
            logger_phot.warning(f"배경 sigma가 매우 작습니다 ({bkg_sigma:.2e}). 임의의 최소값(1.0)으로 설정합니다.")
            bkg_sigma = 1.0

        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_nsigma * bkg_sigma)
        sources = daofind(image_data) 
        
        if sources is None or len(sources) == 0:
            logger_phot.info("DAOStarFinder로 별이 탐지되지 않았습니다.")
            return None
        
        logger_phot.info(f"DAOStarFinder로 {len(sources)}개의 별 탐지 완료.")
        return sources 
    except Exception as e:
        logger_phot.error(f"DAOStarFinder 별 탐지 중 오류: {e}", exc_info=True)
        return None


def perform_aperture_photometry_on_detections(image_data, detections_table, aperture_radius_px, annulus_rin_factor=1.5, annulus_rout_factor=2.5):
    if image_data is None or detections_table is None or len(detections_table) == 0:
        logger_phot.warning("측광을 위한 이미지 데이터 또는 탐지된 별 목록이 없습니다.")
        return None

    logger_phot.info(f"{len(detections_table)}개의 탐지된 별에 대해 조리개 측광 시작. 조리개 반경: {aperture_radius_px} px")
    
    positions = np.transpose((detections_table['xcentroid'], detections_table['ycentroid']))
    apertures = CircularAperture(positions, r=aperture_radius_px)
    
    r_in = aperture_radius_px * annulus_rin_factor
    r_out = aperture_radius_px * annulus_rout_factor
    if r_out <= r_in: r_out = r_in + max(1.0, aperture_radius_px * 0.5) 
    
    # annuli = CircularAnnulus(positions, r_in=r_in, r_out=r_out) # photutils 1.7+
    
    phot_table_initial = aperture_photometry(image_data, apertures)
    
    final_fluxes = []
    background_totals = []

    sigma_clip_bg = SigmaClip(sigma=3.0, maxiters=5) 
    bkg_estimator = np.median

    for i in range(len(positions)):
        try:
            single_annulus = CircularAnnulus(positions[i], r_in=r_in, r_out=r_out)
            annulus_mask = single_annulus.to_mask(method='center')
            
            if annulus_mask is None:
                logger_phot.warning(f"별 {i}에 대한 Annulus 마스크 생성 실패. 배경 0으로 가정.")
                bg_mean_per_pixel = 0.0
            else:
                annulus_data_raw = annulus_mask.multiply(image_data)
                if annulus_data_raw is None:
                    logger_phot.warning(f"별 {i}의 Annulus 데이터가 이미지와 곱해지지 않음. 배경 0으로 가정.")
                    bg_mean_per_pixel = 0.0
                else:
                    annulus_pixels = annulus_data_raw[annulus_mask.data > 0]
                    annulus_pixels_finite = annulus_pixels[np.isfinite(annulus_pixels)]
                    if annulus_pixels_finite.size == 0:
                        logger_phot.warning(f"별 {i}의 Annulus 내 유효 픽셀 없어 배경 추정 불가. 배경 0으로 가정.")
                        bg_mean_per_pixel = 0.0
                    else:
                        clipped_bg_pixels = sigma_clip_bg(annulus_pixels_finite, masked=False, return_bounds=False)
                        bg_mean_per_pixel = bkg_estimator(clipped_bg_pixels) if clipped_bg_pixels.size > 0 else 0.0
            
            ap_area = apertures[i].area if hasattr(apertures[i], 'area') else np.pi * aperture_radius_px**2
            bg_total_in_aperture = bg_mean_per_pixel * ap_area
            
            source_total_flux = phot_table_initial['aperture_sum'][i]
            net_flux = source_total_flux - bg_total_in_aperture
            
            final_fluxes.append(net_flux)
            background_totals.append(bg_total_in_aperture)
        except Exception as e_phot_single:
            logger_phot.error(f"별 {i} (위치: {positions[i]}) 측광 중 오류: {e_phot_single}", exc_info=False)
            final_fluxes.append(np.nan) 
            background_totals.append(np.nan)

    detections_table['net_flux'] = final_fluxes
    detections_table['background_total_in_aperture'] = background_totals
    detections_table['aperture_radius_phot'] = aperture_radius_px 

    logger_phot.info("조리개 측광 완료.")
    return detections_table