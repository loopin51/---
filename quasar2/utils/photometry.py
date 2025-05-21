

# ==============================================================================
# File: utils_photometry.py
# Description: 별 탐지 및 조리개 측광 관련 함수 모음.
# ==============================================================================
import numpy as np
from astropy.stats import mad_std, SigmaClip
from scipy.ndimage import label, find_objects
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.detection import DAOStarFinder 
import logging

logger_phot = logging.getLogger(__name__)

def detect_stars_extinction(image_data, fwhm_dao=3.0, threshold_nsigma_dao=5.0): # 파라미터명 변경 및 fwhm 추가
    """
    DAOStarFinder를 사용하여 이미지에서 별을 탐지합니다 (대기소광 분석용).
    이 함수는 이제 DAOStarFinder를 직접 호출하고 그 결과를 반환합니다.

    Args:
        image_data (np.ndarray): 별을 탐지할 2D 이미지 데이터.
        fwhm_dao (float): DAOStarFinder용 FWHM (픽셀 단위).
        threshold_nsigma_dao (float): DAOStarFinder용 탐지 임계값 (배경 표준편차의 배수).

    Returns:
        astropy.table.Table or None: DAOStarFinder가 반환하는 별들의 정보 테이블.
                                     탐지 실패 시 None.
    """
    logger_phot.debug(f"별 탐지 시작 (DAOStarFinder for extinction). FWHM={fwhm_dao}, Threshold Nsigma={threshold_nsigma_dao}")
    if image_data is None:
        logger_phot.warning("별 탐지를 위한 이미지 데이터가 없습니다.")
        return None
    
    # DAOStarFinder 호출
    # background_sigma는 detect_stars_dao 내부에서 mad_std로 자동 계산됨
    sources_table = detect_stars_dao(image_data, fwhm=fwhm_dao, threshold_nsigma=threshold_nsigma_dao)
    
    if sources_table is None or len(sources_table) == 0:
        logger_phot.info("DAOStarFinder (for extinction)로 별이 탐지되지 않았습니다.")
        return None
    
    logger_phot.info(f"DAOStarFinder (for extinction)로 {len(sources_table)}개의 별 탐지 완료.")
    return sources_table

def find_brightest_star_extinction(stars):
    """
    탐지된 별 목록에서 가장 밝은 별(flux가 가장 큰 별)을 찾습니다.

    Args:
        stars (list): detect_stars_extinction 함수에서 반환된 별 정보 딕셔너리 리스트.

    Returns:
        dict or None: 가장 밝은 별의 정보 딕셔너리, 별이 없으면 None.
    """
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
    """
    주어진 별 위치에 대해 조리개 측광(aperture photometry)을 수행하여 flux를 계산합니다.
    배경 하늘 값은 별 주변의 annulus 영역에서 추정하여 <0xEC><0x8A><0xA5>니다.

    Args:
        image_data (np.ndarray): 측광을 수행할 2D 이미지 데이터.
        star_position (dict): 별의 정보 ('xcentroid', 'ycentroid', 'radius' 포함).

    Returns:
        tuple: (final_flux, aperture_radius, background_total_in_aperture)
               계산 실패 시 (None, None, None).
    """
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
            logger_phot.warning(f"별의 중심 ({star_position['xcentroid']:.1f}, {star_position['ycentroid']:.1f})이 이미지 경계 밖에 있습니다.")
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
    """
    DAOStarFinder를 사용하여 이미지에서 별을 탐지합니다.

    Args:
        image_data (np.ndarray): 별을 탐지할 2D 이미지 데이터.
        fwhm (float): 별의 FWHM (Full Width at Half Maximum) 추정값 (픽셀 단위).
        threshold_nsigma (float): 탐지 임계값 (배경 표준편차의 배수).
        background_sigma (float, optional): 배경의 표준편차. None이면 mad_std로 추정. Defaults to None.

    Returns:
        astropy.table.Table or None: 탐지된 별들의 정보 테이블, 실패 시 None.
                                     테이블은 'xcentroid', 'ycentroid', 'flux' 등의 컬럼을 포함.
    """
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
    """
    탐지된 별 목록에 대해 조리개 측광을 수행합니다.

    Args:
        image_data (np.ndarray): 측광을 수행할 2D 이미지 데이터.
        detections_table (astropy.table.Table): DAOStarFinder 등에서 반환된 별 탐지 테이블.
                                                'xcentroid', 'ycentroid' 컬럼 필요.
        aperture_radius_px (float): 측광 조리개 반경 (픽셀 단위).
        annulus_rin_factor (float, optional): 배경 annulus 내부 반경 계수 (조리개 반경 대비). Defaults to 1.5.
        annulus_rout_factor (float, optional): 배경 annulus 외부 반경 계수 (조리개 반경 대비). Defaults to 2.5.

    Returns:
        astropy.table.Table or None: 측광 결과가 추가된 테이블 ('net_flux', 'background_total_in_aperture', 'aperture_radius_phot').
                                     실패 시 None.
    """
    if image_data is None or detections_table is None or len(detections_table) == 0:
        logger_phot.warning("측광을 위한 이미지 데이터 또는 탐지된 별 목록이 없습니다.")
        return None

    logger_phot.info(f"{len(detections_table)}개의 탐지된 별에 대해 조리개 측광 시작. 조리개 반경: {aperture_radius_px} px")
    
    positions = np.transpose((detections_table['xcentroid'], detections_table['ycentroid']))
    apertures = CircularAperture(positions, r=aperture_radius_px)
    
    r_in = aperture_radius_px * annulus_rin_factor
    r_out = aperture_radius_px * annulus_rout_factor
    if r_out <= r_in: r_out = r_in + max(1.0, aperture_radius_px * 0.5) 
    
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

def find_brightest_star_extinction(sources_table, fwhm_for_radius_approx=3.0): # 입력 타입을 테이블로, fwhm 인자 추가
    """
    DAOStarFinder 결과 테이블에서 가장 밝은 별(flux가 가장 큰 별)을 찾아 
    필요한 정보(xcentroid, ycentroid, flux, radius(근사치))를 딕셔너리로 반환합니다.

    Args:
        sources_table (astropy.table.Table): DAOStarFinder에서 반환된 별 정보 테이블.
                                             'flux', 'xcentroid', 'ycentroid', 'id' 컬럼 필요.
        fwhm_for_radius_approx (float): 별의 FWHM 추정값. 이를 기반으로 반경을 근사합니다.

    Returns:
        dict or None: 가장 밝은 별의 정보 딕셔너리, 별이 없거나 테이블이 유효하지 않으면 None.
                      딕셔너리는 'flux', 'xcentroid', 'ycentroid', 'radius', 'label' 키를 가짐.
    """
    # if sources_table is None or not isinstance(sources_table, Table) or len(sources_table) == 0:
    #     logger_phot.warning("별 목록 테이블이 비어있거나 유효하지 않아 가장 밝은 별을 찾을 수 없습니다.")
    #     return None
    
    if 'flux' not in sources_table.colnames or \
       'xcentroid' not in sources_table.colnames or \
       'ycentroid' not in sources_table.colnames or \
       'id' not in sources_table.colnames: # DAOStarFinder는 'id' 컬럼을 가짐
        logger_phot.error("'flux', 'xcentroid', 'ycentroid', 'id' 컬럼 중 일부가 탐지된 별 테이블에 없습니다.")
        return None
        
    try:
        sources_table.sort('flux', reverse=True)
        brightest_star_row = sources_table[0]
        
        # FWHM으로부터 sigma를 계산하고, 이를 radius로 사용 (조리개 측광 시 이 반경의 배수를 사용)
        # sigma = fwhm / (2 * sqrt(2 * log(2))) approx fwhm / 2.355
        # 여기서는 간단히 FWHM의 절반 정도를 기본 반경으로 사용하거나,
        # calculate_flux_extinction에서 조리개 크기를 직접 설정하도록 유도할 수 있음.
        # 우선은 이전과 유사하게 임의의 값을 사용하되, fwhm 기반으로 변경
        # radius_approx = fwhm_for_radius_approx / 2.0 # 간단한 근사
        radius_approx = max(2.0, fwhm_for_radius_approx / 2.0) # 최소 2픽셀 보장

        result_dict = {
            'flux': brightest_star_row['flux'],
            'xcentroid': brightest_star_row['xcentroid'],
            'ycentroid': brightest_star_row['ycentroid'],
            'radius': radius_approx, 
            'label': brightest_star_row['id'] 
        }
        logger_phot.debug(f"가장 밝은 별 탐지 (DAO): ID {result_dict['label']}, Flux {result_dict['flux']:.2e}, ApproxRadius {radius_approx:.2f}")
        return result_dict
    except Exception as e:
        logger_phot.error(f"가장 밝은 별 탐색 중 오류 (DAO): {e}", exc_info=True)
        return None

# calculate_flux_extinction 함수는 star_position['radius']를 사용하므로, 
# find_brightest_star_extinction에서 반환된 'radius' 값을 사용하게 됩니다.