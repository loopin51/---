# ==============================================================================
# File: utils_photometry.py
# Description: Functions for star detection and aperture photometry.
# ==============================================================================
import numpy as np
from astropy.stats import mad_std, SigmaClip
from scipy.ndimage import label, find_objects
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
import logging
from photutils.detection import DAOStarFinder # DAOStarFinder 추가

logger_phot = logging.getLogger(__name__)

def detect_stars_extinction(image_data, threshold_factor_input=31):
    """ 별 탐지 함수 """
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
            
            if np.sum(star_pixels_image_values) > 1e-9: # 가중치가 0이 아닐 때만 가중 평균
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
    """ 가장 밝은 별(최대 flux)을 찾는 함수 """
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
    """ 측광을 수행하여 별의 flux 계산 """
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
                    clipped_pixels = sigma_clip_bg(annulus_pixels_finite, masked=False, return_bounds=False)
                    background_mean_per_pix = bkg_estimator(clipped_pixels) if clipped_pixels.size > 0 else 0.0
        
        aperture_area = aperture.area if hasattr(aperture, 'area') else np.pi * aperture_radius**2
        background_total_in_aperture = background_mean_per_pix * aperture_area
        
        source_flux_total = phot_table['aperture_sum_0'][0]
        final_flux = source_flux_total - background_total_in_aperture
        
        logger_phot.info(f"Flux 계산 완료: Total={source_flux_total:.2e}, Bg_in_Aperture={background_total_in_aperture:.2e}, Final={final_flux:.2e}")
        return final_flux, aperture_radius, background_total_in_aperture
    except Exception as e:
        logger_phot.error(f"Flux 계산 중 오류: {e}", exc_info=True)
        return None, None, None


# ==============================================================================
# File: utils_astro.py
# Description: General astronomical calculation functions.
# ==============================================================================
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u
from sklearn.linear_model import LinearRegression
import logging

logger_astro = logging.getLogger(__name__)

def calculate_altitude_extinction(header):
    """ FITS 헤더로부터 고도(altitude) 계산 """
    logger_astro.debug("고도 계산 시도...")
    if header is None:
        logger_astro.warning("고도 계산을 위한 헤더 정보가 없습니다.")
        return None
    try:
        # 직접적인 고도 키워드 우선 탐색 (사용자 Canvas 문서에서 언급된 키워드 포함)
        alt_keys = ['CENTALT', 'TELALT', 'ALTITUDE', 'OBJALT', 'ALTAZALT', 'ALT-OBJ', 'ALT']
        for key in alt_keys:
            altitude_val = header.get(key)
            if altitude_val is not None:
                try:
                    alt = float(altitude_val)
                    logger_astro.info(f"헤더에서 '{key}' 발견: {alt} deg")
                    return alt
                except ValueError:
                    logger_astro.warning(f"헤더 키워드 '{key}'의 값 '{altitude_val}'을 float으로 변환 불가.")
                    continue # 다음 키워드 시도

        # 계산에 필요한 키워드 (사용자 Canvas 문서에서 언급된 키워드 및 일반적인 변형 포함)
        obs_date = header.get('DATE-OBS', header.get('UTCDATE')) # UTCDATE도 고려
        
        lat_str_keys = ['OBS-LAT', 'LATITUDE', 'SITELAT', 'GEOLAT', 'OBSGEO-B']
        lon_str_keys = ['OBS-LONG', 'LONGITUD', 'SITELONG', 'GEOLON', 'OBSGEO-L']
        ra_str_keys = ['RA', 'OBJRA', 'TELRA', 'CAT-RA', 'RA_OBJ']
        dec_str_keys = ['DEC', 'OBJDEC', 'TELDEC', 'CAT-DEC', 'DEC_OBJ']

        lat_str, lon_str, ra_str, dec_str = None, None, None, None
        for keys in [lat_str_keys, lon_str_keys, ra_str_keys, dec_str_keys]:
            val = None
            for key in keys:
                val = header.get(key)
                if val is not None: break
            if keys is lat_str_keys: lat_str = val
            elif keys is lon_str_keys: lon_str = val
            elif keys is ra_str_keys: ra_str = val
            elif keys is dec_str_keys: dec_str = val
        
        site_elev = header.get('SITEELEV', header.get('OBSGEO-H', 0.0)) # 관측지 고도 (m)

        # 필수 정보 누락 시 오류 발생
        missing_keys = []
        if not obs_date: missing_keys.append("DATE-OBS/UTCDATE")
        if lat_str is None: missing_keys.append("관측지 위도")
        if lon_str is None: missing_keys.append("관측지 경도")
        if ra_str is None: missing_keys.append("목표 RA")
        if dec_str is None: missing_keys.append("목표 DEC")
        if missing_keys:
            raise ValueError(f"고도 계산에 필요한 FITS 헤더 키워드 부족: {', '.join(missing_keys)}")

        observer_lat = float(lat_str)
        observer_lon = float(lon_str)
        observer_height = float(site_elev) * u.m
        
        # RA, DEC 파싱 (다양한 형식 지원 시도)
        try:
            # SkyCoord는 '12 34 56.7 +12 34 56.7' 또는 '12h34m56.7s +12d34m56.7s' 등 파싱
            # 단위가 명시되지 않은 숫자 값은 도(degree)로 간주될 수 있으므로 주의
            # RA는 시간각(hourangle) 단위, DEC는 도(degree) 단위가 일반적
            ra_unit_in = u.hourangle if (isinstance(ra_str, str) and ('h' in ra_str.lower() or 'm' in ra_str.lower() or ':' in ra_str)) else u.deg
            dec_unit_in = u.deg # DEC는 보통 도 단위
            
            target_coord_icrs = SkyCoord(ra_str, dec_str, unit=(ra_unit_in, dec_unit_in), frame='icrs')
        except Exception as e_coord:
            raise ValueError(f"RA/DEC 파싱 오류 ('{ra_str}', '{dec_str}'): {e_coord}")

        location = EarthLocation(lat=observer_lat * u.deg, lon=observer_lon * u.deg, height=observer_height)
        # DATE-OBS 형식은 ISO 8601 (YYYY-MM-DDTHH:MM:SS.sss)이 표준적
        try:
            observation_time = Time(obs_date, format='isot', scale='utc')
        except ValueError: # 다른 시간 형식 시도 (예: JD)
            try:
                observation_time = Time(float(obs_date), format='jd', scale='utc')
                logger_astro.info(f"DATE-OBS를 JD로 해석: {obs_date}")
            except ValueError:
                raise ValueError(f"DATE-OBS ('{obs_date}') 시간 형식 인식 불가. ISO 8601 또는 JD 필요.")
        
        altaz_frame = AltAz(obstime=observation_time, location=location)
        target_altaz = target_coord_icrs.transform_to(altaz_frame)
        
        calculated_alt = target_altaz.alt.deg
        logger_astro.info(f"고도 계산 완료: {calculated_alt:.2f} deg (헤더 정보 기반)")
        return calculated_alt
    except Exception as e:
        logger_astro.warning(f"고도 계산 중 오류: {e}", exc_info=False)
        return None

def calculate_airmass_extinction(header):
    """ FITS 헤더로부터 대기질량(airmass) 계산 """
    logger_astro.debug("대기질량 계산 시도...")
    if header is None:
        logger_astro.warning("대기질량 계산을 위한 헤더 정보가 없습니다.")
        return None
    try:
        airmass_val = header.get('AIRMASS', header.get('SECZ')) # SECZ도 일반적 키워드
        if airmass_val is not None:
            try:
                am = float(airmass_val)
                logger_astro.info(f"헤더에서 AIRMASS/SECZ 발견: {am}")
                return am
            except ValueError:
                 logger_astro.warning(f"헤더 키워드 AIRMASS/SECZ의 값 '{airmass_val}'을 float으로 변환 불가.")
        
        altitude = calculate_altitude_extinction(header)
        if altitude is None or altitude <= 0.5: # 고도가 매우 낮으면 (0.5도 이하) 계산 불안정
            logger_astro.warning(f"유효한 고도({altitude})를 얻을 수 없거나 너무 낮아 대기질량 계산 불가.")
            return None if altitude is None else 38.0 # 매우 낮은 고도에 대한 근사 최대값

        alt_rad = np.radians(altitude)
        if np.sin(alt_rad) < 1e-6 : 
            logger_astro.warning(f"고도({altitude:.2f} deg)가 0에 매우 가까워 대기질량 계산이 불안정. 매우 큰 값 반환.")
            return 38.0 
            
        sec_z = 1.0 / np.sin(alt_rad) # sec(z) = 1/cos(zenith_angle) = 1/sin(altitude)

        # Pickering (2002) 공식 (z < 85도, 즉 고도 > 5도에서 비교적 정확)
        if altitude < 5: 
             logger_astro.warning(f"고도({altitude:.2f} deg)가 5도 미만으로 매우 낮음. Pickering 공식의 정확도 저하 가능. sec(z)={sec_z:.2f} 값을 사용하거나 더 복잡한 모델 필요.")
             # return sec_z # 단순 sec(z) 사용 또는 아래 공식 계속 진행
        
        term1 = sec_z - 1.0
        # 공식이 발산하지 않도록 term1이 과도하게 커지는 것 방지 (sec_z 상한 설정)
        if sec_z > 40: # sec_z가 매우 크면 (고도가 매우 낮으면)
            logger_astro.warning(f"sec(z)={sec_z:.1f}가 매우 커서 Pickering 공식이 불안정할 수 있음. sec(z) 값으로 제한.")
            calculated_airmass = sec_z
        else:
            calculated_airmass = sec_z - 0.0018167 * term1 - 0.002875 * (term1**2) - 0.0008083 * (term1**3)
        
        calculated_airmass = np.clip(calculated_airmass, 1.0, 40.0) # 일반적인 최대값 범위로 클립

        logger_astro.info(f"대기질량 계산 완료: {calculated_airmass:.4f} (고도: {altitude:.2f} deg 사용)")
        return calculated_airmass
    except Exception as e:
        logger_astro.warning(f"대기질량 계산 중 오류: {e}", exc_info=False)
        return None

def calculate_instrumental_magnitude(flux):
    """ Flux로부터 기기등급 계산 """
    if flux is None or not np.isfinite(flux) or flux <= 1e-9: # 매우 작은 양수 flux도 log10에서 문제 가능
        logger_astro.warning(f"유효하지 않거나 매우 작은 flux 값({flux})으로 기기등급 계산 불가.")
        return None
    try:
        magnitude = -2.5 * np.log10(flux)
        logger_astro.debug(f"기기등급 계산: flux={flux:.2e}, mag={magnitude:.4f}")
        return magnitude
    except Exception as e: 
        logger_astro.error(f"기기등급 계산 중 오류 (flux={flux}): {e}", exc_info=True)
        return None

def perform_linear_regression_extinction(airmasses, magnitudes):
    """ 대기질량과 기기등급으로 선형회귀 수행 """
    logger_astro.debug(f"선형 회귀 시작: {len(airmasses)}개의 데이터 포인트.")
    
    valid_points_indices = [
        i for i, (am, mg) in enumerate(zip(airmasses, magnitudes))
        if am is not None and mg is not None and np.isfinite(am) and np.isfinite(mg) and am > 0 # airmass는 양수여야 함
    ]

    if len(valid_points_indices) < 2:
        logger_astro.warning(f"선형 회귀를 위한 유효 데이터 포인트 부족 ({len(valid_points_indices)}개). 최소 2개 필요.")
        return None, None, None, None 

    x_data = np.array([airmasses[i] for i in valid_points_indices])
    y_data = np.array([magnitudes[i] for i in valid_points_indices])
    
    try:
        x_np = x_data.reshape(-1, 1)
        y_np = y_data

        model = LinearRegression().fit(x_np, y_np)
        slope = model.coef_[0]
        intercept = model.intercept_
        r_squared = model.score(x_np, y_np) # 결정 계수 (R-squared)
        
        logger_astro.info(f"선형 회귀 완료: slope(k)={slope:.4f}, intercept(m0)={intercept:.4f}, R^2={r_squared:.4f}")
        return slope, intercept, r_squared, model
    except Exception as e:
        logger_astro.error(f"선형 회귀 중 오류: {e}", exc_info=True)
        return None, None, None, None

def detect_stars_dao(image_data, fwhm, threshold_nsigma, background_sigma=None):
    """ DAOStarFinder를 사용하여 별을 탐지합니다. """
    logger_phot.debug(f"DAOStarFinder 별 탐지 시작. FWHM={fwhm}, Threshold Nsigma={threshold_nsigma}")
    if image_data is None:
        logger_phot.warning("DAOStarFinder를 위한 이미지 데이터가 없습니다.")
        return None
    
    if not np.all(np.isfinite(image_data)):
        logger_phot.warning("이미지 데이터에 NaN/Inf가 포함되어 있습니다. 0으로 대체합니다.")
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        if background_sigma is None:
            # 배경의 표준편차 (sigma)를 mad_std로 추정
            bkg_sigma = mad_std(image_data) # astropy.stats.mad_std 사용
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
    탐지된 별들에 대해 조리개 측광을 수행합니다.
    detections_table: DAOStarFinder 등에서 반환된 Astropy Table (xcentroid, ycentroid 컬럼 필요)
    aperture_radius_px: 조리개 반경 (픽셀 단위)
    annulus_rin_factor: 조리개 반경 대비 내부 Annulus 반경 배수
    annulus_rout_factor: 조리개 반경 대비 외부 Annulus 반경 배수
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
    
    annuli = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    
    # 초기 측광 (배경 미차감)
    phot_table_initial = aperture_photometry(image_data, apertures)
    
    final_fluxes = []
    background_totals = []

    sigma_clip_bg = SigmaClip(sigma=3.0, maxiters=5) # astropy.stats.SigmaClip
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