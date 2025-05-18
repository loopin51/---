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
from astropy.wcs import WCS # WCS 변환용
from astroquery.simbad import Simbad # SIMBAD 질의용
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

def convert_pixel_to_wcs(x_coords, y_coords, fits_header):
    """ 픽셀 좌표를 WCS (RA, Dec)로 변환합니다. """
    if fits_header is None:
        logger_astro.warning("WCS 변환을 위한 FITS 헤더가 없습니다.")
        return None, None # (ra_array, dec_array)
    try:
        w = WCS(fits_header)
        if not w.is_celestial:
            logger_astro.warning("FITS 헤더에 유효한 천체 WCS 정보가 없습니다.")
            return None, None
        
        # DAOStarFinder 등에서 나온 0-indexed pixel coordinates를 가정 (FITS 표준은 1-indexed)
        # w.all_pix2world의 origin 인자: 0은 0-indexed, 1은 1-indexed
        world_coords = w.all_pix2world(x_coords, y_coords, 0) 
        ra_deg = world_coords[0]
        dec_deg = world_coords[1]
        logger_astro.debug(f"{len(x_coords) if isinstance(x_coords, (list, np.ndarray)) else 1}개 좌표 WCS 변환 완료.")
        return ra_deg, dec_deg
    except Exception as e:
        logger_astro.error(f"픽셀-WCS 변환 중 오류: {e}", exc_info=True)
        return None, None

def calculate_standard_magnitude(instrumental_mag, airmass, k_coeff, m0_coeff):
    """ 기기등급, 대기질량, 소광계수, 영점으로부터 표준등급 계산 """
    if instrumental_mag is None or airmass is None or k_coeff is None or m0_coeff is None or \
       not np.isfinite(instrumental_mag) or not np.isfinite(airmass) or \
       not np.isfinite(k_coeff) or not np.isfinite(m0_coeff):
        logger_astro.debug(f"표준등급 계산 위한 입력값 부족/유효하지 않음: inst_mag={instrumental_mag}, airmass={airmass}, k={k_coeff}, m0={m0_coeff}")
        return np.nan # NaN 반환
    try:
        standard_mag = instrumental_mag - (k_coeff * airmass) - m0_coeff
        logger_astro.debug(f"표준등급 계산: m_inst={instrumental_mag:.3f}, X={airmass:.3f}, k={k_coeff:.3f}, m0={m0_coeff:.3f} -> M_std={standard_mag:.3f}")
        return standard_mag
    except Exception as e:
        logger_astro.error(f"표준등급 계산 중 오류: {e}", exc_info=True)
        return np.nan

def query_simbad_for_object(ra_deg, dec_deg, radius_arcsec=5.0):
    """ 주어진 RA, Dec 좌표 근처의 천체를 SIMBAD에서 검색합니다. """
    if ra_deg is None or dec_deg is None or not np.isfinite(ra_deg) or not np.isfinite(dec_deg):
        return "좌표 없음"
    try:
        logger_astro.debug(f"SIMBAD 질의: RA={ra_deg:.5f}, Dec={dec_deg:.5f}, Radius={radius_arcsec} arcsec")
        simbad_query = Simbad()
        simbad_query.add_votable_fields('otype', 'ids') 
        
        coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
        # SIMBAD 요청 타임아웃 설정 (초 단위)
        simbad_query.TIMEOUT = 10 # 10초
        result_table = simbad_query.query_region(coord, radius=radius_arcsec * u.arcsec)
        
        if result_table is None or len(result_table) == 0:
            logger_astro.debug("SIMBAD: 해당 좌표 근처 천체 없음.")
            return "N/A"
        else:
            main_id = result_table['MAIN_ID'][0].decode('utf-8') if isinstance(result_table['MAIN_ID'][0], bytes) else str(result_table['MAIN_ID'][0])
            obj_type_bytes = result_table['OTYPE'][0]
            obj_type = obj_type_bytes.decode('utf-8') if isinstance(obj_type_bytes, bytes) else str(obj_type_bytes)
            
            ids_bytes = result_table['IDS'][0]
            other_ids_str = ids_bytes.decode('utf-8') if isinstance(ids_bytes, bytes) else str(ids_bytes)
            
            logger_astro.info(f"SIMBAD 결과: ID={main_id}, Type={obj_type}")
            return f"{main_id} ({obj_type})"
    except Exception as e: # TimeoutError 등 네트워크 관련 오류도 포함
        logger_astro.error(f"SIMBAD 질의 중 오류 (RA={ra_deg}, Dec={dec_deg}): {e}", exc_info=False)
        return "SIMBAD 오류"

def match_stars_by_coords(coords_ref, coords_target, tolerance_arcsec):
    """
    두 SkyCoord 객체 간의 별들을 WCS 좌표 기준으로 매칭합니다.
    tolerance_arcsec: 매칭 허용 오차 (arcsec 단위).
    반환값: (매칭된 ref 인덱스 리스트, 매칭된 target 인덱스 리스트, 각 매칭의 거리(arcsec))
    """
    if coords_ref is None or coords_target is None or len(coords_ref) == 0 or len(coords_target) == 0:
        return [], [], []

    try:
        # SkyCoord.match_to_catalog_sky는 ref의 각 요소에 대해 target에서 가장 가까운 것을 찾음
        idx_target, sep2d_angle, _ = coords_ref.match_to_catalog_sky(coords_target)
        
        matched_ref_indices = []
        matched_target_indices = []
        matched_separations_arcsec = []
        
        separation_tolerance = tolerance_arcsec * u.arcsec
        
        for i_ref in range(len(coords_ref)):
            if sep2d_angle[i_ref] < separation_tolerance:
                # 중복 매칭 방지: 하나의 target 별이 여러 ref 별에 매칭되지 않도록 (선택적)
                # 여기서는 가장 가까운 매칭만 고려하므로, match_to_catalog_sky의 기본 동작을 따름
                matched_ref_indices.append(i_ref) # ref 카탈로그에서의 원래 인덱스
                matched_target_indices.append(idx_target[i_ref]) # target 카탈로그에서의 원래 인덱스
                matched_separations_arcsec.append(sep2d_angle[i_ref].arcsec)
        
        logger_astro.info(f"{len(matched_ref_indices)}개의 별 매칭 성공 (허용오차: {tolerance_arcsec} arcsec).")
        return matched_ref_indices, matched_target_indices, matched_separations_arcsec
    except Exception as e:
        logger_astro.error(f"별 좌표 매칭 중 오류: {e}", exc_info=True)
        return [], [], []