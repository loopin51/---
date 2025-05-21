# ==============================================================================
# File: utils_astro.py
# Description: 일반적인 천문 계산 관련 유틸리티 함수 모음.
# 고도, 대기질량, 기기등급, 표준등급 계산, SIMBAD 질의, 좌표 매칭 등.
# ==============================================================================
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u
from astropy.wcs import WCS
from sklearn.linear_model import LinearRegression
from astroquery.simbad import Simbad
import logging

logger_astro = logging.getLogger(__name__)

def calculate_altitude_extinction(header):
    """
    FITS 헤더 정보로부터 천체의 고도(altitude)를 계산합니다.
    헤더에 직접 고도 정보가 있으면 사용하고, 없으면 관측 시간, 위치, 천체 좌표를 이용해 계산합니다.

    Args:
        header (astropy.io.fits.Header): FITS 헤더 객체.

    Returns:
        float or None: 계산된 고도 (도 단위), 실패 시 None.
    """
    logger_astro.debug("고도 계산 시도...")
    if header is None: logger_astro.warning("고도 계산용 헤더 없음."); return None
    try:
        alt_keys = ['CENTALT', 'TELALT', 'ALTITUDE', 'OBJALT', 'ALTAZALT', 'ALT-OBJ', 'ALT']
        for key in alt_keys:
            altitude_val = header.get(key)
            if altitude_val is not None:
                try: alt = float(altitude_val); logger_astro.info(f"헤더 '{key}': {alt} deg"); return alt
                except ValueError: logger_astro.warning(f"헤더 '{key}' 값 '{altitude_val}' float 변환 불가."); continue

        obs_date = header.get('DATE-OBS', header.get('UTCDATE'))
        lat_str_keys = ['OBS-LAT', 'LATITUDE', 'SITELAT', 'GEOLAT', 'OBSGEO-B']; lon_str_keys = ['OBS-LONG', 'LONGITUD', 'SITELONG', 'GEOLON', 'OBSGEO-L']
        ra_str_keys = ['RA', 'OBJRA', 'TELRA', 'CAT-RA', 'RA_OBJ']; dec_str_keys = ['DEC', 'OBJDEC', 'TELDEC', 'CAT-DEC', 'DEC_OBJ']
        lat_str, lon_str, ra_str, dec_str = None, None, None, None
        for keys_list, var_name_str in [(lat_str_keys, "lat_str"), (lon_str_keys, "lon_str"), (ra_str_keys, "ra_str"), (dec_str_keys, "dec_str")]:
            val_found = None; 
            for key_item in keys_list: val_found = header.get(key_item); 
            if val_found is not None: break 
            if var_name_str == "lat_str": lat_str = val_found
            elif var_name_str == "lon_str": lon_str = val_found
            elif var_name_str == "ra_str": ra_str = val_found
            elif var_name_str == "dec_str": dec_str = val_found

        site_elev = header.get('SITEELEV', header.get('OBSGEO-H', 0.0)) 
        missing = [n for v,n in [(obs_date,"DATE-OBS/UTCDATE"),(lat_str,"위도"),(lon_str,"경도"),(ra_str,"RA"),(dec_str,"DEC")] if not v or v is None]
        if missing: raise ValueError(f"고도 계산 필요 키워드 부족: {', '.join(missing)}")
        obs_lat, obs_lon, obs_h = float(lat_str), float(lon_str), float(site_elev)*u.m
        try:
            ra_u = u.hourangle if (isinstance(ra_str, str) and any(c in ra_str.lower() for c in 'hms:')) else u.deg
            dec_u = u.deg
            coord = SkyCoord(str(ra_str), str(dec_str), unit=(ra_u, dec_u), frame='icrs')
        except Exception as e_coord: raise ValueError(f"RA/DEC 파싱 오류 ('{ra_str}', '{dec_str}'): {e_coord}")
        loc = EarthLocation(lat=obs_lat*u.deg, lon=obs_lon*u.deg, height=obs_h)
        try: obs_time = Time(obs_date, format='isot', scale='utc')
        except ValueError: 
            try: obs_time = Time(float(obs_date), format='jd', scale='utc'); logger_astro.info(f"DATE-OBS JD로 해석: {obs_date}")
            except ValueError: raise ValueError(f"DATE-OBS ('{obs_date}') 시간 형식 인식 불가.")
        altaz_frame = AltAz(obstime=obs_time, location=loc)
        alt = coord.transform_to(altaz_frame).alt.deg
        logger_astro.info(f"고도 계산 완료: {alt:.2f} deg"); return alt
    except Exception as e: logger_astro.warning(f"고도 계산 중 오류: {e}", exc_info=False); return None

def calculate_airmass_extinction(header):
    """
    FITS 헤더 정보로부터 천체의 대기질량(airmass)을 계산합니다.
    헤더에 직접 대기질량 정보가 있으면 사용하고, 없으면 고도를 이용해 Pickering (2002) 공식을 사용합니다.

    Args:
        header (astropy.io.fits.Header): FITS 헤더 객체.

    Returns:
        float or None: 계산된 대기질량, 실패 시 None.
    """
    logger_astro.debug("대기질량 계산 시도...")
    if header is None: logger_astro.warning("대기질량 계산용 헤더 없음."); return None
    try:
        airmass_val = header.get('AIRMASS', header.get('SECZ')) 
        if airmass_val is not None:
            try: am = float(airmass_val); logger_astro.info(f"헤더 AIRMASS/SECZ: {am}"); return am
            except ValueError: logger_astro.warning(f"헤더 AIRMASS/SECZ 값 '{airmass_val}' float 변환 불가.")
        alt = calculate_altitude_extinction(header)
        if alt is None or alt <= 0.5: logger_astro.warning(f"유효 고도({alt}) 부족/낮음."); return None if alt is None else 38.0
        alt_rad = np.radians(alt)
        if np.sin(alt_rad) < 1e-6 : logger_astro.warning(f"고도({alt:.2f} deg) 0에 가까움."); return 38.0
        sec_z = 1.0 / np.sin(alt_rad)
        if alt < 5: logger_astro.warning(f"고도({alt:.2f} deg) 5도 미만, Pickering 공식 정확도 저하 가능.")
        term1 = sec_z - 1.0
        if sec_z > 40: logger_astro.warning(f"sec(z)={sec_z:.1f} 매우 큼. sec(z) 값으로 제한."); calculated_airmass = sec_z
        else: calculated_airmass = sec_z - 0.0018167*term1 - 0.002875*(term1**2) - 0.0008083*(term1**3)
        calculated_airmass = np.clip(calculated_airmass, 1.0, 40.0)
        logger_astro.info(f"대기질량 계산: {calculated_airmass:.4f} (고도: {alt:.2f} deg)"); return calculated_airmass
    except Exception as e: logger_astro.warning(f"대기질량 계산 오류: {e}", exc_info=False); return None

def calculate_instrumental_magnitude(flux):
    """
    측광된 flux 값으로부터 기기 등급(instrumental magnitude)을 계산합니다.

    Args:
        flux (float): 별의 net flux 값.

    Returns:
        float or None: 계산된 기기 등급, flux가 유효하지 않으면 None.
    """
    if flux is None or not np.isfinite(flux) or flux <= 1e-9: 
        logger_astro.warning(f"유효X/매우 작은 flux ({flux})로 기기등급 계산 불가.")
        return None
    try: mag = -2.5 * np.log10(flux); logger_astro.debug(f"기기등급: flux={flux:.2e}, mag={mag:.4f}"); return mag
    except Exception as e: logger_astro.error(f"기기등급 계산 오류 (flux={flux}): {e}", exc_info=True); return None

def perform_linear_regression_extinction(airmasses, magnitudes):
    """
    주어진 대기질량(X)과 기기등급(m_inst) 데이터로부터 선형 회귀를 수행하여
    대기소광계수(k)와 영점(m0)을 계산합니다. (m_inst = m0 + k*X)

    Args:
        airmasses (list or np.ndarray): 대기질량 값들의 리스트.
        magnitudes (list or np.ndarray): 해당 기기등급 값들의 리스트.

    Returns:
        tuple: (slope, intercept, r_squared, model)
               slope (float): 대기소광계수 k.
               intercept (float): 영점 m0.
               r_squared (float): 결정 계수 R².
               model (sklearn.linear_model.LinearRegression): 학습된 회귀 모델.
               실패 시 (None, None, None, None).
    """
    logger_astro.debug(f"선형 회귀 시작: {len(airmasses)}개의 데이터 포인트.")
    valid_indices = [i for i, (am,mg) in enumerate(zip(airmasses,magnitudes)) if am is not None and mg is not None and np.isfinite(am) and np.isfinite(mg) and am > 0]
    if len(valid_indices) < 2: logger_astro.warning(f"선형 회귀 유효 데이터 부족 ({len(valid_indices)}개)."); return None,None,None,None
    x_data = np.array([airmasses[i] for i in valid_indices]); y_data = np.array([magnitudes[i] for i in valid_indices])
    try:
        x_np = x_data.reshape(-1,1); y_np = y_data
        model = LinearRegression().fit(x_np, y_np)
        slope, intercept, r_sq = model.coef_[0], model.intercept_, model.score(x_np, y_np)
        logger_astro.info(f"선형 회귀: k={slope:.4f}, m0={intercept:.4f}, R²={r_sq:.4f}"); return slope, intercept, r_sq, model
    except Exception as e: logger_astro.error(f"선형 회귀 오류: {e}", exc_info=True); return None,None,None,None

def convert_pixel_to_wcs(x_coords, y_coords, fits_header):
    """
    FITS 헤더의 WCS 정보를 사용하여 픽셀 좌표를 천구 좌표(RA, Dec)로 변환합니다.

    Args:
        x_coords (float or list/np.ndarray): 변환할 X 픽셀 좌표(들).
        y_coords (float or list/np.ndarray): 변환할 Y 픽셀 좌표(들).
        fits_header (astropy.io.fits.Header): WCS 정보가 포함된 FITS 헤더.

    Returns:
        tuple: (ra_degrees, dec_degrees)
               변환 실패 시 (None, None).
    """
    if fits_header is None: logger_astro.warning("WCS 변환용 FITS 헤더 없음."); return None, None
    try:
        w = WCS(fits_header)
        if not w.is_celestial: logger_astro.warning("FITS 헤더에 유효 천체 WCS 정보 없음."); return None, None
        world_coords = w.all_pix2world(x_coords, y_coords, 0) 
        ra_deg, dec_deg = world_coords[0], world_coords[1]
        logger_astro.debug(f"{len(x_coords) if isinstance(x_coords, (list, np.ndarray)) else 1}개 좌표 WCS 변환 완료."); return ra_deg, dec_deg
    except Exception as e: logger_astro.error(f"픽셀-WCS 변환 오류: {e}", exc_info=True); return None, None

def calculate_standard_magnitude(instrumental_mag, airmass, k_coeff, m0_coeff):
    """
    기기 등급, 대기질량, 대기소광계수, 영점을 이용하여 표준 등급을 계산합니다.
    M_std = m_inst - k*X - m0

    Args:
        instrumental_mag (float): 기기 등급.
        airmass (float): 대기질량 (X).
        k_coeff (float): 대기소광계수 (k).
        m0_coeff (float): 영점 (m0).

    Returns:
        float or np.nan: 계산된 표준 등급, 입력값 유효하지 않으면 np.nan.
    """
    if instrumental_mag is None or airmass is None or k_coeff is None or m0_coeff is None or \
       not np.isfinite(instrumental_mag) or not np.isfinite(airmass) or \
       not np.isfinite(k_coeff) or not np.isfinite(m0_coeff):
        logger_astro.debug(f"표준등급 계산 입력값 부족/유효X: inst_mag={instrumental_mag}, airmass={airmass}, k={k_coeff}, m0={m0_coeff}")
        return np.nan
    try:
        standard_mag = instrumental_mag - (k_coeff * airmass) - m0_coeff
        logger_astro.debug(f"표준등급: m_inst={instrumental_mag:.3f}, X={airmass:.3f}, k={k_coeff:.3f}, m0={m0_coeff:.3f} -> M_std={standard_mag:.3f}")
        return standard_mag
    except Exception as e: logger_astro.error(f"표준등급 계산 오류: {e}", exc_info=True); return np.nan

def query_simbad_for_object(ra_deg, dec_deg, radius_arcsec=5.0):
    """
    주어진 천구 좌표(RA, Dec) 근처의 천체를 SIMBAD에서 조회합니다.

    Args:
        ra_deg (float): RA (도 단위).
        dec_deg (float): Dec (도 단위).
        radius_arcsec (float, optional): 검색 반경 (arcsecond 단위). Defaults to 5.0.

    Returns:
        str: 찾은 천체의 주 ID와 타입 (예: "M 31 (Galaxy)").
             못 찾거나 오류 시 "N/A", "좌표 없음", "SIMBAD 오류" 등 반환.
    """
    if ra_deg is None or dec_deg is None or not np.isfinite(ra_deg) or not np.isfinite(dec_deg): return "좌표 없음"
    try:
        logger_astro.debug(f"SIMBAD 질의: RA={ra_deg:.5f}, Dec={dec_deg:.5f}, Radius={radius_arcsec} arcsec")
        simbad_query = Simbad()
        simbad_query.add_votable_fields('otype', 'ids') 
        coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
        simbad_query.TIMEOUT = 10 
        result_table = simbad_query.query_region(coord, radius=radius_arcsec * u.arcsec)
        
        if result_table is None or len(result_table) == 0: 
            logger_astro.debug("SIMBAD: 해당 좌표 근처 천체 없음."); return "N/A"
        else:
            main_id = result_table['MAIN_ID'][0].decode('utf-8') if isinstance(result_table['MAIN_ID'][0], bytes) else str(result_table['MAIN_ID'][0])
            obj_type_bytes = result_table['OTYPE'][0]
            obj_type = obj_type_bytes.decode('utf-8') if isinstance(obj_type_bytes, bytes) else str(obj_type_bytes)
            logger_astro.info(f"SIMBAD 결과: ID={main_id}, Type={obj_type}"); return f"{main_id} ({obj_type})"
    except Exception as e: 
        logger_astro.error(f"SIMBAD 질의 오류 (RA={ra_deg}, Dec={dec_deg}): {e}", exc_info=False); return "SIMBAD 오류"

def match_stars_by_coords(coords_ref, coords_target, tolerance_arcsec):
    """
    두 별 목록의 천구 좌표를 비교하여 서로 매칭되는 별들의 인덱스를 찾습니다.

    Args:
        coords_ref (astropy.coordinates.SkyCoord): 기준 별 목록의 SkyCoord 객체.
        coords_target (astropy.coordinates.SkyCoord): 대상 별 목록의 SkyCoord 객체.
        tolerance_arcsec (float): 매칭 허용 오차 (arcsecond 단위).

    Returns:
        tuple: (matched_ref_indices, matched_target_indices, matched_separations_arcsec)
               각 리스트는 매칭된 별들의 인덱스와 각도 거리를 담고 있음.
    """
    if coords_ref is None or coords_target is None or len(coords_ref) == 0 or len(coords_target) == 0: return [], [], []
    try:
        idx_target, sep2d_angle, _ = coords_ref.match_to_catalog_sky(coords_target)
        matched_ref_indices, matched_target_indices, matched_separations_arcsec = [], [], []
        separation_tolerance = tolerance_arcsec * u.arcsec 

        for i_ref in range(len(coords_ref)):
            if sep2d_angle[i_ref] < separation_tolerance: 
                matched_ref_indices.append(i_ref)
                matched_target_indices.append(idx_target[i_ref])
                matched_separations_arcsec.append(sep2d_angle[i_ref].arcsec)
        logger_astro.info(f"{len(matched_ref_indices)}개 별 매칭 성공 (허용오차: {tolerance_arcsec} arcsec)."); return matched_ref_indices, matched_target_indices, matched_separations_arcsec
    except Exception as e: 
        logger_astro.error(f"별 좌표 매칭 오류: {e}", exc_info=True); return [], [], []
