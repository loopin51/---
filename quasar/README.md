# Stellar Photometry & H‑R Diagram Pipeline

> **Modules:** `preprocess_methods.py`, `stellar_photometry_hr.py`, `main.py`
>
> **UI:** Gradio (launch with `python main.py`)

---

## 프로젝트 개요

이 프로젝트는 **광학 천체 사진(FITS)** 데이터로부터 

1. **대기 소광 계수**(k<sub>B</sub>, k<sub>V</sub>) 추정  → 표준별을 이용한 영점 보정
2. **표준별 플럭스·색지수 보정** → 관측별 B, V 등급 계산
3. **ROI(관심 영역) 지정** → 별 검출·광도 측정·B‑V 색지수 산출
4. **CSV 카탈로그**, **필드 이미지(별 번호 포함)**, **H‑R 다이어그램** 자동 생성
5. 모든 과정을 **Gradio 웹 UI**에서 "업로드 → 슬라이더 → 실행 → 다운로드" 식으로 시각화

각 기능은 독립적인 파이썬 모듈로 분리되어 있으며, 종단 간 처리 파이프라인을 한 번의 클릭으로 수행할 수 있습니다.

---

## 코드 아키텍처

| 모듈                             | 핵심 역할                                                  | 주요 함수·클래스                                                                       |
| ------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------------------------- |
| **`preprocess_methods.py`**    | ✦ FITS I/O  ✦ 마스터 프레임 생성  ✦ 별 검출  ✦ 대기 소광 & 표준별 보정     | `build_calibration()` / `detect_stars()` / `estimate_extinction_coefficients()` |
| **`stellar_photometry_hr.py`** | ✦ 과학 프레임 보정  ✦ ROI 내 별 매칭  ✦ B, V 등급·색지수  ✦ CSV·이미지 생성 | `detect_stars()` (간단 버전) / `match_catalog()` / `plot_field()` / `plot_hr()`     |
| **`main.py`**                  | ✦ Gradio UI  ✦ ROI 슬라이더  ✦ 실시간 Preview  ✦ 결과 다운로드      | `make_preview()` / `run_pipeline()` / RangeSlider & File components             |

---

## 설치 & 실행

```bash
pip install astropy photutils scikit-learn matplotlib pandas gradio
python main.py       # 웹 브라우저 자동 실행
```

---

## 주요 처리 단계 & 코드 인용

### 1. 마스터 프레임 생성

```python
# preprocess_methods.py
stack = np.stack([load_fits_file(f)[0] for f in file_list])
return np.median(stack, axis=0)            # create_master_frame()
```

*Bias / Dark / Flat* 프레임을 중앙값 스택으로 합성하여 노이즈를 최소화합니다.

### 2. 별 검출 & 플럭스 측정

```python
# preprocess_methods.py
finder = DAOStarFinder(fwhm=3, threshold=threshold_factor*std)
src = finder(image)
...
flux = calc_flux(img, x_cen, y_cen, radius)   # aperture + annulus
```

DAOStarFinder로 후보 위치를 얻고, **원형 Aperture + Annulus** 방식으로 배경을 제거한 플럭스를 계산합니다.

### 3. 대기 소광 계수(k) 추정

```python
# preprocess_methods.py
def estimate_extinction_coefficients(results):
    x = np.array([r[1] for r in results]).reshape(-1,1)  # airmass
    y = np.array([r[0] for r in results])               # instrumental mag
    model = LinearRegression().fit(x, y)
    return model.coef_[0]                               # slope = k
```

동일 표준별의 **instrumental mag vs. airmass** 선형 회귀로 1차 소광 계수를 구합니다.

### 4. 표준별 영점 보정 & 관측 등급 계산

```python
# stellar_photometry_hr.py
def mag(flux, std, k, airm):
    return std['catalog_m'] - 2.5*np.log10(flux/std['flux']) - k*airm
```

추정된 k, 표준별 카탈로그 등급·플럭스를 이용해 관측별 B, V 등급을 보정합니다.

### 5. ROI 선택 & 실시간 Preview

```python
# main.py
roi_x = gr.RangeSlider(...)
roi_y = gr.RangeSlider(...)
roi_preview = gr.Image()
sci_up.change(update_preview, [sci_up, roi_x, roi_y], roi_preview)
```

슬라이더 값을 바꿀 때마다 첫 번째 Science FITS를 읽어 초록색 사각형으로 ROI를 표시합니다.

`update_preview()` 구현:

```python
plt.gca().add_patch(
    plt.Rectangle((x0, y0), x1-x0, y1-y0, ec='lime', fc='none')
)
```

### 6. CSV 카탈로그 & H‑R 다이어그램

```python
# stellar_photometry_hr.py
rows.sort(key=lambda s: (-s['y_V'], s['x_V']))   # 위→아래, 왼→오
pd.DataFrame(rows).to_csv(csv_path, index=False)
...
plt.scatter(BV, Vabs); plt.gca().invert_yaxis()
```

ROI 내 별을 정렬 후 CSV 파일을 기록하고, **B‑V vs. Absolute Magnitude** 산점도로 H‑R 도표를 저장합니다.

### 7. Gradio 출력 & 다운로드

```python
# main.py
run_btn.click(run_pipeline, ..., outputs=[field_img, mag_html, hr_img,
                                          csv_dl, field_dl, hr_dl])
```

* `field_img` : ROI 필드 PNG
* `mag_html` : 데이터프레임 HTML 미리보기
* `hr_img` : H‑R 다이어그램
* `csv_dl`, `field_dl`, `hr_dl` : 다운로드 가능한 파일 링크

---

## 기능 요약

* **FITS 업로드** : Extinction·Calibration·Science 프레임 다중 업로드 지원
* **표준별 & 목표 클러스터 선택** : 내부 DB(`fetch_standard_info`, `fetch_cluster_distance`)
* **대기 소광 자동 추정** : 선형 회귀로 k<sub>B</sub>, k<sub>V</sub>
* **플럭스‑>등급 보정** : 표준별 영점 + ROI airmass
* **실시간 ROI 미리보기** : 초록 사각형으로 시각 확인
* **별 번호·색지수 필드 PNG 생성**
* **CSV 카탈로그 & HTML 미리보기**
* **H‑R 다이어그램 (Absolute Mag)** 생성
* **모든 산출물 다운로드 버튼 제공**

---

## 프로젝트 디렉터리 구조 예시

```
project_root/
├── data/                  # (선택) 샘플 FITS·보정 프레임 보관용
│   ├── extinction/
│   ├── calibration/
│   │   ├── bias/
│   │   ├── dark/
│   │   ├── flat_b/
│   │   └── flat_v/
│   └── science/           # 대상 B/V 관측 이미지
├── outputs/               # 실행 시 자동 생성 (CSV, PNG 등)
├── preprocess_methods.py  # 소광·표준별∙유틸 함수 모듈
├── stellar_photometry_hr.py # CLI 파이프라인 (모듈화 버전)
├── main.py                # Gradio UI 진입점
├── README.md              # 프로젝트 설명서 (본 파일)
└── requirements.txt       # 의존 라이브러리 목록
```

> **TIP**: `main.py` 를 실행하면 `outputs/` 폴더가 없을 경우 자동으로 만들어집니다. 별도의 테스트 데이터를 `data/` 아래에 보관해두면 샘플 분석 및 데모가 편리합니다.

---

## 향후 개선 아이디어

* 표준별·클러스터 데이터를 외부 JSON/CSV 로딩으로 확장
* GUI에 **progress bar** 및 **에러 핸들링 메시지** 표시
* PSF FWHM 정밀 측정을 위해 `photutils.psf` 2‑D PSF 모델 적용
* GPU 가속(Numba/CuPy)으로 대용량 배치 처리 속도 개선

프로젝트 구조와 각 기능의 소스 위치가 명확히 정리되었으므로, 필요 시 원하는 부분만 쉽게 확장·수정할 수 있습니다. 즐거운 관측 데이터 분석 되세요! 🎉
