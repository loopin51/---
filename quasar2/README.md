
# 천체사진 처리 및 분석 도구 (Astro Photometry Toolbox) v0.17.3

## 1. 개요 (Overview)

본 프로젝트는 아마추어 천문학자 및 천문학 연구자들이 FITS (Flexible Image Transport System) 형식의 천체 이미지를 보다 쉽게 처리하고 분석할 수 있도록 돕기 위해 개발된 Gradio 기반의 웹 애플리케이션입니다. 사용자는 이 도구를 통해 기본적인 이미지 보정(BIAS, DARK, FLAT), 별 탐지, 측광, 대기 소광 계수 계산, 표준 등급 변환, H-R도(색-등급도) 작성 등 다양한 천체사진 분석 작업을 수행할 수 있습니다.

이 버전(v0.17.3)은 `ccdproc.ccd_process` 함수 대신 **NumPy를 이용한 수동 보정 로직**을 핵심으로 사용하며, 마스터 플랫은 "예비 마스터 플랫"을 생성한 후 LIGHT 프레임 보정 시 실시간으로 최종 마스터 플랫을 만들어 적용하는 방식을 채택하고 있습니다.

복잡한 명령어나 스크립트 작성 없이 웹 UI를 통해 직관적으로 데이터를 업로드하고 처리 결과를 확인할 수 있도록 설계되었습니다.

## 2. 주요 기능 (Features)

본 애플리케이션은 여러 탭으로 구성되어 있으며, 각 탭은 특정 천체사진 처리 및 분석 단계를 담당합니다.

### 탭 1: 마스터 프레임 생성

* **Master BIAS 생성:** 여러 BIAS 프레임을 결합하여 노이즈가 적은 마스터 BIAS 프레임을 생성합니다. (중앙값 결합, 시그마 클리핑 적용)

* **Master DARK 생성:** 여러 DARK 프레임을 노출 시간별로 그룹화하고, 각 그룹에 대해 마스터 BIAS를 <0xEC><0x8A><0xA5> 후 결합하여 BIAS가 보정된 마스터 DARK 프레임을 생성합니다.

* **예비 Master FLAT 생성:** 여러 FLAT 프레임을 필터별로 그룹화하여, **BIAS나 DARK 보정 없이** Raw Flat들만 결합한 "예비 마스터 플랫"을 생성합니다. 이 예비 플랫은 이후 LIGHT 프레임 보정 시 실시간으로 최종 보정됩니다.

* 생성된 모든 마스터 프레임(BIAS, 모든 DARK, 모든 예비 FLAT)은 UI를 통해 직접 다운로드할 수 있습니다.

* 처리 중 업로드된 원본 임시 파일들은 마스터 프레임 생성이 성공적으로 완료되면 자동으로 삭제되어 저장 공간을 확보합니다.

### 탭 2: LIGHT 프레임 보정

* 사용자가 업로드한 LIGHT 프레임(관측 대상 이미지)에 대해 NumPy 기반의 수동 보정 로직을 사용하여 BIAS, DARK, FLAT 보정을 수행합니다.

* 탭 1에서 생성된 마스터 프레임 또는 사용자가 직접 업로드한 마스터 프레임을 사용하여 보정을 진행합니다.

* **수동 보정 로직:**

  1. `CalibratedLight = RawLight - MasterBias` (BIAS 보정)

  2. `CalibratedLight = CalibratedLight - MasterDark_{LightExpTime}` (DARK 보정)

  3. **최종 마스터 플랫 실시간 생성:**

     * `Flat_{temp1} = PrelimMasterFlat_{Filter} - MasterBias`

     * `Flat_{temp2} = Flat_{temp1} - MasterDark_{LightExpTime}` (LIGHT 프레임 노출시간에 맞는 DARK 사용, 필요시 스케일링)

     * `MasterFlat_{final} = Flat_{temp2} / Median(Flat_{temp2})` (중앙값으로 정규화)

  4. `CalibratedLight = CalibratedLight / MasterFlat_{final}` (FLAT 보정)

* 보정된 첫 번째 LIGHT 프레임의 미리보기를 제공하며, 다양한 스트레칭 옵션을 지원합니다.

* 보정된 모든 LIGHT 프레임은 다운로드 가능합니다.

### 탭 3: 대기 소광 계수 계산

* 여러 장의 LIGHT 프레임(일반적으로 동일한 별을 다른 고도에서 촬영)을 사용하여 대기 소광 계수(k)와 장비 영점(m0)을 계산합니다.

* 각 LIGHT 프레임은 탭 2와 동일한 NumPy 기반 수동 보정 로직으로 보정됩니다.

* 보정된 이미지에서 `DAOStarFinder`를 사용하여 별을 탐지하고, 그중 가장 밝은 별의 기기 등급을 계산합니다.

* FITS 헤더 정보로부터 각 프레임의 대기 질량(Airmass)을 계산합니다.

* (대기질량, 기기 등급) 데이터 포인트를 사용하여 선형 회귀 분석을 수행하여 대기 소광 계수와 영점을 도출합니다.

* 결과는 그래프(기기 등급 vs. 대기 질량)와 함께 요약 정보 및 상세 데이터 테이블로 제공됩니다.

### 탭 4: 상세 측광 및 카탈로그 분석

* B 필터와 V 필터로 촬영된 LIGHT 프레임들에 대해 상세한 별 탐지 및 측광을 수행합니다.

* **표준별 처리:**

  * 사용자가 표준별 FITS 파일과 알려진 표준 등급을 제공하면, 이를 NumPy 기반 수동 보정 로직으로 보정하고 측광하여 해당 필터의 유효 영점(m0_eff)을 계산합니다.

  * 알려진 등급이 없으면 SIMBAD 조회를 시도합니다 (현재는 ID만 표시).

* **대상별 처리:**

  * 각 LIGHT 프레임은 NumPy 기반 수동 보정 로직으로 보정됩니다.

  * `DAOStarFinder`를 사용하여 별을 탐지하고, 사용자가 설정한 ROI(Region of Interest) 내의 별들만 선택합니다.

  * 선택된 별들에 대해 조리개 측광을 수행하여 기기 등급을 계산합니다.

  * FITS 헤더의 WCS 정보를 사용하여 픽셀 좌표를 천구 좌표(RA, Dec)로 변환합니다.

  * 계산된 유효 영점과 대기 소광 계수를 사용하여 표준 등급(B등급, V등급)을 계산합니다.

* B 필터와 V 필터에서 공통으로 관측된 별들을 좌표 기반으로 매칭하여 B-V 색지수를 계산합니다.

* 각 별에 대해 SIMBAD 카탈로그 정보를 조회하여 ID와 타입을 가져옵니다.

* 최종 결과는 밝기 순으로 정렬된 테이블과 CSV 파일로 제공됩니다.

* 첫 번째 LIGHT 프레임에 대해 ROI 및 탐지/측광된 별들을 시각화한 미리보기 이미지를 제공합니다.

### 탭 5: H-R도 (색-등급도) 그리기

* 탭 4에서 생성된 CSV 파일을 업로드받습니다.

* CSV 파일에서 V 표준 등급('StdMag V')과 B-V 색지수('B-V') 데이터를 추출합니다.

* X축을 B-V 색지수, Y축을 V 등급으로 하는 H-R도(색-등급도)를 생성하여 표시합니다.

* 별의 색상은 B-V 값에 따라 푸른색에서 붉은색으로 변하도록 색상 맵을 적용합니다.

* SIMBAD ID가 있는 별은 점 옆에 텍스트로 ID를 표시합니다.

## 3. 기술 스택 (Tech Stack)

* **언어 (Language):** Python 3

* **웹 프레임워크 (Web Framework):** Gradio

* **천문학 라이브러리 (Astronomy Libraries):**

  * Astropy: FITS 파일 처리, WCS 변환, 단위 처리, 좌표계 변환 등 천문학 기본 기능

  * CCDProc: 마스터 프레임 결합 (`combine`, `subtract_bias` 등 일부 기능 활용)

  * Photutils: 별 탐지 (DAOStarFinder), 조리개 측광

  * Astroquery: SIMBAD 카탈로그 조회

* **데이터 처리 및 시각화 (Data Handling & Visualization):**

  * NumPy: 수치 연산 (핵심 보정 로직에 사용)

  * Pandas: 데이터 테이블 처리 (CSV 읽기/쓰기)

  * Matplotlib: 그래프 및 H-R도 생성

  * Pillow (PIL): 이미지 미리보기 생성 및 드로잉

* **기타 (Others):**

  * Scikit-learn: 선형 회귀 분석 (대기 소광 계수 계산 시)

## 4. 설치 및 실행 방법 (Installation and Setup)

### 4.1. 필수 조건 (Prerequisites)

* Python 3.8 이상

* pip (Python 패키지 관리자)

### 4.2. 설치 과정 (Installation Steps)

1. **프로젝트 클론 (Clone the project):**

   ```
   git clone <repository_url>
   cd <project_directory_name>
   '''


2. **가상 환경 생성 및 활성화 (Create and activate a virtual environment - 권장):**

   ```
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate  # Windows
   
   ```

3. **필요한 라이브러리 설치 (Install dependencies):**
   프로젝트 루트 디렉토리에 `requirements.txt` 파일이 있다면 다음 명령으로 설치합니다. (만약 없다면 아래 라이브러리들을 직접 설치)

   ```
   pip install -r requirements.txt
   
   ```

   또는 직접 설치:

   ```
   pip install gradio astropy ccdproc photutils astroquery numpy pandas matplotlib pillow scikit-learn
   ```

### 4.3. 실행 (Running the Application)

프로젝트 루트 디렉토리에서 다음 명령을 실행합니다:

    ```
    python astro_app_main.py

    ``` 
애플리케이션이 실행되면, 터미널에 로컬 URL(보통 `http://127.0.0.1:7860`)이 표시됩니다. 웹 브라우저에서 이 주소로 접속하면 애플리케이션 UI를 사용할 수 있습니다.

**임시 파일 디렉토리:**

* 애플리케이션이 생성하는 파일(마스터 프레임, CSV 등)은 `astro_app_main.py` 파일 위치 하위의 `gradio_temp_outputs` 폴더에 저장됩니다.

* Gradio가 파일 업로드 시 사용하는 임시 파일들은 `astro_app_main.py` 파일 위치 하위의 `gradio_temp_files` 폴더에 저장됩니다.

* 이 두 임시 디렉토리는 애플리케이션 종료 시 자동으로 삭제됩니다.

## 5. 사용 방법 (Usage)

애플리케이션을 실행하면 여러 탭이 있는 웹 UI가 나타납니다. 각 탭은 특정 작업을 수행합니다.

### 탭 1: 마스터 프레임 생성

1. "BIAS 프레임 업로드", "DARK 프레임 업로드", "FLAT 프레임 업로드" 버튼을 사용하여 각각의 Raw FITS 파일들을 업로드합니다.

   * DARK 프레임은 서로 다른 노출 시간의 파일들을 함께 업로드할 수 있습니다.

   * FLAT 프레임은 서로 다른 필터의 파일들을 함께 업로드할 수 있습니다.

2. "마스터 프레임 생성 시작" 버튼을 클릭합니다.

3. 생성된 마스터 BIAS, 모든 마스터 DARK, 모든 예비 마스터 FLAT은 파일 다운로드 컴포넌트를 통해 개별적으로 다운로드할 수 있습니다.

### 탭 2: LIGHT 프레임 보정

1. 보정할 LIGHT 프레임(들)을 업로드합니다.

2. (선택) 탭 1의 마스터 프레임 대신 사용할 프레임들을 직접 업로드합니다.

   * Master BIAS: 단일 파일

   * Master DARK: 여러 노출 시간의 Raw DARK 파일들

   * Master FLAT B/V: 필터별 Raw 예비 플랫 파일

3. "LIGHT 프레임 보정 시작" 버튼을 클릭합니다.

4. 첫 번째 보정 이미지 미리보기가 표시되고, 모든 보정된 LIGHT 프레임은 다운로드 가능합니다.

### 탭 3: 대기 소광 계수 계산

1. 분석할 LIGHT 프레임들(다른 고도에서 촬영된 동일 대상)을 업로드합니다.

2. (선택) 마스터 프레임들을 직접 업로드합니다 (탭 2와 유사).

3. 별 탐지 관련 파라미터(FWHM, 임계값)를 설정합니다. (현재 코드에서는 FWHM은 3.0으로 고정, 임계값은 슬라이더로 조절)

4. "대기소광계수 계산 시작" 버튼을 클릭합니다.

5. 결과 그래프, 요약, 데이터 테이블이 표시됩니다.

### 탭 4: 상세 측광 및 카탈로그 분석

1. B, V 필터 LIGHT 프레임들을 각각 업로드합니다.

2. (선택) 표준별 FITS 파일 및 알려진 등급을 입력합니다.

3. (선택) 마스터 프레임들을 직접 업로드합니다.

4. 대기소광계수, 기본 영점, 별 탐지/측광 파라미터, ROI, SIMBAD 검색 반경을 설정합니다.

5. "상세 측광 분석 시작" 버튼을 클릭합니다.

6. 결과 테이블, CSV 다운로드, 측광 결과 미리보기 이미지가 제공됩니다.

### 탭 5: H-R도 (색-등급도) 그리기

1. 탭 4에서 생성된 CSV 파일을 업로드합니다.

2. "H-R도 그리기 시작" 버튼을 클릭합니다.

3. V 등급 대 B-V 색지수 그래프(H-R도)가 표시됩니다.

## 6. 천문학적 배경 지식 (Astronomical Background)

### 6.1. FITS 파일 보정의 원리 (NumPy 기반 수동 보정)

천체 관측용 CCD 카메라는 완벽하지 않으며, 촬영된 원본 이미지(Raw LIGHT frame)에는 다양한 오류와 노이즈가 포함됩니다. 이러한 효과를 제거하여 실제 천체의 신호를 정확히 얻기 위한 과정을 "보정(Calibration)"이라고 합니다.

* **BIAS 프레임 (Bias Frame):**

  * **정의:** CCD를 읽을 때 발생하는 기본적인 전자적 오프셋입니다. 노출 시간이 0초일 때도 존재합니다.

  * **목적:** 모든 이미지에 공통적으로 더해지는 이 기준 레벨을 제거합니다.

  * **생성:** 매우 짧은 노출 시간으로 여러 장을 촬영 후, 중앙값 결합하여 "마스터 BIAS"를 만듭니다.

* **DARK 프레임 (Dark Frame):**

  * **정의:** CCD 픽셀에서 열에 의해 자발적으로 생성되는 전자(암전류)로 인한 신호입니다. 빛이 없는 상태에서 LIGHT 프레임과 동일한 노출 시간 및 온도에서 촬영합니다.

  * **목적:** 노출 시간 동안 쌓인 암전류를 제거합니다.

  * **생성:** 여러 Raw DARK에서 마스터 BIAS를 <0xEC><0x8A><0xA5> 후, 동일 노출 시간의 프레임들을 중앙값 결합하여 "마스터 DARK"를 만듭니다.

* **FLAT 프레임 (Flat-Field Frame) 및 예비 마스터 플랫:**

  * **정의:** 광학계의 비네팅, CCD 픽셀 감도 차이, 먼지 등으로 인한 이미지 밝기 불균일성을 나타냅니다. 균일한 광원을 촬영하여 얻습니다.

  * **예비 마스터 플랫 생성 (본 프로젝트 방식):** 필터별로 Raw FLAT 프레임들을 **BIAS나 DARK 보정 없이** 단순 결합(평균)하여 "예비 마스터 플랫"을 만듭니다. 이 예비 플랫은 필터별 시스템 반응 패턴만 담고 있습니다.

  * **최종 마스터 플랫 (LIGHT 프레임 보정 시 실시간 생성):**

    1. 해당 LIGHT 프레임의 필터에 맞는 예비 마스터 플랫을 가져옵니다.

    2. 예비 마스터 플랫에서 마스터 BIAS를 <0xEC><0x8A><0xA5>니다: `$Flat_{B} = PrelimFlat - MasterBias$`

    3. 그 결과에서 **LIGHT 프레임의 노출 시간과 동일한 마스터 DARK**를 (필요시 스케일링하여) <0xEC><0x8A><0xA5>니다: `$Flat_{BD} = Flat_{B} - MasterDark_{LightExpTime, ScaledToFlatExpIfNeeded}$`

       * 스케일링: `MasterDark_{Scaled} = MasterDark_{LightExpTime} \times (FlatExpTime / DarkExpTime)`

    4. 최종적으로 이 플랫의 \*\*중앙값(Median)\*\*으로 나누어 정규화합니다: `$MasterFlat_{final} = Flat_{BD} / Median(Flat_{BD})$`

* **LIGHT 프레임 보정 수식 (NumPy 기반 수동 보정):**
  `$CalibratedLight = (RawLight - MasterBias - MasterDark_{LightExpTime}) / MasterFlat_{final}$`

### 6.2. 측광 (Photometry)의 원리

측광은 천체의 밝기(광도 또는 플럭스)를 측정하는 기술입니다.

* **조리개 측광 (Aperture Photometry):**

  * 별 주변에 가상의 원형 "조리개(aperture)"를 설정하고, 조리개 내 픽셀 값 합산.

  * 배경 하늘의 밝기를 보정하기 위해, 별 주변 "Annulus(고리)" 영역의 평균 픽셀 값을 계산하여 조리개 영역에서 빼줌.

  * `Net Flux = (Total Flux in Aperture) - (Mean Sky per Pixel * Aperture Area)`

* **기기 등급 (Instrumental Magnitude):**

  * 측정된 별의 플럭스로부터 계산: `$m_{inst} = -2.5 \times \log_{10}(\text{Flux})$`

  * 대기 효과나 장비 감도가 보정되지 않은 상대적 밝기.

### 6.3. 대기 소광 (Atmospheric Extinction)

지구 대기는 별빛을 흡수하고 산란시켜 별을 실제보다 어둡게 만듭니다.

* **원인:** 레일리 산란, 미 산란, 흡수 등.

* **대기 소광 계수 (k):** 단위 대기 질량당 별빛이 어두워지는 정도. 파장에 따라 다름.

* **대기 질량 (X):** 관측 천체가 통과하는 대기의 상대적 두께. `$X \approx \sec(z)$` (z는 천정 거리).

* **보정 관계식:** `$M_{std} = m_{inst} - k \times X - C$` (C는 영점 상수).

### 6.4. H-R도 (Hertzsprung-Russell Diagram) / 색-등급도 (Color-Magnitude Diagram)

* **정의:** 별의 절대 등급(광도) 대 분광형(표면 온도) 또는 색지수를 나타낸 산점도.

* **의미:** 별의 물리적 특성과 진화 단계를 보여줍니다. 주계열, 거성, 백색왜성 등이 특정 영역에 분포.

* **색지수 (B-V):** B 필터 등급과 V 필터 등급의 차이. 별의 온도를 나타내는 지표 (작을수록 뜨겁고 푸른 별).

## 7. 향후 개선 사항 (Future Improvements)

* **UI/UX 개선:** 사용자 편의성 증대.

* **성능 최적화:** 대용량 파일 처리 속도 향상.

* **고급 측광:** PSF 측광 등.

* **다양한 카탈로그 지원.**

* **시각화 다양화.**

* **자동화 기능.**

* **오류 처리 및 로깅 상세화.**

## 8. 기여 방법 (Contributing)

1. Github 저장소에서 이슈(Issue) 생성.

2. 프로젝트 포크(Fork) 후 변경 사항 적용 및 풀 리퀘스트(Pull Request).

## 9. 라이선스 (License)

본 프로젝트는 MIT 라이선스 하에 배포됩니다. (실제 라이선스 파일 추가 필요)

---

이 README가 프로젝트를 이해하고 사용하는 데 도움이 되기를 바랍니다!
```