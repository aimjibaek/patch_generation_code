### 명령어 설명

1. python 환경에서 설치해야하는 명령어

   pip install opencv-python
   pip install openslide-python
   pip install tqdm
   pip install matplotlib
2. 리눅스 자체에서 설치해야하는 명령어 (에러 발생할 경우)
   opencv
   에러: libGL.so.1 cannot open shared object file 해결: sudo apt-get install libgl1-mesa-glx

   openslide
   에러: libopenslide.so.0: cannot open shared object file: No such file or directory
   해결: sudo apt-get install openslide-tools
   위의 명령어로 해결 안될경우: sudo apt-get install python-openslide

### 코드 파일 설명

1. random_patch_maker.ipynb: 패치 샘플링을 하기 위한 파일입니다.
   패치 필터: PatchFilter.py
   svs 불러오기: custom_openslide.py # 다만 사용할 수 있는 전체 슬라이드 개수가 달라질 경우 세팅을 다시해야되는 부분이 있어서(이 경우 custom_openslide.py의 함수도 변경을 해주어야 함) 동일한 함수를 ipynb 파일상에서 직접 선언하여 사용 (tqdm을 사용하기 위해 함수가 현재 사용중인 18292개의 슬라이드에 맞춰져있음)
   패치샘플링: RandomPatchMaker.py

   is_save=True인 경우 샘플링한 조직 패치와 좌표를 저장합니다.
   show_result=True인 경우 샘플링된 패치의 위치가 나타나있는 전체 슬라이드의 사진을 보여주고 저장합니다.
   patches_num=5000인 경우 5000장을 샘플링합니다.
2. anlayze_patches_lab.ipynb : 특이케이스가 포함된 슬라이드에서 각 지역별 패치를 뽑고 분석하기 위한 코드입니다. / py 파일들을 불러와서 사용합니다.
   패치 필터: PatchFilter.py
   svs 불러오기: custom_openslide.py
   패치 통계값 분석: AnalyzePatchesStatistics.py
   svs의 특정지역에서 샘플링하기: AnalyzeSVS.py
3. analyze_patches_filter_test.ipynb : 2번 코드와 유사한 코드로 각기다른 슬라이드들의 지역별 패치를 저장해서 한꺼번에 필터를 적용하여 분석하기 위한 코드입니다.
4. custom_openslide.py의 show_svs_in_grid 함수
   여러개의 svs를 한번에 시각화하기 위한 함수입니다. (특이케이스 분석을 위한)

### 패치 필터 설명 (./classes/PatchFilter.py)

is_tissue : 메인 샘플링 알고리즘
장점1: 흰색이 많이 포함된 지방조직 등의 조직을 배제하는 경향이 적다
장점2: 색깔있는 마킹을 상대적으로 잘 배제한다
단점1: 노이즈를 조직으로 많이 샘플링하는 문제가 몇몇 특이케이스 슬라이드에 존재한다.('TCGA-DQ-5624-01Z-00-DX1', 'TCGA-GS-A9TY-01A-01-TS1') (노이즈가 지방과 흡사한 경우)
단점2: 조직의 가장자리에서 샘플링된 경우를 지방조직과 잘 구분하지 못한다

is_tissue_prev_filter, is_tissue_ash_nash_filter : 비교군 샘플링 알고리즘
장점: (색깔있는 마킹을 제외한) 노이즈를 잘 배제한다
단점: 흰색이 많이 포함된 조직을 배제하는 경향이 크다(장점과의 trade-off 관계가 있다)
