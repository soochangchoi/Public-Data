import xarray as xr
import netCDF4
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # 또는 'Qt5Agg'
import matplotlib.font_manager as fm
plt.rc('font', family='Malgun Gothic')

data = xr.open_dataset("hgt.mon.mean.nc")

syr = 1979
fyr = 2000
nyr = fyr - syr + 1
hgtRaw = data['hgt'].isel(level=0).sel(lat=slice(None, 20), time=slice(f"{syr}-01", f"{fyr}-12")).values
import numpy as np
"""
평균값과 아노말리값을 저장할 배열의 차원 설정을 위해 lat, lon 배열의 크기 선언
"""
lat = data['lat'].sel(lat=slice(None, 20)).values
lon = data['lon'].values
nlat, nlon = len(lat), len(lon)

"""
hgtClim 변수에는 모든 기간에 대해 1, 2, 3 ... 10, 11, 12월의 평균값을 저장
hgtAno는 실제값(hgtRaw)과 평균값(hgtClim)의 차이
1년에 무조건 12달이 있으므로 i%12를 활용
"""
hgtClim = np.zeros((12, nlat, nlon))
for i in range(nyr*12):
    hgtClim[i%12, :, :] += hgtRaw[i, :, :]
hgtClim /= nyr

hgtAno = np.zeros(hgtRaw.shape)
for i in range(nyr*12):
    hgtAno[i, :, :] = hgtRaw[i,:,:] - hgtClim[i%12, :,:]
    
    
lat_rad = np.radians(lat)  
area_weighting = np.sqrt(np.maximum(np.cos(lat_rad), 0))  # 음수 방지

hgtAnoWgt = hgtAno * area_weighting[:, np.newaxis]  # 차원 맞춰서 곱하기

import numpy as np
from eofs.standard import Eof

solver = Eof(hgtAnoWgt)

eof1 = solver.eofsAsCovariance(neofs=1) # neofs=1, 첫 번째 공간 패턴만 반환,
# eofsAsCovariance >>> Covariance matrix로 EOF 분석 수행 (표준편차로도 가능)
pc1 = solver.pcs(npcs=1, pcscaling=1) # npcs=1, 첫 번째 pc 시계열만 반환

"""
첫 째만 반환해서 차원이 [시간, 1]이므로 1을 없애주기 위해 squeeze() 적용하여
배열의 차원을 [시간]으로 만듦
"""
eof1 = eof1.squeeze()
pc1 = pc1.squeeze()

from sklearn.linear_model import LinearRegression
loadingMap = np.zeros((nlat, nlon))
for ilat in range(nlat):
    for ilon in range(nlon):
        # 각 격자의 시계열 (hgtAno[:, lat, lon])과 pc1의 선형 회귀 수행
        y = hgtAno[:, ilat, ilon]  # 각 격자의 시계열
        X = pc1.reshape(-1, 1)  # pc1을 2D 배열로 변환 (독립 변수)
        
        # 회귀 모델 생성 및 학습
        model = LinearRegression()
        model.fit(X, y)
        
        # 회귀 계수 저장
        loadingMap[ilat, ilon] = model.coef_[0]  # 각 격자의 회귀 계수
        

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={
    'projection': ccrs.NorthPolarStereo() # 북극에서 바라보는 projection
    })
contourf = ax.contourf(lon, lat, loadingMap, transform=ccrs.PlateCarree())
cbar = fig.colorbar(contourf, ax=ax, orientation='vertical')

import matplotlib.colors as mcolors

colors = [
    (0.078431373, 0.392156863, 0.823529412),  # -45 ~ -40
    (0.156862745, 0.509803922, 0.941176471),  # -40 ~ -35
    (0.235294118, 0.588235294, 0.960784314),  # -35 ~ -30
    (0.31372549, 0.647058824, 0.960784314),   # -30 ~ -25
    (0.470588235, 0.725490196, 0.976470588),  # -25 ~ -20
    (0.588235, 0.823529, 0.976470588),        # -20 ~ -15
    (0.705882353, 0.941176471, 0.976470588),  # -15 ~ -10 (7번째 색상)
    (0.870588235, 1, 1),                      # -10 ~ -5
    (1, 1, 1),                                # -5 ~ 5
    (1, 0.976470588, 0.666666667),            # 5 ~ 10
    (1, 0.752941176, 0.235294118),            # 10 ~ 15
    (1, 0.376470588, 0),                      # 15 ~ 20
    (0.882352941, 0.078431373, 0),             # 20 ~ 25
]

custom_cmap = mcolors.ListedColormap(colors, name='custom_cmap')

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.NorthPolarStereo()})
ax.coastlines(resolution='110m', linewidth=1, color='black')  
levels = [-45, -40, -35, -30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25]

"""
lon의 끝부분에 360을 추가함
이를 newlon으로 정의
loadingMap도 첫 번째 열을 복제하여 끝에 추가
"""
newlon = []
for i in lon:
    newlon.append(i)
newlon.append(360)
newLoadingMap = np.hstack([loadingMap, loadingMap[:, :1]])  

"""
newLoadingMap에 minus를 곱해줘야 NOAA에서 그린 그림과 부호가 맞음
level 범위는 -45
"""
contourf  = ax.contourf(newlon, lat, -newLoadingMap[:, :], transform=ccrs.PlateCarree(), cmap=custom_cmap, levels=levels, vmin=-46, vmax=25)

# 그리드 라인 지정정
from matplotlib.ticker import FixedLocator
gridlines = ax.gridlines(color='gray', linestyle=':', linewidth=0.5)
gridlines.xlocator = FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150,180])
gridlines.ylocator = FixedLocator([20, 30, 40, 50, 60, 70, 80])

eofvar = solver.varianceFraction(neigs=1) * 100 


cbar = fig.colorbar(contourf, ax=ax, orientation='vertical')

cbar.set_ticks(levels[1:-1])
cbar.set_ticklabels(levels[1:-1])
plt.title(f'지구온난화와 북극 해빙 면적 감소 (2021년 기준)', fontsize=18)

plt.show()