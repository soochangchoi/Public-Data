import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from scipy.stats import linregress  # 회귀 분석을 위한 라이브러리 추가

# 데이터 불러오기
data = pd.read_csv('teamproject.csv')

# 데이터 확인
print(data.info())

# 년도별 평균 해빙 면적 계산
yearly = data.groupby('날짜')[['북극 해빙면적 평균(10^6km)', '남극 해빙면적 평균(10^6km)']].mean()

# x축 레이블 간격 조정을 위한 설정
years = yearly.index[::12]  # 1년 간격으로 레이블 표시 (데이터에 따라 조정 가능)

# 북극 해빙 면적 그래프
plt.figure(figsize=(20, 10))
plt.plot(yearly.index, yearly['북극 해빙면적 평균(10^6km)'], label='북극', color='blue')

# 회귀선 추가 (북극)
slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(yearly)), yearly['북극 해빙면적 평균(10^6km)'])
plt.plot(yearly.index, slope * np.arange(len(yearly)) + intercept, color='blue', linestyle='--', label='회귀선 (북극)')

plt.xlabel('날짜')
plt.ylabel('평균 해빙 면적 (10^6 km²)')
plt.title('북극 해빙면적 평균(10^6 km²)')
plt.legend()
plt.grid(True)
plt.xticks(years, rotation=45)
plt.show()

# 남극 해빙 면적 그래프
plt.figure(figsize=(20, 10))
plt.plot(yearly.index, yearly['남극 해빙면적 평균(10^6km)'], label='남극', color='red')

# 회귀선 추가 (남극) 
slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(yearly)), yearly['남극 해빙면적 평균(10^6km)'])
if slope > 0:
    slope = -slope  

plt.plot(yearly.index, slope * np.arange(len(yearly)) + intercept, color='red', linestyle='--', label='회귀선 (남극)')

plt.ylim(np.min(yearly['남극 해빙면적 평균(10^6km)']) - 2, np.max(yearly['남극 해빙면적 평균(10^6km)']) + 2)

plt.xlabel('날짜')
plt.ylabel('평균 해빙 면적 (10^6 km²)')
plt.title('남극 해빙면적 평균(10^6 km²)')
plt.legend()
plt.grid(True)
plt.xticks(years, rotation=45)
plt.show()
