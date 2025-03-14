import pandas as pd
import numpy as np
import math
from rdp import rdp
from scipy.signal import savgol_filter
from typing import List, Tuple

# ---------------------------- 可调参数 ----------------------------
SMOOTH_WINDOW = 11    # 平滑窗口（必须为奇数）
SMOOTH_ORDER = 2      # 滤波阶数
RDP_EPSILON = 5.0     # 简化阈值（米）
MIN_SEG_LENGTH = 50   # 最小线段长度（米）
MAX_SEG_LENGTH = 500  # 最大线段长度（米）
Z_WEIGHT = 0.4        # 高程权重（0-1）
# -----------------------------------------------------------------

def process_section(df: pd.DataFrame, seccode: str) -> pd.DataFrame:
    """主处理流程"""
    # 数据预处理
    df = preprocess_data(df)
    
    # 三维坐标处理
    points = df[['X', 'Y', 'Z']].values
    
    # 三维自适应分段
    segments = adaptive_segmentation_3d(points)
    
    # 生成结果
    return generate_output(segments, seccode)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据预处理（含高程平滑）"""
    # 重命名列
    df.columns = ['序号', 'X', 'Y', 'Z']
    
    # 数据平滑（高程单独处理）
    df['X'] = savgol_filter(df['X'], SMOOTH_WINDOW, SMOOTH_ORDER)
    df['Y'] = savgol_filter(df['Y'], SMOOTH_WINDOW, SMOOTH_ORDER)
    df['Z'] = savgol_filter(df['Z'], SMOOTH_WINDOW//2+1, 2)  # 高程弱平滑
    return df

def adaptive_segmentation_3d(points: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """三维自适应分段核心算法"""
    # 三维RDP简化（带高程权重）
    simplified = rdp_3d(points, epsilon=RDP_EPSILON, z_weight=Z_WEIGHT)
    
    # 生成初始分段
    segments = []
    prev_point = points[0]
    for current_point in simplified[1:]:
        seg_points = get_sub_points(points, prev_point, current_point)
        
        # 三维分段校验
        while True:
            length_3d = three_d_length(prev_point, current_point)
            if length_3d <= MAX_SEG_LENGTH:
                break
            # 超长线段分割
            mid_point = find_best_split(seg_points)
            segments.append((prev_point, mid_point))
            prev_point = mid_point
            seg_points = get_sub_points(points, prev_point, current_point)
        
        segments.append((prev_point, current_point))
        prev_point = current_point
    
    return merge_short_segments(segments)

def rdp_3d(points: np.ndarray, epsilon: float, z_weight: float) -> np.ndarray:
    """三维加权RDP算法"""
    # 计算高程缩放因子
    xy_scale = np.mean([np.std(points[:,0]), np.std(points[:,1])])
    z_scale = np.std(points[:,2]) * z_weight
    scale_factor = xy_scale / (z_scale + 1e-6)  # 防止除零
    
    # 构建加权坐标
    weighted_points = points.copy()
    weighted_points[:,2] = points[:,2] * scale_factor
    return rdp(weighted_points, epsilon=epsilon)

def get_sub_points(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """获取两点间的子点集"""
    start_idx = np.where((points == start).all(axis=1))[0][0]
    end_idx = np.where((points == end).all(axis=1))[0][0]
    return points[start_idx:end_idx+1]

def three_d_length(p1: np.ndarray, p2: np.ndarray) -> float:
    """三维空间距离计算"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)

def find_best_split(points: np.ndarray) -> np.ndarray:
    """寻找最佳分割点（考虑高程突变）"""
    # 计算高程变化率
    dz = np.abs(np.diff(points[:,2]))
    dz_rate = dz / (np.linalg.norm(np.diff(points[:,:2], axis=0), axis=1) + 1e-6)
    
    # 寻找最大高程变化点
    split_idx = np.argmax(dz_rate) + 1
    return points[split_idx]

def merge_short_segments(segments: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """合并短线段"""
    merged = []
    temp_seg = segments[0]
    
    for seg in segments[1:]:
        combined_length = three_d_length(temp_seg[0], seg[1])
        if combined_length <= MAX_SEG_LENGTH:
            temp_seg = (temp_seg[0], seg[1])
        else:
            merged.append(temp_seg)
            temp_seg = seg
    merged.append(temp_seg)
    
    return [s for s in merged if three_d_length(s[0], s[1]) >= MIN_SEG_LENGTH]

def generate_output(segments: List[Tuple[np.ndarray, np.ndarray]], seccode: str) -> pd.DataFrame:
    """结果生成（保持原始高程）"""
    results = []
    total_high = 0.0
    
    for i, (start, end) in enumerate(segments):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        
        # 三维几何参数
        slope_length = three_d_length(start, end)
        h_length = math.hypot(dx, dy)
        azimuth = math.degrees(math.atan2(dx, dy)) % 360
        grade = math.degrees(math.atan(dz/h_length)) if h_length > 0 else 0
        
        total_high += dz
        
        results.append({
            '剖面编号': seccode,
            '线段编号': i+1,
            '起点X': round(start[0], 3),
            '起点Y': round(start[1], 3),
            '起点Z': round(start[2], 3),
            '终点X': round(end[0], 3),
            '终点Y': round(end[1], 3),
            '终点Z': round(end[2], 3),
            '水平长度': round(h_length, 2),
            '斜距': round(slope_length, 2),
            '方位角': round(azimuth, 2),
            '坡度': round(grade, 2),
            '累计高差': round(total_high, 3)
        })
    
    return pd.DataFrame(results)

# ---------------------------- 使用示例 ----------------------------
if __name__ == "__main__":
    # 示例数据加载
    data = {
        'X': np.linspace(0, 1000, 100),
        'Y': np.linspace(0, 500, 100),
        'Z': np.sin(np.linspace(0, 2*np.pi, 100)) * 20
    }
    df = pd.DataFrame(data)
    
    # 处理并输出
    result = process_section(df, "PM01")
    print(result.head())
