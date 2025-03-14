# -*- coding: utf-8 -*-
"""
日期：2025-03-08
版本：0.7
功能：根据空间坐标点自动分段生成导线，斜距为5的倍数，最后一段小于50米时延伸至50米，所有线段方向与整体趋势一致
"""

import pandas as pd
import numpy as np
import math
from rdp import rdp
from scipy.signal import savgol_filter

# ---------------------------- 可调参数区域 ----------------------------
SMOOTH_WINDOW = 9    # 平滑滤波窗口大小（必须为奇数）
SMOOTH_ORDER = 2     # 平滑滤波多项式阶数
RDP_EPSILON = 7.0    # 线段简化阈值（单位：米）
MIN_SEG_LENGTH = 150 # 最小线段斜距（米）
MAX_SEG_LENGTH = 400 # 最大线段斜距（米）
LENGTH_STEP = 5      # 长度标准化步长（斜距为5的倍数）
MAX_ANGLE_DIFF = 45  # 允许的最大方位角差（度，严格控制回头路）
# ---------------------------------------------------------------------

def smooth_coordinates(df):
    """数据平滑处理：使用Savitzky-Golay滤波器消除噪声"""
    df = df.rename(columns={df.columns[0]: '点序号', df.columns[1]: 'X', 
                           df.columns[2]: 'Y', df.columns[3]: 'Z'})
    for col in ['X', 'Y', 'Z']:
        df[col] = savgol_filter(df[col], window_length=SMOOTH_WINDOW, polyorder=SMOOTH_ORDER)
    return df

def calculate_azimuth(p1, p2):
    """计算两点之间的方位角"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dx, dy)) % 360

def slope_length(p1, p2):
    """计算三维斜距"""
    return np.linalg.norm(p2 - p1)

def is_direction_consistent(azimuth, overall_azimuth):
    """检查线段方位角是否与整体方向一致"""
    angle_diff = min(abs(azimuth - overall_azimuth), 360 - abs(azimuth - overall_azimuth))
    return angle_diff <= MAX_ANGLE_DIFF

def adaptive_segmentation(points, df, overall_azimuth):
    """自适应分段：确保方向一致性并按斜距分割"""
    simplified = rdp(points, epsilon=RDP_EPSILON)
    indices = [np.where((points == p).all(axis=1))[0][0] for p in simplified]
    indices = sorted(indices)
    
    # 强制包含起点和终点
    if indices[0] != 0:
        indices.insert(0, 0)
    if indices[-1] != len(df) - 1:
        indices.append(len(df) - 1)
    
    segments = []
    prev_idx = indices[0]
    
    i = 1
    while i < len(indices):
        current_idx = indices[i]
        start_point = df.iloc[prev_idx][['X', 'Y', 'Z']].values
        end_point = df.iloc[current_idx][['X', 'Y', 'Z']].values
        seg_azimuth = calculate_azimuth(start_point[:2], end_point[:2])
        
        if is_direction_consistent(seg_azimuth, overall_azimuth):
            seg_length = slope_length(start_point, end_point)
            if seg_length <= MAX_SEG_LENGTH:
                segments.append((prev_idx, current_idx))
                prev_idx = current_idx
                i += 1
            else:
                # 超长线段，尝试分割
                sub_segs = split_segment(prev_idx, current_idx, df, overall_azimuth)
                segments.extend(sub_segs)
                prev_idx = sub_segs[-1][1]
                i += 1
        else:
            # 方向不一致，跳到下一个点
            i += 1
            if i == len(indices):
                # 如果到达最后一个点仍不一致，直接连接终点
                end_idx = len(df) - 1
                if prev_idx != end_idx:
                    segments.append((prev_idx, end_idx))
    
    return segments

def split_segment(start_idx, end_idx, df, overall_azimuth, depth=0):
    """递归分割超长或方向不一致的线段"""
    if depth > 20:
        print("达到最大递归深度，强制终止分割")
        return [(start_idx, end_idx)]
    
    start_point = df.iloc[start_idx][['X', 'Y', 'Z']].values
    end_point = df.iloc[end_idx][['X', 'Y', 'Z']].values
    seg_length = slope_length(start_point, end_point)
    seg_azimuth = calculate_azimuth(start_point[:2], end_point[:2])
    
    if seg_length <= MAX_SEG_LENGTH and is_direction_consistent(seg_azimuth, overall_azimuth):
        return [(start_idx, end_idx)]
    
    # 在start_idx和end_idx之间寻找合适的分割点
    mid_idx = (start_idx + end_idx) // 2
    if mid_idx == start_idx or mid_idx == end_idx:
        return [(start_idx, end_idx)]
    
    segments = []
    mid_point = df.iloc[mid_idx][['X', 'Y', 'Z']].values
    first_azimuth = calculate_azimuth(start_point[:2], mid_point[:2])
    second_azimuth = calculate_azimuth(mid_point[:2], end_point[:2])
    
    if is_direction_consistent(first_azimuth, overall_azimuth):
        first_length = slope_length(start_point, mid_point)
        if first_length > MAX_SEG_LENGTH:
            segments.extend(split_segment(start_idx, mid_idx, df, overall_azimuth, depth + 1))
        else:
            segments.append((start_idx, mid_idx))
    else:
        segments.extend(split_segment(start_idx, mid_idx, df, overall_azimuth, depth + 1))
    
    if is_direction_consistent(second_azimuth, overall_azimuth):
        second_length = slope_length(mid_point, end_point)
        if second_length > MAX_SEG_LENGTH:
            segments.extend(split_segment(mid_idx, end_idx, df, overall_azimuth, depth + 1))
        else:
            segments.append((mid_idx, end_idx))
    else:
        segments.extend(split_segment(mid_idx, end_idx, df, overall_azimuth, depth + 1))
    
    return segments

def adjust_segments(original_segments, df):
    """线段标准化：斜距调整为5的倍数，最后一段小于50米时延伸至50米"""
    adjusted = []
    current_pos = df[['X', 'Y', 'Z']].iloc[0].values
    last_idx = 0
    
    for i, (start_idx, end_idx) in enumerate(original_segments):
        if start_idx != last_idx:
            print(f"Warning: Segment {i} starts at {start_idx}, expected {last_idx}. Adjusting.")
            start_idx = last_idx
        last_idx = end_idx
        
        original_end = df[['X', 'Y', 'Z']].iloc[end_idx].values
        vec = original_end - current_pos
        raw_slope_l = np.linalg.norm(vec)
        
        is_last = (i == len(original_segments) - 1)
        if is_last and raw_slope_l < 50:
            target_slope_l = 50  # 最后一段小于50米时延伸至50米
        else:
            target_slope_l = round(raw_slope_l / LENGTH_STEP) * LENGTH_STEP
            if target_slope_l < MIN_SEG_LENGTH:
                target_slope_l = MIN_SEG_LENGTH
            elif target_slope_l > MAX_SEG_LENGTH:
                target_slope_l = MAX_SEG_LENGTH
        
        if raw_slope_l > 0:
            ratio = target_slope_l / raw_slope_l
            new_end = current_pos + vec * ratio
        else:
            new_end = current_pos.copy()
        
        adjusted.append((current_pos.copy(), new_end))
        current_pos = new_end
    
    return adjusted

def generate_output(segments, seccode):
    """生成输出表格：包含测量参数"""
    results = []
    total_high = 0.0
    
    for i, (start, end) in enumerate(segments):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        
        h_l = math.hypot(dx, dy)
        slope_l = math.sqrt(h_l**2 + dz**2)
        azimuth = math.degrees(math.atan2(dx, dy)) % 360 if h_l > 0 else 0
        grade = math.degrees(math.atan(dz/h_l)) if h_l > 0 else 0
        total_high += dz
        
        segment_data = {
            'SORT_ID': f"{i}", 'SECCODE': seccode, 'SECPOINT': f"{i}-{i+1}",
            'AZIMUTH': round(azimuth, 4), 'GRADE': round(grade, 4),
            'SLOPE_L': round(slope_l, 3), 'HIGH': round(dz, 3),
            'TOTAL_HIGH': round(total_high, 3), 'H_L': round(h_l, 3),
            'FROM_X': round(start[0], 3), 'FROM_Y': round(start[1], 3), 'FROM_Z': round(start[2], 3),
            'TO_X': round(end[0], 3), 'TO_Y': round(end[1], 3), 'TO_Z': round(end[2], 3)
        }
        results.append(segment_data)
    
    return pd.DataFrame(results)

def process_section(input_df, seccode_input):
    """核心处理函数"""
    df = smooth_coordinates(input_df)
    points = df[['X', 'Y']].values
    
    # 计算整体方位角
    start_point = df[['X', 'Y', 'Z']].iloc[0].values
    end_point = df[['X', 'Y', 'Z']].iloc[-1].values
    overall_azimuth = calculate_azimuth(start_point[:2], end_point[:2])
    
    raw_segments = adaptive_segmentation(points, df, overall_azimuth)
    
    if not raw_segments:
        raise RuntimeError("未生成有效分段")
    
    adjusted_segments = adjust_segments(raw_segments, df)
    result_df = generate_output(adjusted_segments, seccode_input)
    return result_df

def main():
    seccode = input("请输入剖面编号（例如，PM01）：")
    df = pd.read_excel('GPS_cleaned.xlsx', sheet_name='Sheet1', header=0)
    result_df = process_section(df, seccode)
    result_df.to_excel('section_output.xlsx', index=False)
    print("处理完成！输出文件为section_output.xlsx")

if __name__ == "__main__":
    main()
