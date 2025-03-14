# -*- coding: utf-8 -*-
"""
作者：张涛
日期：2025-03-08
版本：0.4
功能：根据空间坐标点自动分段生成导线，斜距为5的倍数，最后一段小于50米时延伸至50米
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
MIN_SEG_LENGTH = 150  # 最小线段斜距（米）
MAX_SEG_LENGTH = 400 # 最大线段斜距（米）
LENGTH_STEP = 5      # 长度标准化步长（斜距为5的倍数）
# ---------------------------------------------------------------------

def smooth_coordinates(df):
    """数据平滑处理：使用Savitzky-Golay滤波器消除噪声"""
    df = df.rename(columns={df.columns[0]: '点序号', df.columns[1]: 'X', 
                           df.columns[2]: 'Y', df.columns[3]: 'Z'})
    for col in ['X', 'Y', 'Z']:
        df[col] = savgol_filter(df[col], window_length=SMOOTH_WINDOW, polyorder=SMOOTH_ORDER)
    return df

def adaptive_segmentation(points, df):
    """自适应分段：使用RDP算法识别转折点，并按斜距分割"""
    simplified = rdp(points, epsilon=RDP_EPSILON)
    indices = [np.where((points == p).all(axis=1))[0][0] for p in simplified]
    indices = sorted(indices)
    
    segments = []
    prev_idx = indices[0]
    
    for current_idx in indices[1:]:
        seg_points = df.iloc[prev_idx:current_idx+1]
        seg_length = slope_length(seg_points.iloc[0][['X', 'Y', 'Z']].values, 
                                 seg_points.iloc[-1][['X', 'Y', 'Z']].values)
        
        if seg_length > MAX_SEG_LENGTH:
            sub_segs = split_segment(seg_points, prev_idx, df)
            segments.extend(sub_segs)
        elif seg_length >= MIN_SEG_LENGTH:
            segments.append((prev_idx, current_idx))
        prev_idx = current_idx
    
    if not segments or segments[-1][1] != len(df)-1:
        segments.append((prev_idx, len(df)-1))
    
    return segments

def split_segment(seg_points, start_idx, df, depth=0):
    """递归分割超长线段"""
    if depth > 20:
        print("达到最大递归深度，强制终止分割")
        return [(start_idx, start_idx + len(seg_points) - 1)]
    
    if len(seg_points) < 2:
        return []

    split_idx = 1
    for i in range(1, len(seg_points)-1):
        front_len = slope_length(seg_points.iloc[0][['X', 'Y', 'Z']].values, 
                                seg_points.iloc[i][['X', 'Y', 'Z']].values)
        back_len = slope_length(seg_points.iloc[i][['X', 'Y', 'Z']].values, 
                               seg_points.iloc[-1][['X', 'Y', 'Z']].values)
        if front_len <= MAX_SEG_LENGTH and back_len <= MAX_SEG_LENGTH:
            split_idx = i
            break
    
    first_seg = (start_idx, start_idx + split_idx)
    second_seg = (start_idx + split_idx, start_idx + len(seg_points)-1)
    
    segments = []
    front_points = df.iloc[first_seg[0]:first_seg[1]+1]
    if slope_length(front_points.iloc[0][['X', 'Y', 'Z']].values, 
                   front_points.iloc[-1][['X', 'Y', 'Z']].values) > MAX_SEG_LENGTH:
        segments.extend(split_segment(front_points, first_seg[0], df, depth+1))
    else:
        segments.append(first_seg)
    
    back_points = df.iloc[second_seg[0]:second_seg[1]+1]
    if slope_length(back_points.iloc[0][['X', 'Y', 'Z']].values, 
                   back_points.iloc[-1][['X', 'Y', 'Z']].values) > MAX_SEG_LENGTH:
        segments.extend(split_segment(back_points, second_seg[0], df, depth+1))
    else:
        segments.append(second_seg)
    
    return segments

def adjust_segments(original_segments, df):
    """线段标准化：斜距调整为5的倍数，最后一段小于50米时延伸至50米"""
    adjusted = []
    current_pos = df[['X', 'Y', 'Z']].iloc[0].values
    
    for i, (start_idx, end_idx) in enumerate(original_segments):
        original_end = df[['X', 'Y', 'Z']].iloc[end_idx].values
        vec = original_end - current_pos
        raw_slope_l = np.linalg.norm(vec)  # 计算三维斜距
        
        is_last = (i == len(original_segments) - 1)
        if is_last:
            if raw_slope_l < MIN_SEG_LENGTH:
                target_slope_l = MIN_SEG_LENGTH  # 最后一段小于50米时延伸至50米
            else:
                target_slope_l = round(raw_slope_l / LENGTH_STEP) * LENGTH_STEP
        else:
            target_slope_l = round(raw_slope_l / LENGTH_STEP) * LENGTH_STEP
            if target_slope_l < MIN_SEG_LENGTH:
                target_slope_l = MIN_SEG_LENGTH
            elif target_slope_l > MAX_SEG_LENGTH:
                target_slope_l = MAX_SEG_LENGTH
        
        if raw_slope_l > 0:
            ratio = target_slope_l / raw_slope_l
            new_end = current_pos + vec * ratio  # 按原始方向调整终点
        else:
            new_end = current_pos.copy()
        
        adjusted.append((current_pos.copy(), new_end))
        current_pos = new_end  # 更新起点为上一段终点
    
    return adjusted

def generate_output(segments, seccode):
    """生成输出表格：包含测量参数"""
    results = []
    total_high = 0.0
    
    for i, (start, end) in enumerate(segments):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        
        h_l = math.hypot(dx, dy)  # 水平投影距离
        slope_l = math.sqrt(h_l**2 + dz**2)  # 三维斜距
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

def slope_length(p1, p2):
    """计算三维斜距"""
    return np.linalg.norm(p2 - p1)

def process_section(input_df, seccode_input):
    """核心处理函数"""
    df = smooth_coordinates(input_df)
    points = df[['X', 'Y']].values
    raw_segments = adaptive_segmentation(points, df)
    
    if raw_segments:
        raw_segments[0] = (0, raw_segments[0][1])  # 强制包含起点
        raw_segments[-1] = (raw_segments[-1][0], len(df)-1)  # 强制包含终点
    else:
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
