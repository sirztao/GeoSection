# -*- coding: utf-8 -*-
"""
作者：张涛
日期：2025-03-08
版本：0.2
野外地质剖面分段处理程序（模块化改造版）
功能：根据空间坐标点自动分段生成导线，生成测量线段参数表
"""

import pandas as pd
import numpy as np
import math
from rdp import rdp
from scipy.signal import savgol_filter

# ---------------------------- 可调参数区域 ----------------------------
SMOOTH_WINDOW = 9    # 平滑滤波窗口大小（必须为奇数，最开始默认为5）
SMOOTH_ORDER = 2     # 平滑滤波多项式阶数（通常为2或3）
RDP_EPSILON = 7.0    # 线段简化阈值（单位：米，越大简化程度越高，最开始默认为5）
MIN_SEG_LENGTH = 50  # 允许的最小线段长度（米）
MAX_SEG_LENGTH = 500 # 允许的最大线段长度（米）
MAX_RECURSION = 20   # 最大递归深度防止栈溢出
LENGTH_STEP = 5      # 长度标准化步长（5的倍数）
# ---------------------------------------------------------------------

def print_data_format_warning():
    warning_msg = """
╔══════════════ 警告 WARNING ══════════════════╗
║                                                                              ║
║   ██ 重要提示：使用本程序前请 严格 遵守以下数据格式要求！否则将导致 ██   ║
║   ██ 程序运行失败，并可能造成不可逆的数据损失！后果自负！█████████║
║                                                                              ║
╠══════════════强制数据格式 ══════════════════ ╣
║ ► 列顺序必须为（共4列，不可增减、不可调序）：                               ║
║                                                                              ║
║     [1] 点序号 (整数)       → 示例：1, 2, 3...                              ║
║     [2] X坐标 (东方向，米)   → 示例：123456.789                             ║
║     [3] Y坐标 (北方向，米)   → 示例：987654.321                             ║
║     [4] Z坐标 (海拔，米)     → 示例：152.25                                 ║
║                                                                              ║
╠════════════注意事项 ══════════════════════ ╣
║ ► 文件要求：                                                                ║
║   ✓ 使用UTF-8编码           ✓ 只能使用xlsx格式                              ║
║                                                                              ║
║                                                                              ║
╚═══════════════════════════════ ═══════╝
"""
    print(warning_msg)

def smooth_coordinates(df):
    """
    数据平滑处理函数
    功能：使用Savitzky-Golay滤波器消除测量噪声
    输入要求：原始数据列顺序为[序号, X(东), Y(北), Z(高程)]
    """
    # 重命名列以统一处理
    df = df.rename(columns={
        df.columns[0]: '点序号',
        df.columns[1]: 'X',
        df.columns[2]: 'Y',
        df.columns[3]: 'Z'
    })
    
    # 对每个坐标轴进行平滑处理
    for col in ['X', 'Y', 'Z']:
        df[col] = savgol_filter(df[col], 
                              window_length=SMOOTH_WINDOW,
                              polyorder=SMOOTH_ORDER)
    return df

def adaptive_segmentation(points):
    """
    自适应分段主函数
    处理流程：
    1. 使用RDP算法识别地形转折点
    2. 递归分割超出长度限制的线段
    3. 强制包含首尾点保证路径完整性
    """
    # 第一阶段：RDP算法简化路径
    simplified = rdp(points, epsilon=RDP_EPSILON)
    indices = [np.where((points == p).all(axis=1))[0][0] for p in simplified]
    indices = sorted(indices)
    
    segments = []
    prev_idx = indices[0]
    
    # 第二阶段：处理各简化段
    for current_idx in indices[1:]:
        seg_points = points[prev_idx:current_idx+1]
        seg_length = segment_length(seg_points[0], seg_points[-1])
        
        if seg_length > MAX_SEG_LENGTH:
            sub_segs = split_segment(seg_points, prev_idx)
            segments.extend(sub_segs)
        elif seg_length >= MIN_SEG_LENGTH:
            segments.append((prev_idx, current_idx))
        prev_idx = current_idx
    
    # 强制包含最后一个点
    if not segments or segments[-1][1] != len(points)-1:
        segments.append((prev_idx, len(points)-1))
    
    return segments

def split_segment(points, start_idx, depth=0):
    """
    递归分割函数（带深度控制）
    分割策略：
    1. 优先寻找符合长度约束的分割点
    2. 次优选择使两段最接近目标长度的点
    """
    if depth > MAX_RECURSION:
        print(f"达到最大递归深度{MAX_RECURSION}，强制终止分割")
        return [(start_idx, start_idx + len(points) - 1)]
    
    if len(points) < 2:
        return []

    # 寻找最佳分割点
    split_idx = 1
    for i in range(1, len(points)-1):
        front_len = segment_length(points[0], points[i])
        back_len = segment_length(points[i], points[-1])
        if front_len <= MAX_SEG_LENGTH and back_len <= MAX_SEG_LENGTH:
            split_idx = i
            break
    
    # 定义子段
    first_seg = (start_idx, start_idx + split_idx)
    second_seg = (start_idx + split_idx, start_idx + len(points)-1)
    
    segments = []
    # 处理前半段
    if segment_length(points[0], points[split_idx]) > MAX_SEG_LENGTH:
        segments.extend(split_segment(points[:split_idx+1], start_idx, depth+1))
    else:
        segments.append(first_seg)
    
    # 处理后半段
    if segment_length(points[split_idx], points[-1]) > MAX_SEG_LENGTH:
        segments.extend(split_segment(points[split_idx:], first_seg[1], depth+1))
    else:
        segments.append(second_seg)
    
    return segments

def adjust_segments(original_segments, df):
    """
    线段标准化函数
    核心功能：
    1. 将线段长度调整为LENGTH_STEP的整数倍
    2. 保证各线段首尾相接
    3. 最后一段保留原始长度
    """
    adjusted = []
    current_pos = df[['X', 'Y', 'Z']].iloc[0].values
    
    for i, (start_idx, end_idx) in enumerate(original_segments):
        original_end = df[['X', 'Y', 'Z']].iloc[end_idx].values
        
        # 计算原始长度
        vec = original_end - current_pos
        raw_length = np.linalg.norm(vec)
        
        # 确定目标长度
        is_last = (i == len(original_segments)-1)
        target_length = round(raw_length) if is_last else round(raw_length/LENGTH_STEP)*LENGTH_STEP
        
        # 调整终点位置
        if raw_length > 0:
            ratio = target_length / raw_length
            new_end = current_pos + vec * ratio
        else:
            new_end = current_pos.copy()
        
        adjusted.append((current_pos.copy(), new_end))
        current_pos = new_end
    
    return adjusted

def generate_output(segments, seccode):
    """
    生成最终输出结果的核心函数
    功能：将处理后的线段数据转换为包含测量参数的表格
    
    参数：
        segments - 处理后的线段列表，每个元素为元组 (起点坐标, 终点坐标)
                  示例：[(array([x1,y1,z1]), array([x2,y2,z2])), ...]
        seccode - 剖面编号（外部传入）
    
    返回：
        pd.DataFrame - 包含所有测量参数的表格数据
    """
    results = []
    total_high = 0.0  # 累计高程变化量
    
    # 遍历所有线段，i为线段索引（从0开始）
    for i, (start, end) in enumerate(segments):
        # 计算三维坐标差
        dx = end[0] - start[0]  # 东方向变化量
        dy = end[1] - start[1]  # 北方向变化量
        dz = end[2] - start[2]  # 高程变化量
        
        # 计算基本几何参数
        h_l = math.hypot(dx, dy)          # 水平投影长度（米）
        slope_l = math.sqrt(h_l**2 + dz**2) # 三维斜距（米）
        
        # 计算方位角（0°为正北，顺时针增加）
        azimuth = math.degrees(math.atan2(dx, dy)) % 360 if h_l > 0 else 0
        
        # 计算坡度角（正值为上坡，负值为下坡） 
        grade = math.degrees(math.atan(dz/h_l)) if h_l > 0 else 0
        
        # 累计高程变化（用于TOTAL_HIGH字段）
        total_high += dz
        
        # 构建结果字典（注意字段顺序决定输出列顺序）
        segment_data = {
            #先填写序号
            'SORT_ID': f"{i}", #这里是导线号
            'SECCODE': seccode, #剖面号
            # 新增字段：线段编号（格式：0-1,1-2...）
            # 表示线段在序列中的顺序位置，与物理坐标无关
            'SECPOINT': f"{i}-{i+1}",  
            
            # 方向参数
            'AZIMUTH': round(azimuth, 4),     # 方位角（度，保留4位小数）
            'GRADE': round(grade, 4),        # 坡度角（度，保留4位小数）
            
            # 距离参数
            'SLOPE_L': round(slope_l, 3),    # 三维斜距（米，保留3位小数）
            # 高程参数
            'HIGH': round(dz, 3),            # 本段高程变化（米，保留3位小数）
            'TOTAL_HIGH': round(total_high, 3), # 累计高程变化（米，保留3位小数）
            'H_L': round(h_l, 3),            # 水平投影长度（米，保留3位小数）
            
            # 固定值字段
            'CUR_ID': 0,  # 当前ID（按需求固定为0）
            
            # 起点坐标（东、北、高程）
            'FROM_X': round(start[0], 3),  
            'FROM_Y': round(start[1], 3),
            'FROM_Z': round(start[2], 3),
            
            # 终点坐标（东、北、高程） 
            'TO_X': round(end[0], 3),
            'TO_Y': round(end[1], 3),
            'TO_Z': round(end[2], 3)
        }
        
        results.append(segment_data)
    
    # 转换为DataFrame并确保列顺序
    # 注意：字典键的顺序决定列顺序，Python 3.7+保证插入顺序
    return pd.DataFrame(results)

def segment_length(p1, p2):
    """计算二维水平投影距离"""
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def process_section(input_df, seccode_input):
    """
    核心处理函数（新增）
    :param input_df: 包含X,Y,Z列的DataFrame（点序号列可选）
    :param seccode_input: 剖面编号（如PM01）
    :return: 处理结果DataFrame
    """
    # 确保列名正确（兼容清洗后的数据）
    required_cols = ['X', 'Y', 'Z']
    if not all(col in input_df.columns for col in required_cols):
        raise ValueError("输入数据必须包含X,Y,Z列")

    # 数据预处理
    df = input_df.copy()
    df = smooth_coordinates(df)
    
    # 生成初始分段
    points = df[['X', 'Y']].values
    raw_segments = adaptive_segmentation(points)
    
    # 强制首尾点匹配
    if raw_segments:
        raw_segments[0] = (0, raw_segments[0][1])
        raw_segments[-1] = (raw_segments[-1][0], len(df)-1)
    else:
        raise RuntimeError("未生成有效分段")

    # 线段标准化处理
    adjusted_segments = adjust_segments(raw_segments, df)
    
    # 生成输出结果（传入seccode）
    result_df = generate_output(adjusted_segments, seccode_input)
    return result_df

def main():
    # 保留独立运行能力（兼容旧模式）
    print_data_format_warning()
    seccode = input("请输入剖面编号（例如，PM01）：")
    df = pd.read_excel('GPS_cleaned.xlsx', sheet_name='Sheet1', header=0)
    result_df = process_section(df, seccode)
    result_df.to_excel('section_output.xlsx', index=False)
    print("处理完成！输出文件为section_output.xlsx，请查收！！")

if __name__ == "__main__":
    main()
