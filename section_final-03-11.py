import pandas as pd
import numpy as np
import math
from rdp import rdp
from scipy.signal import savgol_filter

# ---------------------------- 参数配置区域 ----------------------------
SMOOTH_WINDOW = 5     # 平滑窗口（必须为奇数）
SMOOTH_ORDER = 2      # 滤波多项式阶数
RDP_EPSILON = 5.0     # RDP简化阈值（米）
MIN_SEG_LENGTH = 100   # 最小线段长度（米）
MAX_SEG_LENGTH = 500  # 最大线段长度（米）
MAX_RECURSION = 20    # 最大递归深度
LENGTH_STEP = 5       # 长度标准化步长
# ---------------------------------------------------------------------

def print_data_format_warning():
    """数据格式警告提示"""
    print("""
╔══════════════ 警告 ════════════════╗
║ 请确认输入数据包含以下列（顺序敏感）：       ║
║ 1.点序号 2.东坐标X 3.北坐标Y 4.高程Z ║
╚════════════════════════════════════╝""")

def smooth_coordinates(df):
    """坐标平滑处理"""
    df = df.rename(columns={
        df.columns[0]: '点序号',
        df.columns[1]: 'X',
        df.columns[2]: 'Y',
        df.columns[3]: 'Z'
    })
    
    for col in ['X', 'Y', 'Z']:
        df[col] = savgol_filter(df[col], 
                              window_length=SMOOTH_WINDOW,
                              polyorder=SMOOTH_ORDER)
    return df

def adaptive_segmentation(points):
    """自适应分段核心逻辑"""
    total_length = segment_length(points[0], points[-1])
    epsilon = max(RDP_EPSILON, total_length * 0.03)
    
    simplified = rdp(points, epsilon=epsilon)
    indices = sorted(np.where((points == s).all(axis=1))[0][0] for s in simplified)
    
    segments = []
    prev_idx = indices[0]
    
    for current_idx in indices[1:]:
        seg_points = points[prev_idx:current_idx+1]
        seg_length = segment_length(seg_points[0], seg_points[-1])
        
        if seg_length > MAX_SEG_LENGTH:
            sub_segs = split_segment(seg_points, prev_idx)
            segments.extend(sub_segs)
        elif seg_length >= MIN_SEG_LENGTH:
            segments.append((prev_idx, current_idx))
        else:
            if segments:
                segments[-1] = (segments[-1][0], current_idx)
            else:
                segments.append((prev_idx, current_idx))
        prev_idx = current_idx
    
    # 最终段强制校验
    last_start, last_end = prev_idx, len(points)-1
    last_length = segment_length(points[last_start], points[last_end])
    if last_length > MAX_SEG_LENGTH:
        sub_segs = split_segment(points[last_start:last_end+1], last_start)
        segments.extend(sub_segs)
    elif last_length >= MIN_SEG_LENGTH:
        segments.append((last_start, last_end))
    else:
        if segments:
            segments[-1] = (segments[-1][0], last_end)
        else:
            segments.append((last_start, last_end))
    
    return segments

def split_segment(points, start_idx, depth=0):
    """智能分割函数"""
    if depth > MAX_RECURSION or len(points) < 4:
        return [(start_idx, start_idx + len(points)-1)]
    
    # 寻找合法分割点
    valid_splits = []
    for i in range(1, len(points)-1):
        front_len = segment_length(points[0], points[i])
        back_len = segment_length(points[i], points[-1])
        if (MIN_SEG_LENGTH <= front_len <= MAX_SEG_LENGTH and
            MIN_SEG_LENGTH <= back_len <= MAX_SEG_LENGTH):
            valid_splits.append(i)
    
    if valid_splits:
        split_idx = valid_splits[len(valid_splits)//2]
    else:
        split_idx = max(range(1, len(points)-1),
                       key=lambda x: segment_length(points[0], points[x]))
    
    # 处理子段
    first_seg = (start_idx, start_idx + split_idx)
    second_seg = (start_idx + split_idx, start_idx + len(points)-1)
    
    segments = []
    front_seg = points[:split_idx+1]
    if segment_length(front_seg[0], front_seg[-1]) > MAX_SEG_LENGTH:
        segments.extend(split_segment(front_seg, start_idx, depth+1))
    else:
        segments.append(first_seg)
    
    back_seg = points[split_idx:]
    if segment_length(back_seg[0], back_seg[-1]) > MAX_SEG_LENGTH:
        segments.extend(split_segment(back_seg, first_seg[1], depth+1))
    else:
        segments.append(second_seg)
    
    return segments

def adjust_segments(original_segments, df):
    """长度标准化（最后段取整）"""
    adjusted = []
    current_pos = df[['X', 'Y', 'Z']].iloc[0].values
    
    for i, (start_idx, end_idx) in enumerate(original_segments):
        original_end = df[['X', 'Y', 'Z']].iloc[end_idx].values
        vec = original_end - current_pos
        raw_length = np.linalg.norm(vec[:2])  # 水平投影长度
        
        is_last = (i == len(original_segments)-1)
        
        if is_last:
            # 最后段处理：取整并强制限制
            temp_length = min(MAX_SEG_LENGTH, max(MIN_SEG_LENGTH, raw_length))
            target_length = round(temp_length)
            target_length = min(MAX_SEG_LENGTH, max(MIN_SEG_LENGTH, target_length))
        else:
            # 常规处理：步长倍数
            target_length = round(raw_length/LENGTH_STEP)*LENGTH_STEP
            target_length = min(MAX_SEG_LENGTH, max(MIN_SEG_LENGTH, target_length))
        
        if raw_length > 0:
            ratio = target_length / raw_length
            new_end = current_pos + vec * ratio
        else:
            new_end = current_pos.copy()
        
        adjusted.append((current_pos.copy(), new_end))
        current_pos = new_end
    
    return adjusted

def generate_output(segments, seccode):
    """结果生成器"""
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
        
        results.append({
            'SORT_ID': f"{i}",
            'SECCODE': seccode,
            'SECPOINT': f"{i}-{i+1}",  
            'AZIMUTH': round(azimuth, 4),
            'GRADE': round(grade, 4),
            'SLOPE_L': round(slope_l, 3),
            'HIGH': round(dz, 3),
            'TOTAL_HIGH': round(total_high, 3),
            'H_L': round(h_l, 3),
            'CUR_ID': 0,
            'FROM_X': round(start[0], 3),
            'FROM_Y': round(start[1], 3),
            'FROM_Z': round(start[2], 3),
            'TO_X': round(end[0], 3),
            'TO_Y': round(end[1], 3),
            'TO_Z': round(end[2], 3)
        })
    
    return pd.DataFrame(results)

def segment_length(p1, p2):
    """二维距离计算"""
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def process_section(input_df, seccode_input):
    """处理流程控制器"""
    required_cols = ['X', 'Y', 'Z']
    if not all(col in input_df.columns for col in required_cols):
        raise ValueError("输入数据必须包含X,Y,Z列")

    df = input_df.copy()
    df = smooth_coordinates(df)
    
    points = df[['X', 'Y']].values
    raw_segments = adaptive_segmentation(points)
    
    # 首尾点强制包含
    if raw_segments:
        raw_segments[0] = (0, raw_segments[0][1])
        if raw_segments[-1][1] != len(df)-1:
            raw_segments.append((raw_segments[-1][1], len(df)-1))
    
    adjusted_segments = adjust_segments(raw_segments, df)
    return generate_output(adjusted_segments, seccode_input)

def main():
    """独立运行入口"""
    print_data_format_warning()
    seccode = input("请输入剖面编号（例如PM01）：")
    df = pd.read_excel('GPS_cleaned.xlsx', sheet_name='Sheet1')
    result_df = process_section(df, seccode)
    result_df.to_excel('section_output.xlsx', index=False)
    print("处理完成！结果已保存至section_output.xlsx")

if __name__ == "__main__":
    main()
