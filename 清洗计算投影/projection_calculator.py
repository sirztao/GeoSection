# projection_calculator.py
import numpy as np
import pandas as pd
import argparse

def is_point_on_segment(P, S, E, tol=1e-6):
    """判断点是否在线段S-E上"""
    vec_SE = E - S
    vec_SP = P - S
    
    if np.linalg.norm(vec_SE) < tol:
        return (np.linalg.norm(P - S) < tol), 0.0
    
    t = np.dot(vec_SP, vec_SE) / np.dot(vec_SE, vec_SE)
    if t < -tol or t > 1 + tol:
        return False, None
    
    projection = S + t * vec_SE
    if np.linalg.norm(projection - P) < tol:
        return True, max(0.0, min(t, 1.0))
    return False, None

def calculate_projection(lines_df, points_df, 
                         line_cols=['FROM_X','FROM_Y','FROM_Z','TO_X','TO_Y','TO_Z'], 
                         point_cols=['序号','X','Y','Z']):
    """投影计算主函数"""
    # 构建线段数据
    segments = []
    for idx, row in lines_df.iterrows():
        start = np.array([row[line_cols[0]], row[line_cols[1]], row[line_cols[2]]], dtype=np.float64)
        end = np.array([row[line_cols[3]], row[line_cols[4]], row[line_cols[5]]], dtype=np.float64)
        seg_id = f"{idx}-{idx+1}"
        segments.append({'id': seg_id, 'start': start, 'end': end})
    
    # 确定线段AB方向向量
    S_AB = segments[0]['start']
    E_AB = segments[-1]['end']
    AB_dir = E_AB - S_AB
    
    # 处理地质点
    results = []
    for _, row in points_df.iterrows():
        p_id = row[point_cols[0]]
        P = np.array([row[point_cols[1]], row[point_cols[2]], row[point_cols[3]]], dtype=np.float64)
        
        # 检查是否直接在线段上
        found = False
        for i, seg in enumerate(segments):
            S, E = seg['start'], seg['end']
            is_on, t = is_point_on_segment(P, S, E)
            if is_on:
                seg_length = np.linalg.norm(E - S)
                distance = t * seg_length
                # 端点归属处理
                if np.isclose(distance, seg_length, atol=1e-6) and i < len(segments)-1:
                    results.append((segments[i+1]['id'], p_id, 0))
                else:
                    results.append((seg['id'], p_id, int(round(distance))))
                found = True
                break
        if found:
            continue
        
        # 垂面投影计算
        plane_d = np.dot(AB_dir, P)
        for i, seg in enumerate(segments):
            seg_id = seg['id']
            S = seg['start']
            E = seg['end']
            vec_seg = E - S
            denominator = np.dot(AB_dir, vec_seg)
            
            if np.isclose(denominator, 0):
                s_dot = np.dot(AB_dir, S)
                e_dot = np.dot(AB_dir, E)
                if np.isclose(s_dot, plane_d):
                    results.append((seg_id, p_id, 0))
                    found = True
                    break
                elif np.isclose(e_dot, plane_d):
                    if i < len(segments)-1:
                        results.append((segments[i+1]['id'], p_id, 0))
                    else:
                        seg_length = np.linalg.norm(vec_seg)
                        results.append((seg_id, p_id, int(round(seg_length))))
                    found = True
                    break
            else:
                numerator = plane_d - np.dot(AB_dir, S)
                t = numerator / denominator
                if 0 <= t <= 1:
                    seg_length = np.linalg.norm(vec_seg)
                    distance = t * seg_length
                    # 端点处理
                    if np.isclose(distance, seg_length, atol=1e-6):
                        if i < len(segments)-1:
                            results.append((segments[i+1]['id'], p_id, 0))
                        else:
                            results.append((seg_id, p_id, int(round(seg_length))))
                    else:
                        results.append((seg_id, p_id, int(round(distance))))
                    found = True
                    break
        if not found:
            raise ValueError(f"垂面无交点: 点{p_id}")

    # 创建DataFrame并排序
    result_df = pd.DataFrame(results, columns=['线号', '点序号', '位置'])
    
    # 生成排序键
    def get_segment_order(seg_id):
        return tuple(map(int, seg_id.split('-')))
    
    # 复合排序：线号自然序 → 位置升序
    result_df['_sort_key'] = result_df['线号'].apply(get_segment_order)
    result_df = result_df.sort_values(
        by=['_sort_key', '位置'], 
        ascending=[True, True]
    ).drop('_sort_key', axis=1)
    
    return result_df.reset_index(drop=True)

if __name__ == "__main__":
    # 命令行参数配置
    parser = argparse.ArgumentParser(description='地质点投影计算工具')
    parser.add_argument('--lines', default='lines.xlsx', help='线段数据文件（默认：lines.xlsx）')
    parser.add_argument('--points', default='Gpoint.xlsx', help='地质点数据文件（默认：Gpoint.xlsx）')
    parser.add_argument('--output', default='projection_output.xlsx', help='输出文件（默认：projection_output.xlsx）')
    parser.add_argument('--line_cols', nargs=6, 
                        default=['FROM_X','FROM_Y','FROM_Z','TO_X','TO_Y','TO_Z'],
                        help='线段列名：起点X 起点Y 起点Z 终点X 终点Y 终点Z（默认：FROM_X FROM_Y FROM_Z TO_X TO_Y TO_Z）')
    parser.add_argument('--point_cols', nargs=4, 
                        default=['序号','X','Y','Z'],
                        help='地质点列名：序号 X Y Z（默认：序号 X Y Z）')
    args = parser.parse_args()

    # 读取数据
    try:
        lines_df = pd.read_excel(args.lines)
        points_df = pd.read_excel(args.points)
    except FileNotFoundError as e:
        print(f"错误：未找到文件 - {e}")
        exit(1)

    # 执行计算
    try:
        result_df = calculate_projection(
            lines_df=lines_df,
            points_df=points_df,
            line_cols=args.line_cols,
            point_cols=args.point_cols
        )
    except KeyError as e:
        print("列名错误！请检查以下配置：")
        print(f"线段列名配置：{args.line_cols}")
        print(f"地质点列名配置：{args.point_cols}")
        exit(1)
    except ValueError as ve:
        print(f"计算错误：{str(ve)}")
        exit(1)

    # 保存结果
    result_df.to_excel(args.output, index=False)
    print(f"✅ 计算结果已保存至：{args.output}")
