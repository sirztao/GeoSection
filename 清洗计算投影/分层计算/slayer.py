import numpy as np
import pandas as pd

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

def generate_layered_output(projection_df, lines_df):
    """生成分层结构（严格按空间切割顺序）"""
    # 生成线段元数据
    segments = []
    for idx in range(len(lines_df)):
        seg_id = f"{idx}-{idx+1}"
        start = lines_df.iloc[idx][['FROM_X','FROM_Y','FROM_Z']].values
        end = lines_df.iloc[idx][['TO_X','TO_Y','TO_Z']].values
        segments.append({
            'id': seg_id,
            'seccode': lines_df.iloc[idx]['SECCODE'],
            'length': np.linalg.norm(end - start),
            'order': idx  # 线段原始顺序
        })
    
    # 构建分层点集合
    cutting_points = []
    
    # 第一步：收集所有线段的强制起点
    for seg in segments:
        cutting_points.append({
            'type': 'segment_start',
            'SECPOINT': seg['id'],
            'SLOPE_L': 0,
            'SECCODE': seg['seccode'],
            'global_order': seg['order'] * 100000  # 确保起点优先
        })
    
    # 第二步：添加投影点并标记位置
    for _, proj in projection_df.iterrows():
        seg_order = int(proj['线号'].split('-')[0])
        cutting_points.append({
            'type': 'projection',
            'SECPOINT': proj['线号'],
            'SLOPE_L': proj['位置'],
            'SECCODE': proj['SECCODE'],
            'global_order': seg_order * 100000 + proj['位置']
        })
    
    # 第三步：全局排序（先按线段顺序，再按位置）
    sorted_points = sorted(cutting_points, key=lambda x: x['global_order'])
    
    # 第四步：生成分层编码
    layer_records = []
    current_layer = 1
    
    # 添加初始起点
    first_seg = segments[0]
    layer_records.append({
        'SECCODE': first_seg['seccode'],
        'SECPOINT': first_seg['id'],
        'SLOPE_L': 0,
        'LAYCODE': current_layer
    })
    
    # 遍历所有切割点
    for point in sorted_points:
        # 跳过重复起点
        if point['type'] == 'segment_start' and point['SECPOINT'] == layer_records[-1]['SECPOINT']:
            continue
            
        # 新线段起点处理
        if point['type'] == 'segment_start':
            # 继承前一线段的最后一个分层
            layer_records.append({
                'SECCODE': point['SECCODE'],
                'SECPOINT': point['SECPOINT'],
                'SLOPE_L': 0,
                'LAYCODE': current_layer
            })
        else:
            current_layer += 1
            layer_records.append({
                'SECCODE': point['SECCODE'],
                'SECPOINT': point['SECPOINT'],
                'SLOPE_L': point['SLOPE_L'],
                'LAYCODE': current_layer
            })
    
    # 构建最终输出
    result_df = pd.DataFrame(layer_records)
    result_df['SORT_ID'] = range(len(result_df))
    
    return result_df[['SORT_ID', 'SECCODE', 'SECPOINT', 'LAYCODE', 'SLOPE_L']]

def calculate_projection(lines_df, points_df):
    """执行投影计算"""
    # 构建线段数据
    segments = []
    for idx, row in lines_df.iterrows():
        start = np.array([row['FROM_X'], row['FROM_Y'], row['FROM_Z']], dtype=np.float64)
        end = np.array([row['TO_X'], row['TO_Y'], row['TO_Z']], dtype=np.float64)
        seg_id = f"{idx}-{idx+1}"
        segments.append({
            'id': seg_id,
            'seccode': row['SECCODE'],
            'start': start,
            'end': end,
            'length': np.linalg.norm(end - start)
        })
    
    results = []
    for _, row in points_df.iterrows():
        P = np.array([row['X'], row['Y'], row['Z']], dtype=np.float64)
        found = False
        
        # 第一轮检查：直接在线段上
        for i, seg in enumerate(segments):
            S, E = seg['start'], seg['end']
            is_on, t = is_point_on_segment(P, S, E)
            if is_on:
                distance = t * seg['length']
                if np.isclose(distance, seg['length'], atol=1e-6) and i < len(segments)-1:
                    results.append({
                        '线号': segments[i+1]['id'],
                        '位置': 0,
                        'SECCODE': segments[i+1]['seccode']
                    })
                else:
                    results.append({
                        '线号': seg['id'],
                        '位置': int(round(distance)),
                        'SECCODE': seg['seccode']
                    })
                found = True
                break
        
        if not found:
            # 垂面投影计算
            AB_dir = segments[-1]['end'] - segments[0]['start']
            plane_d = np.dot(AB_dir, P)
            
            for i, seg in enumerate(segments):
                S, E = seg['start'], seg['end']
                vec_seg = E - S
                denominator = np.dot(AB_dir, vec_seg)
                
                if np.isclose(denominator, 0):
                    s_dot = np.dot(AB_dir, S)
                    e_dot = np.dot(AB_dir, E)
                    if np.isclose(s_dot, plane_d):
                        results.append({'线号': seg['id'], '位置':0, 'SECCODE':seg['seccode']})
                        found = True
                        break
                    elif np.isclose(e_dot, plane_d):
                        if i < len(segments)-1:
                            results.append({
                                '线号': segments[i+1]['id'],
                                '位置':0,
                                'SECCODE':segments[i+1]['seccode']
                            })
                        else:
                            results.append({
                                '线号': seg['id'],
                                '位置':int(round(seg['length'])),
                                'SECCODE':seg['seccode']
                            })
                        found = True
                        break
                else:
                    t = (plane_d - np.dot(AB_dir, S)) / denominator
                    if 0 <= t <= 1:
                        distance = t * seg['length']
                        results.append({
                            '线号': seg['id'],
                            '位置': int(round(distance)),
                            'SECCODE': seg['seccode']
                        })
                        found = True
                        break
            
            if not found:
                raise ValueError(f"点 {row['序号']} 无法投影")

    return pd.DataFrame(results)

if __name__ == "__main__":
    # 配置参数
    LINES_FILE = 'section.xlsx'
    POINTS_FILE = 'points.xlsx'
    OUTPUT_FILE = 'slayer.xlsx'
    
    try:
        # 读取数据
        lines_df = pd.read_excel(LINES_FILE)
        points_df = pd.read_excel(POINTS_FILE)
        
        # 检查必要列
        if 'SECCODE' not in lines_df.columns:
            raise KeyError("section.xlsx中缺少SECCODE列")
            
        # 执行投影计算
        projection_df = calculate_projection(lines_df, points_df)
        
        # 生成分层结果
        result_df = generate_layered_output(projection_df, lines_df)
        
        # 保存结果
        result_df.to_excel(OUTPUT_FILE, index=False)
        print(f"✅ 结果已成功保存至 {OUTPUT_FILE}")
        
    except FileNotFoundError as e:
        print(f"文件不存在：{str(e)}")
    except KeyError as e:
        print(f"列错误：{str(e)}")
    except ValueError as e:
        print(f"计算错误：{str(e)}")
    except Exception as e:
        print(f"未知错误：{str(e)}")
