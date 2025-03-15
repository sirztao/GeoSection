# data_cleaner.py
import pandas as pd
import numpy as np

class GPSCleaner:
    def __init__(self, input_path):
        self.input_path = input_path

    def project_point(self, point, A, B):
        AB = B - A
        AP = point[:2] - A
        t = np.dot(AP, AB) / np.dot(AB, AB)
        return A + t * AB

    def vertical_distance(self, point, A, B):
        AB = B - A
        AP = point[:2] - A
        cross = AB[0]*AP[1] - AB[1]*AP[0]
        return np.abs(cross) / np.linalg.norm(AB)

    def clean_data(self, threshold=20):
        # 读取原始数据
        df = pd.read_excel(self.input_path, header=None, skiprows=1)
        df.columns = ['点序号', 'X', 'Y', 'Z']
        
        # 获取首尾点
        start_point = df.iloc[0][['X','Y','Z']].values
        end_point = df.iloc[-1][['X','Y','Z']].values
        A = start_point[:2]
        B = end_point[:2]

        # 排序逻辑
        points_xy = df[['X','Y']].values
        AB = B - A
        AB_sq = np.dot(AB, AB)
        df['t'] = [np.dot(p - A, AB)/AB_sq for p in points_xy]
        sorted_df = df.sort_values('t').reset_index(drop=True)

        # 清洗准备
        points = sorted_df[['X','Y','Z']].values
        v_dists = [self.vertical_distance(p, A, B) for p in points]
        points_with_dist = list(zip(points, v_dists))

        # 清洗过程
        cleaned = self._clean_adjacent(points_with_dist, A, B, threshold)
        cleaned_points = np.array([p for p, _ in cleaned])

        # 去重和排序
        final_df = pd.DataFrame(cleaned_points, columns=['X','Y','Z'])
        final_df = final_df.drop_duplicates()
        final_df['t'] = [np.dot(p[:2]-A, AB)/AB_sq for p in final_df.values]
        final_df = final_df.sort_values('t').reset_index(drop=True)
        
        # 强制保留首尾点
        if not np.array_equal(final_df.iloc[0][['X','Y','Z']].values, start_point):
            final_df = pd.concat([pd.DataFrame([start_point], columns=['X','Y','Z']), final_df])
        if not np.array_equal(final_df.iloc[-1][['X','Y','Z']].values, end_point):
            final_df = pd.concat([final_df, pd.DataFrame([end_point], columns=['X','Y','Z'])])

        # 生成最终数据
        final_df['点序号'] = range(1, len(final_df)+1)
        return final_df[['点序号','X','Y','Z']]

    def _clean_adjacent(self, points_with_dist, A, B, threshold):
        while True:
            modified = False
            new_points = []
            i = 0
            n = len(points_with_dist)
            while i < n:
                if i == n-1:
                    new_points.append(points_with_dist[i])
                    break
                p1, d1 = points_with_dist[i]
                p2, d2 = points_with_dist[i+1]
                
                q1 = self.project_point(p1, A, B)
                q2 = self.project_point(p2, A, B)
                dist = np.linalg.norm(q2 - q1)
                
                if dist < threshold:
                    if d1 <= d2:
                        new_points.append((p1, d1))
                    else:
                        new_points.append((p2, d2))
                    i += 2
                    modified = True
                else:
                    new_points.append((p1, d1))
                    i += 1
            if not modified:
                break
            points_with_dist = new_points
        return points_with_dist

# ===================== 独立运行支持 =====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPS数据清洗工具")
    parser.add_argument("-i", "--input", type=str, default="GPS.xlsx",
                        help="输入文件名（默认：GPS.xlsx）")
    parser.add_argument("-t", "--threshold", type=float, default=20,
                        help="投影距离阈值（默认：10米）")
    args = parser.parse_args()

    print(f"""
    ╔══════════════════════════════════════════════╗
    ║                                              ║
    ║           独立数据清洗模式启动               ║
    ║        输入文件：{args.input.ljust(20)}        ║
    ║        清洗阈值：{str(args.threshold).ljust(20)}米  ║
    ║                                              ║
    ╚══════════════════════════════════════════════╝
    """)

    try:
        cleaner = GPSCleaner(args.input)
        cleaned_df = cleaner.clean_data(args.threshold)
        
        output_file = args.input.replace(".xlsx", f"_cleaned.xlsx")
        cleaned_df.to_excel(output_file, index=False)
        print(f"\n✅ 清洗完成！结果已保存至：{output_file}")
        
    except Exception as e:
        print(f"\n❌ 处理失败：{str(e)}")
        print("请检查：")
        print(f"1. 文件 {args.input} 是否存在")
        print("2. 数据格式是否符合要求（4列：点序号,X,Y,Z）")