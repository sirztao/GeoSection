import os
import sys
import pandas as pd
from data_cleaner import GPSCleaner
from section_final import process_section
from projection_calculator import calculate_projection, generate_layered_output
from rich import print

def main():
    # 用户界面初始化
    print("╔═══════════════════════════════════════════════════════╗")
    print("║                 地质剖面数据处理系统                    ║")
    print("╚═══════════════════════════════════════════════════════╝")

    try:
        # -------------------- 阶段1: 数据清洗 --------------------
        print("\n📍 [1/4] 数据清洗")
        input_file = input("请输入GPS数据文件名（默认：GPS.xlsx）：").strip() or "GPS.xlsx"
        
        cleaner = GPSCleaner(input_path=input_file)
        cleaned_df = cleaner.clean_data(threshold=20)
        cleaned_file = input_file.replace('.xlsx', '_cleaned.xlsx')
        cleaned_df.to_excel(cleaned_file, index=False)
        print(f"✅ 清洗数据已保存至：{cleaned_file}")

        # -------------------- 阶段2: 参数输入 --------------------
        print("\n📍 [2/4] 参数设置")
        while True:
            seccode = input("请输入剖面编号（如PM01）：").strip().upper()
            if len(seccode) >= 4 and seccode.isalnum():
                break
            print("⚠️ 编号需至少4位字母数字组合")

        # -------------------- 阶段3: 生成剖面 --------------------
        print("\n📍 [3/4] 生成剖面数据")
        result_df = process_section(cleaned_df, seccode)
        output_file = f"{seccode}_section.xlsx"
        result_df.to_excel(output_file, index=False)
        print(f"✅ 剖面数据已保存至：{output_file}")

        # -------------------- 阶段4: 投影计算 --------------------
        print("\n📍 [4/4] 投影计算")
        gpoints_df = pd.read_excel("points.xlsx")
        
        # 执行投影计算（关键参数修正）
        projection_df = calculate_projection(
            lines_df=result_df,
            points_df=gpoints_df,
            line_cols=['FROM_X','FROM_Y','FROM_Z','TO_X','TO_Y','TO_Z'],
            point_cols=['序号','X','Y','Z']
        )
        
        # 生成分层数据
        layered_df = generate_layered_output(projection_df, result_df)
        
        # 保存结果文件
        projection_file = f"{seccode}_projection.xlsx"
        layered_file = f"{seccode}_layered.xlsx"
        projection_df.to_excel(projection_file, index=False)
        layered_df.to_excel(layered_file, index=False)
        
        print(f"\n🎉 处理完成！最终结果：\n- {projection_file}\n- {layered_file}")

    # -------------------- 异常处理 --------------------
    except FileNotFoundError as e:
        print(f"\n❌ 文件错误：{str(e)}")
        print("请检查：1.文件是否存在 2.文件名是否正确 3.文件是否被占用")
    except KeyError as e:
        print(f"\n❌ 数据列错误：{str(e)}")
        print("请检查输入文件的列名是否符合要求")
    except Exception as e:
        print(f"\n❌ 系统错误：{str(e)}")
        print("请联系技术支持并提供错误截图")

if __name__ == "__main__":
    main()
