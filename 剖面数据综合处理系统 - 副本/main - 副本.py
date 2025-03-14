# main.py
import os
import sys
import pandas as pd
from data_cleaner import GPSCleaner
from section_final import process_section
from projection_calculator import calculate_projection, generate_layered_output

def main():
    # 用户界面初始化
    print("╔═══════════════════════════════════════════════════════╗")
    print("║                                                       ║")
    print("║          地质剖面数据处理系统 - 主流程启动            ║")
    print("║                                                       ║")
    print("╚═══════════════════════════════════════════════════════╝")

    try:
        # -------------------- 阶段1: 数据清洗 --------------------
        print("\n📍 [阶段 1/4] 数据清洗准备")
        default_file = 'GPS.xlsx'
        input_file = input(f"请输入要处理的文件名（默认 {default_file}，输入q退出）：").strip()
        
        if input_file.lower() == 'q':
            print("❌ 用户终止程序")
            sys.exit(0)
            
        input_file = input_file if input_file else default_file
        print(f"   • 使用文件：{input_file}")
        
        print("\n🛠️  开始数据清洗...")
        cleaner = GPSCleaner(input_path=input_file)
        cleaned_df = cleaner.clean_data(threshold=20)
        print("   ✅ 清洗完成！")

        # 保存中间数据
        cleaned_file = input_file.replace('.xlsx', '_cleaned.xlsx')
        print(f"\n💾 保存清洗结果至：{cleaned_file}")
        cleaned_df.to_excel(cleaned_file, index=False)
        print("   ✅ 中间数据保存成功！")

        # -------------------- 阶段2: 用户输入 --------------------
        print("\n📍 [阶段 2/4] 参数输入")
        while True:
            seccode = input("请输入剖面编号（例如 PM01，输入q退出）：").strip()
            if seccode.lower() == 'q':
                print("❌ 用户终止程序")
                sys.exit(0)
            if len(seccode) >= 4 and seccode.isalnum():
                break
            print("⚠️  编号格式错误！要求：至少4位字母数字组合（如PM01）")

        # 数据验证
        required_cols = ['X', 'Y', 'Z']
        if not all(col in cleaned_df.columns for col in required_cols):
            print("❌ 数据错误：缺少必要的坐标列（X, Y, Z）")
            sys.exit(1)

        # 生成剖面数据
        print("\n📍 [阶段 3/4] 核心处理")
        print("   🚀 开始生成剖面数据...")
        result_df = process_section(cleaned_df, seccode)
        print("   ✅ 剖面数据生成完成！")

        # 保存剖面数据
        output_file = f"{seccode}_section_output.xlsx"
        if os.path.exists(output_file):
            overwrite = input(f"⚠️  文件 {output_file} 已存在，是否覆盖？(y/n): ").strip().lower()
            if overwrite != 'y':
                print("❌ 用户选择不覆盖文件，程序终止")
                sys.exit(0)

        print(f"\n💾 保存剖面数据至：{output_file}")
        result_df.to_excel(output_file, index=False)
        print("   ✅ 剖面数据保存成功！")

        # -------------------- 阶段4: 投影计算 --------------------
        print("\n📍 [阶段 4/4] 投影计算")
        print("   🚀 开始进行地质点投影计算...")    
        # 读取地质点数据
        gpoint_file = "points.xlsx"
        print(f"\n📖 读取地质点数据：{gpoint_file}")
        try:
            gpoints_df = pd.read_excel(gpoint_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"地质点文件 {gpoint_file} 不存在")

        # 定义列名映射（根据实际数据修改！）
        LINE_COLS = ['FROM_X', 'FROM_Y', 'FROM_Z', 'TO_X', 'TO_Y', 'TO_Z']  # 线段数据列名
        POINT_COLS = ['序号', 'X', 'Y', 'Z']  # 地质点数据列名

        # 执行投影计算
        projection_df = calculate_projection(
            lines_df=result_df,
            points_df=gpoints_df,
            line_cols=LINE_COLS,
            point_cols=POINT_COLS
        )

        # 新增：生成分层数据
        print("\n🔧 正在生成分层结构...")
        try:
            # 确保lines_df包含SECCODE列
            if 'SECCODE' not in result_df.columns:
                raise KeyError("导线数据中缺少SECCODE列")
    
            layered_df = generate_layered_output(projection_df, result_df)
        except Exception as e:
            print(f"分层生成失败：{str(e)}")
            exit(1)

        # 保存投影结果
        projection_file = f"{seccode}_projection_output.xlsx"
        print(f"\n💾 保存投影结果至：{projection_file}")
        projection_df.to_excel(projection_file, index=False)
        print("   ✅ 投影结果保存成功！")

        # 新增：保存分层结果
        layered_file = f"{seccode}_layered_output.xlsx"
        print(f"\n💾 保存分层结果至：{layered_file}")
        layered_df.to_excel(layered_file, index=False)
        print("   ✅ 分层结果保存成功！")

        # 最终报告
        print("\n🎉 所有处理已完成！")
        print(f"   • 清洗数据：{cleaned_file}")
        print(f"   • 剖面数据：{output_file}")
        print(f"   • 投影结果：{projection_file}")

    # -------------------- 异常处理 --------------------
    except FileNotFoundError as e:
        print("\n⚠️  文件错误！".ljust(50, ' '))
        print(f"   ❌ {str(e)}")
        print("   请确认：")
        print("   1. 文件与程序处于同一目录")
        print("   2. 文件扩展名正确(.xlsx)")
        print("   3. 文件名拼写无误")

    except PermissionError:
        print("\n⚠️  权限错误！".ljust(50, ' '))
        print("   ❌ 无法写入文件，请关闭所有Excel文件后重试")

    except KeyError as e:
        print("\n⚠️  列名错误！".ljust(50, ' '))
        print(f"   ❌ 未找到列：{str(e)}")
        print("   请检查配置：")
        print(f"   线段列名：{LINE_COLS}")
        print(f"   地质点列名：{POINT_COLS}")

    except ValueError as ve:
        print("\n⚠️  数据格式错误！".ljust(50, ' '))
        print(f"   ❌ {str(ve)}")
        print("   建议检查：")
        print("   1. 坐标列是否为数值型")
        print("   2. 数据是否存在空值")

    except Exception as e:
        print("\n⚠️  系统错误！".ljust(50, ' '))
        print(f"   ❌ 发生未知错误：{str(e)}")
        print("   建议：")
        print("   1. 检查输入数据完整性")
        print("   2. 联系技术支持")

if __name__ == "__main__":
    main()
