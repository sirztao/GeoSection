```markdown
# Projection Calculator

用于计算点集到线集的最短垂足投影的 Python 工具。

## 功能特性
- 支持三维空间坐标计算
- 自定义输入数据列名
- 输出包含原始点、投影点、所属线段等信息
- 兼容 Pandas DataFrame 输入/输出

## 安装依赖
```bash
pip install pandas numpy
```

## 快速开始
### 数据准备
1. **线集数据 (`lines_df`)**：需包含以下默认列（可自定义）：
   - `FROM_X`, `FROM_Y`, `FROM_Z` : 线段起点坐标
   - `TO_X`, `TO_Y`, `TO_Z`     : 线段终点坐标

2. **点集数据 (`points_df`)**：需包含以下默认列（可自定义）：
   - `序号` : 点唯一标识
   - `X`, `Y`, `Z` : 点坐标

### 基础用法
```python
import pandas as pd
from projection_calculator import calculate_projection

# 加载数据
lines_df = pd.read_csv('lines.csv')
points_df = pd.read_csv('points.csv')

# 计算投影
projection_df = calculate_projection(lines_df, points_df)

# 保存结果
projection_df.to_csv('output.csv', index=False)
```

### 自定义列名
```python
projection_df = calculate_projection(
    lines_df,
    points_df,
    line_cols=['start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z'],
    point_cols=['id', 'coord_x', 'coord_y', 'coord_z']
)
```

## 输出说明
结果 DataFrame 包含以下列：
| 列名               | 描述                          |
|--------------------|-----------------------------|
| `原始点_序号`       | 原始点标识符                  |
| `原始点_X/Y/Z`      | 原始点坐标                    |
| `投影点_X/Y/Z`      | 投影点坐标                    |
| `所属线段_起点`     | 投影所在线段的起点坐标 (X,Y,Z) |
| `所属线段_终点`     | 投影所在线段的终点坐标 (X,Y,Z) |
| `投影距离`          | 原始点到投影点的欧氏距离       |

## 高级配置
### 参数说明
```python
def calculate_projection(
    lines_df: pd.DataFrame,
    points_df: pd.DataFrame,
    line_cols: list = ['FROM_X','FROM_Y','FROM_Z','TO_X','TO_Y','TO_Z'],
    point_cols: list = ['序号','X','Y','Z']
) -> pd.DataFrame:
    """
    :param lines_df: 线段数据 DataFrame
    :param points_df: 点集数据 DataFrame
    :param line_cols: 线段数据列名 [起点X, 起点Y, 起点Z, 终点X, 终点Y, 终点Z]
    :param point_cols: 点集数据列名 [ID列, X坐标, Y坐标, Z坐标]
    :return: 包含投影信息的 DataFrame
    """
```

## 注意事项
1. 确保输入数据不包含空值
2. Z 坐标可为零值（适用于二维场景）
3. 默认列名可通过参数自定义
4. 建议使用 Python 3.8+ 运行

## 示例数据
见 `examples/` 目录：
- `lines_sample.csv` : 线段数据示例
- `points_sample.csv` : 点集数据示例
- `output_sample.csv` : 输出结果示例

## 许可证
MIT License
```
