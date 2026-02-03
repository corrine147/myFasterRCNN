import json
import os

def read_json_file(file_path):
    """
    读取JSON文件并返回解析后的数据
    
    参数:
        file_path (str): JSON文件的路径
    
    返回:
        list: 解析后的JSON数据列表，如果出错返回空列表
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return []
    
    try:
        # 打开并读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            # 加载JSON数据
            data = json.load(f)
            
        # 验证数据格式是否为列表
        if not isinstance(data, list):
            print("错误：JSON文件的根数据类型不是列表")
            return []
        
        print(f"成功读取JSON文件，共包含 {len(data)} 个标注对象")
        return data
    
    except json.JSONDecodeError as e:
        print(f"JSON解析错误：{e}")
        print("提示：请检查JSON文件的语法是否完整（比如末尾是否有未闭合的括号/引号）")
        return []
    except Exception as e:
        print(f"读取文件时发生未知错误：{e}")
        return []

def analyze_annotation_data(data):
    """
    分析标注数据，提取关键信息
    """
    if not data:
        return
    
    # 统计不同类别的数量
    category_count = {}
    # 存储所有边界框信息
    all_bboxes = []
    
    for idx, item in enumerate(data):
        # 获取类别名称
        category = item.get('category_name', '未知类别')
        # 获取边界框信息
        bbox = item.get('bbox_corners', [])
        # 获取文件名
        filename = item.get('filename', '未知文件')
        
        # 统计类别
        if category in category_count:
            category_count[category] += 1
        else:
            category_count[category] = 1
        
        # 存储边界框信息（带索引和类别）
        all_bboxes.append({
            'index': idx,
            'category': category,
            'filename': filename,
            'bbox': bbox
        })
        
        # 打印单个对象的关键信息示例（只打印前3个）
        if idx < 3:
            print(f"\n第 {idx+1} 个标注对象：")
            print(f"  类别: {category}")
            print(f"  文件: {filename}")
            print(f"  边界框: {bbox}")
            print(f"  激光雷达点数: {item.get('num_lidar_pts', 0)}")
    
    # 打印统计信息
    print("\n=== 数据统计 ===")
    for category, count in category_count.items():
        print(f"  {category}: {count} 个")

# 主程序
if __name__ == "__main__":
    # 替换为你的JSON文件路径
    json_file_path = "/home/fxf/data/nuScenes-mini/v1.0-mini/image_annotations.json"
    
    # 读取JSON文件
    annotation_data = read_json_file(json_file_path)
    
    # 分析数据
    if annotation_data:
        analyze_annotation_data(annotation_data)