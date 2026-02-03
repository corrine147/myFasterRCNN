import json
import os
import cv2  # 新增：处理图片和绘制标注框
import numpy as np  # 辅助cv2处理图片

def read_json_file(file_path):
    """
    读取JSON文件并返回解析后的数据
    
    参数:
        file_path (str): JSON文件的路径
    
    返回:
        list: 解析后的JSON数据列表，如果出错返回空列表
    """
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
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
    
    category_count = {}
    all_bboxes = []
    
    for idx, item in enumerate(data):
        category = item.get('category_name', '未知类别')
        bbox = item.get('bbox_corners', [])
        filename = item.get('filename', '未知文件')
        
        if category in category_count:
            category_count[category] += 1
        else:
            category_count[category] = 1
        
        all_bboxes.append({
            'index': idx,
            'category': category,
            'filename': filename,
            'bbox': bbox
        })
        
        if idx < 3:
            print(f"\n第 {idx+1} 个标注对象：")
            print(f"  类别: {category}")
            print(f"  文件: {filename}")
            print(f"  边界框: {bbox}")
            print(f"  激光雷达点数: {item.get('num_lidar_pts', 0)}")
    
    print("\n=== 数据统计 ===")
    for category, count in category_count.items():
        print(f"  {category}: {count} 个")

# ===================== 新增核心函数 =====================
def draw_annotation_on_image(annotation_data, root_path="/home/fxf/data/nuScenes-mini/"):
    """
    根据标注数据，拼接图片绝对路径，读取图片并绘制bbox标注框，最后显示图片
    参数:
        annotation_data (list): read_json_file返回的标注数据列表
        root_path (str): 图片根目录的绝对路径，默认拼接指定路径
    说明:
        1. bbox_corners为nuscenes导出格式，是[左上x, 左上y, 右下x, 右下y]的四坐标
        2. 按**图片维度**去重绘制（一个图片的所有标注框一次性绘制）
        3. 按q键关闭当前图片，继续显示下一张；按其他键直接关闭所有窗口
    """
    if not annotation_data:
        print("无标注数据，无法绘制图片")
        return
    
    # 步骤1：按filename分组（一个图片对应多个标注框，避免重复读取）
    image_annotations = {}
    for item in annotation_data:
        filename = item.get('filename', '未知文件')
        if filename not in image_annotations:
            image_annotations[filename] = []
        image_annotations[filename].append({
            'category': item.get('category_name', '未知类别'),
            'bbox': item.get('bbox_corners', [0, 0, 0, 0])  # 四坐标，默认空框
        })
    
    print(f"\n=== 开始绘制标注图片 ===")
    print(f"共检测到 {len(image_annotations)} 张待标注图片，按q键切换下一张，其他键退出")
    
    # 步骤2：遍历每个图片，拼接路径、读取、绘制标注框
    for filename, annos in image_annotations.items():
        # 拼接图片绝对路径（root_path + 原filename，自动处理路径分隔符）
        img_abs_path = os.path.join(root_path, filename)
        # 检查图片是否存在
        if not os.path.exists(img_abs_path):
            print(f"警告：图片 {img_abs_path} 不存在，跳过绘制")
            continue
        
        # 读取图片（cv2默认BGR格式，不影响绘制，显示正常）
        img = cv2.imread(img_abs_path)
        if img is None:
            print(f"警告：图片 {img_abs_path} 读取失败，跳过绘制")
            continue
        
        # 遍历当前图片的所有标注框，逐个绘制
        for anno in annos:
            category = anno['category']
            bbox = anno['bbox']
            # 提取bbox四坐标（确保是数值类型，防止绘制报错）
            x1, y1, x2, y2 = map(int, bbox)  # nuscenes的bbox_corners是[左上x,左上y,右下x,右下y]
            # 绘制矩形框：参数（图片，左上点，右下点，颜色(BGR)，线宽）
            # 颜色选红色(0,0,255)，线宽2，醒目且不遮挡内容
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 在框的左上角添加类别文字标注：参数（图片，文字，坐标，字体，字号，颜色，线宽）
            cv2.putText(
                img, category, (x1, y1 - 10),  # 文字在框上方10个像素，避免遮挡
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,  # 字体+字号
                (0, 0, 255), 2  # 红色文字，线宽2
            )
        
        # 调整窗口大小（避免图片过大/过小，按原图比例缩放至800x600左右）
        cv2.namedWindow(f"标注图片: {filename}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"标注图片: {filename}", 800, 600)
        # 显示图片
        cv2.imshow(f"标注图片: {filename}", img)
        
        # 等待按键：0表示无限等待，按q键继续下一张，其他键退出
        key = cv2.waitKey(0) & 0xFF
        if key != ord('q'):  # 非q键，直接退出所有图片显示
            print("用户手动退出图片显示")
            break
        # 关闭当前窗口，准备显示下一张
        cv2.destroyWindow(f"标注图片: {filename}")
    
    # 最后释放所有cv2窗口资源，防止内存泄漏
    cv2.destroyAllWindows()
    print("=== 图片标注绘制完成 ===")

# 主程序
if __name__ == "__main__":
    # 替换为你的JSON文件路径
    json_file_path = "/home/fxf/data/nuScenes-mini/v1.0-mini/image_annotations.json"
    # 图片根目录（可修改，默认与代码中拼接的一致）
    img_root_path = "/home/fxf/data/nuScenes-mini/"
    
    # 1. 读取JSON文件
    annotation_data = read_json_file(json_file_path)
    
    # 2. 分析数据
    if annotation_data:
        analyze_annotation_data(annotation_data)
        # 3. 新增：绘制图片标注框并显示
        draw_annotation_on_image(annotation_data, img_root_path)