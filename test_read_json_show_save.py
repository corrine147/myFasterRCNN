import json
import os
import cv2
import numpy as np

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
    分析标注数据，提取关键信息并返回类别统计、带索引标注信息
    返回:
        tuple: (category_count, all_bboxes)
            category_count (dict): 类别-数量统计
            all_bboxes (list): 带索引的详细标注信息
    """
    if not data:
        return {}, []
    
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
    return category_count, all_bboxes

def select_filter_labels(category_count):
    """
    标签筛选交互函数：选择需要保留的标注类别，多个用逗号分隔
    参数:
        category_count (dict): 数据中的类别-数量统计，用于校验标签是否存在
    返回:
        list: 选中的需要保留的label列表，空列表表示保留所有类别
    """
    if not category_count:
        print("无类别数据，跳过标签筛选")
        return []
    
    print("\n=== 标签筛选 ===")
    print(f"当前数据包含的所有类别：{', '.join(category_count.keys())}")
    print("筛选规则：")
    print("  1. 输入需要保留的类别（多个用英文逗号分隔，如car,truck）：仅绘制这些类别的标注框")
    print("  2. 直接回车/输入all/ALL：保留所有类别，不筛选")
    print("  3. 输入q/Q：退出程序")
    
    while True:
        user_input = input("\n请输入需要保留的类别：").strip()
        # 退出程序
        if user_input in ['q', 'Q']:
            print("用户退出程序")
            exit()
        # 保留所有类别
        if not user_input or user_input in ['all', 'ALL']:
            print("选择保留所有类别，不进行标签筛选")
            return []
        # 按逗号分割筛选标签，去除空值和空格
        filter_labels = [label.strip() for label in user_input.split(',') if label.strip()]
        if not filter_labels:
            print("输入为空，请重新输入（或直接回车保留所有）")
            continue
        # 校验标签是否存在
        valid_labels = [label for label in filter_labels if label in category_count]
        invalid_labels = [label for label in filter_labels if label not in category_count]
        # 提示无效标签
        if invalid_labels:
            print(f"警告：以下类别不存在于数据中，已自动过滤：{', '.join(invalid_labels)}")
        # 有效标签非空
        if valid_labels:
            print(f"成功选择保留类别：{', '.join(valid_labels)}，仅绘制这些类别的标注框")
            return valid_labels
        # 所有标签都无效，重新选择
        else:
            print("无有效类别，请重新输入（可直接回车保留所有）")

def select_single_sample(annotation_data, all_bboxes):
    """
    样本选择交互函数：支持按索引/文件名选择单个样本，或批量处理所有图片
    参数:
        annotation_data (list): 原始标注数据
        all_bboxes (list): 带索引的标注信息
    返回:
        dict: 选中的单个样本数据，None表示批量处理所有图片
    """
    if not annotation_data:
        return None
    
    print("\n=== 样本选择 ===")
    total_imgs = len(set([x['filename'] for x in all_bboxes]))
    print(f"当前共有 {len(annotation_data)} 个标注对象，{total_imgs} 张图片")
    print("选择方式：")
    print("  1. 输入数字（如5）：查看并保存对应索引的单个标注样本（索引从0开始）")
    print("  2. 输入文件名（如samples/CAM_FRONT/xxx.jpg）：查看并保存该图片的所有标注")
    print("  3. 输入b/B/直接回车：批量处理所有图片（自动保存所有标注图，无显示交互）")
    print("  4. 输入q/Q：退出程序")
    
    while True:
        user_input = input("\n请输入你的选择：").strip()
        # 退出程序
        if user_input in ['q', 'Q']:
            print("用户退出程序")
            exit()
        # 批量处理所有图片
        if not user_input or user_input in ['b', 'B']:
            print("选择批量处理所有图片，将自动绘制并保存所有标注图")
            return None
        # 按索引选择单个样本
        if user_input.isdigit():
            idx = int(user_input)
            if 0 <= idx < len(annotation_data):
                single_sample = annotation_data[idx]
                print(f"成功选择第 {idx} 个样本，文件：{single_sample.get('filename', '未知文件')}，类别：{single_sample.get('category_name', '未知类别')}")
                return {'type': 'index', 'data': single_sample}
            else:
                print(f"索引超出范围！请输入0到 {len(annotation_data)-1} 之间的数字")
                continue
        # 按文件名选择单个图片
        else:
            file_matched = [item for item in annotation_data if item.get('filename', '') == user_input]
            if file_matched:
                print(f"成功选择文件 {user_input}，该图片共有 {len(file_matched)} 个标注对象")
                return {'type': 'file', 'data': file_matched}
            else:
                print(f"文件名 {user_input} 不存在，请检查输入是否与JSON中filename完全一致")
                continue

def draw_annotation_on_image(annotation_data, root_path="/home/fxf/data/nuScenes-mini/", 
                             save_dir="nuscenes_annotated_imgs", single_sample=None,
                             filter_labels=[]):
    """
    核心绘制函数：支持标签筛选、单样本/批量处理，自动保存所有绘制后的图片
    参数:
        annotation_data (list): 原始标注数据
        root_path (str): 图片根目录绝对路径
        save_dir (str): 标注图保存目录（自动创建）
        single_sample (dict): 单样本数据，None则批量处理
        filter_labels (list): 需要保留的标签列表，空列表表示保留所有
    说明:
        1. 批量模式：自动保存所有图片，无显示、无交互，快速处理大批量数据
        2. 单样本模式：显示图片+按键交互（q键关闭），同时自动保存
        3. 标签筛选：仅绘制filter_labels中的类别，无关类别标注框忽略
    """
    if not annotation_data:
        print("无标注数据，无法绘制图片")
        return
    
    # 自动创建保存目录（支持相对/绝对路径）
    os.makedirs(save_dir, exist_ok=True)
    save_dir_abs = os.path.abspath(save_dir)
    print(f"\n=== 开始处理标注图片 ===")
    print(f"标注图将统一保存至：{save_dir_abs}")
    
    # 步骤1：按文件名分组标注数据，同时应用标签筛选
    image_annotations = {}
    # 处理单样本模式
    if single_sample is not None:
        data_source = [single_sample['data']] if single_sample['type'] == 'index' else single_sample['data']
        for item in data_source:
            filename = item.get('filename', '未知文件')
            category = item.get('category_name', '未知类别')
            # 标签筛选：不在保留列表则跳过
            if filter_labels and category not in filter_labels:
                continue
            bbox = item.get('bbox_corners', [0,0,0,0])
            if filename not in image_annotations:
                image_annotations[filename] = []
            image_annotations[filename].append({'category': category, 'bbox': bbox})
    # 处理批量模式
    else:
        for item in annotation_data:
            filename = item.get('filename', '未知文件')
            category = item.get('category_name', '未知类别')
            # 标签筛选：不在保留列表则跳过
            if filter_labels and category not in filter_labels:
                continue
            bbox = item.get('bbox_corners', [0,0,0,0])
            if filename not in image_annotations:
                image_annotations[filename] = []
            image_annotations[filename].append({'category': category, 'bbox': bbox})
    
    # 无符合条件的标注图片
    if not image_annotations:
        print("警告：无符合标签筛选条件的标注图片，处理结束")
        return
    print(f"共检测到 {len(image_annotations)} 张符合条件的图片，开始绘制并保存...")
    
    # 步骤2：遍历绘制并保存所有图片
    processed_count = 0
    for filename, annos in image_annotations.items():
        # 拼接图片绝对路径
        img_abs_path = os.path.join(root_path, filename)
        if not os.path.exists(img_abs_path):
            print(f"[跳过] 图片不存在：{img_abs_path}")
            continue
        # 读取图片（cv2.imread返回None表示读取失败）
        img = cv2.imread(img_abs_path)
        if img is None:
            print(f"[跳过] 图片读取失败：{img_abs_path}")
            continue
        
        # 绘制筛选后的标注框和类别文字
        for anno in annos:
            category = anno['category']
            bbox = anno['bbox']
            # 确保bbox是4个数值，避免绘制报错
            if len(bbox) != 4:
                print(f"[警告] {filename} 中类别{category}的bbox格式错误，跳过绘制：{bbox}")
                continue
            x1, y1, x2, y2 = map(int, bbox)
            # 绘制红色矩形框（BGR格式，线宽2）
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 绘制类别文字（框上方10像素，防止超出图片边界）
            text_pos = (x1, max(y1 - 10, 10))
            cv2.putText(img, category, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        
        # 保存标注后的图片（取原图片名，自动覆盖同名文件）
        save_filename = os.path.basename(filename)
        save_abs_path = os.path.join(save_dir, save_filename)
        cv2.imwrite(save_abs_path, img)
        processed_count += 1
        print(f"[保存成功] {processed_count}/{len(image_annotations)} | {save_abs_path}")
        
        # 单样本模式：显示图片并保留按键交互
        if single_sample is not None:
            cv2.namedWindow(f"标注预览: {filename}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"标注预览: {filename}", 800, 600)
            cv2.imshow(f"标注预览: {filename}", img)
            print(f"\n单样本预览：按任意键关闭图片窗口")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # 批量模式：处理完成后提示统计信息
    if not single_sample:
        print(f"\n=== 批量处理完成 ===")
        print(f"总检测符合条件图片：{len(image_annotations)} 张")
        print(f"成功绘制并保存图片：{processed_count} 张")
        print(f"标注图保存目录：{save_dir_abs}")
    # 单样本模式：提示保存完成
    else:
        print(f"\n=== 单样本处理完成 ===")
        print(f"标注图已保存至：{save_dir_abs}")

# 主程序入口
if __name__ == "__main__":
    # ===================== 路径配置（根据你的实际环境修改！）=====================
    JSON_FILE_PATH = "/home/fxf/data/nuScenes-mini/v1.0-mini/image_annotations.json"  # JSON标注文件路径
    IMG_ROOT_PATH = "/home/fxf/data/nuScenes-mini/"  # 图片根目录绝对路径
    SAVE_DIR = "nuscenes_filtered_annotations"  # 标注图保存目录（可改绝对路径）
    # ==========================================================================

    # 1. 读取JSON标注数据
    annotation_data = read_json_file(JSON_FILE_PATH)
    if not annotation_data:
        print("无标注数据，程序退出")
        exit()
    
    # 2. 分析数据，获取类别统计和带索引标注信息
    category_count, all_bboxes = analyze_annotation_data(annotation_data)
    
    # 3. 标签筛选：选择需要保留的标注类别
    filter_labels = select_filter_labels(category_count)
    
    # 4. 样本选择：单样本查看/批量处理所有图片
    single_sample = select_single_sample(annotation_data, all_bboxes)
    
    # 5. 核心处理：绘制标注框、自动保存图片（单样本含预览，批量无交互）
    draw_annotation_on_image(
        annotation_data=annotation_data,
        root_path=IMG_ROOT_PATH,
        save_dir=SAVE_DIR,
        single_sample=single_sample,
        filter_labels=filter_labels
    )