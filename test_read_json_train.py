import json
import os
import random
from typing import List, Dict, Tuple, Any
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from tqdm import tqdm  # Progress bar library

# --------------------------------- 全局配置 ---------------------------------
DATA_ROOT = "/home/fxf/data/nuScenes-mini"  # 改为你的数据集实际路径
JSON_PATH = os.path.join(DATA_ROOT, "v1.0-mini/image_annotations.json")
IMG_FORMAT = ".jpg"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 4
NUM_WORKERS = 0
EPOCHS = 10
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
RESIZE_SIZE = (640, 480)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 固定随机种子（保证实验可复现）
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# --------------------------------- 标注格式转换（框→图） --------------------------
def convert_box_anno_to_image_anno(box_anno_list: List[Dict]) -> List[Dict]:
    """
    将原JSON中「一个框一个字典」的标注，转换为「一张图一个字典」的标注
    转换后每个字典包含：filename + 该图所有bbox_corners + 该图所有category_names
    """
    image_anno_dict = {}
    for single_box_anno in box_anno_list:
        # 提取单框标注信息
        img_filename = single_box_anno.get("filename", "unknown")
        bbox = single_box_anno.get("bbox_corners", [0, 0, 0, 0])
        category = single_box_anno.get("category_name", "unknown")
        
        # 按图片文件名分组，整合同图的所有框和类别
        if img_filename not in image_anno_dict:
            image_anno_dict[img_filename] = {
                "filename": img_filename,
                "bbox_corners": [],
                "category_names": []
            }
        image_anno_dict[img_filename]["bbox_corners"].append(bbox)
        image_anno_dict[img_filename]["category_names"].append(category)
    
    # 字典转列表，最终返回「一张图一个字典」的列表
    image_anno_list = list(image_anno_dict.values())
    print(f"标注格式转换完成：{len(box_anno_list)}个框标注 → {len(image_anno_list)}张图片标注")
    return image_anno_list

# --------------------------------------- 文件IO函数 -------------------------------------
def read_json_file(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("Error: JSON root is not a list")
            return []
        print(f"成功读取JSON，共{len(data)}个框标注")
        return data
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return []
    except Exception as e:
        print(f"文件读取未知错误: {e}")
        return []

# ------------------------------------------- 可视化函数 -----------------------------------
def visualize_annotation(img_path: str, bboxes: List[List[float]], categories: List[str],
                         save_img: bool = False, save_path: str = "vis_gt_1.jpg") -> None:
    """单图绘制所有真实标注框，绿色框+红色类别名"""
    # 修复重复后缀
    if img_path.endswith(f"{IMG_FORMAT}{IMG_FORMAT}"):
        img_path = img_path[:-len(IMG_FORMAT)]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: 读取图片失败 {img_path}")
        return
    # 遍历绘制所有框和类别
    for bbox, cat in zip(bboxes, categories):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色真实框
        cv2.putText(img, cat, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # 防窗口卡死逻辑
    cv2.imshow("Ground Truth (Multi-Box)", img)
    cv2.waitKey(30000)
    cv2.destroyWindow("Ground Truth (Multi-Box)")
    cv2.waitKey(1)
    # 保存图片
    if save_img:
        cv2.imwrite(save_path, img)
        print(f"真实标注可视化图保存至: {save_path}")

@torch.no_grad()
def visualize_prediction(model: torch.nn.Module, img_path: str, gt_bboxes: List[List[float]], gt_categories: List[str],
                         transform: transforms.Compose, device: torch.device,
                         cat2idx: Dict, idx2cat: Dict, save_img: bool = False, save_path: str = "vis_pred.jpg") -> None:
    """单图绘制所有真实框+所有高置信度预测框，绿色GT+红色Pred"""
    # 路径修复
    if img_path.endswith(f"{IMG_FORMAT}{IMG_FORMAT}"):
        img_path = img_path[:-len(IMG_FORMAT)]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: 读取预测图片失败 {img_path}")
        return
    # 图片预处理（模型输入格式）
    pil_img = Image.open(img_path).convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    # 模型推理
    model.eval()
    outputs = model(img_tensor)
    outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
    pred = outputs[0]
    # 绘制所有真实框
    for gt_bbox, gt_cat in zip(gt_bboxes, gt_categories):
        x1, y1, x2, y2 = map(int, gt_bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"GT:{gt_cat}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # 绘制高置信度预测框（置信度>0.5，过滤低置信无效框）
    conf_thresh = 0.5
    valid_idx = pred["scores"] > conf_thresh
    pred_boxes = pred["boxes"][valid_idx]
    pred_labels = pred["labels"][valid_idx]
    pred_scores = pred["scores"][valid_idx]
    # 遍历绘制有效预测框
    if len(pred_boxes) > 0:
        for box, label_idx, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box.numpy().astype(int)
            cat_name = idx2cat.get(label_idx.item()-1, "unknown")  # 减1去除背景索引
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色预测框
            cv2.putText(img, f"{cat_name}:{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(img, "No Pred (conf<0.5)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    # 防窗口卡死
    cv2.imshow("Pred vs GT (Multi-Box)", img)
    cv2.waitKey(30000)
    cv2.destroyWindow("Pred vs GT (Multi-Box)")
    cv2.waitKey(1)
    # 保存图片
    if save_img:
        cv2.imwrite(save_path, img)
        print(f"预测对比图保存至: {save_path}")

# ----------------------------------- 自定义数据集类 -----------------------------------
class NuScenes2DDataset(Dataset):
    """
    标准目标检测数据集类：单张图片=1个样本，包含该图所有标注框和标签
    样本格式：(图像张量[C,H,W], 目标字典{boxes[N,4], labels[N,]})
    """
    def __init__(self, image_anno_list: List[Dict], data_root: str, transform: transforms.Compose = None):
        self.image_anno_list = image_anno_list  # 图式标注列表（一张图一个字典）
        self.data_root = data_root
        self.transform = transform
        # 提取所有唯一类别，构建类别-索引映射
        self.categories = self._get_unique_categories()
        self.cat2idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx2cat = {idx: cat for cat, idx in self.cat2idx.items()}
        self.num_classes = len(self.categories) + 1  # Faster R-CNN要求0为背景，类别从1开始
        print(f"检测到{len(self.categories)}个类别: {self.categories}")
        print(f"Faster R-CNN总类别数（含背景）: {self.num_classes}")

    def _get_unique_categories(self) -> List[str]:
        """从图式标注中提取所有唯一类别"""
        cat_set = set()
        for img_anno in self.image_anno_list:
            for cat in img_anno.get("category_names", ["unknown"]):
                cat_set.add(cat)
        return sorted(list(cat_set))

    def _get_img_abs_path(self, filename: str) -> str:
        """获取图片绝对路径，修复后缀缺失"""
        img_path = os.path.join(self.data_root, filename)
        if not img_path.endswith(IMG_FORMAT):
            img_path += IMG_FORMAT
        return img_path

    def __len__(self) -> int:
        """样本数=图片总数，而非框数！"""
        return len(self.image_anno_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        按索引获取单个样本：单张图 + 该图所有框(N,4) + 该图所有标签(N,)
        """
        img_anno = self.image_anno_list[idx]
        # 1. 读取图片（强制RGB3通道，避免灰度图）
        img_path = self._get_img_abs_path(img_anno["filename"])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size  # 原图宽高，用于框的缩放
        # 2. 提取该图的所有框和类别
        bboxes = np.array(img_anno["bbox_corners"], dtype=np.float32)  # (N,4)
        cat_names = img_anno["category_names"]  # (N,)
        if len(bboxes) == 0:
            raise ValueError(f"图片{img_path}无标注框，跳过！")
        # 3. 框缩放（适配图像Resize，保证框与图像对应）
        if self.transform:
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    resize_w, resize_h = t.size
                    # 所有框按相同比例缩放 (N,4) * (4,) → (N,4)
                    scale = np.array([resize_w/orig_w, resize_h/orig_h, resize_w/orig_w, resize_h/orig_h])
                    bboxes = bboxes * scale
                    break
        # 4. 转换为Torch张量（Faster R-CNN要求格式）
        boxes = torch.from_numpy(bboxes)  # (N,4) 无需unsqueeze！
        labels = [self.cat2idx[cat] + 1 for cat in cat_names]  # 类别索引+1（避开背景0）
        labels = torch.tensor(labels, dtype=torch.int64)  # (N,)
        # 5. 图像预处理
        if self.transform:
            img = self.transform(img)
        # 6. 构建标准目标字典
        target = {"boxes": boxes, "labels": labels}
        return img, target

# ----------------------------------------- 批处理函数 -------------------------------------
def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    标准Faster R-CNN批处理函数：
    图像拼接为张量[B,C,H,W]，目标保留为字典列表（每个字典对应一张图，框数可不同）
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

# ---------------------------------------- DataLoader构建 ----------------------------------
def build_dataloaders(image_anno_list: List[Dict], data_root: str, resize_size: Tuple[int, int] = RESIZE_SIZE,
                      train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO,
                      batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS) -> Tuple[DataLoader, DataLoader, DataLoader, NuScenes2DDataset]:
    # 图像预处理管道（Faster R-CNN适配，无需额外归一化）
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),  # 转[C,H,W]张量，值范围[0,1]
    ])
    # 构建完整数据集
    full_dataset = NuScenes2DDataset(image_anno_list, data_root, transform)
    # 数据集切分（按图片数切分，8:1:1）
    total_size = len(full_dataset)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例和必须为1"
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    print(f"数据集切分：训练{train_size}图 | 验证{val_size}图 | 测试{test_size}图")
    # 构建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, full_dataset

# -------------------------------------------- 模型构建 -----------------------------------
def build_fasterrcnn_model(num_classes: int) -> torch.nn.Module:
    """构建预训练Faster R-CNN，替换预测头适配自定义类别"""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = model.to(DEVICE)
    return model

# -------------------------------------------- 训练/验证函数 ---------------------------------
def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: optim.Optimizer, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Train Epoch [{epoch+1}/{EPOCHS}]")
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        # 前向传播获取损失
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # 反向传播优化
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        # 记录损失
        total_loss += losses.item()
        pbar.set_postfix(Batch_Loss=f"{losses.item():.4f}", Avg_Loss=f"{total_loss/(batch_idx+1):.4f}")
    avg_loss = total_loss / len(loader)
    pbar.close()
    print(f"训练轮次[{epoch+1}/{EPOCHS}] 平均损失: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, epoch: int) -> float:
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Val Epoch [{epoch+1}/{EPOCHS}]")
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        # 训练模式计算损失（Faster R-CNN特性）
        model.train()
        loss_dict = model(images, targets)
        model.eval()
        # 损失求和
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        pbar.set_postfix(Batch_Loss=f"{losses.item():.4f}", Avg_Loss=f"{total_loss/(batch_idx+1):.4f}")
    avg_loss = total_loss / len(loader)
    pbar.close()
    print(f"验证轮次[{epoch+1}/{EPOCHS}] 平均损失: {avg_loss:.4f}\n")
    return avg_loss

# -------------------------------------------- 主训练流程 -----------------------------------
def main_train_pipeline(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer) -> Dict[str, List[float]]:
    loss_record = {"train_loss": [], "val_loss": []}
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        val_loss = validate(model, val_loader, epoch)
        loss_record["train_loss"].append(train_loss)
        loss_record["val_loss"].append(val_loss)
    # 训练完成统计
    print("=== 训练完成 ===")
    print(f"最终训练损失: {loss_record['train_loss'][-1]:.4f}")
    print(f"最终验证损失: {loss_record['val_loss'][-1]:.4f}")
    best_val_loss = min(loss_record['val_loss'])
    best_epoch = loss_record['val_loss'].index(best_val_loss) + 1
    print(f"最佳验证损失: {best_val_loss:.4f} (第{best_epoch}轮)")
    return loss_record

# ------------------------------------------------- 主函数 ----------------------------------
if __name__ == "__main__":
    # Step1: 读取原始框式标注
    box_anno = read_json_file(JSON_PATH)
    if not box_anno:
        print("无标注数据，程序退出")
        exit(1)
    # Step2: 转换为图式标注（核心步骤，实现单图多框为1样本）
    image_anno = convert_box_anno_to_image_anno(box_anno)
    if not image_anno:
        print("标注转换失败，程序退出")
        exit(1)
    # Step3: 可视化前3张图的真实标注（多框）
    print("\n=== 可视化前3张图的真实标注 ===")
    for i in range(min(3, len(image_anno))):
        anno = image_anno[i]
        img_path = os.path.join(DATA_ROOT, anno["filename"])
        visualize_annotation(img_path, anno["bbox_corners"], anno["category_names"],
                             save_img=True, save_path=f"vis_gt_{i+1}.jpg")
    # Step4: 构建DataLoader
    print("\n=== 构建数据加载器 ===")
    train_loader, val_loader, test_loader, full_dataset = build_dataloaders(
        image_anno, DATA_ROOT, RESIZE_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, BATCH_SIZE, NUM_WORKERS
    )
    # Step5: 测试DataLoader（验证单样本多框格式）
    print("\n=== 测试数据加载器（单批次演示） ===")
    for imgs, targets in train_loader:
        print(f"批次图像形状: {imgs.shape} [B,C,H,W]")
        print(f"批次第1张图的框数: {len(targets[0]['boxes'])} | 标签数: {len(targets[0]['labels'])}")
        print(f"批次第1张图第1个框类别: {full_dataset.idx2cat[targets[0]['labels'][0].item()-1]}")
        break
    # Step6: 构建模型
    print("\n=== 构建Faster R-CNN模型 ===")
    model = build_fasterrcnn_model(full_dataset.num_classes)
    # Step7: 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # Step8: 开始训练+验证
    print("\n=== 开始训练Faster R-CNN ===")
    loss_record = main_train_pipeline(model, train_loader, val_loader, optimizer)
    # Step9: 保存模型权重（时间戳命名，避免覆盖）
    model_save_path = f"fasterrcnn_nuscenes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n模型权重保存至: {model_save_path}")
    # Step10: 测试集单样本预测可视化（多框）
    print("\n=== 测试集预测结果可视化 ===")
    test_dataset = test_loader.dataset
    sample_idx = 0
    orig_anno_idx = test_dataset.indices[sample_idx]
    orig_anno = image_anno[orig_anno_idx]
    test_img_path = os.path.join(DATA_ROOT, orig_anno["filename"])
    # 绘制预测框与真实框对比
    visualize_prediction(
        model=model, img_path=test_img_path, gt_bboxes=orig_anno["bbox_corners"], gt_categories=orig_anno["category_names"],
        transform=full_dataset.transform, device=DEVICE, cat2idx=full_dataset.cat2idx, idx2cat=full_dataset.idx2cat,
        save_img=True, save_path="prediction_vs_ground_truth.jpg"
    )
    # 程序结束
    print("\n=== 所有任务完成（单图多框为1样本训练模式） ===")
