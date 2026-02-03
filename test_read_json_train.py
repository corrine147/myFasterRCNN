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

# -------------------------- Global Configuration --------------------------
# Root path of nuScenes mini dataset (modify to your actual path)
DATA_ROOT = "/home/fxf/data/nuScenes-mini"
# Path of the exported 2D annotation JSON file
JSON_PATH = os.path.join(DATA_ROOT, "v1.0-mini/image_annotations.json")
# Image format (consistent with nuScenes dataset)
IMG_FORMAT = ".jpg"
# Random seed for reproducibility
SEED = 42
# Train/Val/Test split ratio (e.g. 8:1:1)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
# Training hyperparameters
BATCH_SIZE = 4
NUM_WORKERS = 0
EPOCHS = 10  # Total training epochs
LR = 0.005   # Learning rate
MOMENTUM = 0.9  # SGD momentum
WEIGHT_DECAY = 0.0005  # Weight decay for regularization
# Image resize size (for neural network input)
RESIZE_SIZE = (640, 480)
# Device: use GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Set random seed for all libraries (reproducibility)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# -------------------------- File IO Functions --------------------------
def read_json_file(file_path: str) -> List[Dict]:
    """
    Read JSON file and return parsed data list
    Args:
        file_path (str): Path of the JSON file
    Returns:
        List[Dict]: Parsed JSON data list, empty list if error occurs
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("Error: Root data type of JSON file is not list")
            return []
        
        print(f"Successfully read JSON file, total {len(data)} annotation objects")
        return data
    
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print("Hint: Check JSON syntax (e.g. unclosed brackets/quotes at the end)")
        return []
    except Exception as e:
        print(f"Unknown Error when reading file: {e}")
        return []

# -------------------------- Visualization Function (Fixed Window Stuck) --------------------------
def visualize_annotation(img_path: str, bbox: List[float], category: str, save_img: bool = False, save_path: str = "vis_annotation.jpg") -> None:
    """
    Visualize annotation bbox on the corresponding image
    Fixed: Window destroy logic to avoid program stuck, add timeout mechanism
    Args:
        img_path (str): Absolute path of the image
        bbox (List[float]): Bbox coordinates from bbox_corners [x1, y1, x2, y2]
        category (str): Category name of the annotation
        save_img (bool): Whether to save the visualized image
        save_path (str): Path to save the visualized image
    """
    # Fix duplicate suffix issue (xxx.jpg.jpg -> xxx.jpg)
    if img_path.endswith(f"{IMG_FORMAT}{IMG_FORMAT}"):
        img_path = img_path[:-len(IMG_FORMAT)]
    # Read image with OpenCV (BGR format)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Failed to read image {img_path}")
        return
    
    # Convert bbox coordinates to integer (cv2 only supports integer coordinates)
    x1, y1, x2, y2 = map(int, bbox)
    # Draw rectangle (green, line width 2) - Ground Truth
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Add category text (red, font size 0.8, line width 2)
    cv2.putText(img, category, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Show image (optimized window logic to avoid stuck)
    cv2.imshow("Ground Truth Annotation", img)
    cv2.waitKey(30000)  # Wait for key press (30s timeout, auto exit)
    cv2.destroyWindow("Ground Truth Annotation")  # Destroy specified window
    cv2.waitKey(1)  # Clear key buffer to avoid subsequent block
    
    # Save image if needed
    if save_img:
        cv2.imwrite(save_path, img)
        print(f"Visualized image saved to {save_path}")

# -------------------------- Prediction Visualization (Fixed Stuck + No Prediction Error) --------------------------
@torch.no_grad()
def visualize_prediction(model: torch.nn.Module, img_path: str, gt_bbox: List[float], gt_category: str,
                         transform: transforms.Compose, device: torch.device,
                         cat2idx: Dict, idx2cat: Dict, save_img: bool = False, save_path: str = "vis_prediction.jpg") -> None:
    """
    Visualize model prediction bbox and ground truth bbox on the original image
    Fixed: Window stuck + no prediction box index error
    Args:
        model: Trained Faster R-CNN model
        img_path: Absolute path of the input image
        gt_bbox: Ground truth bbox from annotation
        gt_category: Ground truth category name
        transform: Image transform pipeline
        device: Running device (cuda/cpu)
        cat2idx: Category name to index mapping
        idx2cat: Index to category name mapping
        save_img: Whether to save the result image
        save_path: Path to save the visualization result
    """
    # Fix duplicate image suffix issue
    if img_path.endswith(f"{IMG_FORMAT}{IMG_FORMAT}"):
        img_path = img_path[:-len(IMG_FORMAT)]
    
    # Read original image for drawing (BGR)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Failed to read image {img_path} for prediction visualization")
        return
    
    # Preprocess image for model input (RGB -> tensor)
    pil_img = Image.open(img_path).convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # Model inference (eval mode)
    model.eval()
    outputs = model(img_tensor)
    outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
    pred = outputs[0]
    
    # Get top-1 prediction (highest score) - avoid no prediction error
    if len(pred["boxes"]) > 0 and len(pred["scores"]) > 0:
        max_score_idx = torch.argmax(pred["scores"])
        pred_box = pred["boxes"][max_score_idx].cpu().numpy().astype(int)
        pred_label_idx = pred["labels"][max_score_idx].cpu().item() - 1  # Minus background index (0)
        pred_category = idx2cat.get(pred_label_idx, "unknown")
        pred_score = pred["scores"][max_score_idx].cpu().item()
        
        # Draw prediction bbox (RED) and label with score
        cv2.rectangle(img, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 0, 255), 2)
        pred_text = f"{pred_category}: {pred_score:.2f}"
        cv2.putText(img, pred_text, (pred_box[0], pred_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        # Add hint text when model has no prediction
        cv2.putText(img, "Model: No Prediction", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    
    # Draw ground truth bbox (GREEN) and label
    gt_x1, gt_y1, gt_x2, gt_y2 = map(int, gt_bbox)
    cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 2)
    cv2.putText(img, f"GT: {gt_category}", (gt_x1, gt_y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Show and save (optimized window logic)
    cv2.imshow("Prediction vs Ground Truth", img)
    cv2.waitKey(30000)  # 30s timeout auto exit
    cv2.destroyWindow("Prediction vs Ground Truth")  # Destroy specified window
    cv2.waitKey(1)  # Clear key buffer
    
    if save_img:
        cv2.imwrite(save_path, img)
        print(f"Prediction visualization saved to {save_path}")

# -------------------------- Custom Dataset Class --------------------------
class NuScenes2DDataset(Dataset):
    """
    Custom Dataset for nuScenes 2D annotation data
    Compatible with PyTorch DataLoader, adapted for Faster R-CNN target format
    """
    def __init__(self, annotation_data: List[Dict], data_root: str, transform: transforms.Compose = None):
        """
        Initialize the dataset
        Args:
            annotation_data (List[Dict]): Parsed JSON annotation data list
            data_root (str): Root path of the nuScenes dataset
            transform (transforms.Compose): Image transform pipeline
        """
        self.annotation_data = annotation_data
        self.data_root = data_root
        self.transform = transform
        # Get unique categories and build mapping
        self.categories = self._get_unique_categories()
        self.cat2idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx2cat = {idx: cat for cat, idx in self.cat2idx.items()}
        # Faster R-CNN requires class index start from 1 (0 for background)
        self.num_classes = len(self.categories) + 1
        print(f"Found {len(self.categories)} unique categories: {self.categories}")
        print(f"Faster R-CNN num_classes (include background): {self.num_classes}")

    def _get_unique_categories(self) -> List[str]:
        """Get sorted unique category names from annotation data"""
        categories = set()
        for item in self.annotation_data:
            cat = item.get('category_name', 'unknown')
            categories.add(cat)
        return sorted(list(categories))

    def _get_img_abs_path(self, filename: str) -> str:
        """Get absolute image path, fix duplicate suffix issue"""
        img_path = os.path.join(self.data_root, filename)
        if img_path.endswith(IMG_FORMAT) and filename.endswith(IMG_FORMAT):
            return img_path
        return img_path + IMG_FORMAT

    def __len__(self) -> int:
        """Return total number of annotation samples"""
        return len(self.annotation_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get single sample for Faster R-CNN
        Target format: dict{'boxes': (1,4), 'labels': (1,)}
        """
        item = self.annotation_data[idx]
        # Read image (ensure 3 RGB channels)
        img_path = self._get_img_abs_path(item.get('filename', 'unknown'))
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size

        # Process bbox (rescale to resize size if transform exists)
        bbox = np.array(item.get('bbox_corners', [0,0,0,0]), dtype=np.float32)
        if self.transform and hasattr(self.transform, 'transforms'):
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    resize_w, resize_h = t.size
                    bbox = bbox * np.array([resize_w/img_w, resize_h/img_h, resize_w/img_w, resize_h/img_h])
                    break
        boxes = torch.from_numpy(bbox).unsqueeze(0)  # Shape: (1,4)

        # Process label (Faster R-CNN: class index start from 1)
        cat_name = item.get('category_name', 'unknown')
        cat_idx = self.cat2idx.get(cat_name, 0) + 1  # +1 for background (0)
        labels = torch.tensor([cat_idx], dtype=torch.int64)  # Shape: (1,)

        # Apply image transform pipeline
        if self.transform:
            img = self.transform(img)

        # Target format for Faster R-CNN (batch will be list of dicts)
        target = {'boxes': boxes, 'labels': labels}
        return img, target

# -------------------------- Collate Function for Faster R-CNN --------------------------
def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """Custom collate function for Faster R-CNN (support variable target format)"""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

# -------------------------- DataLoader Build Function --------------------------
def build_dataloaders(annotation_data: List[Dict], data_root: str, resize_size: Tuple[int, int] = RESIZE_SIZE,
                      train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO,
                      batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS) -> Tuple[DataLoader, DataLoader, DataLoader, NuScenes2DDataset]:
    """Build train/val/test DataLoader with custom collate_fn for Faster R-CNN"""
    # Image transform pipeline (Faster R-CNN compatible, no extra norm for raw training)
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),  # Convert to (C, H, W) tensor, value range [0,1]
    ])
    
    # Create full dataset
    full_dataset = NuScenes2DDataset(annotation_data, data_root, transform)
    # Split dataset (ensure ratio sum is 1.0)
    total_size = len(full_dataset)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Train/Val/Test ratio sum must be 1.0"
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    print(f"Dataset split: Train({train_size}) | Val({val_size}) | Test({test_size})")
    
    # Build DataLoader (shuffle train set, no shuffle for val/test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader, full_dataset

# -------------------------- Build Faster R-CNN Model --------------------------
def build_fasterrcnn_model(num_classes: int) -> torch.nn.Module:
    """
    Build pre-trained Faster R-CNN ResNet50 FPN model
    Replace the default predictor with custom one for nuScenes dataset
    Args:
        num_classes (int): Number of classes (include background)
    Returns:
        torch.nn.Module: Custom Faster R-CNN model
    """
    # Load pre-trained Faster R-CNN model from torchvision
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # Get input features of the final box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the default predictor with custom one (adapt to our num_classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Move model to target device (GPU/CPU)
    model = model.to(DEVICE)
    return model

# -------------------------- Training & Validation Function --------------------------
def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: optim.Optimizer, epoch: int) -> float:
    """
    Train the model for one epoch with tqdm progress bar
    Args:
        model (torch.nn.Module): Faster R-CNN model
        loader (DataLoader): Train DataLoader
        optimizer (optim.Optimizer): SGD optimizer for training
        epoch (int): Current epoch number
    Returns:
        float: Average training loss of the epoch
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    # Create tqdm progress bar (show epoch info and batch progress)
    progress_bar = tqdm(loader, desc=f"Train Epoch [{epoch+1}/{EPOCHS}]", unit="batch")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # Move image and target data to device (GPU/CPU)
        images = images.to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        # Forward pass: get loss dictionary (classification + regression loss)
        loss_dict = model(images, targets)
        # Sum all losses for backpropagation
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass and optimize
        optimizer.zero_grad()  # Clear previous gradients
        losses.backward()      # Compute gradients via backpropagation
        optimizer.step()       # Update model weights
        
        # Accumulate total loss and update progress bar
        total_loss += losses.item()
        progress_bar.set_postfix({
            "Batch Loss": f"{losses.item():.4f}",
            "Avg Loss": f"{total_loss/(batch_idx+1):.4f}"
        })
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(loader)
    progress_bar.close()
    print(f"Train Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()  # Disable gradient computation to save memory and speed up
def validate(model: torch.nn.Module, loader: DataLoader, epoch: int) -> float:
    """
    Validate the model for one epoch with tqdm progress bar
       Fixed: Universal loss calculation for all torchvision versions (no model.model attribute)
    Args:
        model (torch.nn.Module): Faster R-CNN model
        loader (DataLoader): Val DataLoader
        epoch (int): Current epoch number
    Returns:
        float: Average validation loss of the epoch
    """
    total_loss = 0.0
    # Create tqdm progress bar for validation
    progress_bar = tqdm(loader, desc=f"Val Epoch [{epoch+1}/{EPOCHS}]", unit="batch")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # Move data to target device
        images = images.to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        # train!�eval	@torch.no_grad()	
        model.train()
        loss_dict = model(images, targets)
        model.eval()  # eval
        
        # Sum losses
        losses = sum(loss for loss in loss_dict.values())
        # Accumulate total loss
        total_loss += losses.item()
        # Update progress bar with current loss
        progress_bar.set_postfix({
            "Batch Loss": f"{losses.item():.4f}",
            "Avg Loss": f"{total_loss/(batch_idx+1):.4f}"
        })
    
    # Calculate average validation loss
    avg_loss = total_loss / len(loader)
    progress_bar.close()
    print(f"Val Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {avg_loss:.4f}\n")
    return avg_loss

# -------------------------- Main Training Pipeline --------------------------
def main_train_pipeline(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer) -> Dict[str, List[float]]:
    """
    Main training & validation loop, record loss history for each epoch
    Args:
        model (torch.nn.Module): Faster R-CNN model
        train_loader (DataLoader): Train DataLoader
        val_loader (DataLoader): Val DataLoader
        optimizer (optim.Optimizer): SGD optimizer
    Returns:
        Dict[str, List[float]]: Recorded train/validation loss history
    """
    # Initialize loss record dictionary
    loss_record = {
        'train_loss': [],
        'val_loss': []
    }
    # Training loop over all epochs
    for epoch in range(EPOCHS):
        # Train one epoch and get average train loss
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        # Validate one epoch and get average val loss
        val_loss = validate(model, val_loader, epoch)
        # Record loss for current epoch
        loss_record['train_loss'].append(train_loss)
        loss_record['val_loss'].append(val_loss)
    
    # Print final training results
    print("=== Training Complete ===")
    print(f"Final Train Loss: {loss_record['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {loss_record['val_loss'][-1]:.4f}")
    print(f"Best Val Loss: {min(loss_record['val_loss']):.4f} (Epoch: {loss_record['val_loss'].index(min(loss_record['val_loss']))+1})")
    return loss_record

# -------------------------- Main Function --------------------------
if __name__ == "__main__":
    # Step 1: Read JSON annotation file
    annotation_data = read_json_file(JSON_PATH)
    if not annotation_data:
        print("Error: No annotation data found, exit program")
        exit(1)
    
    # Step 2: Visualize first 3 ground truth annotations (fixed path issue)
    print("\n=== Start Visualizing Ground Truth Annotations (First 3) ===")
    for idx in range(min(3, len(annotation_data))):
        item = annotation_data[idx]
        img_filename = item.get('filename', 'unknown')
        img_abs_path = os.path.join(DATA_ROOT, img_filename)
        bbox = item.get('bbox_corners', [0,0,0,0])
        category = item.get('category_name', 'unknown')
        print(f"Visualizing {idx+1}-th annotation: {category} | {img_abs_path}")
        visualize_annotation(img_abs_path, bbox, category, save_img=True, save_path=f"vis_gt_{idx+1}.jpg")
    
    # Step 3: Build train/val/test DataLoader with Faster R-CNN collate_fn
    print("\n=== Building DataLoader ===")
    train_loader, val_loader, test_loader, full_dataset = build_dataloaders(
        annotation_data, DATA_ROOT, RESIZE_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, BATCH_SIZE, NUM_WORKERS
    )
    
    # Step 4: Test DataLoader (iterate one batch for demo)
    print("\n=== Testing DataLoader (One Batch Demo) ===")
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx+1} - Image Shape: {imgs.shape}")  # (B, C, H, W)
        print(f"Batch {batch_idx+1} - Target Boxes Shape: {targets[0]['boxes'].shape}")  # (1,4) per sample
        print(f"Batch {batch_idx+1} - Target Labels Shape: {targets[0]['labels'].shape}")  # (1,) per sample
        # Convert label index back to category name
        cat_idx = targets[0]['labels'].item() - 1  # Minus background index
        cat_name = full_dataset.idx2cat.get(cat_idx, 'unknown')
        print(f"Batch {batch_idx+1} - Sample 1 Category: {cat_name}")
        break  # Only iterate one batch for demo
    
    # Step 5: Build custom Faster R-CNN model for nuScenes dataset
    print("\n=== Building Faster R-CNN Model ===")
    model = build_fasterrcnn_model(full_dataset.num_classes)
    
    # Step 6: Define SGD optimizer (standard for Faster R-CNN training)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    # Step 7: Start full training & validation pipeline
    print("\n=== Start Training Faster R-CNN ===")
    loss_record = main_train_pipeline(model, train_loader, val_loader, optimizer)
    
    # Step 8: Save trained model weights (with timestamp to avoid overwriting)
    model_save_path = f"fasterrcnn_nuscenes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTrained model weights saved to: {model_save_path}")
    
    # Step 9: Visualize model prediction vs ground truth on test set sample
    print("\n=== Visualizing Model Prediction Result ===")
    test_dataset = test_loader.dataset
    sample_idx = 0  # Select the first sample in test set for visualization
    # Get original annotation data of the test sample
    orig_annot_index = test_dataset.indices[sample_idx]
    orig_annot = annotation_data[orig_annot_index]
    test_img_filename = orig_annot.get('filename', 'unknown')
    test_img_path = os.path.join(DATA_ROOT, test_img_filename)
    test_gt_bbox = orig_annot.get('bbox_corners', [0,0,0,0])
    test_gt_category = orig_annot.get('category_name', 'unknown')
    # Visualize prediction and ground truth comparison
    visualize_prediction(
        model=model, img_path=test_img_path, gt_bbox=test_gt_bbox, gt_category=test_gt_category,
        transform=full_dataset.transform, device=DEVICE,
        cat2idx=full_dataset.cat2idx, idx2cat=full_dataset.idx2cat,
        save_img=True, save_path="prediction_vs_ground_truth.jpg"
    )
    
    # Final prompt
    print("\n=== All Tasks Completed Successfully ===")