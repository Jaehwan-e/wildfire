# 필수 임포트
from super_gradients.training import Trainer, models
from super_gradients.training import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
import torch

import numpy
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
torch.serialization.add_safe_globals([numpy.ndarray])

# 데이터셋 파라미터 및 데이터 로더 정의
dataset_params = {
    'data_dir': './dataset/D-Fire',
    'train_images_dir': 'train/images',
    'train_labels_dir': 'train/labels',
    'val_images_dir': 'val/images',
    'val_labels_dir': 'val/labels',
    'classes': ['class1', 'class2']
}

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': 32,
        'num_workers': 4
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': 32,
        'num_workers': 4
    }
)

trainer = Trainer(
    experiment_name='yolo_nas_local',
    ckpt_root_dir='./checkpoints/'
)

net = models.get(Models.YOLO_NAS_M, pretrained_weights=None, num_classes=len(dataset_params['classes']))

train_params = {
    'early_stopping': True,
    'early_stopping_params': {
        'early_stopping_metric': 'mAP@0.50',  # 성능 기준 metric (train_params['metric_to_watch']와 동일하게!)
        'early_stopping_mode': 'max',         # max: 성능이 올라야 함, min: 성능이 낮아져야 함
        'early_stopping_patience': 30,        # 개선 없으면 몇 epoch 후 중단할지
        'early_stopping_min_delta': 0.001     # 개선으로 인정할 최소 차이
    },
    'silent_mode': False,
    'average_best_models': True,
    'warmup_mode': 'linear_epoch_step',
    'warmup_initial_lr': 1e-6,
    'lr_warmup_epochs': 3,
    'initial_lr': 5e-4,
    'lr_mode': 'cosine',
    'cosine_final_lr_ratio': 0.1,
    'optimizer': 'Adam',
    'optimizer_params': {'weight_decay': 0.0001},
    'zero_weight_decay_on_bias_and_bn': True,
    'ema': True,
    'ema_params': {'decay': 0.9, 'decay_type': 'threshold'},
    'max_epochs': 500,
    'mixed_precision': True,
    'loss': PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    'valid_metrics_list': [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    'metric_to_watch': 'mAP@0.50',
}

print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

trainer.train(model=net, training_params=train_params, train_loader=train_data, valid_loader=val_data)