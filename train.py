# 필수 라이브러리 및 모듈 임포트
from super_gradients.training import Trainer, models  # 모델 학습을 위한 Trainer와 모델 관리 모듈
from super_gradients.common.object_names import Models  # 사용할 모델의 이름들
from super_gradients.training.losses import PPYoloELoss  # PP-YOLOE에서 사용하는 손실 함수
from super_gradients.training.metrics import DetectionMetrics_050  # mAP@0.5 측정용 메트릭 클래스
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback  # 후처리 콜백 (NMS 등)
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,  # COCO YOLO 포맷 학습용 데이터 로더
    coco_detection_yolo_format_val,    # COCO YOLO 포맷 검증용 데이터 로더
)
import torch
import numpy

# PyTorch의 pickle serialization 에러 방지를 위한 numpy 관련 safe globals 등록
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
torch.serialization.add_safe_globals([numpy.ndarray])

# 데이터셋 파라미터 정의 (이미지, 라벨 경로 및 클래스 목록)
dataset_params = {
    'data_dir': './dataset/D-Fire',         # 데이터셋 루트 디렉토리
    'train_images_dir': 'train/images',     # 학습 이미지 디렉토리
    'train_labels_dir': 'train/labels',     # 학습 라벨 디렉토리
    'val_images_dir': 'val/images',         # 검증 이미지 디렉토리
    'val_labels_dir': 'val/labels',         # 검증 라벨 디렉토리
    'classes': ['smoke', 'fire']            # 클래스 이름 리스트
}

# 학습 데이터 로더 생성
train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': 32,     # 배치 사이즈
        'num_workers': 4      # 데이터 로딩에 사용할 워커 수
    }
)

# 검증 데이터 로더 생성
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

# Trainer 객체 생성 (모델 학습을 관리할 객체)
trainer = Trainer(
    experiment_name='yolo_nas_local',  # 실험 이름 (결과 디렉터리에 사용됨)
    ckpt_root_dir='./checkpoints/'     # 체크포인트 저장 디렉토리
)

# 모델 불러오기 (YOLO-NAS-M, 사전학습 없이 시작, 클래스 수 지정)
net = models.get(
    Models.YOLO_NAS_M, 
    pretrained_weights=None, 
    num_classes=len(dataset_params['classes'])
)

# 학습 파라미터 정의
train_params = {
    # 조기 종료 설정: 성능 향상이 없으면 학습 조기 종료
    'early_stopping': True,
    'early_stopping_params': {
        'early_stopping_metric': 'mAP@0.50',       # 기준 메트릭
        'early_stopping_mode': 'max',              # 최대화가 목표
        'early_stopping_patience': 30,             # 성능 향상 없을 경우 30 epoch 후 종료
        'early_stopping_min_delta': 0.001          # 성능 향상으로 인정될 최소 수치
    },
    'silent_mode': False,                          # 로그 출력 여부
    'average_best_models': True,                   # 최고의 모델 평균 저장 여부
    'warmup_mode': 'linear_epoch_step',            # 워밍업 방식 (선형 증가)
    'warmup_initial_lr': 1e-6,                     # 워밍업 시작 learning rate
    'lr_warmup_epochs': 3,                         # 워밍업 적용 epoch 수
    'initial_lr': 5e-4,                            # 초기 learning rate
    'lr_mode': 'cosine',                           # learning rate 스케줄링 (cosine decay)
    'cosine_final_lr_ratio': 0.1,                  # 최종 learning rate 비율
    'optimizer': 'Adam',                           # 옵티마이저
    'optimizer_params': {'weight_decay': 0.0001},  # 옵티마이저 추가 파라미터
    'zero_weight_decay_on_bias_and_bn': True,      # Bias 및 BatchNorm에 가중치 감쇠 제외
    'ema': True,                                   # EMA(지수이동평균) 사용
    'ema_params': {'decay': 0.9, 'decay_type': 'threshold'},  # EMA decay 설정
    'max_epochs': 500,                             # 최대 epoch 수
    'mixed_precision': True,                       # Mixed Precision 학습 활성화 (성능 향상 및 메모리 절약)

    # 손실 함수 정의
    'loss': PPYoloELoss(
        use_static_assigner=False,                 # 동적 어사이너 사용 (PPYOLOE 특징)
        num_classes=len(dataset_params['classes']),
        reg_max=16                                 # regression 분해를 위한 최대 bin 수
    ),

    # 검증 시 사용할 메트릭 리스트 정의 (mAP@0.5 기준)
    'valid_metrics_list': [
        DetectionMetrics_050(
            score_thres=0.1,           # 예측 결과의 score threshold
            top_k_predictions=300,     # 최대 top-k 예측값 평가
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,    # 정답 박스 정규화 여부
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,  # 후처리 시 score 임계값 (더 낮은 것 포함 가능)
                nms_top_k=1000,        # NMS 대상 top-K 후보
                max_predictions=300,   # 최종 예측 최대 수
                nms_threshold=0.7      # NMS에서 사용할 IoU 임계값
            )
        )
    ],
    'metric_to_watch': 'mAP@0.50'      # early stopping 및 best model 판단 기준
}

# GPU 사용 가능 여부 확인 및 정보 출력
print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# 학습 실행
trainer.train(
    model=net, 
    training_params=train_params, 
    train_loader=train_data, 
    valid_loader=val_data
)
