import wandb
import subprocess
import multiprocessing


def train_with_torchrun():
    # 设置环境变量 CUDA_VISIBLE_DEVICES 为传入的 GPU ID
    cmd = f"torchrun --nproc_per_node=1 oversampling.py"
    subprocess.run(cmd, shell=True)


# 定义 Sweep 配置
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'lr': {'values': [5e-5]},
        'batch_size': {'values': [8]},
        'num_epochs': {'values': [15]},
        'dropout': {'values': [0]},
        'sample_rate': {'values': [0.1, 0.2, 0.5, 1]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="food_product_2")
wandb.agent(sweep_id, function=train_with_torchrun)

# from send_emails import send_email

# send_email()
