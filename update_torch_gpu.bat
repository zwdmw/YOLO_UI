@echo off
echo 正在为yolo环境安装PyTorch GPU版本...

call conda activate yolo
if %errorlevel% neq 0 (
    echo 环境yolo不存在，请确认环境名称正确
    pause
    exit /b 1
)

echo 卸载现有的PyTorch CPU版本...
call conda remove --force pytorch torchvision torchaudio cpuonly -y
if %errorlevel% neq 0 (
    echo 卸载PyTorch失败
    pause
    exit /b 1
)

echo 安装PyTorch GPU版本...
call conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
if %errorlevel% neq 0 (
    echo 安装PyTorch GPU版本失败
    pause
    exit /b 1
)

echo 安装完成，验证PyTorch GPU版本...
python -c "import torch; print('PyTorch版本:',torch.__version__); print('CUDA可用:',torch.cuda.is_available()); print('CUDA版本:',torch.version.cuda if torch.cuda.is_available() else '不可用')"

echo.
echo 如果显示CUDA可用为True，说明GPU版本安装成功。
pause 