# CUDA-Kernel

## Nsight性能测评

1. 在 WSL 终端中，直接使用 `ncu` 命令行工具运行你的程序：
   ```bash
   # 需要加上 sudo 来获取 GPU 硬件计数器的读取权限
   sudo /usr/local/cuda/bin/ncu -o my_report ./你的可执行文件
   ```
2. 运行完毕后，当前目录下会生成一个 `my_report.ncu-rep` 文件。
3. 在 Windows 的 Nsight Compute 主界面中，点击左上角 `File` -> `Open File...`。
4. 在文件浏览器的地址栏直接输入 `\\wsl$\Ubuntu\home\你的路径` 找到报告文件并打开，这样可以完全绕过连接配置的问题。
