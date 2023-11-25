set IMAGE_PATH="D:\BaiduNetdiskDownload\hloc-1124"
set GSCODE="D:\nerfData\gs-nerf"
set HLOCCODE="D:\nerfData\hloc"

docker run --name hloc_xxx -it --rm --gpus all --shm-size 12G -p 8888:8888 -v "C:\Users\bigy2\.cache":/root/.cache -v %HLOCCODE%:/root/code -v %IMAGE_PATH%:/root/data wsandy95/hloc:latest /bin/bash -c "cd /root/code && python3 runSFM.py -i /root/data/input/ -o /root/data/sfm"

docker run --name train_60b96908w064c4b6cb9c064f771fa8119 -it --rm --gpus all -v %IMAGE_PATH%\sfm\colmap:/root/data -v %GSCODE%:/root/code 192.168.1.42:5000/cuda:11.1-devel-ubuntu2204-gsnerf  /bin/bash  -c "source ~/miniconda3/bin/activate && conda init bash && conda activate gaussian_splatting && cd /root/code && python train.py -s /root/data -m /root/data/output "


@REM @REM @REM ----viewer
set TrainOutputPath=%IMAGE_PATH%\sfm\colmap
@REM "D:\BaiduNetdiskDownload\hloc-1124\sfm\colmap"
set VIEWCODE="D:\nerfData\gaussian-splatting-lightning"
docker run --name  viewer_97f459dee22843aaa728ded6999cf6ab -it --rm --gpus all -p 8080:8080 -v %VIEWCODE%:/root/code -v %TrainOutputPath%:/root/data 192.168.1.42:5000/cuda:11.1-devel-ubuntu2204-nerfviewer-ffmpeg  /bin/bash  -c "source ~/miniconda3/bin/activate && conda init bash && conda activate gspl  && cd /root/code && python viewer.py  /root/data/output/point_cloud/iteration_30000/point_cloud.ply " 


docker run --name  viewer_97f459dee22843aaa728ded6999cf6ab -it --rm --gpus all -p 8080:8080 -v %VIEWCODE%:/root/code -v %TrainOutputPath%:/root/data 192.168.1.42:5000/cuda:11.1-devel-ubuntu2204-nerfviewer-ffmpeg  /bin/bash  -c "source ~/miniconda3/bin/activate && conda init bash && conda activate gspl  && cd /root/code && python /root/code/render.py /root/data/output/point_cloud/iteration_30000/point_cloud.ply --camera-path-filename /root/data/render/camera_paths/2023-11-24-11-36-26.json --output-path  /root/data/render/video/2023-11-24-11-36-26.mp4"

