#超分辨率

### 文件说明
- checkpoint_scale2 : 放大2倍的checkpoint
- checkpoint_scale4: 放大4倍的checkpoint
- model.py : 模型
- main.py : 运行
- super_resolve.py : 运行保存的模型(checkpoint)
- dataset.py : 数据,自动下载BSD300保存在dataset文件夹中


### 运行
- train模型:
	python main.py  --upscale_factor 4 --nEpochs 60 --batch_size 16
- 测试超分辨率:
	python super_resolve.py --input_img 12003.jpg --model checkpoint_scale4/model_epoch_59.pth --output_img new_scale4.png
	
