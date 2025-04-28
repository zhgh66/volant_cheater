# volant_cheater

仅作为学习神经网络的实践作业 不用于违法途径尤其是游戏作弊

## V1.0 2025.4.28 12.27

使用yoloV8模型 目前只训练了一个大概 
数据集一共只有1990张截图 是自己打出来的 但是github这里太大了上传不了 要的话可以留言 现在在找集锦pov继续截图增大数据量
数据集里的类有三个 分别是enermy enermy_head teammate 主要的重点是enermy_head

可视化输出就是yolo里面提供的 加上一个F1指数 因为之前没怎么涉及 不太清楚该怎么修改模型的参数 有熟练的可以留言
目前的想法是先增加数据量 顺便通过高斯加噪增强模型的性能

<img width="716" alt="0f2fa8ca616ea0c862c8d50aeb47550b" src="https://github.com/user-attachments/assets/162e8e8f-88cb-45be-8f92-11a05cf48ccf" />
<img width="737" alt="a8b9c3fc5b5b8df5614a77635acb8240" src="https://github.com/user-attachments/assets/4dd790d6-38d3-48f0-b328-7ab9e46e3714" />
<img width="933" alt="616e229eee8300c27f72cc765b065dbe" src="https://github.com/user-attachments/assets/9305101d-a4f4-413d-ba75-19cb829ada9f" />

以后可能会在模型性能更好后再继续设计监控和移动鼠标功能

## V1.1 2025.4.28 23.33

之前的多尺度训练完全错误 yolo有自带的多尺度训练的入口 这次使用了这个 并且将batch设置为了6 因为我的电脑用的是3060 6GB的版本 所以只能设置小一点 运行时的GPUmen控制在5上下 达到更高效训练的目的

设置了一部分数据增强功能 就是有点调节不明白 还在摸索 

由于数据集量不够大 设置了一开始的高斯加噪 应该对于提升有点作用

本来晚上有事就干脆设置了200轮 20轮没提升的早停 但是实际上很早就收束了 没有想象的用时久

这次的训练结果对于原来的有很大的提升 主要是val的box_loss和dfl_loss有了显著的下降趋势 比起之前那个版本可以看出了下降的趋势而不是乱波动 在想是哪一部份更改的作用 

这次就用原版yolo的输出显示了 不用tensorboard输出一大堆

![results](https://github.com/user-attachments/assets/b224c03f-227b-4792-a21e-c1bdb7560662)

希望有熟悉yolo调参的大佬能动动手打开一下V1.1中的yolov8m_optimized_enemy文件夹 里面有一系列完整的输出 帮我提一些建议 更新后的训练文件是train_gaosi（随便用中文拼音命名的）训练好的模型在yolov8m_optimized_enemy/weights中 
