# Eddy-Detection-Traditional-AI-
本项目是对 Kang &amp; Curchitser (2013) 在 *Journal of Geophysical Research: Oceans* 发表的论文《Gulf Stream eddy characteristics in a high-resolution ocean model》中所述涡旋检测算法的Python实现。  该算法采用一种混合方法，结合了基于物理参数（Okubo-Weiss, SSH）和基于流场几何特征的涡旋识别技术，以提高检测的准确性和鲁棒性。最终，项目将每日的海洋模式数据（如海面高度、速度场）处理成一个包含原始特征和涡旋标签（暖涡、冷涡、背景）的HDF5数据集，非常适合用于机器学习模型的训练。
