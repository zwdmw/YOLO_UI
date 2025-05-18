from PyQt5.QtWidgets import QWidget, QToolTip
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont

class TooltipManager:
    """管理UI元素的工具提示"""
    
    @staticmethod
    def apply_tooltips(window):
        """为主窗口的UI元素添加工具提示"""
        # 设置工具提示全局字体
        QToolTip.setFont(QFont('Segoe UI', 9))
        
        # 应用主窗口工具提示
        TooltipManager.apply_training_tab_tooltips(window.training_tab)
        TooltipManager.apply_testing_tab_tooltips(window.testing_tab)
        TooltipManager.apply_inference_tab_tooltips(window.inference_tab)
        TooltipManager.apply_settings_tab_tooltips(window.settings_tab)
    
    @staticmethod
    def apply_training_tab_tooltips(tab):
        """为训练标签页添加工具提示"""
        if not tab:
            return
            
        # 常用训练标签页控件的工具提示
        if hasattr(tab, 'dataset_path_edit'):
            tab.dataset_path_edit.setToolTip("指定数据集的路径<br><b>快捷键:</b> 无")
            
        if hasattr(tab, 'model_combo'):
            tab.model_combo.setToolTip("选择要训练的模型架构<br><b>快捷键:</b> 无")
            
        if hasattr(tab, 'epochs_spin'):
            tab.epochs_spin.setToolTip("训练的轮数<br><b>提示:</b> 较大的轮数通常能获得更好的结果，但需要更长的训练时间")
            
        if hasattr(tab, 'batch_size_spin'):
            tab.batch_size_spin.setToolTip("每批次训练的样本数量<br><b>提示:</b> 较大的批量可以加速训练，但需要更多的GPU内存")
            
        if hasattr(tab, 'start_training_btn'):
            tab.start_training_btn.setToolTip("开始训练模型<br><b>确保所有参数设置正确</b>")
            
        if hasattr(tab, 'stop_training_btn'):
            tab.stop_training_btn.setToolTip("停止当前训练进程<br><b>警告:</b> 训练停止后无法恢复")
            
        if hasattr(tab, 'clear_log_btn'):
            tab.clear_log_btn.setToolTip("清除训练日志<br><b>快捷键:</b> Ctrl+L")
    
    @staticmethod
    def apply_testing_tab_tooltips(tab):
        """为测试标签页添加工具提示"""
        if not tab:
            return
            
        # 常用测试标签页控件的工具提示
        if hasattr(tab, 'test_dataset_path_edit'):
            tab.test_dataset_path_edit.setToolTip("指定测试数据集的路径<br><b>快捷键:</b> 无")
            
        if hasattr(tab, 'model_path_edit'):
            tab.model_path_edit.setToolTip("指定训练好的模型权重文件路径<br><b>提示:</b> 选择合适的模型权重以匹配您的测试数据")
            
        if hasattr(tab, 'conf_threshold_spin'):
            tab.conf_threshold_spin.setToolTip("检测置信度阈值<br><b>提示:</b> 较高的值会减少误报，但可能会增加漏报")
            
        if hasattr(tab, 'start_testing_btn'):
            tab.start_testing_btn.setToolTip("开始评估模型性能<br><b>确保所有参数设置正确</b>")
            
        if hasattr(tab, 'stop_testing_btn'):
            tab.stop_testing_btn.setToolTip("停止当前测试进程<br><b>警告:</b> 测试停止后无法恢复")
            
        if hasattr(tab, 'clear_test_log_btn'):
            tab.clear_test_log_btn.setToolTip("清除测试日志<br><b>快捷键:</b> Ctrl+L")
    
    @staticmethod
    def apply_inference_tab_tooltips(tab):
        """为推理标签页添加工具提示"""
        if not tab:
            return
            
        # 常用推理标签页控件的工具提示
        if hasattr(tab, 'inference_source_edit'):
            tab.inference_source_edit.setToolTip("指定推理源（图像、视频或文件夹路径）<br><b>快捷键:</b> 无")
            
        if hasattr(tab, 'inference_model_path_edit'):
            tab.inference_model_path_edit.setToolTip("指定用于推理的模型权重文件路径<br><b>提示:</b> 选择合适的模型以获得最佳结果")
            
        if hasattr(tab, 'start_inference_btn'):
            tab.start_inference_btn.setToolTip("开始对选定的源进行推理<br><b>确保所有参数设置正确</b>")
            
        if hasattr(tab, 'stop_inference_btn'):
            tab.stop_inference_btn.setToolTip("停止当前推理进程<br><b>警告:</b> 推理停止后无法恢复")
            
        if hasattr(tab, 'save_results_btn'):
            tab.save_results_btn.setToolTip("保存推理结果<br><b>提示:</b> 结果将保存在指定的输出目录中")
    
    @staticmethod
    def apply_settings_tab_tooltips(tab):
        """为设置标签页添加工具提示"""
        if not tab:
            return
            
        # 常用设置标签页控件的工具提示
        if hasattr(tab, 'device_combo'):
            tab.device_combo.setToolTip("选择运行环境<br><b>提示:</b> GPU通常比CPU快很多")
            
        if hasattr(tab, 'output_dir_edit'):
            tab.output_dir_edit.setToolTip("指定保存输出结果的目录<br><b>快捷键:</b> 无")
            
        if hasattr(tab, 'save_settings_btn'):
            tab.save_settings_btn.setToolTip("保存当前设置<br><b>快捷键:</b> Ctrl+S")
            
        if hasattr(tab, 'reset_settings_btn'):
            tab.reset_settings_btn.setToolTip("重置所有设置为默认值<br><b>警告:</b> 此操作无法撤销")
            
        if hasattr(tab, 'theme_combo') and tab.theme_combo:
            tab.theme_combo.setToolTip("选择应用程序的主题<br><b>提示:</b> 暗色主题可减轻眼睛疲劳")
    
    @staticmethod
    def show_temporary_tooltip(widget, message, duration=3000):
        """显示临时的工具提示"""
        position = widget.mapToGlobal(widget.rect().topRight())
        QToolTip.showText(position, message, widget)
        
        # 创建定时器在指定时间后隐藏工具提示
        QTimer.singleShot(duration, lambda: QToolTip.hideText()) 