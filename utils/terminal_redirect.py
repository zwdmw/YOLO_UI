import sys
import traceback
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication

class TerminalRedirect(QObject):
    """Redirect stdout and stderr to a PyQt signal for UI integration."""
    outputWritten = pyqtSignal(str)
    
    def __init__(self, stream=None):
        super().__init__()
        self.stream = stream
        self.buffer = ""
    
    def write(self, text):
        """Write text to stream and emit signal."""
        try:
            if self.stream:
                self.stream.write(text)
                self.stream.flush()
            
            self.buffer += text
            if '\n' in text:
                self.outputWritten.emit(self.buffer)
                self.buffer = ""
        except Exception as e:
            # If something goes wrong, try to at least write to the original stream
            try:
                if self.stream:
                    self.stream.write(f"Error in TerminalRedirect.write: {str(e)}\n")
                    self.stream.write(traceback.format_exc())
                    self.stream.flush()
            except Exception:
                pass  # Last resort: simply ignore if we can't handle the error
    
    def flush(self):
        """Flush the stream."""
        try:
            if self.stream:
                self.stream.flush()
            if self.buffer:
                self.outputWritten.emit(self.buffer)
                self.buffer = ""
        except Exception as e:
            # If something goes wrong, try to at least write to the original stream
            try:
                if self.stream:
                    self.stream.write(f"Error in TerminalRedirect.flush: {str(e)}\n")
                    self.stream.flush()
            except Exception:
                pass  # Last resort: simply ignore if we can't handle the error

class TerminalManager:
    """Manage terminal redirection for the application."""
    
    def __init__(self):
        # Save original streams before any redirection
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        # Create redirects pointing to the original streams
        self.stdout_redirect = TerminalRedirect(self.original_stdout)
        self.stderr_redirect = TerminalRedirect(self.original_stderr)
        # 默认不连接到UI
        self.ui_connected = False
    
    def start_redirection(self):
        """Start redirection of stdout and stderr."""
        try:
            sys.stdout = self.stdout_redirect
            sys.stderr = self.stderr_redirect
            print("Terminal redirection started successfully")
        except Exception as e:
            # If redirection fails, write to original stdout
            self.original_stdout.write(f"Failed to start terminal redirection: {str(e)}\n")
            self.original_stdout.write(traceback.format_exc())
            self.original_stdout.flush()
    
    def stop_redirection(self):
        """Stop redirection and restore original streams."""
        try:
            # Flush any pending output
            if hasattr(sys.stdout, 'flush'):
                sys.stdout.flush()
            if hasattr(sys.stderr, 'flush'):
                sys.stderr.flush()
            
            # Restore original streams
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
        except Exception as e:
            # If something goes wrong, try to at least write to the original stream
            try:
                self.original_stdout.write(f"Error stopping terminal redirection: {str(e)}\n")
                self.original_stdout.write(traceback.format_exc())
                self.original_stdout.flush()
            except Exception:
                pass  # Last resort: simply ignore if we can't handle the error
    
    def connect_to_text_edit(self, text_edit):
        """Connect the redirection signals to a QTextEdit widget."""
        try:
            if text_edit is None:
                # 如果没有提供text_edit，仍然可以继续重定向，但不会显示到UI
                self.ui_connected = False
                return
                
            self.stdout_redirect.outputWritten.connect(lambda text: self._append_to_text_edit(text_edit, text))
            self.stderr_redirect.outputWritten.connect(lambda text: self._append_to_text_edit(text_edit, text, True))
            self.ui_connected = True
        except Exception as e:
            # If something goes wrong, try to at least write to the original stream
            try:
                self.original_stdout.write(f"Error connecting terminal redirection: {str(e)}\n")
                self.original_stdout.write(traceback.format_exc())
                self.original_stdout.flush()
            except Exception:
                pass  # Last resort: simply ignore if we can't handle the error
    
    def _append_to_text_edit(self, text_edit, text, is_error=False):
        """Append text to the text edit widget."""
        try:
            # 检查是否连接到UI
            if not self.ui_connected:
                return
                
            # Check if the text_edit exists and is valid
            if text_edit is None or not hasattr(text_edit, 'textCursor'):
                return
                
            cursor = text_edit.textCursor()
            cursor.movePosition(cursor.End)
            
            # 获取当前主题
            app = QApplication.instance()
            theme = app.property('theme') if app.property('theme') else 'tech'
            
            # 根据主题设置文本颜色
            if is_error:
                if theme == 'light':
                    text_edit.setTextColor(QColor(220, 53, 69))  # 浅色主题下的错误红色
                elif theme == 'dark':
                    text_edit.setTextColor(QColor(255, 107, 107))  # 深色主题下的错误红色
                else:  # tech theme
                    text_edit.setTextColor(QColor(255, 82, 82))  # 科技感主题下的错误红色
            else:
                if theme == 'light':
                    text_edit.setTextColor(QColor(33, 37, 41))  # 浅色主题下的文本黑色
                elif theme == 'dark':
                    text_edit.setTextColor(QColor(220, 230, 240))  # 深色主题下的文本浅色
                else:  # tech theme
                    text_edit.setTextColor(QColor(220, 230, 240))  # 科技感主题下的文本浅色
                
            # Append the text
            text_edit.append(text)
            text_edit.setTextCursor(cursor)
            text_edit.ensureCursorVisible()
        except Exception as e:
            # If something goes wrong, try to at least write to the original stream
            try:
                self.original_stdout.write(f"Error appending to text edit: {str(e)}\n")
                self.original_stdout.write(traceback.format_exc())
                self.original_stdout.flush()
            except Exception:
                pass  # Last resort: simply ignore if we can't handle the error 