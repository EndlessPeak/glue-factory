"""
Based on sacred/stdout_capturing.py in project Sacred
https://github.com/IDSIA/sacred

Author: Paul-Edouard Sarlin (skydes)
"""

from __future__ import division, print_function, unicode_literals

import os
import subprocess
import sys
from contextlib import contextmanager
from threading import Timer


def apply_backspaces_and_linefeeds(text):
    """
    Interpret backspaces and linefeeds in text like a terminal would.
    Interpret text like a terminal by removing backspace and linefeed
    characters and applying them line by line.
    If final line ends with a carriage it keeps it to be concatenable with next
    output chunk.
    """
    orig_lines = text.split("\n")
    orig_lines_len = len(orig_lines)
    new_lines = []
    for orig_line_idx, orig_line in enumerate(orig_lines):
        chars, cursor = [], 0
        orig_line_len = len(orig_line)
        for orig_char_idx, orig_char in enumerate(orig_line):
            if orig_char == "\r" and (
                orig_char_idx != orig_line_len - 1
                or orig_line_idx != orig_lines_len - 1
            ):
                cursor = 0
            elif orig_char == "\b":
                cursor = max(0, cursor - 1)
            else:
                if (
                    orig_char == "\r"
                    and orig_char_idx == orig_line_len - 1
                    and orig_line_idx == orig_lines_len - 1
                ):
                    cursor = len(chars)
                if cursor == len(chars):
                    chars.append(orig_char)
                else:
                    chars[cursor] = orig_char
                cursor += 1
        new_lines.append("".join(chars))
    return "\n".join(new_lines)


def flush():
    """Try to flush all stdio buffers, both from python and from C."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported


# 复制 stdout 和 stderr 到文件中(同时可以显示在控制台上)
# 受以下链接启发:
# http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
# http://stackoverflow.com/a/651718/1388435
# http://stackoverflow.com/a/22434262/1388435
@contextmanager
def capture_outputs(filename):
    """
    1. 使用文件描述符级别的操作来复制标准输出和标准错误
       确保所有的输出(包括底层C库的输出)都能被捕获
    2. 依赖 tee 命令
       它可以同时将输出写入文件和标准输出/错误，确保控制台和文件的内容完全一致
    3. contextlib.contextmanager 是一个装饰器
       用于简化上下文管理器的创建
       上下文管理器的主要目的是管理资源的分配和释放，如文件的打开和关闭
       当本函数被 @contextmanager 装饰时，其 yield 关键字可以分隔资源的初始化和资源清理部分
    """
    # 以追加模式打开指定的日志文件，文件对象赋值给 target
    with open(str(filename), "a+") as target:
        # 分别设置标准输出和标准错误的文件描述符
        original_stdout_fd = 1
        original_stderr_fd = 2
        # 获取打开的文件的文件描述符
        target_fd = target.fileno()

        # 保存原始的标准输出和标准错误的文件描述符的拷贝
        # os.dup 复制文件描述符并返回新的文件描述符
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        # 创建子进程来复制输出
        # tee 读取标准输入的数据，然后同时写入标准输出和指定的文件
        tee_stdout = subprocess.Popen(
            # 将输入复制到标准输出和标准错误中
            # -a 表示追加， -i 表示忽略中断信号
            ["tee", "-a", "-i", "/dev/stderr"],
            # 启动新会话
            start_new_session=True,
            # 将父进程的输入管道连接到子进程的标准输入，这样父进程可以向子进程传递数据
            stdin=subprocess.PIPE,
            # 将子进程的标准错误重定向到日志文件（这句和下面那句似乎有些冗余）
            stderr=target_fd,
            # 将子进程的标准输出重定向到文件描述符1，即标准输出
            stdout=1,
        )
        tee_stderr = subprocess.Popen(
            ["tee", "-a", "-i", "/dev/stderr"],
            start_new_session=True,
            stdin=subprocess.PIPE,
            stderr=target_fd,
            # 将子进程的标准输出重定向到文件描述符2，即标准错误
            stdout=2,
        )

        # 刷新缓冲区
        flush()
        # 获取 tee 进程标准输入的文件描述符，将标准输出和标准错误重定向到 tee 的标准输入中
        os.dup2(tee_stdout.stdin.fileno(), original_stdout_fd)
        os.dup2(tee_stderr.stdin.fileno(), original_stderr_fd)

        try:
            # 进入上下文
            yield
        finally:
            # 再次刷新缓冲区
            flush()

            # 关闭 tee 的标准输入，通知 tee 即将结束
            tee_stdout.stdin.close()
            tee_stderr.stdin.close()

            # 恢复原始标准输出和标准错误的文件描述符
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)

            # 等待 tee 进程结束或者超时结束
            # 定义超时时终结 tee 进程的函数
            def kill_tees():
                tee_stdout.kill()
                tee_stderr.kill()
            # 创建计时器，在1秒后调用
            tee_timer = Timer(1, kill_tees)
            try:
                # 启动计时器
                tee_timer.start()
                # 等待 tee 进程结束
                tee_stdout.wait()
                tee_stderr.wait()
            finally:
                # 取消计时器，如果 tee 进程已经结束则无须终止
                tee_timer.cancel()

            # 关闭文件描述符
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

    # 清理日志文件
    # 打开日志文件读取内容
    with open(str(filename), "r") as target:
        text = target.read()
    # 处理回退和换行符来确保文件内容正确
    text = apply_backspaces_and_linefeeds(text)
    # 处理后的内容写回文件
    with open(str(filename), "w") as target:
        target.write(text)

# 下面是 ChatGPT 提供的简化版实现思路
# @contextmanager
# def capture_outputs(output_file):
#     # 保存原始的标准输出和标准错误
#     original_stdout = sys.stdout
#     original_stderr = sys.stderr
    
#     # 打开日志文件
#     with open(output_file, 'w') as f:
#         # 创建一个可以写入文件和控制台的自定义流
#         class StreamDuplicator:
#             def __init__(self, stream1, stream2):
#                 self.stream1 = stream1
#                 self.stream2 = stream2
            
#             def write(self, message):
#                 self.stream1.write(message)
#                 self.stream2.write(message)
            
#             def flush(self):
#                 self.stream1.flush()
#                 self.stream2.flush()

#         # 将标准输出和标准错误重定向到自定义流
#         sys.stdout = StreamDuplicator(original_stdout, f)
#         sys.stderr = StreamDuplicator(original_stderr, f)
        
#         try:
#             yield
#         finally:
#             # 恢复原始的标准输出和标准错误
#             sys.stdout = original_stdout
#             sys.stderr = original_stderr
