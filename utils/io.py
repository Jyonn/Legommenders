import json
import pickle

import yaml
from typing import Protocol, cast


# 定义一个支持 write 方法的协议类型，用于静态类型检查
class SupportsWriteStr(Protocol):
    def write(self, __s: str) -> object:
        ...


class SupportsWriteBytes(Protocol):
    def write(self, __s: bytes) -> object:
        ...


# 加载 JSON 文件为 Python 对象
def json_load(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# 从 JSON 字符串反序列化为 Python 对象
def json_loads(s: str):
    return json.loads(s)


# 将 Python 对象序列化为 JSON 字符串（带缩进，中文可读）
def json_dumps(obj, indent=2) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False)


# 保存 Python 对象为 JSON 文件（带缩进，中文可读）
def json_save(obj, filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, cast(SupportsWriteStr, f), indent=2, ensure_ascii=False)


# 加载 JSON Lines 格式文件（每行是一个 JSON）
def jsonl_load(filepath: str):
    lines = file_load(filepath).split('\n')  # 读取整个文件并按行分割
    lines = list(filter(lambda line: line.strip(), lines))  # 去掉空行
    lines = list(map(lambda line: json_loads(line), lines))  # 每行解析成 JSON 对象
    return lines


# 加载 YAML 文件为 Python 对象
def yaml_load(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# 从 YAML 字符串反序列化为 Python 对象
def yaml_loads(s: str):
    return yaml.safe_load(s)


# 将 Python 对象序列化为 YAML 字符串（支持中文）
def yaml_dumps(obj) -> str:
    return yaml.dump(obj, indent=2, allow_unicode=True)


# 保存 Python 对象为 YAML 文件
def yaml_save(obj, filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(obj, cast(SupportsWriteStr, f), indent=2, allow_unicode=True)


# 加载文本文件内容为字符串
def file_load(filepath: str, binary=False) -> str:
    with open(filepath, 'rb' if binary else 'r', encoding='utf-8') as f:
        return f.read()


# 保存文本到文件，可选择是否追加
def file_save(filepath: str, content: str, append=False):
    with open(filepath, 'a+' if append else 'w', encoding='utf-8') as f:
        f.write(content)


def pkl_load(filepath: str):
    return pickle.load(open(filepath, "rb"))


def pkl_save(obj, filepath: str):
    pickle.dump(obj, cast(SupportsWriteBytes, open(filepath, "wb")))
