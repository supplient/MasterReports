# Generate markdown index
# usage: python generate_markdown_index.py

# e.g.
# # blogs
# [C++检查模板参数的成员函数是否具有指定签名.md](blogs\C++检查模板参数的成员函数是否具有指定签名.md)

# [CUDA中原子锁的实现.md](blogs\CUDA中原子锁的实现.md)

# [GPU数据结构设计模式.md](blogs\GPU数据结构设计模式.md)

# [并行问题_offset2index.md](blogs\并行问题_offset2index.md)

# # ContactCut探索
# [[Deprecated]基于簇的形变的问题.md](ContactCut探索\[Deprecated]基于簇的形变的问题.md)

# [变换矩阵与特征分解.md](ContactCut探索\变换矩阵与特征分解.md)

# [基于形变程度的破坏检测.md](ContactCut探索\基于形变程度的破坏检测.md)

# [README.md](README.md)

# # 小论文内容的问题
# [关于内容的问题.md](小论文内容的问题\关于内容的问题.md)


import os
import logging

def getTree(path):
    tree = []

    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path, file)
        if os.path.isdir(filepath):
            sub_tree = getTree(filepath)
            if len(sub_tree) > 0:
                tree.append((file, sub_tree))
        else:
            filename, ext = os.path.splitext(file)
            if ext == ".md":
                tree.append((file, None))

    return tree

def logTree(tree, logger, level=1, path=""):
    for dirname, subtree in tree:
        if dirname == "node_modules":
            continue

        subpath = os.path.join(path, dirname)
        if not subtree:
            logger.info(
                "[" + dirname + "](" + subpath + ")\n"
            )
        else:
            logger.info(
                "#"*level + " " + dirname
            )
            logTree(subtree, logger, level=level+1, path=subpath)

if __name__ == "__main__":
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("README.md", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    tree = getTree(".")
    logTree(tree, logger)
