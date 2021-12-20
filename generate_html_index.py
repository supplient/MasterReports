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
            if ext == ".html":
                tree.append((file, None))

    return tree

def logTree(tree, logger, level=1, path=""):
    for dirname, subtree in tree:
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
    fh = logging.FileHandler("README.md", mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    tree = getTree(".")
    logTree(tree, logger)
