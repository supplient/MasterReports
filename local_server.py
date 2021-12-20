from http.server import SimpleHTTPRequestHandler
import socketserver
import argparse
import os
from http import HTTPStatus
import urllib

####################
# 浏览器解析html文件中的"../lib/view.js"这类地址时，是从url出发的。
# 例如如果当前url是127.0.0.1:8080/test/abc的话，那么就是将/作为根目录，而/test/abc作为当前目录计算../lib/view.js，结果也就是/test/lib/view.js
# 而如果url是127.0.0.1:8080/abc的话，那么/test就是根目录，/test/abc依然是当前目录，计算结果依然是/test/lib/view.js
# 而若url直接是127.0.0.1:8080，那就是把/test/abc作为根目录，所以找不到上一目录了，结果变成了/lib/view.js，结果就会变成404了。
####################

class MyHandler(SimpleHTTPRequestHandler):
    directory = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=MyHandler.directory, **kwargs)

    def send_head(self):
        """Override for do not respond 304(which use browser cache)
        """
        path = self.translate_path(self.path)
        f = None
        if os.path.isdir(path):
            parts = urllib.parse.urlsplit(self.path)
            if not parts.path.endswith('/'):
                # redirect browser - doing basically what apache does
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                new_parts = (parts[0], parts[1], parts[2] + '/',
                             parts[3], parts[4])
                new_url = urllib.parse.urlunsplit(new_parts)
                self.send_header("Location", new_url)
                self.end_headers()
                return None
            for index in "index.html", "index.htm":
                index = os.path.join(path, index)
                if os.path.exists(index):
                    path = index
                    break
            else:
                return self.list_directory(path)
        ctype = self.guess_type(path)
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            fs = os.fstat(f.fileno())
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", ctype)
            self.send_header("Content-Length", str(fs[6]))
            self.send_header("Last-Modified",
                self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f
        except:
            f.close()
            raise

def startServe(directory, address, port):
    HANDLER = MyHandler
    HANDLER.extensions_map.update({
        ".js": "application/javascript",
        ".mp4": "video/mp4"
    })
    HANDLER.directory = directory

    print("Serving " + directory + " at http://" + address + ":" + str(port))
    with socketserver.TCPServer((address, port), HANDLER) as httpd:
        print("Start serve")
        httpd.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start a local server"
    )
    parser.add_argument(
        "webroot",
        nargs="?",
        default=os.getcwd(),
        help="The directory's path which should be webroot, like ./mypage",
    )
    parser.add_argument(
        "--addr",
        default="127.0.0.1",
        dest="address",
        help="IP address, like 127.0.0.1",
    )
    parser.add_argument(
        "-p",
        default=8080,
        type=int,
        dest="port",
        help="IP port, like 8080",
    )
    args = parser.parse_args()
    startServe(args.webroot, args.address, args.port)