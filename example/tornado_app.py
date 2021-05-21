import tornado.web
import h5py
import os.path
import sys
import tornado.ioloop
from h5core.responses import DatasetResponse, ResolvedEntityResponse, create_response
from h5core.encoders import encode


class BaseHandler(tornado.web.RequestHandler):
    def initialize(self, base_dir) -> None:
        self.base_dir = base_dir

    def get(self, file_path):
        path = self.get_query_argument("path", None)
        format = self.get_query_argument("format", None)

        with h5py.File(os.path.join(self.base_dir, file_path), "r") as h5file:
            response = self.get_response(h5file, path)

        encoded_content_chunks, headers = encode(response, format)

        for key, value in headers.items():
            self.set_header(key, value)
        for chunk in encoded_content_chunks:
            self.write(chunk)
        self.finish()

    def get_response(self, h5file, path):
        raise NotImplementedError


class AttributeHandler(BaseHandler):
    def get_response(self, h5file, path):
        response = create_response(h5file, path)
        assert isinstance(response, ResolvedEntityResponse)
        return response.attributes()


class DataHandler(BaseHandler):
    def get_response(self, h5file, path):
        selection = self.get_query_argument("selection", None)

        response = create_response(h5file, path)
        assert isinstance(response, DatasetResponse)
        return response.data(selection)


class MetadataHandler(BaseHandler):
    def get_response(self, h5file, path):
        response = create_response(h5file, path)
        return response.metadata()


PORT = 8888

if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    app = tornado.web.Application(
        [
            (r"/attr/(.*)", AttributeHandler, {"base_dir": base_dir}),
            (r"/data/(.*)", DataHandler, {"base_dir": base_dir}),
            (r"/meta/(.*)", MetadataHandler, {"base_dir": base_dir}),
        ],
        debug=True,
    )
    app.listen(PORT)
    print(f"App is listening on port {PORT} from {base_dir}...")
    tornado.ioloop.IOLoop.current().start()
