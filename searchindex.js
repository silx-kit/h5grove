Search.setIndex({"docnames": ["api", "example", "index", "integration", "low_level"], "filenames": ["api.md", "example.md", "index.md", "integration.md", "low_level.md"], "titles": ["h5grove endpoints API", "Example implementations", "h5grove, core utilities to serve HDF5 file contents", "Integration in an existing application", "h5grove reference"], "terms": {"us": [1, 3, 4], "ar": [1, 2, 3, 4], "given": [1, 4], "folder": 1, "These": [1, 4], "function": [1, 3, 4], "backend": [1, 2, 3], "make": [1, 2, 3], "util": [1, 3, 4], "provid": [1, 2, 3, 4], "h5grove": [1, 3], "packag": [1, 2], "The": [1, 2, 3, 4], "endpoint": [1, 2, 3, 4], "api": [1, 2, 3, 4], "i": [1, 2, 3, 4], "describ": 1, "here": [1, 2, 3], "base": [1, 3], "server": 1, "sampl": 1, "code": 1, "usag": [1, 3], "fastapi_app": 1, "py": 1, "h": 1, "p": 1, "port": 1, "ip": 1, "basedir": [1, 3], "listen": 1, "default": [1, 4], "8888": 1, "localhost": 1, "directori": [1, 3], "from": [1, 3, 4], "which": [1, 3, 4], "retriev": 1, "hdf5": [1, 3, 4], "file": [1, 3], "flask_app": 1, "compress": [1, 2], "enabl": 1, "http": [1, 4], "fals": [1, 3, 4], "tornado_app": 1, "python": [2, 4], "design": [2, 4], "attribut": [2, 4], "metadata": [2, 4], "data": [2, 3, 4], "h5py": [2, 4], "latest": 2, "releas": 2, "avail": [2, 3], "There": 2, "sever": 2, "out": 2, "can": [2, 3, 4], "howev": 2, "thei": 2, "dedic": [2, 3], "usecas": 2, "settl": 2, "one": 2, "implement": [2, 3, 4], "hamper": 2, "reusabl": 2, "In": 2, "addit": 2, "some": 2, "problem": 2, "aris": 2, "constantli": 2, "when": [2, 4], "To": 2, "name": [2, 4], "few": 2, "resolv": [2, 4], "extern": [2, 4], "link": [2, 4], "deal": 2, "slice": [2, 4], "dataset": [2, 4], "encod": 2, "effici": 2, "consist": 2, "look": [2, 4], "you": [2, 4], "nan": 2, "infin": 2, "json": [2, 3, 4], "aim": 2, "build": [2, 3, 4], "block": 2, "solv": 2, "common": [2, 3], "reus": 2, "exist": 2, "new": [2, 3], "pip": 2, "low": 2, "level": [2, 4], "whatev": 2, "choos": 2, "web": [2, 3, 4], "framework": 2, "requir": 2, "follow": 2, "wai": 2, "fastapi": 2, "flask": 2, "tornado": 2, "simpli": 2, "exampl": [2, 4], "we": 2, "might": 2, "interest": 2, "add": 2, "capabl": 2, "an": [2, 4], "applic": 2, "If": [2, 3, 4], "doe": 2, "suit": 2, "modul": [2, 3], "your": 2, "own": 2, "let": 2, "handl": [2, 4], "task": 2, "resolut": [2, 4], "histori": 2, "track": 2, "page": [2, 3], "integr": 2, "refer": 2, "see": 3, "ad": 3, "contain": [3, 4], "It": 3, "apirout": 3, "includ": [3, 4], "import": 3, "router": 3, "set": [3, 4], "app": 3, "configur": 3, "serv": [3, 4], "base_dir": 3, "o": [3, 4], "path": [3, 4], "abspath": 3, "option": 3, "include_rout": 3, "helper": 3, "async": 3, "get_attr": 3, "depend": 3, "add_base_path": 3, "attr_kei": [3, 4], "queri": [3, 4], "none": [3, 4], "attr": 3, "handler": 3, "paramet": [3, 4], "str": [3, 4], "list": [3, 4], "get_data": 3, "dtype": [3, 4], "origin": [3, 4], "format": 3, "flatten": [3, 4], "select": [3, 4], "bool": [3, 4], "get_meta": 3, "resolve_link": [3, 4], "only_valid": [3, 4], "meta": 3, "get_stat": 3, "stat": [3, 4], "rout": 3, "object": 3, "defin": [3, 4], "scope": 3, "receiv": 3, "send": 3, "return": [3, 4], "type": [3, 4], "where": 3, "blueprint": 3, "regist": 3, "__name__": 3, "config": 3, "h5_base_dir": 3, "register_blueprint": 3, "reli": 3, "being": 3, "url_rul": 3, "attr_rout": 3, "data_rout": 3, "meta_rout": 3, "paths_rout": 3, "stats_rout": 3, "map": 3, "url": 3, "get_handl": 3, "construct": 3, "pass": 3, "directli": [3, 4], "argument": 3, "h5grove_handl": 3, "allow_origin": 3, "On": 3, "Or": 3, "add_handl": 3, "class": [3, 4], "attributehandl": 3, "request": [3, 4], "kwarg": 3, "httpserverrequest": 3, "ani": [3, 4], "basehandl": 3, "initi": 3, "prepar": 3, "call": 3, "begin": 3, "befor": 3, "get": [3, 4], "post": 3, "etc": 3, "overrid": 3, "thi": [3, 4], "method": [3, 4], "perform": 3, "regardless": 3, "asynchron": 3, "support": [3, 4], "def": 3, "decor": 3, "gen": 3, "coroutin": 3, "await": 3, "execut": 3, "proce": 3, "until": 3, "done": 3, "version": 3, "3": 3, "1": [3, 4], "write_error": 3, "status_cod": 3, "custom": 3, "error": 3, "mai": 3, "write": 3, "render": 3, "set_head": 3, "produc": 3, "output": 3, "usual": 3, "wa": 3, "caus": 3, "uncaught": 3, "except": 3, "httperror": 3, "exc_info": 3, "tripl": 3, "note": 3, "current": 3, "purpos": 3, "like": [3, 4], "sy": 3, "traceback": 3, "format_exc": 3, "int": [3, 4], "datahandl": 3, "metadatahandl": 3, "statisticshandl": 3, "allow": 3, "cor": 3, "tupl": [3, 4], "dict": [3, 4], "create_cont": 4, "wrapper": 4, "desir": 4, "entiti": 4, "decompress": 4, "hdf5plugin": 4, "h5file": 4, "linkresolut": 4, "factori": 4, "soft": 4, "open": 4, "tell": 4, "should": 4, "onli": 4, "valid": 4, "rais": 4, "patherror": 4, "cannot": 4, "found": 4, "linkerror": 4, "all": 4, "typeerror": 4, "encount": 4, "unsupport": 4, "hold": 4, "inform": 4, "expos": 4, "relev": 4, "through": 4, "non": 4, "For": 4, "data_stat": 4, "statist": 4, "comput": 4, "plug": 4, "so": 4, "take": 4, "more": 4, "externallinkcont": 4, "externallink": 4, "kind": 4, "external_link": 4, "depth": 4, "target_fil": 4, "target_path": 4, "properti": 4, "last": 4, "member": 4, "target": 4, "softlinkcont": 4, "softlink": 4, "soft_link": 4, "datasetcont": 4, "h5py_ent": 4, "filter": 4, "kei": 4, "sequenc": 4, "true": 4, "arrai": 4, "convers": 4, "No": 4, "safe": 4, "convert": 4, "j": 4, "typedarrai": 4, "develop": 4, "mozilla": 4, "org": 4, "fr": 4, "doc": 4, "javascript": 4, "global_object": 4, "numpi": 4, "index": 4, "strict_positive_min": 4, "number": 4, "positive_min": 4, "min": 4, "max": 4, "mean": 4, "std": 4, "float": 4, "attributemetadata": 4, "chunk": 4, "shape": 4, "typemetadata": 4, "groupcont": 4, "group": 4, "recurs": 4, "child": 4, "0": 4, "children": 4, "childmetadata": 4, "appropri": 4, "header": 4, "respons": 4, "orjson": 4, "other": 4, "download": 4, "warn": 4, "Not": 4, "bin": 4, "nd": 4, "scalar": 4, "byte": 4, "csv": 4, "npy": 4, "tiff": 4, "2d": 4, "A": 4, "queryargumenterror": 4, "among": 4, "ones": 4, "abov": 4, "associ": 4, "orjson_default": 4, "serializ": 4, "orjson_encod": 4, "param": 4, "callabl": 4, "github": 4, "com": 4, "ijl": 4, "bin_encod": 4, "ndarrai": 4, "csv_encod": 4, "npy_encod": 4, "tiff_encod": 4}, "objects": {"h5grove.content": [[4, 0, 1, "", "DatasetContent"], [4, 0, 1, "", "ExternalLinkContent"], [4, 0, 1, "", "GroupContent"], [4, 0, 1, "", "SoftLinkContent"], [4, 4, 1, "", "create_content"]], "h5grove.content.DatasetContent": [[4, 1, 1, "", "attributes"], [4, 1, 1, "", "data"], [4, 1, 1, "", "data_stats"], [4, 2, 1, "", "kind"], [4, 1, 1, "", "metadata"], [4, 3, 1, "", "name"], [4, 3, 1, "", "path"]], "h5grove.content.ExternalLinkContent": [[4, 2, 1, "", "kind"], [4, 1, 1, "", "metadata"], [4, 3, 1, "", "name"], [4, 3, 1, "", "path"], [4, 3, 1, "", "target_file"], [4, 3, 1, "", "target_path"]], "h5grove.content.GroupContent": [[4, 1, 1, "", "attributes"], [4, 2, 1, "", "kind"], [4, 1, 1, "", "metadata"], [4, 3, 1, "", "name"], [4, 3, 1, "", "path"]], "h5grove.content.SoftLinkContent": [[4, 2, 1, "", "kind"], [4, 1, 1, "", "metadata"], [4, 3, 1, "", "name"], [4, 3, 1, "", "path"], [4, 3, 1, "", "target_path"]], "h5grove.encoders": [[4, 0, 1, "", "Response"], [4, 4, 1, "", "bin_encode"], [4, 4, 1, "", "csv_encode"], [4, 4, 1, "", "encode"], [4, 4, 1, "", "npy_encode"], [4, 4, 1, "", "orjson_default"], [4, 4, 1, "", "orjson_encode"], [4, 4, 1, "", "tiff_encode"]], "h5grove.encoders.Response": [[4, 2, 1, "", "content"], [4, 2, 1, "", "headers"]], "h5grove": [[3, 5, 0, "-", "fastapi_utils"], [3, 5, 0, "-", "flask_utils"], [3, 5, 0, "-", "tornado_utils"]], "h5grove.fastapi_utils": [[3, 4, 1, "", "get_attr"], [3, 4, 1, "", "get_data"], [3, 4, 1, "", "get_meta"], [3, 4, 1, "", "get_stats"], [3, 6, 1, "", "router"], [3, 6, 1, "", "settings"]], "h5grove.flask_utils": [[3, 6, 1, "", "BLUEPRINT"], [3, 6, 1, "", "URL_RULES"], [3, 4, 1, "", "attr_route"], [3, 4, 1, "", "data_route"], [3, 4, 1, "", "meta_route"], [3, 4, 1, "", "stats_route"]], "h5grove.tornado_utils": [[3, 0, 1, "", "AttributeHandler"], [3, 0, 1, "", "BaseHandler"], [3, 0, 1, "", "DataHandler"], [3, 0, 1, "", "MetadataHandler"], [3, 0, 1, "", "StatisticsHandler"], [3, 4, 1, "", "get_handlers"]], "h5grove.tornado_utils.BaseHandler": [[3, 1, 1, "", "initialize"], [3, 1, 1, "", "prepare"], [3, 1, 1, "", "write_error"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:attribute", "3": "py:property", "4": "py:function", "5": "py:module", "6": "py:data"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "attribute", "Python attribute"], "3": ["py", "property", "Python property"], "4": ["py", "function", "Python function"], "5": ["py", "module", "Python module"], "6": ["py", "data", "Python data"]}, "titleterms": {"h5grove": [0, 2, 4], "endpoint": 0, "api": 0, "exampl": 1, "implement": 1, "fastapi": [1, 3], "name": 1, "argument": 1, "flask": [1, 3], "tornado": [1, 3], "core": 2, "util": 2, "serv": 2, "hdf5": 2, "file": [2, 4], "content": [2, 4], "document": 2, "rational": 2, "instal": 2, "us": 2, "changelog": 2, "quick": 2, "access": 2, "integr": 3, "an": 3, "exist": 3, "applic": 3, "fastapi_util": 3, "refer": [3, 4], "flask_util": 3, "tornado_util": 3, "modul": 4, "creat": 4, "object": 4, "encod": 4, "gener": 4, "binari": 4, "format": 4}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 58}, "alltitles": {"h5grove endpoints API": [[0, "h5grove-endpoints-api"]], "Example implementations": [[1, "example-implementations"]], "FastAPI": [[1, "fastapi"], [3, "fastapi"]], "Named Arguments": [[1, "named-arguments"], [1, "named-arguments"], [1, "named-arguments"]], "Flask": [[1, "flask"], [3, "flask"]], "Tornado": [[1, "tornado"], [3, "tornado"]], "h5grove, core utilities to serve HDF5 file contents": [[2, "h5grove-core-utilities-to-serve-hdf5-file-contents"]], "Documentation": [[2, "documentation"]], "Rationale": [[2, "rationale"]], "Installation": [[2, "installation"]], "Using h5grove": [[2, "using-h5grove"]], "Changelog": [[2, "changelog"]], "Quick access": [[2, "quick-access"]], "Integration in an existing application": [[3, "integration-in-an-existing-application"]], "fastapi_utils reference": [[3, "module-h5grove.fastapi_utils"]], "flask_utils reference": [[3, "module-h5grove.flask_utils"]], "tornado_utils reference": [[3, "module-h5grove.tornado_utils"]], "h5grove reference": [[4, "h5grove-reference"]], "content module": [[4, "content-module"]], "Create a content object": [[4, "create-a-content-object"]], "Content object reference": [[4, "content-object-reference"]], "encoders module": [[4, "encoders-module"]], "General": [[4, "general"], [4, "id1"]], "Binary": [[4, "binary"]], "File formats": [[4, "file-formats"]]}, "indexentries": {"attributehandler (class in h5grove.tornado_utils)": [[3, "h5grove.tornado_utils.AttributeHandler"]], "blueprint (in module h5grove.flask_utils)": [[3, "h5grove.flask_utils.BLUEPRINT"]], "basehandler (class in h5grove.tornado_utils)": [[3, "h5grove.tornado_utils.BaseHandler"]], "datahandler (class in h5grove.tornado_utils)": [[3, "h5grove.tornado_utils.DataHandler"]], "metadatahandler (class in h5grove.tornado_utils)": [[3, "h5grove.tornado_utils.MetadataHandler"]], "statisticshandler (class in h5grove.tornado_utils)": [[3, "h5grove.tornado_utils.StatisticsHandler"]], "url_rules (in module h5grove.flask_utils)": [[3, "h5grove.flask_utils.URL_RULES"]], "attr_route() (in module h5grove.flask_utils)": [[3, "h5grove.flask_utils.attr_route"]], "data_route() (in module h5grove.flask_utils)": [[3, "h5grove.flask_utils.data_route"]], "get_attr() (in module h5grove.fastapi_utils)": [[3, "h5grove.fastapi_utils.get_attr"]], "get_data() (in module h5grove.fastapi_utils)": [[3, "h5grove.fastapi_utils.get_data"]], "get_handlers() (in module h5grove.tornado_utils)": [[3, "h5grove.tornado_utils.get_handlers"]], "get_meta() (in module h5grove.fastapi_utils)": [[3, "h5grove.fastapi_utils.get_meta"]], "get_stats() (in module h5grove.fastapi_utils)": [[3, "h5grove.fastapi_utils.get_stats"]], "h5grove.fastapi_utils": [[3, "module-h5grove.fastapi_utils"]], "h5grove.flask_utils": [[3, "module-h5grove.flask_utils"]], "h5grove.tornado_utils": [[3, "module-h5grove.tornado_utils"]], "initialize() (h5grove.tornado_utils.basehandler method)": [[3, "h5grove.tornado_utils.BaseHandler.initialize"]], "meta_route() (in module h5grove.flask_utils)": [[3, "h5grove.flask_utils.meta_route"]], "module": [[3, "module-h5grove.fastapi_utils"], [3, "module-h5grove.flask_utils"], [3, "module-h5grove.tornado_utils"]], "prepare() (h5grove.tornado_utils.basehandler method)": [[3, "h5grove.tornado_utils.BaseHandler.prepare"]], "router (in module h5grove.fastapi_utils)": [[3, "h5grove.fastapi_utils.router"]], "settings (in module h5grove.fastapi_utils)": [[3, "h5grove.fastapi_utils.settings"]], "stats_route() (in module h5grove.flask_utils)": [[3, "h5grove.flask_utils.stats_route"]], "write_error() (h5grove.tornado_utils.basehandler method)": [[3, "h5grove.tornado_utils.BaseHandler.write_error"]], "datasetcontent (class in h5grove.content)": [[4, "h5grove.content.DatasetContent"]], "externallinkcontent (class in h5grove.content)": [[4, "h5grove.content.ExternalLinkContent"]], "groupcontent (class in h5grove.content)": [[4, "h5grove.content.GroupContent"]], "response (class in h5grove.encoders)": [[4, "h5grove.encoders.Response"]], "softlinkcontent (class in h5grove.content)": [[4, "h5grove.content.SoftLinkContent"]], "attributes() (h5grove.content.datasetcontent method)": [[4, "h5grove.content.DatasetContent.attributes"]], "attributes() (h5grove.content.groupcontent method)": [[4, "h5grove.content.GroupContent.attributes"]], "bin_encode() (in module h5grove.encoders)": [[4, "h5grove.encoders.bin_encode"]], "content (h5grove.encoders.response attribute)": [[4, "h5grove.encoders.Response.content"]], "create_content() (in module h5grove.content)": [[4, "h5grove.content.create_content"]], "csv_encode() (in module h5grove.encoders)": [[4, "h5grove.encoders.csv_encode"]], "data() (h5grove.content.datasetcontent method)": [[4, "h5grove.content.DatasetContent.data"]], "data_stats() (h5grove.content.datasetcontent method)": [[4, "h5grove.content.DatasetContent.data_stats"]], "encode() (in module h5grove.encoders)": [[4, "h5grove.encoders.encode"]], "headers (h5grove.encoders.response attribute)": [[4, "h5grove.encoders.Response.headers"]], "kind (h5grove.content.datasetcontent attribute)": [[4, "h5grove.content.DatasetContent.kind"]], "kind (h5grove.content.externallinkcontent attribute)": [[4, "h5grove.content.ExternalLinkContent.kind"]], "kind (h5grove.content.groupcontent attribute)": [[4, "h5grove.content.GroupContent.kind"]], "kind (h5grove.content.softlinkcontent attribute)": [[4, "h5grove.content.SoftLinkContent.kind"]], "metadata() (h5grove.content.datasetcontent method)": [[4, "h5grove.content.DatasetContent.metadata"]], "metadata() (h5grove.content.externallinkcontent method)": [[4, "h5grove.content.ExternalLinkContent.metadata"]], "metadata() (h5grove.content.groupcontent method)": [[4, "h5grove.content.GroupContent.metadata"]], "metadata() (h5grove.content.softlinkcontent method)": [[4, "h5grove.content.SoftLinkContent.metadata"]], "name (h5grove.content.datasetcontent property)": [[4, "h5grove.content.DatasetContent.name"]], "name (h5grove.content.externallinkcontent property)": [[4, "h5grove.content.ExternalLinkContent.name"]], "name (h5grove.content.groupcontent property)": [[4, "h5grove.content.GroupContent.name"]], "name (h5grove.content.softlinkcontent property)": [[4, "h5grove.content.SoftLinkContent.name"]], "npy_encode() (in module h5grove.encoders)": [[4, "h5grove.encoders.npy_encode"]], "orjson_default() (in module h5grove.encoders)": [[4, "h5grove.encoders.orjson_default"]], "orjson_encode() (in module h5grove.encoders)": [[4, "h5grove.encoders.orjson_encode"]], "path (h5grove.content.datasetcontent property)": [[4, "h5grove.content.DatasetContent.path"]], "path (h5grove.content.externallinkcontent property)": [[4, "h5grove.content.ExternalLinkContent.path"]], "path (h5grove.content.groupcontent property)": [[4, "h5grove.content.GroupContent.path"]], "path (h5grove.content.softlinkcontent property)": [[4, "h5grove.content.SoftLinkContent.path"]], "target_file (h5grove.content.externallinkcontent property)": [[4, "h5grove.content.ExternalLinkContent.target_file"]], "target_path (h5grove.content.externallinkcontent property)": [[4, "h5grove.content.ExternalLinkContent.target_path"]], "target_path (h5grove.content.softlinkcontent property)": [[4, "h5grove.content.SoftLinkContent.target_path"]], "tiff_encode() (in module h5grove.encoders)": [[4, "h5grove.encoders.tiff_encode"]]}})