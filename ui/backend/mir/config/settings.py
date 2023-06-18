import pkg_resources

from mir import __version__
from mir.api.v1 import mir_pb2_grpc as pb2_grpc
from mir.servicer.v1.mir import SearchService

PROJECT_PROG = pkg_resources.get_distribution('mir').project_name
PROJECT_VERSION = __version__
PROJECT_DESCRIPTION = 'MIR gRPC service'

SERVICER_ADDER = pb2_grpc.add_SearchServiceServicer_to_server
SERVICE = SearchService
