# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from mir.api.v1 import mir_pb2 as mir_dot_api_dot_v1_dot_mir__pb2


class SearchServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Phase1Search = channel.unary_unary(
                '/v1.SearchService/Phase1Search',
                request_serializer=mir_dot_api_dot_v1_dot_mir__pb2.Phase1SearchRequest.SerializeToString,
                response_deserializer=mir_dot_api_dot_v1_dot_mir__pb2.Phase1SearchResponse.FromString,
                )
        self.Phase2Search = channel.unary_unary(
                '/v1.SearchService/Phase2Search',
                request_serializer=mir_dot_api_dot_v1_dot_mir__pb2.Phase2SearchRequest.SerializeToString,
                response_deserializer=mir_dot_api_dot_v1_dot_mir__pb2.Phase2SearchResponse.FromString,
                )
        self.Phase3Search = channel.unary_unary(
                '/v1.SearchService/Phase3Search',
                request_serializer=mir_dot_api_dot_v1_dot_mir__pb2.Phase3SearchRequest.SerializeToString,
                response_deserializer=mir_dot_api_dot_v1_dot_mir__pb2.Phase3SearchResponse.FromString,
                )
        self.RandomSimilarPapers = channel.unary_unary(
                '/v1.SearchService/RandomSimilarPapers',
                request_serializer=mir_dot_api_dot_v1_dot_mir__pb2.RandomSimilarPapersRequest.SerializeToString,
                response_deserializer=mir_dot_api_dot_v1_dot_mir__pb2.RandomSimilarPapersResponse.FromString,
                )


class SearchServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Phase1Search(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Phase2Search(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Phase3Search(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RandomSimilarPapers(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SearchServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Phase1Search': grpc.unary_unary_rpc_method_handler(
                    servicer.Phase1Search,
                    request_deserializer=mir_dot_api_dot_v1_dot_mir__pb2.Phase1SearchRequest.FromString,
                    response_serializer=mir_dot_api_dot_v1_dot_mir__pb2.Phase1SearchResponse.SerializeToString,
            ),
            'Phase2Search': grpc.unary_unary_rpc_method_handler(
                    servicer.Phase2Search,
                    request_deserializer=mir_dot_api_dot_v1_dot_mir__pb2.Phase2SearchRequest.FromString,
                    response_serializer=mir_dot_api_dot_v1_dot_mir__pb2.Phase2SearchResponse.SerializeToString,
            ),
            'Phase3Search': grpc.unary_unary_rpc_method_handler(
                    servicer.Phase3Search,
                    request_deserializer=mir_dot_api_dot_v1_dot_mir__pb2.Phase3SearchRequest.FromString,
                    response_serializer=mir_dot_api_dot_v1_dot_mir__pb2.Phase3SearchResponse.SerializeToString,
            ),
            'RandomSimilarPapers': grpc.unary_unary_rpc_method_handler(
                    servicer.RandomSimilarPapers,
                    request_deserializer=mir_dot_api_dot_v1_dot_mir__pb2.RandomSimilarPapersRequest.FromString,
                    response_serializer=mir_dot_api_dot_v1_dot_mir__pb2.RandomSimilarPapersResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'v1.SearchService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class SearchService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Phase1Search(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/v1.SearchService/Phase1Search',
            mir_dot_api_dot_v1_dot_mir__pb2.Phase1SearchRequest.SerializeToString,
            mir_dot_api_dot_v1_dot_mir__pb2.Phase1SearchResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Phase2Search(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/v1.SearchService/Phase2Search',
            mir_dot_api_dot_v1_dot_mir__pb2.Phase2SearchRequest.SerializeToString,
            mir_dot_api_dot_v1_dot_mir__pb2.Phase2SearchResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Phase3Search(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/v1.SearchService/Phase3Search',
            mir_dot_api_dot_v1_dot_mir__pb2.Phase3SearchRequest.SerializeToString,
            mir_dot_api_dot_v1_dot_mir__pb2.Phase3SearchResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RandomSimilarPapers(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/v1.SearchService/RandomSimilarPapers',
            mir_dot_api_dot_v1_dot_mir__pb2.RandomSimilarPapersRequest.SerializeToString,
            mir_dot_api_dot_v1_dot_mir__pb2.RandomSimilarPapersResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
