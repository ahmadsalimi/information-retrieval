# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: api/v1/mir.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10\x61pi/v1/mir.proto\x12\x02v1\"\x90\x01\n\x13Phase1SearchRequest\x12\"\n\x07\x64\x61taset\x18\x01 \x01(\x0e\x32\x11.v1.Phase1Dataset\x12\r\n\x05query\x18\x02 \x01(\t\x12\x18\n\x10max_result_count\x18\x03 \x01(\x05\x12\x16\n\x0eranking_method\x18\x04 \x01(\t\x12\x14\n\x0ctitle_weight\x18\x05 \x01(\x02\"M\n\x0bPhase1Paper\x12\x0e\n\x06\x64oc_id\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02\x12\r\n\x05title\x18\x03 \x01(\t\x12\x10\n\x08\x61\x62stract\x18\x04 \x01(\t\"N\n\x14Phase1SearchResponse\x12\x17\n\x0f\x63orrected_query\x18\x01 \x01(\t\x12\x1d\n\x04hits\x18\x02 \x03(\x0b\x32\x0f.v1.Phase1Paper\"~\n\x13Phase2SearchRequest\x12\r\n\x05query\x18\x01 \x01(\t\x12\x18\n\x10max_result_count\x18\x02 \x01(\x05\x12\x16\n\x0eranking_method\x18\x03 \x01(\t\x12\x14\n\x0ctitle_weight\x18\x04 \x01(\x02\x12\x10\n\x08\x63\x61tegory\x18\x05 \x01(\t\"p\n\x0bPhase2Paper\x12\x0e\n\x06\x64oc_id\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02\x12\r\n\x05title\x18\x03 \x01(\t\x12\x10\n\x08\x61\x62stract\x18\x04 \x01(\t\x12\x10\n\x08\x63\x61tegory\x18\x05 \x01(\t\x12\x0f\n\x07\x63luster\x18\x06 \x01(\t\"N\n\x14Phase2SearchResponse\x12\x17\n\x0f\x63orrected_query\x18\x01 \x01(\t\x12\x1d\n\x04hits\x18\x02 \x03(\x0b\x32\x0f.v1.Phase2Paper\"\x9f\x02\n\x13Phase3SearchRequest\x12\r\n\x05query\x18\x01 \x01(\t\x12\x18\n\x10max_result_count\x18\x02 \x01(\x05\x12\x16\n\x0eranking_method\x18\x03 \x01(\t\x12\x14\n\x0ctitle_weight\x18\x04 \x01(\x02\x12\x1e\n\x16personalization_weight\x18\x05 \x01(\x02\x12S\n\x17preference_by_professor\x18\x06 \x03(\x0b\x32\x32.v1.Phase3SearchRequest.PreferenceByProfessorEntry\x1a<\n\x1aPreferenceByProfessorEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\"\x93\x01\n\x0bPhase3Paper\x12\x0e\n\x06\x64oc_id\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02\x12\x14\n\x0csearch_score\x18\x03 \x01(\x02\x12\x16\n\x0epagerank_score\x18\x04 \x01(\x02\x12\r\n\x05title\x18\x05 \x01(\t\x12\x10\n\x08\x61\x62stract\x18\x06 \x01(\t\x12\x16\n\x0erelated_topics\x18\x07 \x03(\t\"N\n\x14Phase3SearchResponse\x12\x17\n\x0f\x63orrected_query\x18\x01 \x01(\t\x12\x1d\n\x04hits\x18\x02 \x03(\x0b\x32\x0f.v1.Phase3Paper\"8\n\x1aRandomSimilarPapersRequest\x12\x1a\n\x12number_of_similars\x18\x01 \x01(\x05\"c\n\x0cSimilarPaper\x12\x0e\n\x06\x64oc_id\x18\x01 \x01(\t\x12\r\n\x05title\x18\x02 \x01(\t\x12\x10\n\x08\x63\x61tegory\x18\x03 \x01(\t\x12\x10\n\x08\x61\x62stract\x18\x04 \x01(\t\x12\x10\n\x08\x64istance\x18\x05 \x01(\x02\"n\n\x1bRandomSimilarPapersResponse\x12%\n\x0bquery_paper\x18\x01 \x01(\x0b\x32\x10.v1.SimilarPaper\x12(\n\x0esimilar_papers\x18\x02 \x03(\x0b\x32\x10.v1.SimilarPaper*(\n\rPhase1Dataset\x12\t\n\x05\x41iBio\x10\x00\x12\x0c\n\x08HwSystem\x10\x01\x32\xb8\x02\n\rSearchService\x12\x43\n\x0cPhase1Search\x12\x17.v1.Phase1SearchRequest\x1a\x18.v1.Phase1SearchResponse\"\x00\x12\x43\n\x0cPhase2Search\x12\x17.v1.Phase2SearchRequest\x1a\x18.v1.Phase2SearchResponse\"\x00\x12\x43\n\x0cPhase3Search\x12\x17.v1.Phase3SearchRequest\x1a\x18.v1.Phase3SearchResponse\"\x00\x12X\n\x13RandomSimilarPapers\x12\x1e.v1.RandomSimilarPapersRequest\x1a\x1f.v1.RandomSimilarPapersResponse\"\x00\x62\x06proto3')

_PHASE1DATASET = DESCRIPTOR.enum_types_by_name['Phase1Dataset']
Phase1Dataset = enum_type_wrapper.EnumTypeWrapper(_PHASE1DATASET)
AiBio = 0
HwSystem = 1


_PHASE1SEARCHREQUEST = DESCRIPTOR.message_types_by_name['Phase1SearchRequest']
_PHASE1PAPER = DESCRIPTOR.message_types_by_name['Phase1Paper']
_PHASE1SEARCHRESPONSE = DESCRIPTOR.message_types_by_name['Phase1SearchResponse']
_PHASE2SEARCHREQUEST = DESCRIPTOR.message_types_by_name['Phase2SearchRequest']
_PHASE2PAPER = DESCRIPTOR.message_types_by_name['Phase2Paper']
_PHASE2SEARCHRESPONSE = DESCRIPTOR.message_types_by_name['Phase2SearchResponse']
_PHASE3SEARCHREQUEST = DESCRIPTOR.message_types_by_name['Phase3SearchRequest']
_PHASE3SEARCHREQUEST_PREFERENCEBYPROFESSORENTRY = _PHASE3SEARCHREQUEST.nested_types_by_name['PreferenceByProfessorEntry']
_PHASE3PAPER = DESCRIPTOR.message_types_by_name['Phase3Paper']
_PHASE3SEARCHRESPONSE = DESCRIPTOR.message_types_by_name['Phase3SearchResponse']
_RANDOMSIMILARPAPERSREQUEST = DESCRIPTOR.message_types_by_name['RandomSimilarPapersRequest']
_SIMILARPAPER = DESCRIPTOR.message_types_by_name['SimilarPaper']
_RANDOMSIMILARPAPERSRESPONSE = DESCRIPTOR.message_types_by_name['RandomSimilarPapersResponse']
Phase1SearchRequest = _reflection.GeneratedProtocolMessageType('Phase1SearchRequest', (_message.Message,), {
  'DESCRIPTOR' : _PHASE1SEARCHREQUEST,
  '__module__' : 'api.v1.mir_pb2'
  # @@protoc_insertion_point(class_scope:v1.Phase1SearchRequest)
  })
_sym_db.RegisterMessage(Phase1SearchRequest)

Phase1Paper = _reflection.GeneratedProtocolMessageType('Phase1Paper', (_message.Message,), {
  'DESCRIPTOR' : _PHASE1PAPER,
  '__module__' : 'api.v1.mir_pb2'
  # @@protoc_insertion_point(class_scope:v1.Phase1Paper)
  })
_sym_db.RegisterMessage(Phase1Paper)

Phase1SearchResponse = _reflection.GeneratedProtocolMessageType('Phase1SearchResponse', (_message.Message,), {
  'DESCRIPTOR' : _PHASE1SEARCHRESPONSE,
  '__module__' : 'api.v1.mir_pb2'
  # @@protoc_insertion_point(class_scope:v1.Phase1SearchResponse)
  })
_sym_db.RegisterMessage(Phase1SearchResponse)

Phase2SearchRequest = _reflection.GeneratedProtocolMessageType('Phase2SearchRequest', (_message.Message,), {
  'DESCRIPTOR' : _PHASE2SEARCHREQUEST,
  '__module__' : 'api.v1.mir_pb2'
  # @@protoc_insertion_point(class_scope:v1.Phase2SearchRequest)
  })
_sym_db.RegisterMessage(Phase2SearchRequest)

Phase2Paper = _reflection.GeneratedProtocolMessageType('Phase2Paper', (_message.Message,), {
  'DESCRIPTOR' : _PHASE2PAPER,
  '__module__' : 'api.v1.mir_pb2'
  # @@protoc_insertion_point(class_scope:v1.Phase2Paper)
  })
_sym_db.RegisterMessage(Phase2Paper)

Phase2SearchResponse = _reflection.GeneratedProtocolMessageType('Phase2SearchResponse', (_message.Message,), {
  'DESCRIPTOR' : _PHASE2SEARCHRESPONSE,
  '__module__' : 'api.v1.mir_pb2'
  # @@protoc_insertion_point(class_scope:v1.Phase2SearchResponse)
  })
_sym_db.RegisterMessage(Phase2SearchResponse)

Phase3SearchRequest = _reflection.GeneratedProtocolMessageType('Phase3SearchRequest', (_message.Message,), {

  'PreferenceByProfessorEntry' : _reflection.GeneratedProtocolMessageType('PreferenceByProfessorEntry', (_message.Message,), {
    'DESCRIPTOR' : _PHASE3SEARCHREQUEST_PREFERENCEBYPROFESSORENTRY,
    '__module__' : 'api.v1.mir_pb2'
    # @@protoc_insertion_point(class_scope:v1.Phase3SearchRequest.PreferenceByProfessorEntry)
    })
  ,
  'DESCRIPTOR' : _PHASE3SEARCHREQUEST,
  '__module__' : 'api.v1.mir_pb2'
  # @@protoc_insertion_point(class_scope:v1.Phase3SearchRequest)
  })
_sym_db.RegisterMessage(Phase3SearchRequest)
_sym_db.RegisterMessage(Phase3SearchRequest.PreferenceByProfessorEntry)

Phase3Paper = _reflection.GeneratedProtocolMessageType('Phase3Paper', (_message.Message,), {
  'DESCRIPTOR' : _PHASE3PAPER,
  '__module__' : 'api.v1.mir_pb2'
  # @@protoc_insertion_point(class_scope:v1.Phase3Paper)
  })
_sym_db.RegisterMessage(Phase3Paper)

Phase3SearchResponse = _reflection.GeneratedProtocolMessageType('Phase3SearchResponse', (_message.Message,), {
  'DESCRIPTOR' : _PHASE3SEARCHRESPONSE,
  '__module__' : 'api.v1.mir_pb2'
  # @@protoc_insertion_point(class_scope:v1.Phase3SearchResponse)
  })
_sym_db.RegisterMessage(Phase3SearchResponse)

RandomSimilarPapersRequest = _reflection.GeneratedProtocolMessageType('RandomSimilarPapersRequest', (_message.Message,), {
  'DESCRIPTOR' : _RANDOMSIMILARPAPERSREQUEST,
  '__module__' : 'api.v1.mir_pb2'
  # @@protoc_insertion_point(class_scope:v1.RandomSimilarPapersRequest)
  })
_sym_db.RegisterMessage(RandomSimilarPapersRequest)

SimilarPaper = _reflection.GeneratedProtocolMessageType('SimilarPaper', (_message.Message,), {
  'DESCRIPTOR' : _SIMILARPAPER,
  '__module__' : 'api.v1.mir_pb2'
  # @@protoc_insertion_point(class_scope:v1.SimilarPaper)
  })
_sym_db.RegisterMessage(SimilarPaper)

RandomSimilarPapersResponse = _reflection.GeneratedProtocolMessageType('RandomSimilarPapersResponse', (_message.Message,), {
  'DESCRIPTOR' : _RANDOMSIMILARPAPERSRESPONSE,
  '__module__' : 'api.v1.mir_pb2'
  # @@protoc_insertion_point(class_scope:v1.RandomSimilarPapersResponse)
  })
_sym_db.RegisterMessage(RandomSimilarPapersResponse)

_SEARCHSERVICE = DESCRIPTOR.services_by_name['SearchService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PHASE3SEARCHREQUEST_PREFERENCEBYPROFESSORENTRY._options = None
  _PHASE3SEARCHREQUEST_PREFERENCEBYPROFESSORENTRY._serialized_options = b'8\001'
  _PHASE1DATASET._serialized_start=1443
  _PHASE1DATASET._serialized_end=1483
  _PHASE1SEARCHREQUEST._serialized_start=25
  _PHASE1SEARCHREQUEST._serialized_end=169
  _PHASE1PAPER._serialized_start=171
  _PHASE1PAPER._serialized_end=248
  _PHASE1SEARCHRESPONSE._serialized_start=250
  _PHASE1SEARCHRESPONSE._serialized_end=328
  _PHASE2SEARCHREQUEST._serialized_start=330
  _PHASE2SEARCHREQUEST._serialized_end=456
  _PHASE2PAPER._serialized_start=458
  _PHASE2PAPER._serialized_end=570
  _PHASE2SEARCHRESPONSE._serialized_start=572
  _PHASE2SEARCHRESPONSE._serialized_end=650
  _PHASE3SEARCHREQUEST._serialized_start=653
  _PHASE3SEARCHREQUEST._serialized_end=940
  _PHASE3SEARCHREQUEST_PREFERENCEBYPROFESSORENTRY._serialized_start=880
  _PHASE3SEARCHREQUEST_PREFERENCEBYPROFESSORENTRY._serialized_end=940
  _PHASE3PAPER._serialized_start=943
  _PHASE3PAPER._serialized_end=1090
  _PHASE3SEARCHRESPONSE._serialized_start=1092
  _PHASE3SEARCHRESPONSE._serialized_end=1170
  _RANDOMSIMILARPAPERSREQUEST._serialized_start=1172
  _RANDOMSIMILARPAPERSREQUEST._serialized_end=1228
  _SIMILARPAPER._serialized_start=1230
  _SIMILARPAPER._serialized_end=1329
  _RANDOMSIMILARPAPERSRESPONSE._serialized_start=1331
  _RANDOMSIMILARPAPERSRESPONSE._serialized_end=1441
  _SEARCHSERVICE._serialized_start=1486
  _SEARCHSERVICE._serialized_end=1798
# @@protoc_insertion_point(module_scope)
