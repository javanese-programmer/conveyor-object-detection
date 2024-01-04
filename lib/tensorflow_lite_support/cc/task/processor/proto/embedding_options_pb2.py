# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow_lite_support/cc/task/processor/proto/embedding_options.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow_lite_support/cc/task/processor/proto/embedding_options.proto',
  package='tflite.task.processor',
  syntax='proto2',
  serialized_options=b'\n(org.tensorflow.lite.task.processor.protoP\001',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\nGtensorflow_lite_support/cc/task/processor/proto/embedding_options.proto\x12\x15tflite.task.processor\":\n\x10\x45mbeddingOptions\x12\x14\n\x0cl2_normalize\x18\x01 \x01(\x08\x12\x10\n\x08quantize\x18\x02 \x01(\x08\x42,\n(org.tensorflow.lite.task.processor.protoP\x01'
)




_EMBEDDINGOPTIONS = _descriptor.Descriptor(
  name='EmbeddingOptions',
  full_name='tflite.task.processor.EmbeddingOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='l2_normalize', full_name='tflite.task.processor.EmbeddingOptions.l2_normalize', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='quantize', full_name='tflite.task.processor.EmbeddingOptions.quantize', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=98,
  serialized_end=156,
)

DESCRIPTOR.message_types_by_name['EmbeddingOptions'] = _EMBEDDINGOPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

EmbeddingOptions = _reflection.GeneratedProtocolMessageType('EmbeddingOptions', (_message.Message,), {
  'DESCRIPTOR' : _EMBEDDINGOPTIONS,
  '__module__' : 'tensorflow_lite_support.cc.task.processor.proto.embedding_options_pb2'
  # @@protoc_insertion_point(class_scope:tflite.task.processor.EmbeddingOptions)
  })
_sym_db.RegisterMessage(EmbeddingOptions)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
