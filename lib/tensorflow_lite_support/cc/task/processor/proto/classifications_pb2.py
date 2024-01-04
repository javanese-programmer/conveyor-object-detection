# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow_lite_support/cc/task/processor/proto/classifications.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow_lite_support.cc.task.processor.proto import class_pb2 as tensorflow__lite__support_dot_cc_dot_task_dot_processor_dot_proto_dot_class__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow_lite_support/cc/task/processor/proto/classifications.proto',
  package='tflite.task.processor',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\nEtensorflow_lite_support/cc/task/processor/proto/classifications.proto\x12\x15tflite.task.processor\x1a;tensorflow_lite_support/cc/task/processor/proto/class.proto\"g\n\x0f\x43lassifications\x12-\n\x07\x63lasses\x18\x01 \x03(\x0b\x32\x1c.tflite.task.processor.Class\x12\x12\n\nhead_index\x18\x02 \x01(\x05\x12\x11\n\thead_name\x18\x03 \x01(\t\"W\n\x14\x43lassificationResult\x12?\n\x0f\x63lassifications\x18\x01 \x03(\x0b\x32&.tflite.task.processor.Classifications'
  ,
  dependencies=[tensorflow__lite__support_dot_cc_dot_task_dot_processor_dot_proto_dot_class__pb2.DESCRIPTOR,])




_CLASSIFICATIONS = _descriptor.Descriptor(
  name='Classifications',
  full_name='tflite.task.processor.Classifications',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='classes', full_name='tflite.task.processor.Classifications.classes', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='head_index', full_name='tflite.task.processor.Classifications.head_index', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='head_name', full_name='tflite.task.processor.Classifications.head_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=157,
  serialized_end=260,
)


_CLASSIFICATIONRESULT = _descriptor.Descriptor(
  name='ClassificationResult',
  full_name='tflite.task.processor.ClassificationResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='classifications', full_name='tflite.task.processor.ClassificationResult.classifications', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=262,
  serialized_end=349,
)

_CLASSIFICATIONS.fields_by_name['classes'].message_type = tensorflow__lite__support_dot_cc_dot_task_dot_processor_dot_proto_dot_class__pb2._CLASS
_CLASSIFICATIONRESULT.fields_by_name['classifications'].message_type = _CLASSIFICATIONS
DESCRIPTOR.message_types_by_name['Classifications'] = _CLASSIFICATIONS
DESCRIPTOR.message_types_by_name['ClassificationResult'] = _CLASSIFICATIONRESULT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Classifications = _reflection.GeneratedProtocolMessageType('Classifications', (_message.Message,), {
  'DESCRIPTOR' : _CLASSIFICATIONS,
  '__module__' : 'tensorflow_lite_support.cc.task.processor.proto.classifications_pb2'
  # @@protoc_insertion_point(class_scope:tflite.task.processor.Classifications)
  })
_sym_db.RegisterMessage(Classifications)

ClassificationResult = _reflection.GeneratedProtocolMessageType('ClassificationResult', (_message.Message,), {
  'DESCRIPTOR' : _CLASSIFICATIONRESULT,
  '__module__' : 'tensorflow_lite_support.cc.task.processor.proto.classifications_pb2'
  # @@protoc_insertion_point(class_scope:tflite.task.processor.ClassificationResult)
  })
_sym_db.RegisterMessage(ClassificationResult)


# @@protoc_insertion_point(module_scope)
