/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

export enum OutboundMessageType {
  STOP = 'stop',
}

export enum InboundMessageType {
  UPDATE = 'update',
  METADATA = 'meta',
}

interface UpdatePayload {
  step: number;
  maxStep: number;
}

interface MetaPayload {
  algo: string;
}

export interface UpdateMessage {
  type: InboundMessageType.UPDATE;
  mainPayload: Point[];
  auxPayload: UpdatePayload;
}

export interface MetadataMessage {
  type: InboundMessageType.METADATA;
  mainPayload: Metadata[];
  auxPayload: MetaPayload;
}

export type Message = UpdateMessage | MetadataMessage;

// [x, y] or [x, y, z].
export type Point = [number, number] | [number, number, number];

// Metadata for each embedding point. It is static across dim reduction step.
export interface Metadata {
  label: string;
  // Data URI.
  imageLabel?: string;
  color: number;
}

