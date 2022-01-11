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

import {InboundMessageType, Message, OutboundMessageType} from './types';

let _cellMessenger: MessengerImpl;

export class GlobalMessenger {
  private tx?: (cellId: string, eventType: string) => void;
  private rx?: (varNames: string[]) => Promise<any[]>;
  private messengers = new Map();

  // Colab will listen to the event from the `dom` to receive a message from TS.
  initForColab(dom: HTMLElement) {
    this.tx = (outputCellId: string, eventType: string) => {
      dom.dispatchEvent(new CustomEvent(eventType));
    };
    this.rx = async (dataGlobalVarNames: string[]) => {
      const data = [];
      const anyThis = globalThis as any;
      for (const varWithUrl of dataGlobalVarNames) {
        const url = anyThis[varWithUrl];
        const response = await fetch(url);
        const obj = await response.json();
        data.push(...obj);

        URL.revokeObjectURL(url);
        delete anyThis[varWithUrl];
      }
      return data;
    };
  }

  initForIPython(portNumber: number) {
    if (this.tx && this.rx) return;

    this.tx = (outputCellId: string, eventType: string) => {
      fetch(`${location.protocol}//${location.hostname}:${portNumber}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        mode: 'cors',
        body: JSON.stringify({eventType, cellId: outputCellId}),
      });
    };
    this.rx = async (dataGlobalVarNames) => {
      const data = [];
      const anyThis = globalThis as any;
      for (const varWithData of dataGlobalVarNames) {
        if (Array.isArray(anyThis[varWithData])) {
          data.push(...anyThis[varWithData]);
        }

        delete anyThis[varWithData];
      }
      return data;
    };
  }

  createMessengerForOutputcell(id: string) {
    _cellMessenger = new MessengerImpl((type: OutboundMessageType) =>
      this.sendMessage(id, type)
    );
    this.messengers.set(id, _cellMessenger);
  }

  private sendMessage(outputCellId: string, type: OutboundMessageType) {
    if (!this.tx) {
      throw new Error(
        'Cannot send a message before Notebook has bootstrapped the module'
      );
    }
    this.tx(outputCellId, type);
  }

  async onMessageFromPython(
    cellId: string,
    type: InboundMessageType,
    dataGlobalVarNames: string[],
    otherPayload: any
  ) {
    if (!this.rx) {
      throw new Error('');
    }
    const mainPayload = (await this.rx(dataGlobalVarNames)) ?? [];

    let message = null;
    switch (type) {
      case InboundMessageType.UPDATE: {
        message = {
          type,
          mainPayload,
          auxPayload: otherPayload,
        };
        break;
      }
      case InboundMessageType.METADATA: {
        message = {
          type,
          mainPayload,
          auxPayload: otherPayload,
        };
        break;
      }
      default:
        throw new RangeError(`Unknown message type: ${type}`);
    }

    if (message) {
      this.messengers.get(cellId).invokeCallbacks(message);
    }
  }
}

export interface Messenger {
  sendMessage(type: OutboundMessageType): void;

  addOnMessage(cb: (message: Message) => void): void;
}

class MessengerImpl implements Messenger {
  private readonly callbacks: Array<(message: Message) => void> = [];
  constructor(
    private readonly sendMessageImpl: (type: OutboundMessageType) => void
  ) {}

  sendMessage(type: OutboundMessageType) {
    this.sendMessageImpl(type);
  }

  addOnMessage(cb: (message: Message) => void) {
    this.callbacks.push(cb);
  }

  invokeCallbacks(message: Message) {
    for (const callback of this.callbacks) {
      callback(message);
    }
  }
}

export function getCellMessenger() {
  return _cellMessenger;
}
