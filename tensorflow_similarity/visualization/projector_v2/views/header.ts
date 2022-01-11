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

import {Messenger} from '../lib/ipc';
import {createElement, updateStyles} from '../lib/renderer';
import {InboundMessageType, Message, OutboundMessageType} from '../lib/types';

export class Header {
  private stepTime: number = Date.now();
  private readonly numberFormatter = new Intl.NumberFormat();
  private readonly rtFormatter = new (Intl as any).RelativeTimeFormat();
  private readonly step = createElement('div', null, ['Unknown / Unknown']);
  private readonly runningTime = updateStyles(createElement('div', null), {
    fontSize: '0.85em',
  });

  private timeoutId: number = -1;
  private readonly container = updateStyles(
    createElement('div', null, [
      this.step,
      this.runningTime,
      // spacer.
      createElement('div'),
      createElement(
        'button',
        {
          onPassiveClick: () => {
            this.messenger.sendMessage(OutboundMessageType.STOP);
          },
        },
        ['Stop']
      ),
    ]),
    {
      alignItems: 'center',
      backgroundColor: '#333e',
      color: '#fff',
      display: 'grid',
      gridTemplateColumns: 'auto auto auto 1fr auto',
      columnGap: '10px',
      height: '30px',
      padding: '0 10px',
    }
  );

  constructor(private readonly messenger: Messenger) {
    messenger.addOnMessage((message: Message) => {
      switch (message.type) {
        case InboundMessageType.UPDATE: {
          this.updateSteps(message.auxPayload.step, message.auxPayload.maxStep);
          break;
        }
      }
    });
    this.updateRunningTime();
  }

  private updateRunningTime() {
    this.timeoutId = setTimeout(() => {
      let diff = (Date.now() - this.stepTime) / 1000;
      let unit = 'second';
      if (diff > 60) {
        diff /= 60;
        unit = 'minute';
      }
      if (diff > 60) {
        diff /= 60;
        unit = 'hour';
      }
      if (diff > 24) {
        diff /= 24;
        unit = 'day';
      }
      this.runningTime.textContent = this.rtFormatter.format(
        -Math.round(diff),
        unit
      );
      this.updateRunningTime();
    }, 500);
  }

  getDomElement(): HTMLElement {
    return this.container;
  }

  private updateSteps(stepNumber: number, maxSteps: number): void {
    const formattedStepNumber = this.numberFormatter.format(stepNumber);
    const formattedMaxSteps = this.numberFormatter.format(maxSteps);
    this.step.textContent = `${formattedStepNumber} / ${formattedMaxSteps}`;

    if (stepNumber >= maxSteps) {
      clearTimeout(this.timeoutId);
      this.runningTime.textContent = 'Finished';
    } else {
      this.stepTime = Date.now();
      this.runningTime.textContent = 'moments ago';
    }
  }
}
